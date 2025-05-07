from contextlib import nullcontext
from typing import Callable, cast

import diffusers
import torch
from diffusers import MarigoldDepthPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler, LCMScheduler
from torch.optim import SGD, Adagrad, Adam, Optimizer
from transformers import CLIPTextModel, CLIPTokenizer

import utils

diffusers.utils.logging.disable_progress_bar()

MARIGOLD_CKPT_ORIGINAL = "prs-eth/marigold-v1-0"
MARIGOLD_CKPT_LCM = "prs-eth/marigold-lcm-v1-0"
VAE_CKPT_LIGHT = "madebyollin/taesd"
SUPPORTED_LOSS_FUNCS = ["l1", "l2", "edge", "smooth"]
EPSILON = 1e-7


def get_projection_fn(projection: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns the appropriate logarithmic function based on the specified projection method.

    This function is used to transform depth values into logarithmic space, which can
    improve accuracy for scenes with large depth ranges by giving more precision to
    closer objects and less precision to distant objects.

    Args:
        projection (str): The projection method to use. Supported values are:
            - "log": Natural logarithm (base e)
            - "log10": Base-10 logarithm
            - "linear": Identity function (no logarithmic transformation)

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: A function that applies the specified
        logarithmic transformation to a tensor of depth values.

    Raises:
        ValueError: If an unsupported projection method is provided.
    """  # noqa: E501
    if projection == "log":
        return torch.log
    elif projection == "log10":
        return torch.log10
    elif projection == "linear":
        return lambda x: x
    raise ValueError(f"Unknown projection method: {projection}")


def compute_affine_params(
    affines: torch.Tensor,
    guides: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes optimal affine transformation parameters for a batch of depth maps.

    This function calculates the scale and shift parameters needed to transform
    the network's output depth maps to align with sparse depth measurements.
    It performs a least-squares optimization to find the best affine parameters
    that minimize the difference between the transformed depth maps and the guide
    measurements at valid mask locations.

    The affine transformation is defined as:
        metric_depth = scale * affine_depth + shift

    The optimal parameters are computed using the following formulas:
        scale = cov(affine, guide) / var(affine)
        shift = mean(guide) - scale * mean(affine)

    where cov and var are computed only over valid mask locations.

    Args:
        affines (torch.Tensor): Batch of depth maps to be transformed, shape [N, 1, H, W].
        guides (torch.Tensor): Batch of target depth measurements, shape [N, 1, H, W].
        masks (torch.Tensor): Binary masks indicating valid measurements, shape [N, 1, H, W].

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - scales (torch.Tensor): Scale parameters for each sample in the batch, shape [N].
            - shifts (torch.Tensor): Shift parameters for each sample in the batch, shape [N].
    """  # noqa: E501

    N = affines.shape[0]
    # Flatten H, W dimensions
    affines_flat = affines.view(N, -1)  # [N, H*W]
    guides_flat = guides.view(N, -1)  # [N, H*W]
    masks_flat = masks.view(N, -1)  # [N, H*W]

    # Count valid points for each batch element
    num_valid = masks_flat.sum(dim=1, keepdim=True)  # [N, 1]

    # Check if any mask has no valid points
    if torch.any(num_valid == 0):
        raise ValueError("At least one mask in the batch has no valid points")

    # Use num_valid directly since we've verified no zeros exist
    num_valid_safe = num_valid

    # Calculate sums with mask applied (points outside mask become 0)
    affines_masked_sum = torch.sum(
        affines_flat * masks_flat, dim=1, keepdim=True
    )  # [N, 1]
    guides_masked_sum = torch.sum(
        guides_flat * masks_flat, dim=1, keepdim=True
    )  # [N, 1]

    # Calculate mean of masked elements
    affines_mean = affines_masked_sum / num_valid_safe  # [N, 1]
    guides_mean = guides_masked_sum / num_valid_safe  # [N, 1]

    # Center the masked elements
    affines_centered = (affines_flat - affines_mean) * masks_flat  # [N, H*W]
    guides_centered = (guides_flat - guides_mean) * masks_flat  # [N, H*W]

    # Calculate variance and covariance terms (considering only masked elements)
    vars_ = torch.sum(affines_centered.pow(2), dim=1, keepdim=True)  # [N, 1]
    covs = torch.sum(affines_centered * guides_centered, dim=1, keepdim=True)  # [N, 1]

    # Calculate scale and shift
    scales = covs / (vars_ + EPSILON)  # [N, 1]
    shifts = guides_mean - scales * affines_mean  # [N, 1]

    # Reshape from [N, 1] to [N] and return
    return scales.squeeze(1), shifts.squeeze(1)


def compute_loss(
    denses: torch.Tensor,
    sparses: torch.Tensor,
    masks: torch.Tensor,
    loss_funcs: list[str],
    images: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes a combined loss between dense depth predictions and sparse depth measurements.

    Args:
        denses (torch.Tensor): Predicted dense depth map tensor of shape [N, 1, H, W].
        sparses (torch.Tensor): Sparse depth measurements tensor of shape [N, 1, H, W], with zeros at unmeasured points.
        masks (torch.Tensor): Binary mask tensor of shape [N, 1, H, W] indicating valid sparse depth measurements.
        loss_funcs (list[str]): List of loss function names to apply. Supported values are:
                    - "l1": L1 loss between dense and sparse at measured points
                    - "l2": L2 loss between dense and sparse at measured points
                    - "edge": Edge-aware loss that compares gradients with image gradients
                    - "smooth": Smoothness loss that penalizes depth discontinuities
        images (torch.Tensor | None, optional): RGB or grayscale image tensor of shape [N, C, H, W] where C is 1 or 3.
               Required when using "edge" or "smooth" loss functions. Defaults to None.

    Returns:
        torch.Tensor: A tensor of shape [N] containing the total loss for each sample in the batch.

    Raises:
        ValueError: If loss_funcs is empty, contains an unsupported loss function,
                   or if images is not provided when required for edge/smooth losses.
    """  # noqa: E501
    if len(loss_funcs) == 0:
        raise ValueError("loss_funcs must contain at least one loss function")
    total = torch.zeros(denses.shape[0], device=denses.device)  # [N]

    for loss_func in loss_funcs:
        if loss_func == "l1":
            # Compute L1 loss per sample using masked operations
            l1_loss = torch.abs(denses - sparses)
            l1_loss = l1_loss * masks  # Apply mask
            # Sum over HW dimensions and divide by number of valid points per sample
            total += l1_loss.sum(dim=(1, 2, 3)) / masks.sum(dim=(1, 2, 3))

        elif loss_func == "l2":
            # Compute L2 loss per sample using masked operations
            l2_loss = (denses - sparses) ** 2
            l2_loss = l2_loss * masks  # Apply mask
            # Sum over HW dimensions and divide by number of valid points per sample
            total += l2_loss.sum(dim=(1, 2, 3)) / masks.sum(dim=(1, 2, 3))

        elif loss_func == "edge":
            if images is None:
                raise ValueError("image must be provided for edge loss")

            # Convert to grayscale if needed
            num_channels = images.shape[1]
            if num_channels == 3:
                gray_image = (
                    0.299 * images[:, 0:1]
                    + 0.587 * images[:, 1:2]
                    + 0.114 * images[:, 2:3]
                )  # [N, 1, H, W]
            elif num_channels == 1:
                gray_image = images
            else:
                raise ValueError(f"Image must have 1 or 3 channels, got {num_channels}")

            # Compute gradients for entire batch at once
            grad_pred_x = torch.abs(denses[:, :, :, :-1] - denses[:, :, :, 1:])
            grad_pred_y = torch.abs(denses[:, :, :-1, :] - denses[:, :, 1:, :])
            grad_gray_x = torch.abs(gray_image[:, :, :, :-1] - gray_image[:, :, :, 1:])
            grad_gray_y = torch.abs(gray_image[:, :, :-1, :] - gray_image[:, :, 1:, :])

            # Compute edge loss per sample using reduction over spatial dimensions
            edge_loss_x = torch.abs(grad_pred_x - grad_gray_x).mean(dim=(1, 2, 3))
            edge_loss_y = torch.abs(grad_pred_y - grad_gray_y).mean(dim=(1, 2, 3))
            total += edge_loss_x + edge_loss_y

        elif loss_func == "smooth":
            if images is None:
                raise ValueError("image must be provided for smooth loss")

            # Compute smoothness loss per sample using reduction over spatial dimensions
            loss_h = torch.abs(denses[:, :, :-1, :] - denses[:, :, 1:, :]).mean(
                dim=(1, 2, 3)
            )
            loss_w = torch.abs(denses[:, :, :, :-1] - denses[:, :, :, 1:]).mean(
                dim=(1, 2, 3)
            )
            total += loss_h + loss_w

        else:
            raise ValueError(f"Unknown loss function: {loss_func}")

    return total  # Returns tensor of shape [N]


class MarigoldDepthCompletionPipeline(MarigoldDepthPipeline):
    """
    Pipeline for depth completion using the Marigold model.

    Takes RGB image and sparse depth as input to produce dense depth maps.
    Uses diffusion model to refine depth predictions while preserving sparse
    depth constraints through latent optimization and affine transformation.
    Supports batch processing with optional temporal consistency.
    """  # noqa: E501

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: DDIMScheduler | LCMScheduler,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        prediction_type: str | None = None,
        scale_invariant: bool | None = True,
        shift_invariant: bool | None = True,
        default_denoising_steps: int | None = None,
        default_processing_resolution: int | None = None,
    ) -> None:
        super().__init__(
            unet,
            vae,
            scheduler,
            text_encoder,
            tokenizer,
            prediction_type,
            scale_invariant,
            shift_invariant,
            default_denoising_steps,
            default_processing_resolution,
        )

    def _affine_to_metric(
        self,
        affines: torch.Tensor,
        guides: torch.Tensor,
        masks: torch.Tensor,
        closed_form: bool = False,
        affine_params: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Transforms affine depth maps to metric depth maps using either learned or closed-form parameters.

        This method applies an affine transformation (scale and shift) to convert the network's
        output depth maps into properly scaled metric depth maps that align with sparse depth
        measurements. The transformation can be applied in two ways:

        1. Using learned parameters (when closed_form=False): Applies scale and shift parameters
           that are optimized during the diffusion process.
        2. Using closed-form solution (when closed_form=True): Computes optimal scale and shift
           parameters analytically for each sample in the batch.

        Args:
            affines (torch.Tensor): Batch of affine depth maps with shape [N, 1, H, W].
            guides (torch.Tensor): Batch of normalized sparse depth measurements with shape [N, 1, H, W].
            masks (torch.Tensor): Binary masks indicating valid sparse measurements with shape [N, 1, H, W].
            closed_form (bool, optional): Whether to use closed-form solution for affine parameters.
                Defaults to False.
            affine_params (tuple[torch.Tensor, torch.Tensor] | None, optional): Tuple of (scales, shifts)
                parameters when not using closed-form solution. Each tensor has shape [N, 1, 1, 1].
                Required when closed_form=False. Defaults to None.

        Returns:
            torch.Tensor: Batch of metric depth maps with shape [N, 1, H, W].

        Raises:
            ValueError: If affine_params is None when closed_form is False.
        """  # noqa: E501
        if not closed_form and affine_params is None:
            raise ValueError("affine_params must be provided when closed_form is False")
        N = affines.shape[0]
        if not closed_form:
            assert affine_params is not None
            scales, shifts = affine_params
            mins, maxs = utils.masked_minmax(
                guides.view(N, -1), masks.view(N, -1), dim=-1
            )
            mins = mins.view(N, 1, 1, 1)
            maxs = maxs.view(N, 1, 1, 1)
            return (scales**2) * (maxs - mins) * affines + (shifts**2) * mins
        else:
            scales, shifts = compute_affine_params(affines, guides, masks)
            scales = scales.view(N, 1, 1, 1)
            shifts = shifts.view(N, 1, 1, 1)
            return (scales * affines) + shifts

    def _latent_to_affine(
        self,
        latents: torch.Tensor,
        orig_res: tuple[int, int],
        padding: tuple[int, int],
        interp_mode: str = "bilinear",
    ) -> torch.Tensor:
        """
        Converts latent representations to affine depth maps in the original image resolution.

        This method transforms the latent representations from the diffusion model into
        affine depth maps through the following steps:
        1. Decodes the latent vectors into initial depth predictions
        2. Removes padding that was added during preprocessing
        3. Resizes the depth maps to the original image resolution

        Args:
            latents (torch.Tensor): Batch of latent representations with shape [N, 4, EH, EW],
                where EH and EW are the encoded height and width.
            orig_res (tuple[int, int]): Original resolution (height, width) of the input images.
            padding (tuple[int, int]): Padding values (top, left) that were applied during preprocessing.
            interp_mode (str, optional): Interpolation mode for resizing. Options include
                "bilinear", "bicubic", etc. Defaults to "bilinear".

        Returns:
            torch.Tensor: Batch of affine depth maps with shape [N, 1, H, W], where H and W
                are the original image dimensions.
        """  # noqa: E501
        affine = self.decode_prediction(latents)  # [N, 1, PPH, PPW]
        affine = self.image_processor.unpad_image(affine, padding)
        affine_resized = self.image_processor.resize_antialias(
            affine, orig_res, interp_mode
        )  # [N, 1, H, W]
        return affine_resized

    def __call__(
        self,
        imgs: torch.Tensor,
        sparses: torch.Tensor,
        max_depth: float,
        min_depth: float = 0.0,
        projection: str = "linear",  # "linear", "log", "log10"
        inv: bool = False,
        norm: str = "minmax",
        percentile: tuple[float, float] = (0.01, 0.99),
        pred_latents_prev: torch.Tensor | None = None,
        beta: float = 0.9,
        steps: int = 50,
        resolution: int = 768,
        closed_form: bool | None = None,
        opt: str = "adam",
        lr: tuple[float, float] | None = None,
        kld: bool = False,
        kld_weight: float = 0.1,
        kld_mode: str = "simple",
        interp_mode: str = "bilinear",
        loss_funcs: list[str] | None = None,
        seed: int = 2024,
        train_latents: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Executes depth completion on a batch of RGB images using sparse depth measurements.

        This function implements the primary depth completion algorithm through a diffusion-based approach.
        It refines depth predictions iteratively by optimizing latent representations via a denoising process
        guided by sparse depth measurements.

        Args:
            imgs (torch.Tensor): A batch of RGB images with dimensions [N, C, H, W].
                These are raw images, not normalized to the [0, 1] range.
            sparses (torch.Tensor): A batch of sparse depth maps with dimensions [N, 1, H, W].
                These maps should have zeros at missing positions and positive values at measurement points.
                The depth values are raw and not normalized.
            max_depth (float): The maximum depth value for normalization.
            min_depth (float, optional): The minimum depth value for normalization. Defaults to 0.0.
            projection (str, optional): The method for projecting depth values.
                Options include "linear", "log", or "log10". Using "log" or "log10" transforms depth values
                to log space before processing, which can enhance accuracy for scenes with large depth ranges.
                Defaults to "linear".
            inv (bool, optional): Indicates whether to apply inverse projection (1/depth).
                When set to True, the model operates with inverse depth (disparity), which can enhance
                accuracy for distant objects. Defaults to False.
            norm (str, optional): The normalization method for input sparse depth maps.
                Options include "const", "minmax", or "percentile". Defaults to "minmax".
                Note: When projection="log" or "log10" or inv=True, norm="const" is not allowed.
            percentile (tuple[float, float], optional): The percentile values (min, max) used for depth normalization
                when norm="percentile". Values should be in the range [0, 1]. For example, (0.01, 0.99) means
                the depth range is determined by the 1st and 99th percentiles of the sparse depth values.
                This helps exclude outliers when normalizing depth. Defaults to (0.01, 0.99).
                This option is valid only when norm="percentile".
            pred_latents_prev (torch.Tensor | None, optional): Previous prediction latents
                with dimensions [N, 4, EH, EW] from a prior frame or iteration.
                This enables temporal consistency when processing video sequences. Defaults to None.
            beta (float, optional): The momentum factor for prediction latents between frames.
                Must be within the range (0, 1). Higher values give more weight to new latents,
                while lower values retain more information from previous frames. Defaults to 0.9.
                This option is ignored when pred_latents_prev is None.
            steps (int, optional): The number of denoising steps.
                Higher values yield better quality but result in slower inference. Defaults to 50.
            resolution (int, optional): The resolution for internal processing.
                Higher values yield better quality but consume more memory. Defaults to 768.
            closed_form (bool | None, optional): Whether to use closed-form solution for affine parameters.
                When True, computes optimal affine parameters analytically rather than through optimization.
                If None, it will be set to the opposite of train_latents. When train_latents=False,
                closed_form must be True. Defaults to None.
            opt (str, optional): The optimizer to use for latent optimization.
                Options include "adam", "sgd", or "adagrad". Defaults to "adam".
                This option is ignored when train_latents=False.
            lr (tuple[float, float] | None, optional): Learning rates for (latent, scaling).
                If None, defaults to (0.05, 0.005).
                This option is ignored when train_latents=False.
            kld (bool, optional): Indicates whether to apply a KL divergence penalty to
                keep prediction latents close to N(0,1). Defaults to False.
                This option is ignored when train_latents=False.
            kld_weight (float, optional): The weight for the KL divergence penalty.
                Used only when kld is True. Defaults to 0.1.
                This option is ignored when train_latents=False.
            kld_mode (str, optional): The KL divergence mode. Options include:
                - "simple": Uses a simplified penalty based on the squared L2 norm of latents.
                  This is the fastest but least accurate approximation of KL divergence.
                - "strict": Computes the proper forward KL divergence between the latent distribution and N(0,1).
                  This is more accurate but slightly more computationally expensive.
                Defaults to "simple". This option is ignored when train_latents=False.
            interp_mode (str, optional): The interpolation mode for resizing.
                Options include "bilinear", "bicubic", etc. Defaults to "bilinear".
            loss_funcs (list[str] | None, optional): The loss functions to use.
                If None, defaults to ["l1", "l2"]. Supported options include
                "l1", "l2", "edge", and "smooth". When using "edge" or "smooth",
                the RGB image is used to guide depth discontinuities. Defaults to None.
                This option is ignored when train_latents=False.
            seed (int, optional): The random seed for initializing the diffusion process generator and ensuring reproducibility.
                Defaults to 2024.
            train_latents (bool, optional): Whether to optimize the latent representations during inference.
                When False, the model will use the closed-form solution for affine parameters without optimizing latents.
                Setting to False can speed up inference at the cost of potentially lower quality. Defaults to True.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - A dense depth prediction with dimensions [N, 1, H, W] in metric units (same as input sparse depth)
                - Prediction latents with dimensions [N, 4, EH, EW] that can be used for temporal consistency
                  in subsequent frames
        """  # noqa: E501

        # Check input shapes
        if (
            imgs.ndim != 4
            or sparses.ndim != 4
            or (imgs.shape[0] != sparses.shape[0])
            or (imgs.shape[-2:] != sparses.shape[-2:])
        ):
            raise ValueError(
                "Shape of image must be [N, C, H, W] and shape of sparse must be "
                f"[N, 1, H, W], but got image.shape: "
                f"{imgs.shape} and sparse.shape: {sparses.shape}"
            )
        N, _, H, W = imgs.shape
        EH = resolution * H // (8 * max(H, W))
        EW = resolution * W // (8 * max(H, W))
        if pred_latents_prev is not None:
            if pred_latents_prev.ndim != 4 or pred_latents_prev.shape != (N, 4, EH, EW):
                raise ValueError(
                    "Shape of pred_latents_prev must be [N, 4, EH, EW], but got "
                    f"{pred_latents_prev.shape}"
                )

        # Check closed_form and train_latents
        if closed_form is None:
            closed_form = not train_latents
        elif not closed_form and not train_latents:
            raise ValueError(
                "Closed form solution must be enabled when trainable "
                "latents are not used. Set closed_form=True when "
                "train_latents=False, or just leave closed_form=None. "
            )

        # Check if beta is in (0, 1)
        if not (0 < beta < 1):
            raise ValueError(f"beta must be in (0, 1), but got {beta}")

        # Check percentile lies in [0, 1]
        if norm == "percentile" and not all(0 <= p <= 1 for p in percentile):
            raise ValueError(f"percentile must be in [0, 1], but got {percentile}")

        # Check projection method
        if projection not in ["linear", "log", "log10"]:
            raise ValueError(f"Unknown projection method: {projection}")

        # Check if min_depth > 0 when projection is "log"
        if (projection in ["log", "log10"] or inv) and min_depth <= EPSILON:
            raise ValueError(
                f"min_depth must be > {EPSILON} when "
                f"projection is 'log' or 'log10' or inv is True, "
                f"but got {min_depth}"
            )

        # Set learning rates
        if lr is None:
            lr_latent = 0.05
            lr_scaling = 0.005
        else:
            lr_latent, lr_scaling = lr

        # Set loss functions
        if loss_funcs is None:
            loss_funcs = ["l1", "l2"]
        else:
            for func in loss_funcs:
                if func not in SUPPORTED_LOSS_FUNCS:
                    raise ValueError(f"Unknown loss function: {func}")

        # Preprocess inputs
        with torch.no_grad():
            # Create random generator
            generator = torch.Generator(device=self.device).manual_seed(seed)

            # Create empty text embedding if not created
            if self.empty_text_embedding is None:
                text_inputs = self.tokenizer(
                    "",
                    padding="do_not_pad",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                self.empty_text_embedding: torch.Tensor = self.text_encoder(
                    text_inputs.input_ids.to(self.device)
                )[0]

            # Tile empty text conditioning
            batch_empty_text_embedding = self.empty_text_embedding.repeat(N, 1, 1)

            # Create common prediction latents
            pred_latents_common = torch.randn(
                (N, 4, EH, EW),
                device=imgs.device,
                dtype=self.dtype,
                generator=generator,
            )  # [N, 4, EH, EW]

            # Preprocess input images
            imgs_resized, padding, orig_res = self.image_processor.preprocess(
                imgs,
                processing_resolution=resolution,
                device=self.device,
                dtype=self.dtype,
            )  # [N, C, PPH, PPW]
            orig_res = cast(tuple[int, int], orig_res)

            # Get latent encodings
            img_latents, _ = self.prepare_latents(
                imgs_resized, None, generator, 1, N
            )  # [N, 4, EH, EW], [N, 4, EH, EW]
            if pred_latents_prev is not None:
                pred_latents = (
                    beta * pred_latents_common + (1 - beta) * pred_latents_prev
                )
            else:
                pred_latents = pred_latents_common

            # Calculate min & max depth values for each sample in the batch
            masks = sparses > 0
            if norm == "minmax":
                min_depths, max_depths = utils.masked_minmax(
                    sparses.view(N, -1), masks.view(N, -1), dim=-1
                )
                min_depths = min_depths.view(N, 1, 1, 1)
                max_depths = max_depths.view(N, 1, 1, 1)
            elif norm == "percentile":
                p = torch.tensor(percentile, device=sparses.device)
                ranges = torch.stack(
                    [
                        torch.quantile(
                            s[m],
                            p,
                        )
                        for s, m in zip(sparses, masks, strict=True)
                    ]
                )  # [N, 2]
                min_depths = ranges[:, 0].view(N, 1, 1, 1)
                max_depths = ranges[:, 1].view(N, 1, 1, 1)
            elif norm == "const":
                min_depths = torch.full((N, 1, 1, 1), min_depth, device=sparses.device)
                max_depths = torch.full((N, 1, 1, 1), max_depth, device=sparses.device)
            else:
                raise ValueError(f"Unknown norm method: {norm}")

            # Clamp depth values to [min_depth, max_depth]
            sparses_clamped = sparses.clamp(min=min_depths, max=max_depths)
            if norm in ["minmax", "percentile"]:
                min_depths = min_depths.clamp(min=min_depth)
                max_depths = max_depths.clamp(max=max_depth)

            # Project depth values to specified space
            proj_fn = get_projection_fn(projection)
            min_depths_proj = proj_fn(min_depths)
            max_depths_proj = proj_fn(max_depths)
            sparses_clamped_proj = proj_fn(sparses_clamped)

            # Inverse projected depth values if necessary
            if inv:
                min_depths_proj, max_depths_proj = (
                    1 / max_depths_proj,
                    1 / min_depths_proj,
                )
                sparses_clamped_proj = 1 / sparses_clamped_proj

            # Normalize sparse depth maps to [0, 1]
            sparses_normed = (sparses_clamped_proj - min_depths_proj) / (
                max_depths_proj - min_depths_proj
            )

        # Set current prediction latents as trainable params if train_latents is True
        if train_latents:
            pred_latents = torch.nn.Parameter(pred_latents)

        # Set scaling params
        affine_params = (
            (
                # scale
                torch.nn.Parameter(torch.ones(N, 1, 1, 1, device=self.device)),
                # shift
                torch.nn.Parameter(torch.zeros(N, 1, 1, 1, device=self.device)),
            )
            if not closed_form and train_latents
            else None
        )

        # Set up optimizer
        optimizer: Optimizer | None = None
        if train_latents:
            param_groups = [
                {"params": [pred_latents], "lr": lr_latent},
            ]
            if affine_params is not None:
                param_groups.append({"params": list(affine_params), "lr": lr_scaling})
            if opt == "adam":
                optimizer = Adam(param_groups)
            elif opt == "sgd":
                optimizer = SGD(param_groups)
            elif opt == "adagrad":
                optimizer = Adagrad(param_groups)
            else:
                raise ValueError(f"Unknown optimizer: {opt}")

        # Denoising loop
        # NOTE: grad calculation is needed when train_latents is True
        context = nullcontext() if train_latents else torch.no_grad()
        with context:
            self.scheduler.set_timesteps(steps, device=self.device)
            for t in self.scheduler.timesteps:
                if optimizer is not None:
                    # Clear gradients
                    optimizer.zero_grad()

                # Forward pass through the U-Net
                latents = torch.cat(
                    [img_latents, pred_latents], dim=1
                )  # [N, 8, EH, EW]
                pred_noises: torch.Tensor = self.unet(
                    latents,
                    t,
                    encoder_hidden_states=batch_empty_text_embedding,
                    return_dict=False,
                )[
                    0
                ]  # [N, 4, EH, EW]

                if optimizer is not None:
                    # Compute noise to later rescale the depth latent gradient
                    with torch.no_grad():
                        a_prod_t = cast(float, self.scheduler.alphas_cumprod[t])
                        b_prod_t = 1 - a_prod_t
                        pred_epsilons = (a_prod_t**0.5) * pred_noises + (
                            b_prod_t**0.5
                        ) * pred_latents  # [N, 4, EH, EW]

                    # Preview the final output depth with Tweedie's formula
                    previews = cast(
                        torch.Tensor,
                        self.scheduler.step(
                            pred_noises, t, pred_latents, generator=generator
                        ).pred_original_sample,
                    )  # [N, 4, EH, EW]

                    # Predict affine form of dense depth maps
                    denses_affine = self._latent_to_affine(
                        previews,
                        orig_res,
                        padding,
                        interp_mode=interp_mode,
                    )  # [N, 1, H, W]

                    # Predict scaled & shifted dense depth maps
                    denses_normed = self._affine_to_metric(
                        denses_affine,
                        sparses_normed,
                        masks,
                        affine_params=affine_params,
                        closed_form=closed_form,
                    ).clamp(
                        min=0.0, max=1.0
                    )  # [N, 1, H, W]

                    # Convert depth space
                    if projection != "linear":
                        denses_normed = (
                            denses_normed * (max_depths - min_depths) + min_depths
                        )
                        denses_normed = get_projection_fn(projection)(denses_normed)
                        if inv:
                            denses_normed = 1 / denses_normed
                        denses_normed = (denses_normed - min_depths_proj) / (
                            max_depths_proj - min_depths_proj
                        )
                    elif inv:
                        denses_normed = (
                            denses_normed * (max_depths - min_depths) + min_depths
                        )
                        denses_normed = 1 / denses_normed
                        denses_normed = (denses_normed - min_depths_proj) / (
                            max_depths_proj - min_depths_proj
                        )

                    # Compute loss
                    losses = compute_loss(
                        denses_normed, sparses_normed, masks, loss_funcs, images=imgs
                    )

                    # NOTE: Add KL divergence penalty to keep
                    # the distribution of pred_latent close to N(0,1)
                    if kld:
                        kld_losses = utils.kld_stdnorm(
                            pred_latents, reduction="none", mode=kld_mode
                        ).reshape(N, 1, 1, 1)
                        losses = losses + kld_weight * kld_losses

                    # Backprop
                    losses.backward(torch.ones_like(losses))  # Preserve batch dimension

                    # NOTE: Scale grads of pred_latents by the norm of pred_epsilons
                    # for stable optimization
                    with torch.no_grad():
                        assert pred_latents.grad is not None
                        pred_epsilon_norms = torch.linalg.norm(
                            pred_epsilons.view(N, -1), dim=1
                        )  # [N]
                        pred_latent_grad_norms = torch.linalg.norm(
                            pred_latents.grad.view(N, -1), dim=1
                        )  # [N]
                        factors = pred_epsilon_norms / pred_latent_grad_norms.clamp(
                            min=EPSILON
                        )  # [N]
                        factors = factors.view(N, 1, 1, 1)  # [N, 1, 1, 1]
                        # Scaling
                        pred_latents.grad *= factors  # [N, 4, EH, EW]

                    # Backprop
                    optimizer.step()

                    # Update latent with regular denoising diffusion step
                    # NOTE: update only data, not grads
                    with torch.no_grad():
                        pred_latents.data = self.scheduler.step(
                            pred_noises, t, pred_latents, generator=generator
                        ).prev_sample
                else:
                    # Update latent with regular denoising diffusion step
                    pred_latents = self.scheduler.step(
                        pred_noises, t, pred_latents, generator=generator
                    ).prev_sample

        # Compute final dense depth maps
        with torch.no_grad():
            pred_latents_detached = pred_latents.detach()
            denses_affine = self._latent_to_affine(
                pred_latents_detached,
                orig_res,
                padding,
                interp_mode=interp_mode,
            )  # [N, 1, H, W]
            denses_normed = self._affine_to_metric(
                denses_affine,
                sparses_normed,
                masks,
                affine_params=affine_params,
                closed_form=closed_form,
            )  # [N, 1, H, W]
            # Decode
            denses_normed = denses_normed.clamp(min=0.0, max=1.0)
            denses = denses_normed * (max_depths - min_depths) + min_depths
        return denses, pred_latents_detached
