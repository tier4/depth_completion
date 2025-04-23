from typing import cast

import diffusers
import torch
from diffusers import MarigoldDepthPipeline

import utils

diffusers.utils.logging.disable_progress_bar()

MARIGOLD_CKPT_ORIGINAL = "prs-eth/marigold-v1-0"
MARIGOLD_CKPT_LCM = "prs-eth/marigold-lcm-v1-0"
VAE_CKPT_LIGHT = "madebyollin/taesd"
SUPPORTED_LOSS_FUNCS = ["l1", "l2", "edge", "smooth"]
EPSILON = 1e-7


def compute_loss(
    dense: torch.Tensor,
    sparse: torch.Tensor,
    loss_funcs: list[str],
    image: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes a combined loss between dense depth predictions and sparse depth measurements.

    Args:
        dense: Predicted dense depth map tensor of shape [N, 1, H, W].
        sparse: Sparse depth measurements tensor of shape [N, 1, H, W], with zeros at unmeasured points.
        loss_funcs: List of loss function names to apply. Supported values are:
                    - "l1": L1 loss between dense and sparse at measured points
                    - "l2": L2 loss between dense and sparse at measured points
                    - "edge": Edge-aware loss that compares gradients with image gradients
                    - "smooth": Smoothness loss that penalizes depth discontinuities
        image: Optional RGB or grayscale image tensor of shape [N, C, H, W] where C is 1 or 3.
               Required when using "edge" or "smooth" loss functions.

    Returns:
        A tensor of shape [N] containing the total loss for each sample in the batch.

    Raises:
        ValueError: If loss_funcs is empty, contains an unsupported loss function,
                   or if image is not provided when required for edge/smooth losses.
    """  # noqa: E501
    if len(loss_funcs) == 0:
        raise ValueError("loss_funcs must contain at least one loss function")
    total = torch.zeros(dense.shape[0], device=dense.device)  # [N]
    sparse_mask = sparse > 0

    for loss_func in loss_funcs:
        if loss_func == "l1":
            # Compute L1 loss per sample using masked operations
            l1_loss = torch.abs(dense - sparse)
            l1_loss = l1_loss * sparse_mask  # Apply mask
            # Sum over HW dimensions and divide by number of valid points per sample
            total += l1_loss.sum(dim=(1, 2, 3)) / sparse_mask.sum(dim=(1, 2, 3))

        elif loss_func == "l2":
            # Compute L2 loss per sample using masked operations
            l2_loss = (dense - sparse) ** 2
            l2_loss = l2_loss * sparse_mask  # Apply mask
            # Sum over HW dimensions and divide by number of valid points per sample
            total += l2_loss.sum(dim=(1, 2, 3)) / sparse_mask.sum(dim=(1, 2, 3))

        elif loss_func == "edge":
            if image is None:
                raise ValueError("image must be provided for edge loss")

            # Convert to grayscale if needed
            num_channels = image.shape[1]
            if num_channels == 3:
                gray_image = (
                    0.299 * image[:, 0:1]
                    + 0.587 * image[:, 1:2]
                    + 0.114 * image[:, 2:3]
                )  # [N, 1, H, W]
            elif num_channels == 1:
                gray_image = image
            else:
                raise ValueError(f"Image must have 1 or 3 channels, got {num_channels}")

            # Compute gradients for entire batch at once
            grad_pred_x = torch.abs(dense[:, :, :, :-1] - dense[:, :, :, 1:])
            grad_pred_y = torch.abs(dense[:, :, :-1, :] - dense[:, :, 1:, :])
            grad_gray_x = torch.abs(gray_image[:, :, :, :-1] - gray_image[:, :, :, 1:])
            grad_gray_y = torch.abs(gray_image[:, :, :-1, :] - gray_image[:, :, 1:, :])

            # Compute edge loss per sample using reduction over spatial dimensions
            edge_loss_x = torch.abs(grad_pred_x - grad_gray_x).mean(dim=(1, 2, 3))
            edge_loss_y = torch.abs(grad_pred_y - grad_gray_y).mean(dim=(1, 2, 3))
            total += edge_loss_x + edge_loss_y

        elif loss_func == "smooth":
            if image is None:
                raise ValueError("image must be provided for smooth loss")

            # Compute smoothness loss per sample using reduction over spatial dimensions
            loss_h = torch.abs(dense[:, :, :-1, :] - dense[:, :, 1:, :]).mean(
                dim=(1, 2, 3)
            )
            loss_w = torch.abs(dense[:, :, :, :-1] - dense[:, :, :, 1:]).mean(
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

    def _affine_to_metric(
        self,
        dense: torch.Tensor,  # [N, 1, H, W]
        scale: torch.Tensor,  # [N, 1, H, W] or [N, 1, 1, 1]
        shift: torch.Tensor,  # [N, 1, H, W] or [N, 1, 1, 1]
        sparse_range: torch.Tensor,  # [N, 1, 1, 1]
        sparse_min: torch.Tensor,  # [N, 1, 1, 1]
    ) -> torch.Tensor:
        """
        Converts the model's affine-invariant depth representation to metric depth values.

        This method applies an affine transformation to convert the normalized depth predictions
        from the model's internal representation to actual metric depth values that match
        the scale and range of the provided sparse depth measurements. The transformation
        uses learned scale and shift parameters to ensure the output depth values are
        properly calibrated to the input sparse measurements.

        Args:
            dense (torch.Tensor): Normalized depth predictions from the model with shape [N, 1, H, W].
            scale (torch.Tensor): Learned scaling factor with shape [N, 1, H, W] or [N, 1, 1, 1].
            shift (torch.Tensor): Learned shift factor with shape [N, 1, H, W] or [N, 1, 1, 1].
            sparse_range (torch.Tensor): Range (max-min) of sparse depth values with shape [N, 1, 1, 1].
            sparse_min (torch.Tensor): Minimum value of sparse depth with shape [N, 1, 1, 1].

        Returns:
            torch.Tensor: Calibrated metric depth values with shape [N, 1, H, W] that match
                          the scale of the input sparse depth measurements.
        """  # noqa: E501
        return (scale**2) * sparse_range * dense + (shift**2) * sparse_min

    def latent_to_dense(
        self,
        latent: torch.Tensor,  # [N, 4, EH, EW]
        orig_res: tuple[int, int],
        padding: tuple[int, int],
        affine_invariant: bool = False,
        affine_params: tuple[torch.Tensor, torch.Tensor] | None = None,
        sparse_range: tuple[torch.Tensor, torch.Tensor] | None = None,
        interp_mode: str = "bilinear",
    ) -> torch.Tensor:
        """
        Converts latent representation to dense depth map.

        This method decodes the latent representation from the diffusion model into a dense
        depth map, applies unpadding, resizes to the original resolution, and optionally
        applies affine transformation to match the scale of sparse depth measurements.

        Args:
            latent (torch.Tensor): Latent representation with shape [N, 4, EH, EW].
            orig_res (tuple[int, int]): Original resolution (H, W) to resize to.
            padding (tuple[int, int]): Padding values to remove.
            affine_invariant (bool, optional): Whether to apply affine transformation.
                When True, the output will be transformed to match the scale of sparse depth.
                Defaults to False.
            affine_params (tuple[torch.Tensor, torch.Tensor] | None, optional):
                Scale and shift parameters for affine transformation. Required when
                affine_invariant is True. Defaults to None.
            sparse_range (tuple[torch.Tensor, torch.Tensor] | None, optional):
                Min and max values of sparse depth. Required when affine_invariant is True.
                Defaults to None.
            interp_mode (str, optional): Interpolation mode for resizing.
                Options include "bilinear", "bicubic", etc. Defaults to "bilinear".

        Returns:
            torch.Tensor: Dense depth map with shape [N, 1, H, W].
        """  # noqa: E501
        if affine_invariant:
            if affine_params is None or sparse_range is None:
                raise ValueError(
                    "scaling and sparse_range must be "
                    "provided when affine_invariant is True"
                )
        decoded = self.decode_prediction(latent)  # [N, 1, PPH, PPW]
        decoded = self.image_processor.unpad_image(decoded, padding)
        decoded_resized = self.image_processor.resize_antialias(
            decoded, orig_res, interp_mode
        )  # [N, 1, H, W]
        if affine_invariant:
            assert affine_params is not None and sparse_range is not None
            scale, shift = affine_params
            sparse_min, sparse_max = sparse_range
            decoded_resized = self._affine_to_metric(
                decoded_resized, scale, shift, sparse_max - sparse_min, sparse_min
            )
        return decoded_resized  # [N, 1, H, W]

    def __call__(
        self,
        imgs: torch.Tensor,
        sparses: torch.Tensor,
        depth_range: tuple[float, float] | str = "minmax",
        pred_latents_prev: torch.Tensor | None = None,
        percentile: float = 0.05,
        steps: int = 50,
        resolution: int = 768,
        affine_invariant: bool = True,
        opt: str = "adam",
        lr: tuple[float, float] | None = None,
        beta: float = 0.9,
        kl_penalty: bool = False,
        kl_weight: float = 0.1,
        kl_mode: str = "simple-forward",
        interp_mode: str = "bilinear",
        loss_funcs: list[str] | None = None,
        seed: int = 2024,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform depth completion on a batch of RGB images using sparse depth measurements.

        This method implements the core depth completion algorithm using a diffusion-based approach.
        It iteratively refines depth predictions by optimizing latent representations through
        a denoising process guided by sparse depth measurements.

        Args:
            imgs (torch.Tensor): Batch of RGB images with shape [N, C, H, W].
                Raw images, not normalized to [0, 1].
            sparses (torch.Tensor): Batch of sparse depth maps with shape [N, 1, H, W].
                Should have zeros at missing positions and positive values at measurement points.
                Raw depth values, not normalized.
            depth_range (tuple[float, float] | str, optional): Range of depth values to use.
                If tuple, specifies (min_depth, max_depth) directly.
                If "minmax", uses min and max values from input sparse depth.
                If "percentile", uses depth values at specified percentiles to exclude outliers.
                Defaults to "minmax".
            pred_latents_prev (torch.Tensor | None, optional): Previous prediction latents
                with shape [N, 4, EH, EW] from a prior frame or iteration.
                Enables temporal consistency when processing video sequences. Defaults to None.
            percentile (float, optional): Percentile value for determining depth range.
                Only used when depth_range="percentile". Lower values (e.g., 0.05) exclude outliers
                by using the 5th and 95th percentiles. Defaults to 0.05.
            steps (int, optional): Number of denoising steps.
                Higher values give better quality but slower inference. Defaults to 50.
            resolution (int, optional): Resolution for internal processing.
                Higher values give better quality but use more memory. Defaults to 768.
            affine_invariant (bool, optional): Whether to use affine invariant depth completion.
                When True, the model applies affine transformations to handle arbitrary depth scales
                and shifts between the model's internal representation and the input sparse depth.
                This allows the model to work with different depth sensors and units without retraining.
                The model will automatically estimate the appropriate scale and shift parameters
                to align its predictions with the input sparse measurements. Defaults to True.
            opt (str, optional): Optimizer to use ("adam" or "sgd"). Defaults to "adam".
            lr (tuple[float, float] | None, optional): Learning rates for (latent, scaling).
                If None, defaults to (0.05, 0.005).
            beta (float, optional): Momentum factor for prediction latents between frames.
                Must be in range [0, 1]. Higher values give more weight to new latents,
                while lower values preserve more information from previous frames. Defaults to 0.9.
            kl_penalty (bool, optional): Whether to apply KL divergence penalty to
                keep prediction latents close to N(0,1). Defaults to False.
            kl_weight (float, optional): Weight for KL divergence penalty.
                Only used when kl_penalty is True. Defaults to 0.1.
            kl_mode (str, optional): KL divergence mode. Options are:
                - "simple-forward": Uses a simplified penalty based on squared L2 norm of latents.
                  Fastest but least accurate approximation of KL divergence.
                - "forward": Computes proper forward KL divergence between latent distribution and N(0,1).
                  More accurate but slightly more computationally expensive.
                - "symmetric": Computes both forward and backward KL divergence for more robust regularization.
                  Most accurate but most computationally expensive.
                Defaults to "simple-forward".
            interp_mode (str, optional): Interpolation mode for resizing.
                Options include "bilinear", "bicubic", etc. Defaults to "bilinear".
            loss_funcs (list[str] | None, optional): Loss functions to use.
                If None, defaults to ["l1", "l2"]. Supported options are
                "l1", "l2", "edge", and "smooth". When using "edge" or "smooth",
                the RGB image is used to guide depth discontinuities. Defaults to None.
            seed (int, optional): Random seed for reproducibility. Defaults to 2024.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Dense depth prediction with shape [N, 1, H, W] in metric units (same as input sparse depth)
                - Prediction latents with shape [N, 4, EH, EW] that can be used for temporal consistency
                  in subsequent frames
        """  # noqa: E501
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Create empty text embedding if not created
        if self.empty_text_embedding is None:
            with torch.no_grad():
                text_inputs = self.tokenizer(
                    "",
                    padding="do_not_pad",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(self.device)
                self.empty_text_embedding: torch.Tensor = self.text_encoder(
                    text_input_ids
                )[0]

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
        if beta < 0 or beta > 1:
            raise ValueError(f"beta must be in [0, 1], but got {beta}")

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
        with torch.no_grad():
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
        if isinstance(depth_range, str):
            if depth_range == "minmax":
                min_depths, max_depths = utils.masked_minmax(
                    sparses, masks, dims=(1, 2, 3)
                )
                min_depths = min_depths.view(-1, 1, 1, 1)
                max_depths = max_depths.view(-1, 1, 1, 1)
            elif depth_range == "percentile":
                ranges = torch.stack(
                    [
                        torch.quantile(
                            s[m],
                            torch.tensor(
                                [percentile, 1 - percentile], device=sparses.device
                            ),
                        )
                        for s, m in zip(sparses, masks, strict=True)
                    ]
                )  # [N, 2]
                min_depths = ranges[:, 0].view(-1, 1, 1, 1)
                max_depths = ranges[:, 1].view(-1, 1, 1, 1)
            else:
                raise ValueError(f"Unknown depth_range: {depth_range}")
        else:
            min_depths = torch.full((N, 1, 1, 1), depth_range[0], device=sparses.device)
            max_depths = torch.full((N, 1, 1, 1), depth_range[1], device=sparses.device)

        # Normalize sparse depth maps
        sparses_clamped = torch.clamp(sparses, min=min_depths, max=max_depths)
        sparses_normed = (sparses_clamped - min_depths) / (max_depths - min_depths)
        sparses_normed[~masks] = 0  # Set unmeasured regions to 0
        sparses_min, sparses_max = utils.masked_minmax(
            sparses_normed, masks, dims=(1, 2, 3)
        )
        sparse_ranges = (sparses_min, sparses_max) if affine_invariant else None

        # Set current prediction latents as trainable params
        pred_latents = torch.nn.Parameter(pred_latents)  # [N, 4, EH, EW]

        # Set scaling params
        affine_params = (
            (
                # scale
                torch.nn.Parameter(torch.ones(N, 1, 1, 1, device=self.device)),
                # shift
                torch.nn.Parameter(torch.zeros(N, 1, 1, 1, device=self.device)),
            )
            if affine_invariant
            else None
        )

        # Set up optimizer
        optimizer: torch.optim.Optimizer
        param_groups = [
            {"params": [pred_latents], "lr": lr_latent},
        ]
        if affine_params is not None:
            param_groups.append({"params": list(affine_params), "lr": lr_scaling})
        if opt == "adam":
            optimizer = torch.optim.Adam(param_groups)
        elif opt == "sgd":
            optimizer = torch.optim.SGD(param_groups)
        else:
            raise ValueError(f"Unknown optimizer: {opt}")

        # Denoising loop
        self.scheduler.set_timesteps(steps, device=self.device)
        for t in self.scheduler.timesteps:
            optimizer.zero_grad()

            # Forward pass through the U-Net
            latents = torch.cat([img_latents, pred_latents], dim=1)  # [N, 8, EH, EW]
            pred_noises = self.unet(
                latents,
                t,
                encoder_hidden_states=batch_empty_text_embedding,
                return_dict=False,
            )[
                0
            ]  # [N, 4, EH, EW]

            # Compute pred_epsilon to later rescale the depth latent gradient
            with torch.no_grad():
                a_prod_t = self.scheduler.alphas_cumprod[t]
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

            # Predict dense depth maps
            denses_normed = self.latent_to_dense(
                previews,
                orig_res,
                padding,
                affine_invariant=affine_invariant,
                affine_params=affine_params,
                sparse_range=sparse_ranges,
                interp_mode=interp_mode,
            )  # [N, 1, H, W]
            losses = compute_loss(denses_normed, sparses_normed, loss_funcs, image=imgs)

            # NOTE: Add KL divergence penalty to keep
            # the distribution of pred_latent close to N(0,1)
            if kl_penalty:
                # NOTE: Convert to float32 to avoid numerical instability
                pred_latents_fp32 = pred_latents.to(torch.float32)
                if kl_mode == "simple-forward":
                    kl_losses = pred_latents_fp32.square().mean(
                        dim=(1, 2, 3), keepdim=True
                    )
                elif kl_mode in ["forward", "symmetric"]:
                    # KL divergence between N(mu, sigma^2) and N(0, 1)
                    # Forward pass: N(0, 1) -> N(mu, sigma^2)
                    mu = pred_latents_fp32.mean(dim=(1, 2, 3), keepdim=True)
                    var = pred_latents_fp32.var(
                        dim=(1, 2, 3), keepdim=True, unbiased=False
                    )
                    kl_losses = 0.5 * (mu.square() + var - torch.log(var + EPSILON) - 1)
                    if kl_mode == "symmetric":
                        # Backward pass: N(mu, sigma^2) -> N(0, 1)
                        kl_losses += 0.5 * (
                            (mu.square() + 1) / (var + EPSILON)
                            + torch.log(var + EPSILON)
                            - 1
                        )
                else:
                    raise ValueError(
                        f"Invalid kl_mode: {kl_mode}. Use 'simple' or 'strict'"
                    )

                losses = losses + kl_weight * kl_losses

            # Backprop
            losses.backward(torch.ones_like(losses))  # Preserve batch dimension

            # Scale gradients up
            with torch.no_grad():
                assert pred_latents.grad is not None

                # Calculate norms per sample in the batch
                pred_epsilon_norms = torch.linalg.norm(
                    pred_epsilons.view(N, -1), dim=1
                )  # [N]
                pred_latent_grad_norms = torch.linalg.norm(
                    pred_latents.grad.view(N, -1), dim=1
                )  # [N]

                # Calculate scaling factor per sample
                factors = pred_epsilon_norms / torch.clamp(
                    pred_latent_grad_norms, min=EPSILON
                )  # [N]
                # Reshape scaling factor for broadcasting
                factors = factors.view(N, 1, 1, 1)  # [N, 1, 1, 1]

                # Apply per-sample scaling
                pred_latents.grad *= factors  # [N, 4, EH, EW]

            # Backprop
            optimizer.step()

            # Execute update of the latent with regular denoising diffusion step
            with torch.no_grad():
                pred_latents.data = self.scheduler.step(
                    pred_noises, t, pred_latents, generator=generator
                ).prev_sample

        # Compute final dense depth maps
        with torch.no_grad():
            pred_latents_detached = pred_latents.detach()
            denses_normed = self.latent_to_dense(
                pred_latents_detached,
                orig_res,
                padding,
                affine_invariant=affine_invariant,
                affine_params=affine_params,
                sparse_range=sparse_ranges,
                interp_mode=interp_mode,
            )  # [N, 1, H, W]
            # Decode
            denses_normed = torch.clamp(denses_normed, min=0, max=1)
            denses = denses_normed * (max_depths - min_depths) + min_depths
        return denses, pred_latents_detached
