# Copyright 2024 Massimiliano Viola, Kevin Qu, Nando Metzger, Anton Obukhov ETH Zurich.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold-DC#-citation
# More information can be found at https://marigolddepthcompletion.github.io
# ---------------------------------------------------------------------------------
import diffusers
import torch
from diffusers import MarigoldDepthPipeline

diffusers.utils.logging.disable_progress_bar()

MARIGOLD_CKPT_ORIGINAL = "prs-eth/marigold-v1-0"
MARIGOLD_CKPT_LCM = "prs-eth/marigold-lcm-v1-0"
VAE_CKPT_LIGHT = "madebyollin/taesd"
SUPPORTED_LOSS_FUNCS = ["l1", "l2", "edge", "smooth"]


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

        This method applies an affine transformation to convert the depth predictions
        from the model's internal representation to actual metric depth values that match
        the scale and range of the provided sparse depth measurements.

        Args:
            dense (torch.Tensor): Affine-invariant depth predictions from the model with shape [N, 1, H, W].
            scale (torch.Tensor): Scaling factor with shape [N, 1, H, W] or [N, 1, 1, 1].
            shift (torch.Tensor): Shift factor with shape [N, 1, H, W] or [N, 1, 1, 1].
            sparse_range (torch.Tensor): Range of sparse depth values with shape [N, 1, 1, 1].
            sparse_min (torch.Tensor): Minimum value of sparse depth with shape [N, 1, 1, 1].

        Returns:
            torch.Tensor: Metric depth values with shape [N, 1, H, W].
        """  # noqa: E501
        return (scale**2) * sparse_range * dense + (shift**2) * sparse_min

    def latent_to_dense(
        self,
        latent: torch.Tensor,  # [N, 4, EH, EW]
        scale: torch.Tensor,  # [N, 1, H, W] or [N, 1, 1, 1]
        shift: torch.Tensor,  # [N, 1, H, W] or [N, 1, 1, 1]
        sparse_range: torch.Tensor,  # [N, 1, 1, 1]
        sparse_min: torch.Tensor,  # [N, 1, 1, 1]
        padding: tuple,
        original_resolution: tuple,
        interp_mode: str = "bilinear",
    ) -> torch.Tensor:
        """
        Converts latent representation to metric depth values.

        This method decodes the latent representation from the diffusion model,
        removes padding, resizes to the original resolution, and applies an affine
        transformation to convert to metric depth values.

        Args:
            latent (torch.Tensor): Latent representation from the diffusion model with shape [N, 4, EH, EW].
            scale (torch.Tensor): Scaling factor with shape [N, 1, H, W] or [N, 1, 1, 1].
            shift (torch.Tensor): Shift factor with shape [N, 1, H, W] or [N, 1, 1, 1].
            sparse_range (torch.Tensor): Range of sparse depth values with shape [N, 1, 1, 1].
            sparse_min (torch.Tensor): Minimum value of sparse depth with shape [N, 1, 1, 1].
            padding (tuple): Padding values used during processing.
            original_resolution (tuple): Original resolution (height, width) to resize the output to.
            interp_mode (str, optional): Interpolation mode for resizing. Defaults to "bilinear".

        Returns:
            torch.Tensor: Metric depth values with shape [N, 1, H, W].
        """  # noqa: E501
        affine_invariant_pred = self.decode_prediction(latent)  # [N, 1, PPH, PPW]
        affine_invariant_pred = self.image_processor.unpad_image(
            affine_invariant_pred, padding
        )
        affine_invariant_pred = self.image_processor.resize_antialias(
            affine_invariant_pred, original_resolution, interp_mode
        )  # [N, 1, H, W]
        pred = self._affine_to_metric(
            affine_invariant_pred, scale, shift, sparse_range, sparse_min
        )
        return pred  # [N, 1, H, W]

    def __call__(
        self,
        imgs: torch.Tensor,
        sparses: torch.Tensor,
        pred_latents_prev: torch.Tensor | None = None,
        max_depth: float | None = None,
        steps: int = 50,
        resolution: int = 768,
        opt: str = "adam",
        lr: tuple[float, float] | None = None,
        beta: float = 0.9,
        kl_penalty: bool = False,
        kl_weight: float = 0.1,
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
            pred_latents_prev (torch.Tensor | None, optional): Previous prediction latents
                with shape [N, 4, EH, EW] from a prior frame or iteration.
                Enables temporal consistency when processing video sequences. Defaults to None.
            max_depth (float | None, optional): Maximum depth value for normalization.
                If None, uses per-sample maximum. Defaults to None.
            steps (int, optional): Number of denoising steps.
                Higher values give better quality but slower inference. Defaults to 50.
            resolution (int, optional): Resolution for internal processing.
                Higher values give better quality but use more memory. Defaults to 768.
            opt (str, optional): Optimizer to use ("adam" or "sgd"). Defaults to "adam".
            lr (tuple[float, float], optional): Learning rates for (latent, scaling).
                Defaults to (0.05, 0.005).
            beta (float, optional): Momentum factor for prediction latents between frames.
                Must be in range [0, 1]. Higher values give more weight to new latents,
                while lower values preserve more information from previous frames. Defaults to 0.9.
            kl_penalty (bool, optional): Whether to apply KL divergence penalty to
                keep prediction latents close to N(0,1). Defaults to False.
            kl_weight (float, optional): Weight for KL divergence penalty.
                Only used when kl_penalty is True. Defaults to 0.1.
            interp_mode (str, optional): Interpolation mode for resizing.
                Options include "bilinear", "bicubic", etc. Defaults to "bilinear".
            loss_funcs (list[str] | None, optional): Loss functions to use.
                If None, defaults to ["l1", "l2"]. Supported options are
                "l1", "l2", "edge", and "smooth".
            seed (int, optional): Random seed for reproducibility. Defaults to 2024.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Dense depth prediction with shape [N, 1, H, W]
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
        imgs_resized, padding, orig_size = self.image_processor.preprocess(
            imgs,
            processing_resolution=resolution,
            device=self.device,
            dtype=self.dtype,
        )  # [N, C, PPH, PPW]

        # Preprocess sparse depth maps
        if max_depth is None:
            # Normalize by per-sample max
            max_depths = torch.amax(
                sparses, dim=(1, 2, 3), keepdim=True
            )  # [N, 1, 1, 1]
        else:
            # Normalize by absolute max specified by user
            max_depths = (
                torch.ones(len(sparses), 1, 1, 1, device=self.device) * max_depth
            )  # [N, 1, 1, 1]
        sparses_normed = sparses / max_depths

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

        # Set current prediction latents as trainable params
        pred_latents = torch.nn.Parameter(pred_latents)  # [N, 4, EH, EW]

        # Calculate min, max, and range for each sample in the batch [N] -> [N,1,1,1]
        sparse_masks = sparses_normed > 0
        sparse_mins = torch.stack(
            [s[m].min() for s, m in zip(sparses_normed, sparse_masks, strict=True)]
        ).view(-1, 1, 1, 1)
        sparse_maxs = torch.stack(
            [s[m].max() for s, m in zip(sparses_normed, sparse_masks, strict=True)]
        ).view(-1, 1, 1, 1)
        sparse_ranges = sparse_maxs - sparse_mins

        # Set up scaling params
        scale, shift = torch.nn.Parameter(
            torch.ones(N, 1, 1, 1, device=self.device)
        ), torch.nn.Parameter(torch.zeros(N, 1, 1, 1, device=self.device))
        # [N, 1, 1, 1], [N, 1, 1, 1]

        # Set up optimizer
        optimizer: torch.optim.Optimizer
        param_groups = [
            {"params": [scale, shift], "lr": lr_scaling},
            {"params": [pred_latents], "lr": lr_latent},
        ]
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
            previews = self.scheduler.step(
                pred_noises, t, pred_latents, generator=generator
            ).pred_original_sample  # [N, 4, EH, EW]

            # Predict dense depth maps
            denses = self.latent_to_dense(
                previews,
                scale,
                shift,
                sparse_ranges,
                sparse_mins,
                padding,
                orig_size,
                interp_mode,
            )  # [N, 1, H, W]
            losses = compute_loss(denses, sparses_normed, loss_funcs, image=imgs)

            # NOTE: Add KL divergence penalty to keep
            # the distribution of pred_latent close to N(0,1)
            if kl_penalty:
                kl_losses = kl_weight * pred_latents.square().mean(
                    dim=(1, 2, 3), keepdim=True
                )
                losses = losses + kl_losses

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
                    pred_latent_grad_norms, min=1e-8
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
            denses = self.latent_to_dense(
                pred_latents_detached,
                scale,
                shift,
                sparse_ranges,
                sparse_mins,
                padding,
                orig_size,
            )  # [N, 1, H, W]
            # Decode
            denses *= max_depths
        return denses, pred_latents_detached


# End of Selection
