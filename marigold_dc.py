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
import numpy as np
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
                    - "smooth": Smoothness loss (requires image parameter)
        image: Optional RGB or grayscale image tensor of shape [N, C, H, W] where C is 1 or 3.
               Required when using "edge" or "smooth" loss functions.

    Returns:
        A tensor of shape [N] containing the total loss for each sample in the batch.

    Raises:
        ValueError: If loss_funcs is empty, or if image is not provided when required.
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

    This pipeline extends the MarigoldDepthPipeline to perform depth completion,
    which takes an RGB image and sparse depth measurements as input and produces
    a dense depth map as output. The pipeline uses a diffusion model to iteratively
    refine the depth prediction while respecting the sparse depth constraints.

    The depth completion process involves:
    1. Encoding the input image into latent space
    2. Optimizing the latent representation to match sparse depth points
    3. Using a parametrized affine transformation to convert from the model's
       depth representation to metric depth values
    4. Iteratively refining the prediction through a guided diffusion process
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
        the scale and range of the provided sparse depth measurements.

        Args:
            dense (torch.Tensor): Normalized depth predictions from the model with shape [N, 1, H, W].
            scale (torch.Tensor): Scaling factor with shape [N, 1, H, W] or [N, 1, 1, 1].
            shift (torch.Tensor): Shift factor with shape [N, 1, H, W] or [N, 1, 1, 1].
            sparse_range (torch.Tensor): Range of sparse depth values with shape [N, 1, 1, 1].
            sparse_min (torch.Tensor): Minimum value of sparse depth with shape [N, 1, 1, 1].

        Returns:
            torch.Tensor: Metric depth values with shape [N, 1, H, W].
        """  # noqa: E501
        return (scale**2) * sparse_range * dense + (shift**2) * sparse_min

    def latent_to_metric(
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
            interp_mode (str, optional): Interpolation mode for resizing.
                Options are "bilinear" or "nearest". Defaults to "bilinear".

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
        imgs: np.ndarray,
        sparses: np.ndarray,
        steps: int = 50,
        resolution: int = 768,
        seed: int = 2024,
        elemwise_scaling: bool = False,
        interp_mode: str = "bilinear",
        loss_funcs: list[str] | None = None,
        opt: str = "adam",
        lr: tuple[float, float] | None = None,
        beta: float = 0.9,
    ) -> np.ndarray:
        """
        Perform depth completion on a sequence of RGB images using sparse depth measurements.

        Args:
            imgs (np.ndarray): Numpy array of shape [N, L, H, W, C],
                representing a sequence of images with C channels.
            sparses (np.ndarray): Numpy array of shape [N, L, H, W],
                representing a sequence of sparse depth maps.
                Should have zeros at missing positions and positive values at
                measurement points.
            steps (int, optional): Number of denoising steps.
                Higher values give better quality but slower inference.
                Defaults to 50.
            resolution (int, optional): Resolution for internal processing.
                Higher values give better quality but use more memory.
                Defaults to 768.
            seed (int, optional): Random seed for reproducibility. Defaults to 2024.
            elemwise_scaling (bool, optional): Whether to use element-wise scaling
                for the affine transformation. Defaults to False.
            interp_mode (str, optional): Interpolation mode for resizing.
                Options are "bilinear" or "nearest". Defaults to "bilinear".
            loss_funcs (list[str], optional): List of loss functions to use for
                optimization. Options include "l1", "l2", "edge", and "smooth".
                Defaults to ["l1", "l2"].
            opt (str, optional): Optimizer to use for depth completion.
                Options are "adam", "adamw", or "sgd". Defaults to "adam".
            lr (tuple[float, float], optional): Learning rates for (latent, scaling).
                Defaults to (0.05, 0.005).
            beta (float, optional): Momentum factor for prediction latents between
                frames in a sequence. Must be in range [0, 1]. Defaults to 0.9.

        Returns:
            np.ndarray: Dense depth prediction of shape [N, L, H, W].

        Raises:
            ValueError: If input shapes are incompatible or if beta is out of range.
        """
        # Set device
        device: torch.device = self._execution_device

        # Get random generator
        generator = torch.Generator(device=device).manual_seed(seed)

        # Check input shapes
        if (
            imgs.ndim != 5
            or sparses.ndim != 4
            or imgs.shape[0] != sparses.shape[0]
            or imgs.shape[1] != sparses.shape[1]
        ):
            raise ValueError(
                "Shape of image must be [N, L, C, H, W] and shape of sparse must be "
                f"[N, L, H, W], but got image.shape: "
                f"{imgs.shape} and sparse.shape: {sparses.shape}"
            )
        if beta < 0 or beta > 1:
            raise ValueError(f"beta must be in [0, 1], but got {beta}")

        # Convert to torch tensors
        imgs = torch.from_numpy(imgs).permute(
            0, 1, 4, 2, 3
        )  # [N, L, H, W, C] -> [N, L, C, H, W]
        sparses = torch.from_numpy(sparses).unsqueeze(
            2
        )  # [N, L, H, W] -> [N, L, 1, H, W]
        N, L, _, H, W = imgs.shape

        # Move to execution device
        imgs = imgs.to(self._execution_device, non_blocking=True)  # [N, L, C, H, W]
        sparses = sparses.to(
            self._execution_device, non_blocking=True
        )  # [N, L, 1, H, W]

        # Set loss functions
        if loss_funcs is None:
            loss_funcs = ["l1", "l2"]

        # Set learning rates
        if lr is None:
            lr_latent = 0.05
            lr_scaling = 0.005
        else:
            lr_latent, lr_scaling = lr

        # Prepare empty text conditioning
        # Used for all sequence frames
        if self.empty_text_embedding is None:
            with torch.no_grad():
                text_inputs = self.tokenizer(
                    "",
                    padding="do_not_pad",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(device)
                self.empty_text_embedding = self.text_encoder(text_input_ids)[0]
        batch_empty_text_embedding = self.empty_text_embedding.repeat(N, 1, 1)

        # Get prediction lantents common to all sequence frames
        pred_latent_common = torch.randn(
            (
                N,
                4,
                resolution * H // (8 * max(H, W)),
                resolution * W // (8 * max(H, W)),
            ),
            device=imgs.device,
            dtype=self.dtype,
        )  # [N, 4, EH, EW]
        pred_latent_prev: torch.Tensor | None = None

        # Iterate over sequence length
        ret: list[np.ndarray] = []
        for frame_idx in range(L):
            # Get current image and sparse depth
            img: torch.Tensor = imgs[:, frame_idx]  # [N, C, H, W]
            sparse: torch.Tensor = sparses[:, frame_idx]  # [N, 1, H, W]

            # Preprocess input images
            img_resized, padding, orig_size = self.image_processor.preprocess(
                img,
                processing_resolution=resolution,
                device=device,
                dtype=self.dtype,
            )  # [N, C, PPH, PPW]

            # Get latent encodings
            with torch.no_grad():
                img_latent, _ = self.prepare_latents(
                    img_resized, None, generator, 1, N
                )  # [N, 4, EH, EW], [N, 4, EH, EW]
                if pred_latent_prev is not None:
                    pred_latent = (
                        beta * pred_latent_common + (1 - beta) * pred_latent_prev
                    )
                else:
                    pred_latent = pred_latent_common

            # Set current prediction latents as trainable params
            pred_latent = torch.nn.Parameter(pred_latent)  # [N, 4, EH, EW]

            # Calculate min, max, and range for each sample in the batch [N] -> [N,1,1,1]
            sparse_mask = sparse > 0
            sparse_min = torch.stack(
                [s[m].min() for s, m in zip(sparse, sparse_mask, strict=True)]
            ).view(-1, 1, 1, 1)
            sparse_max = torch.stack(
                [s[m].max() for s, m in zip(sparse, sparse_mask, strict=True)]
            ).view(-1, 1, 1, 1)
            sparse_range = sparse_max - sparse_min

            # Set up scaling params
            if elemwise_scaling:
                # Element-wise scaling
                scale, shift = torch.nn.Parameter(
                    torch.ones(N, 1, *orig_size, device=device)
                ), torch.nn.Parameter(
                    torch.zeros(N, 1, *orig_size, device=device)
                )  # [N, 1, H, W], [N, 1, H, W]
            else:
                # Global scaling
                scale, shift = torch.nn.Parameter(
                    torch.ones(N, 1, 1, 1, device=device)
                ), torch.nn.Parameter(torch.zeros(N, 1, 1, 1, device=device))
                # [N, 1, 1, 1], [N, 1, 1, 1]

            # Set up optimizer
            optimizer: torch.optim.Optimizer
            param_groups = [
                {"params": [scale, shift], "lr": lr_scaling},
                {"params": [pred_latent], "lr": lr_latent},
            ]
            if opt == "adam":
                optimizer = torch.optim.Adam(param_groups)
            elif opt == "adamw":
                optimizer = torch.optim.AdamW(param_groups)
            elif opt == "sgd":
                optimizer = torch.optim.SGD(param_groups)
            else:
                raise ValueError(f"Unknown optimizer: {opt}")

            # Denoising loop
            self.scheduler.set_timesteps(steps, device=device)
            for t in self.scheduler.timesteps:
                optimizer.zero_grad()

                # Forward pass through the U-Net
                latent = torch.cat([img_latent, pred_latent], dim=1)  # [N, 8, EH, EW]
                noise = self.unet(
                    latent,
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
                    pred_epsilon = (a_prod_t**0.5) * noise + (
                        b_prod_t**0.5
                    ) * pred_latent  # [N, 4, EH, EW]

                # Preview the final output depth with
                # Tweedie's formula (See Equation 1 of the paper)
                preview = self.scheduler.step(
                    noise, t, pred_latent, generator=generator
                ).pred_original_sample  # [N, 4, EH, EW]

                # Decode to metric space, compute loss with guidance and backpropagate
                dense = self.latent_to_metric(
                    preview,
                    scale,
                    shift,
                    sparse_range,
                    sparse_min,
                    padding,
                    orig_size,
                    interp_mode,
                )  # [N, 1, H, W]
                loss = compute_loss(dense, sparse, loss_funcs, image=img)
                loss.backward(torch.ones_like(loss))  # Preserve batch dimension

                # Scale gradients up
                with torch.no_grad():
                    # Calculate norms per sample in the batch
                    pred_epsilon_norm = torch.linalg.norm(
                        pred_epsilon.view(N, -1), dim=1
                    )  # [N]
                    pred_latent_grad_norm = torch.linalg.norm(
                        pred_latent.grad.view(N, -1), dim=1
                    )  # [N]

                    # Calculate scaling factor per sample
                    factor = pred_epsilon_norm / torch.clamp(
                        pred_latent_grad_norm, min=1e-8
                    )  # [N]
                    # Reshape scaling factor for broadcasting
                    factor = factor.view(N, 1, 1, 1)  # [N, 1, 1, 1]

                    # Apply per-sample scaling
                    pred_latent.grad *= factor  # [N, 4, EH, EW]

                # Backprop
                optimizer.step()

                # Execute update of the latent with regular denoising diffusion step
                with torch.no_grad():
                    pred_latent.data = self.scheduler.step(
                        noise, t, pred_latent, generator=generator
                    ).prev_sample

            # Cache previous prediction latents
            pred_latent_prev = pred_latent.detach()

            # Compute dense maps for the current frame
            with torch.no_grad():
                dense = self.latent_to_metric(
                    pred_latent_prev,
                    scale,
                    shift,
                    sparse_range,
                    sparse_min,
                    padding,
                    orig_size,
                    interp_mode,
                )  # [N, 1, H, W]
                dense_np = self.image_processor.pt_to_numpy(dense)  # [N, H, W, 1]
                dense_np = np.squeeze(dense_np, axis=-1)  # [N, H, W]
                ret.append(np.expand_dims(dense_np, axis=1))  # [N, 1, H, W]
        return np.concatenate(ret, axis=1)  # [N, L, H, W]
