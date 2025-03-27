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
    depth_pred: torch.Tensor,
    depth_target: torch.Tensor,
    loss_funcs: list[str],
    image: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-sample losses between predicted and target depth maps.

    This function computes multiple types of losses between predicted and target depth maps,
    maintaining independence between samples in the batch. Each loss is computed separately
    for each sample without any batch-wise averaging.

    Args:
        depth_pred (torch.Tensor): Predicted depth maps. Shape: [N, 1, H, W]
        depth_target (torch.Tensor): Target depth maps with sparse values.
            Shape: [N, 1, H, W]. Zero values indicate missing measurements.
        loss_funcs (list[str]): List of loss functions to use.
            Available options: ["l1", "l2", "edge", "smooth"]
            - l1: Mean absolute error for valid depth points
            - l2: Mean squared error for valid depth points
            - edge: Gradient similarity between depth and image edges
            - smooth: Spatial smoothness of the depth map
        image (torch.Tensor | None, optional): Input images, required for "edge" and "smooth" losses.
            Shape: [N, C, H, W] where C is 1 or 3. Defaults to None.

    Returns:
        torch.Tensor: Per-sample loss values. Shape: [N]
            Each element represents the total loss for that sample,
            computed as the sum of all specified loss functions.

    Raises:
        ValueError: If loss_funcs is empty
        ValueError: If image is None when using "edge" or "smooth" loss
        ValueError: If image has invalid number of channels
        ValueError: If unknown loss function is specified
    """
    if len(loss_funcs) == 0:
        raise ValueError("loss_funcs must contain at least one loss function")
    total = torch.zeros(depth_pred.shape[0], device=depth_pred.device)  # [N]
    sparse_mask = depth_target > 0

    for loss_func in loss_funcs:
        if loss_func == "l1":
            # Compute L1 loss per sample using masked operations
            l1_loss = torch.abs(depth_pred - depth_target)
            l1_loss = l1_loss * sparse_mask  # Apply mask
            # Sum over HW dimensions and divide by number of valid points per sample
            total += l1_loss.sum(dim=(1, 2, 3)) / sparse_mask.sum(dim=(1, 2, 3))

        elif loss_func == "l2":
            # Compute L2 loss per sample using masked operations
            l2_loss = (depth_pred - depth_target) ** 2
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
            grad_pred_x = torch.abs(depth_pred[:, :, :, :-1] - depth_pred[:, :, :, 1:])
            grad_pred_y = torch.abs(depth_pred[:, :, :-1, :] - depth_pred[:, :, 1:, :])
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
            loss_h = torch.abs(depth_pred[:, :, :-1, :] - depth_pred[:, :, 1:, :]).mean(
                dim=(1, 2, 3)
            )
            loss_w = torch.abs(depth_pred[:, :, :, :-1] - depth_pred[:, :, :, 1:]).mean(
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

    Attributes:
        unet (UNet2DConditionModel): U-Net model for diffusion process
        vae (AutoencoderKL): Variational autoencoder for encoding/decoding images
        scheduler (DDIMScheduler): Scheduler for the diffusion process
        text_encoder (CLIPTextModel): Text encoder for conditioning
        tokenizer (CLIPTokenizer): Tokenizer for text inputs
        image_processor (MarigoldImageProcessor): Processor for image preprocessing
        empty_text_embedding (torch.Tensor): Cached empty text embedding for conditioning
        dtype (torch.dtype): Data type for model computation
        _execution_device (torch.device): Device for model execution
    """

    def __call__(
        self,
        image: np.ndarray | torch.Tensor,
        sparse: np.ndarray | torch.Tensor,
        steps: int = 50,
        resolution: int = 768,
        seed: int = 2024,
        elemwise_scaling: bool = False,
        interp_mode: str = "bilinear",
        loss_funcs: list[str] | None = None,
        aa: bool = False,
        opt: str = "adam",
        lr: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """
        Perform depth completion on an RGB image using sparse depth measurements.

        Args:
            image (np.ndarray | torch.Tensor): Input RGB image of shape [H, W, C] or [N, C, H, W].
            sparse (np.ndarray | torch.Tensor): Sparse depth measurements of shape [H, W] or [N, H, W].
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
            aa (bool, optional): Whether to enable anti-aliasing during processing.
                Defaults to False.
            opt (str, optional): Optimizer to use for depth completion.
                Options are "adam", "adamw", or "sgd". Defaults to "adam".
            lr (tuple[float, float], optional): Learning rates for (latent, scaling).
                Defaults to (0.05, 0.005).

        Returns:
            np.ndarray: Dense depth prediction of shape [H, W] or [N, H, W].

        Raises:
            ValueError: If sparse is not a 2D numpy array.
        """  # noqa: E501
        # Set device
        device: torch.device = self._execution_device

        # Get random generator
        generator = torch.Generator(device=device).manual_seed(seed)

        # Check inputs
        is_batched = False
        if image.ndim == 3:
            if sparse.ndim != 2:
                raise ValueError(
                    "Shape of sparse must be [H, W] if shape of image is [H, W, C], "
                    f"but got shape of sparse: {sparse.shape}"
                )
            # Add batch dimension
            image = image[None]
            sparse = sparse[None]
        elif image.ndim == 4:
            if sparse.ndim != 3:
                raise ValueError(
                    "Shape of sparse must be [N, H, W] if shape of image is "
                    f"[N, C, H, W], but got shape of sparse: {sparse.shape}"
                )
            if image.shape[0] != sparse.shape[0]:
                raise ValueError(
                    "Shape of sparse must be [N, H, W] if shape of image is "
                    f"[N, C, H, W], but got shape of sparse: {sparse.shape}"
                )
            is_batched = True
        else:
            raise ValueError(
                "Shape of image must be [H, W, C] or [N, C, H, W], "
                f"but got shape of image: {image.shape}"
            )
        # Convert to torch tensors if not already
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(
                0, 3, 1, 2
            )  # [N, H, W, C] -> [N, C, H, W]
        if isinstance(sparse, np.ndarray):
            sparse = torch.from_numpy(sparse).unsqueeze(1)  # [N, H, W] -> [N, 1, H, W]

        # Move to execution device
        image = image.to(device, non_blocking=True)
        sparse = sparse.to(device, non_blocking=True)
        batch_size = image.shape[0]

        # Set learning rates
        if lr is None:
            lr_latent = 0.05
            lr_scaling = 0.005
        else:
            lr_latent, lr_scaling = lr

        # Set loss functions
        if loss_funcs is None:
            loss_funcs = ["l1", "l2"]

        # Prepare empty text conditioning
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
        batch_empty_text_embedding = self.empty_text_embedding.repeat(batch_size, 1, 1)

        # Preprocess input images
        # Convert image to channel-first format (N, C, H, W)
        image_resized, padding, original_resolution = self.image_processor.preprocess(
            image,
            processing_resolution=resolution,
            device=device,
            dtype=self.dtype,
        )  # (N, 3, PPH, PPW)

        # Encode input image into latent space
        with torch.no_grad():
            image_latent, pred_latent = self.prepare_latents(
                image_resized, None, generator, 1, batch_size
            )  # [N, 4, EH, EW], [N, 4, EH, EW]

        # Preprocess sparse depth
        sparse_mask = sparse > 0
        guided = sparse[sparse_mask]

        # Set up optimization targets and compute
        # the range and lower bound of the sparse depth
        if elemwise_scaling:
            scale, shift = torch.nn.Parameter(
                torch.ones(batch_size, 1, *original_resolution, device=device)
            ), torch.nn.Parameter(
                torch.ones(batch_size, 1, *original_resolution, device=device)
            )  # [N, 1, H, W], [N, 1, H, W]
        else:
            scale, shift = torch.nn.Parameter(
                torch.ones(batch_size, device=device)
            ), torch.nn.Parameter(torch.ones(batch_size, device=device))
            # [N], [N]
        pred_latent = torch.nn.Parameter(pred_latent)
        sparse_min = guided.min()
        sparse_max = guided.max()
        sparse_range = sparse_max - sparse_min

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

        def affine_to_metric(depth: torch.Tensor) -> torch.Tensor:
            return (scale**2) * sparse_range * depth + (shift**2) * sparse_min

        def latent_to_metric(latent: torch.Tensor) -> torch.Tensor:
            affine_invariant_pred = self.decode_prediction(latent)  # [N, 1, PPH, PPW]
            affine_invariant_pred = self.image_processor.unpad_image(
                affine_invariant_pred, padding
            )
            affine_invariant_pred = self.image_processor.resize_antialias(
                affine_invariant_pred,
                original_resolution,
                interp_mode,
                is_aa=aa,
            )  # [N, 1, H, W]
            pred = affine_to_metric(affine_invariant_pred)
            return pred  # [N, 1, H, W]

        # Denoising loop
        self.scheduler.set_timesteps(steps, device=device)
        for t in self.scheduler.timesteps:
            optimizer.zero_grad()

            # Forward pass through the U-Net
            latent = torch.cat([image_latent, pred_latent], dim=1)  # [N, 8, EH, EW]
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

            # Forward denoising step
            step_output = self.scheduler.step(
                noise, t, pred_latent, generator=generator
            )

            # Preview the final output depth with
            # Tweedie's formula (See Equation 1 of the paper)
            preview = step_output.pred_original_sample  # [N, 4, EH, EW]

            # Decode to metric space, compute loss with guidance and backpropagate
            dense = latent_to_metric(preview)  # [N, 1, H, W]
            loss = compute_loss(dense, sparse, loss_funcs, image=image)
            loss.backward(torch.ones_like(loss))  # Preserve batch dimension

            # Scale gradients up
            with torch.no_grad():
                # Calculate norms per sample in the batch
                pred_epsilon_norm = torch.linalg.norm(
                    pred_epsilon.view(batch_size, -1), dim=1
                )  # [N]
                depth_latent_grad_norm = torch.linalg.norm(
                    pred_latent.grad.view(batch_size, -1), dim=1
                )  # [N]

                # Calculate scaling factor per sample
                scaling_factor = pred_epsilon_norm / torch.clamp(
                    depth_latent_grad_norm, min=1e-8
                )  # [N]

                # Reshape scaling factor for broadcasting
                scaling_factor = scaling_factor.view(
                    batch_size, 1, 1, 1
                )  # [N, 1, 1, 1]

                # Apply per-sample scaling
                pred_latent.grad *= scaling_factor  # [N, 4, EH, EW]

            # Execute the update step through guidance backprop
            optimizer.step()

            # Execute update of the latent with regular denoising diffusion step
            with torch.no_grad():
                pred_latent.data = self.scheduler.step(
                    noise, t, pred_latent, generator=generator
                ).prev_sample

        # Decode predictions from latent into pixel space
        with torch.no_grad():
            dense = latent_to_metric(pred_latent.detach())

        # return Numpy array
        dense = self.image_processor.pt_to_numpy(dense)  # (N, H, W, 1)
        if is_batched:
            return np.squeeze(dense, axis=-1)  # [N, H, W]
        return np.squeeze(dense)  # [H, W]
