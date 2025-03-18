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
    # depth_pred: [N, 1, H, W]
    # depth_target: [N, 1, H, W]
    # image: [N, 3, H, W]
    if len(loss_funcs) == 0:
        raise ValueError("loss_funcs must contain at least one loss function")
    total = torch.tensor(0.0, device=depth_pred.device)
    sparse_mask = depth_target > 0
    for loss_func in loss_funcs:
        if loss_func == "l1":
            # Normal l1 loss function
            total += torch.nn.functional.l1_loss(
                depth_pred[sparse_mask], depth_target[sparse_mask]
            )
        elif loss_func == "l2":
            # Normal l2 loss function
            total += torch.nn.functional.mse_loss(
                depth_pred[sparse_mask], depth_target[sparse_mask]
            )
        elif loss_func == "edge":
            # Edge-preserving loss function using gradients
            # of predicted depth and input image
            if image is None:
                raise ValueError("image must be provided for edge loss")
            num_channels = image.shape[1]
            if num_channels == 3:
                # Convert image to grayscale
                gray_image = (
                    0.299 * image[:, 0:1]
                    + 0.587 * image[:, 1:2]
                    + 0.114 * image[:, 2:3]
                )  # [N, 1, H, W]
            elif num_channels == 1:
                gray_image = image
            else:
                raise ValueError(f"Image must have 1 or 3 channels, got {num_channels}")
            # Gradients of depth map in x and y directions
            grad_pred_x = torch.abs(depth_pred[:, :, :, :-1] - depth_pred[:, :, :, 1:])
            grad_pred_y = torch.abs(depth_pred[:, :, :-1, :] - depth_pred[:, :, 1:, :])

            # Gradients of image (grayscale) in x and y directions
            grad_gray_x = torch.abs(gray_image[:, :, :, :-1] - gray_image[:, :, :, 1:])
            grad_gray_y = torch.abs(gray_image[:, :, :-1, :] - gray_image[:, :, 1:, :])

            # The more similar the gradients of depth and image are,
            # the more aligned the edges are considered to be
            loss = torch.mean(torch.abs(grad_pred_x - grad_gray_x)) + torch.mean(
                torch.abs(grad_pred_y - grad_gray_y)
            )
            total += loss
        elif loss_func == "smooth":
            # Smoothness loss function
            # of predicted depth and input image
            if image is None:
                raise ValueError("image must be provided for smooth loss")
            loss_h = torch.mean(
                torch.abs(depth_pred[:, :, :-1, :] - depth_pred[:, :, 1:, :])
            )
            loss_w = torch.mean(
                torch.abs(depth_pred[:, :, :, :-1] - depth_pred[:, :, :, 1:])
            )
            total += loss_h + loss_w
        else:
            raise ValueError(f"Unknown loss function: {loss_func}")
    return total


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
        empty_text_embedding (torch.Tensor): Cached empty text embedding for conditioning
        tokenizer: Tokenizer for text inputs
        text_encoder: Text encoder model
        unet: U-Net model for denoising
        scheduler: Diffusion scheduler
        image_processor: Processor for image preprocessing and postprocessing
        dtype: Data type for model computation
    """

    def __call__(
        self,
        image: np.ndarray,
        sparse: np.ndarray,
        steps: int = 50,
        resolution: int = 768,
        seed: int = 2024,
        elemwise_scaling: bool = False,
        interp_mode: str = "bilinear",
        loss_funcs: list[str] | None = None,
        aa: bool = False,
    ) -> np.ndarray:
        """
        Perform depth completion on an RGB image using sparse depth measurements.

        Args:
            image (np.ndarray): Input RGB image of shape [H, W, C].
            sparse (np.ndarray): Sparse depth measurements of shape [H, W].
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

        Returns:
            np.ndarray: Dense depth prediction of shape [H, W].

        Raises:
            ValueError: If sparse is not a 2D numpy array.
        """  # noqa: E501

        # Check inputs.
        if sparse.ndim != 2:
            raise ValueError(
                "Sparse depth map must be a 2d numpy "
                "array with zeros at missing positions"
            )

        # Set loss functions
        if loss_funcs is None:
            loss_funcs = ["l1", "l2"]

        # Set device
        device: torch.device = self._execution_device

        # Get random generator
        generator = torch.Generator(device=device).manual_seed(seed)

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
                text_input_ids = text_inputs.input_ids.to(self._execution_device)
                self.empty_text_embedding = self.text_encoder(text_input_ids)[0]

        # Preprocess input images
        # Convert image to channel-first format (N, C, H, W)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).to(device)
        image_tensor_resized, padding, original_resolution = (
            self.image_processor.preprocess(
                image_tensor,
                processing_resolution=resolution,
                device=device,
                dtype=self.dtype,
            )
        )  # (N, 3, PPH, PPW)

        # Encode input image into latent space
        with torch.no_grad():
            image_latent, pred_latent = self.prepare_latents(
                image_tensor_resized, None, generator, 1, 1
            )  # (N * E, 4, h, w), (N * E, 4, h, w)

        # Preprocess sparse depth
        sparse = torch.from_numpy(sparse)[None, None].float().to(device)
        sparse_mask = sparse > 0
        guided_points = sparse[sparse_mask]

        # Set up optimization targets and compute
        # the range and lower bound of the sparse depth
        if elemwise_scaling:
            scale, shift = torch.nn.Parameter(
                torch.ones(original_resolution, device=device)
            ), torch.nn.Parameter(torch.ones(original_resolution, device=device))
        else:
            scale, shift = torch.nn.Parameter(
                torch.ones(1, device=device)
            ), torch.nn.Parameter(torch.ones(1, device=device))
        pred_latent = torch.nn.Parameter(pred_latent)
        sparse_min = guided_points.min()
        sparse_max = guided_points.max()
        sparse_range = sparse_max - sparse_min

        # Set up optimizer
        optimizer = torch.optim.Adam(
            [
                {"params": [scale, shift], "lr": 0.005},
                {"params": [pred_latent], "lr": 0.05},
            ]
        )

        def affine_to_metric(depth: torch.Tensor) -> torch.Tensor:
            return (scale**2) * sparse_range * depth + (shift**2) * sparse_min

        def latent_to_metric(latent: torch.Tensor) -> torch.Tensor:
            affine_invariant_pred = self.decode_prediction(latent)  # (E, 1, PPH, PPW)
            affine_invariant_pred = self.image_processor.unpad_image(
                affine_invariant_pred, padding
            )  # (E, 1, PH, PW)
            affine_invariant_pred = self.image_processor.resize_antialias(
                affine_invariant_pred,
                original_resolution,
                interp_mode,
                is_aa=aa,
            )  # (E, 1, H, W)
            pred = affine_to_metric(affine_invariant_pred)
            return pred

        # Denoising loop
        self.scheduler.set_timesteps(steps, device=device)
        for t in self.scheduler.timesteps:
            optimizer.zero_grad()

            # Forward pass through the U-Net
            latent = torch.cat([image_latent, pred_latent], dim=1)  # (1, 8, h, w)
            noise = self.unet(  # (1, 4, h, w)
                latent,
                t,
                encoder_hidden_states=self.empty_text_embedding,
                return_dict=False,
            )[0]

            # Compute pred_epsilon to later rescale the depth latent gradient
            with torch.no_grad():
                a_prod_t = self.scheduler.alphas_cumprod[t]
                b_prod_t = 1 - a_prod_t
                pred_epsilon = (a_prod_t**0.5) * noise + (b_prod_t**0.5) * pred_latent

            # Forward denoising step
            step_output = self.scheduler.step(
                noise, t, pred_latent, generator=generator
            )

            # Preview the final output depth with
            # Tweedie's formula (See Equation 1 of the paper)
            preview = step_output.pred_original_sample

            # Decode to metric space, compute loss with guidance and backpropagate
            dense = latent_to_metric(preview)
            loss = compute_loss(
                dense,
                sparse,
                loss_funcs,
                image=image_tensor,
            )
            loss.backward()

            # Scale gradients up
            with torch.no_grad():
                pred_epsilon_norm = torch.linalg.norm(pred_epsilon).item()
                depth_latent_grad_norm = torch.linalg.norm(pred_latent.grad).item()
                scaling_factor = pred_epsilon_norm / max(depth_latent_grad_norm, 1e-8)
                pred_latent.grad *= scaling_factor

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
        return dense.squeeze()
