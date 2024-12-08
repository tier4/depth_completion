# Copyright 2024 Massimiliano Viola, Kevin Qu, Anton Obukhov, ETH Zurich.
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

import os
import warnings
import argparse

import diffusers
import numpy as np
import torch
from diffusers import DDIMScheduler, MarigoldDepthPipeline
from PIL import Image

warnings.simplefilter(action="ignore", category=FutureWarning)
diffusers.utils.logging.disable_progress_bar()


class MarigoldDepthCompletionPipeline(MarigoldDepthPipeline):
    def __call__(self, image, sparse_depth, num_inference_steps=50, seed=2024):  # noqa
        # Resolving variables
        device = self._execution_device
        generator = torch.Generator(device=device).manual_seed(seed)

        # Check inputs.
        if num_inference_steps is None:
            raise ValueError("Invalid num_inference_steps")
        if type(sparse_depth) is not np.ndarray or sparse_depth.ndim != 2:
            raise ValueError("Sparse depth should be a 2D numpy ndarray with zeros at missing positions")

        with torch.no_grad():
            # Prepare empty text conditioning
            if self.empty_text_embedding is None:
                prompt = ""
                text_inputs = self.tokenizer(
                    prompt,
                    padding="do_not_pad",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(device)
                self.empty_text_embedding = self.text_encoder(text_input_ids)[0]  # [1,2,1024]

        # Preprocess input images
        image, padding, original_resolution = self.image_processor.preprocess(
            image, processing_resolution=0, device=device, dtype=self.dtype
        )  # [N,3,PPH,PPW]

        if sparse_depth.shape != image.shape[-2:]:
            raise ValueError(
                f"Sparse depth dimensions ({sparse_depth.shape}) must match that of the image ({image.shape[-2:]})"
            )
        with torch.no_grad():
            # Encode input image into latent space
            image_latent, pred_latent = self.prepare_latents(image, None, generator, 1, 1)  # [N*E,4,h,w], [N*E,4,h,w]
        del image

        # Preprocess sparse depth
        sparse_depth = torch.from_numpy(sparse_depth)[None, None].float()
        sparse_depth = sparse_depth.to(device)
        sparse_mask = sparse_depth > 0
        print(f"Using {sparse_mask.int().sum().item()} guidance points")

        # Set up optimization targets

        scale = torch.nn.Parameter(torch.ones(1, device=device), requires_grad=True)
        shift = torch.nn.Parameter(torch.ones(1, device=device), requires_grad=True)
        pred_latent = torch.nn.Parameter(pred_latent, requires_grad=True)

        sparse_range = (sparse_depth[sparse_mask].max() - sparse_depth[sparse_mask].min()).item()
        sparse_lower = (sparse_depth[sparse_mask].min()).item()

        def to_metric(d):
            return (scale**2) * sparse_range * d + (shift**2) * sparse_lower

        optimizer = torch.optim.Adam(
            [
                {"params": [scale, shift], "lr": 0.005},
                {"params": [pred_latent], "lr": 0.05},
            ]
        )

        # Process the denoising loop

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        for iter, t in enumerate(
            self.progress_bar(self.scheduler.timesteps, desc=f"Marigold-DC steps ({str(device)})...")
        ):
            optimizer.zero_grad()

            batch_latent = torch.cat([image_latent, pred_latent], dim=1)  # [1,8,h,w]
            noise = self.unet(batch_latent, t, encoder_hidden_states=self.empty_text_embedding, return_dict=False)[
                0
            ]  # [1,4,h,w]

            # Compute pred_epsilon to later rescale the depth latent gradient
            with torch.no_grad():
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                pred_epsilon = (alpha_prod_t**0.5) * noise + (beta_prod_t**0.5) * pred_latent

            step_output = self.scheduler.step(noise, t, pred_latent, generator=generator)

            # Preview the final output, clip, rescale, resize
            pred_original_sample = step_output.pred_original_sample
            current_x0_depth = self.decode_prediction(pred_original_sample)  # [1,1,h,w], range in [0,1]
            resized_current_x0_depth = torch.nn.functional.interpolate(
                current_x0_depth,
                sparse_depth.shape[-2:],
                mode="bilinear",
                antialias=True,
            )
            resized_current_x0_depth = torch.clamp(resized_current_x0_depth, 0, 1)  # [1,1,h,w]

            current_metric_estimate = to_metric(resized_current_x0_depth)  # [1,1,h,w]

            def loss_l1l2(input, target):
                return torch.nn.functional.l1_loss(input, target) + torch.nn.functional.mse_loss(input, target)

            loss = loss_l1l2(current_metric_estimate[sparse_mask], sparse_depth[sparse_mask])
            loss.backward()

            # Scale gradients up
            with torch.no_grad():
                pred_epsilon_norm = torch.linalg.norm(pred_epsilon).item()
                depth_latent_grad_norm = torch.linalg.norm(pred_latent.grad).item()
                scaling_factor = pred_epsilon_norm / max(depth_latent_grad_norm, 1e-8)
                pred_latent.grad *= scaling_factor

            optimizer.step()

            with torch.no_grad():
                pred_latent.data = self.scheduler.step(noise, t, pred_latent, generator=generator).prev_sample
            del current_x0_depth, pred_original_sample, current_metric_estimate, step_output, resized_current_x0_depth, pred_epsilon, noise
            torch.cuda.empty_cache()

        del image_latent

        # Decode predictions from latent into pixel space
        with torch.no_grad():
            affine_invariant_prediction = self.decode_prediction(pred_latent)  # [E,1,PPH,PPW]
            prediction = to_metric(affine_invariant_prediction).detach()

        # Remove padding. The output shape is (PH, PW)
        prediction = self.image_processor.unpad_image(prediction, padding)  # [E,1,PH,PW]

        # Revert to the input resolution
        prediction = self.image_processor.resize_antialias(
            prediction, original_resolution, "bilinear", is_aa=False
        )  # [1,1,H,W]

        # Prepare the final outputs
        prediction = self.image_processor.pt_to_numpy(prediction)  # [N,H,W,1]

        # Offload all models
        self.maybe_free_model_hooks()

        return prediction.squeeze()


def main():
    parser = argparse.ArgumentParser(description="Marigold-DC Pipeline")

    DEPTH_CHECKPOINT = "prs-eth/marigold-depth-v1-0"
    parser.add_argument("--in-image", type=str, default="data/image.png", help="Input image")
    parser.add_argument("--in-depth", type=str, default="data/sparse_100.npy", help="Input sparse depth")
    parser.add_argument("--out-depth", type=str, default="data/dense_100.npy", help="Output dense depth")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--checkpoint", type=str, default=DEPTH_CHECKPOINT, help="Depth checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    pipe = MarigoldDepthCompletionPipeline.from_pretrained(args.checkpoint, prediction_type="depth").to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    pred = pipe(
        image=Image.open(args.in_image),
        sparse_depth=np.load(args.in_depth),
        num_inference_steps=args.num_inference_steps,
    )

    np.save(args.out_depth, pred)
    vis = pipe.image_processor.visualize_depth(pred, val_min=pred.min(), val_max=pred.max())[0]
    vis.save(os.path.splitext(args.out_depth)[0] + "_vis.jpg")


if __name__ == "__main__":
    main()
