import sys
import time
from pathlib import Path
from typing import Literal

import click
import cv2
import numpy as np
import torch
from diffusers import AutoencoderTiny, DDIMScheduler
from loguru import logger
from PIL import Image

import utils
from marigold_dc import MarigoldDepthCompletionPipeline

NPARRAY_EXTS = [".bl2", ".npz", ".npy"]
MARIGOLD_CKPT_ORIGINAL = "prs-eth/marigold-v1-0"
VAE_CKPT_LIGHT = "madebyollin/taesd"
EPSILON = 1e-6


@click.command()
@click.argument(
    "img_dir",
    type=click.Path(exists=True, path_type=Path, file_okay=False, dir_okay=True),
)
@click.argument(
    "depth_dir",
    type=click.Path(exists=True, path_type=Path, file_okay=False, dir_okay=True),
)
@click.argument("out_dir", type=click.Path(exists=False, path_type=Path))
@click.option(
    "--vae",
    type=click.Choice(["original", "light"]),
    default="light",
    help="VAE model to use for depth completion. "
    "original - The original VAE model from Marigold (e.g. Stable Diffusion VAE). "
    f"light - A lightweight VAE model from {VAE_CKPT_LIGHT}.",
    show_default=True,
)
@click.option(
    "-n",
    "--steps",
    type=click.IntRange(min=1),
    default=50,
    help="Number of denoising steps.",
    show_default=True,
)
@click.option(
    "-r",
    "--res",
    type=click.IntRange(min=1),
    default=768,
    help="Input resolution. Input images will be resized to ${res} x ${res}.",
    show_default=True,
)
@click.option(
    "--max-distance",
    type=click.FloatRange(min=0, min_open=True),
    default=120.0,
    help="Max absolute distance [m] of input sparse depth maps.",
    show_default=True,
)
@click.option(
    "-os",
    "--output-size",
    type=utils.CommaSeparated(int, n=2),
    default=None,
    show_default=True,
)
@click.option(
    "-v",
    "--vis",
    type=bool,
    default=True,
    show_default=True,
    help="Whether to save visualization of output depth maps.",
)
@click.option(
    "--log",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to save logs.",
    show_default=True,
)
@click.option(
    "--log-level",
    type=click.Choice(
        ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
    ),
    default="INFO",
    help="Minimum log level to output.",
    show_default=True,
)
@click.option(
    "-dt",
    "--dtype",
    type=click.Choice(["bf16", "fp32"]),
    default="bf16",
    help="Data type for inference.",
    show_default=True,
)
@click.option(
    "-c",
    "--compress",
    type=click.Choice(["npz", "bl2", "none"]),
    default="bl2",
    help="Specify the compression format for the output depth maps. If none, saves uncompressed.",
    show_default=True,
)
def main(
    img_dir: Path,
    depth_dir: Path,
    out_dir: Path,
    vae: Literal["original", "light"],
    steps: int,
    res: int,
    max_distance: float,
    output_size: list[int] | None,
    vis: bool,
    log: Path | None,
    log_level: Literal[
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    ],
    dtype: Literal["bf16", "fp32"],
    compress: Literal["npz", "bl2", "none"],
) -> None:
    # Set log level
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    # Configure logger if log path is provided
    if log is not None:
        if not log.parent.exists():
            log.parent.mkdir(parents=True)
        logger.add(log, rotation="10 MB", level=log_level)
        logger.info(f"Saving logs to {log}")

    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.critical("CUDA must be available to run this script.")
        sys.exit(1)

    # Get paths of input images
    img_paths_all = utils.get_img_paths(img_dir)

    # Sort input image paths by filename
    img_paths_all.sort(key=lambda x: x.name)

    # Get paths of input depth images
    depth_paths: list[Path] = []
    img_paths: list[Path] = []
    for path in img_paths_all:
        # NOTE:
        # Get corresponding depth file path by checking each supported extension
        # NPARRAY_EXTS contains [".bl2", ".npz", ".npy"]
        # Constructs paths by:
        # 1. Getting relative path from img_dir
        # 2. Changing extension to each supported type
        # 3. Joining with depth_dir
        # Then filters to find first existing path
        depth_path_candidates = list(
            filter(
                lambda p: p.exists(),
                [
                    depth_dir / path.relative_to(img_dir).with_suffix(ext)
                    for ext in NPARRAY_EXTS
                ],
            )
        )
        if len(depth_path_candidates) == 0:
            logger.warning(f"No depth map found for {path} (skipping)")
            continue
        depth_paths.append(depth_path_candidates[0])
        img_paths.append(path)
    assert len(depth_paths) == len(img_paths)
    if len(img_paths) == 0:
        logger.critical("No valid image-depth pairs found")
        sys.exit(1)
    logger.info(f"Found {len(depth_paths):,} input image-depth pairs")

    # Create output directory if it doesn't exist
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
        logger.info(f"Created output directory at {out_dir}")

    # Initialize pipeline
    # NOTE: Do not use float16 as it will make nans in predictions
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
    pipe = MarigoldDepthCompletionPipeline.from_pretrained(
        MARIGOLD_CKPT_ORIGINAL,
        prediction_type="depth",
        torch_dtype=torch_dtype,
    ).to("cuda")
    if vae == "light":
        del pipe.vae
        pipe.vae = AutoencoderTiny.from_pretrained(
            VAE_CKPT_LIGHT, torch_dtype=torch_dtype
        ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    logger.info(f"Initialized inference pipeline (dtype={dtype}, vae={vae})")

    # Evaluation loop
    for i, (img_path, depth_path) in enumerate(
        zip(img_paths, depth_paths, strict=True)
    ):
        logger.info(
            f"[{i+1:,} / {len(img_paths):,}] " f"Processing {img_path} and {depth_path}"
        )

        # Load camera image
        img, is_valid = utils.load_img(img_path, "RGB")
        if not is_valid:
            logger.warning(f"Empty input image found: {img_path} (skipping)")
            continue

        # Load depth map
        depth = utils.load_array(depth_path)
        depth_mask = depth > EPSILON

        # Infer camera category
        camera_category = utils.infer_camera_category(img_path)
        if camera_category is None:
            logger.warning(
                f"Could not infer camera category of this image file: {img_path} (skipping)"
            )
            continue

        # Run inference
        start_time = time.time()
        depth_pred = pipe(
            image=img,
            sparse_depth=depth,
            num_inference_steps=steps,
            processing_resolution=res,
        )
        duration_pred = time.time() - start_time
        logger.debug(f"Predicted depth map in {duration_pred:.2f} seconds")
        if utils.has_nan(depth_pred):
            logger.warning("NaN values found in depth map prediction (skipping)")
            continue

        # Save predicted depth map
        save_dir = (out_dir / "depth" / img_path.relative_to(img_dir)).parent
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
            logger.info(f"Created output directory for saving depth maps at {save_dir}")
        if compress != "none":
            depth_pred_path = save_dir / img_path.with_suffix(f".{compress}").name
        else:
            depth_pred_path = save_dir / img_path.with_suffix(".npy").name

        utils.save_array(
            depth_pred,
            depth_pred_path,
            compress=compress if compress != "none" else None,
        )
        logger.info(f"Saved predicted depth map at {depth_pred_path}")

        # Save visualization of predicted depth map
        if vis:
            depth_visualized = pipe.image_processor.visualize_depth(
                depth, val_min=0, val_max=max_distance
            )[0]
            depth_visualized = np.array(depth_visualized)
            depth_visualized[~depth_mask] = 0
            depth_visualized = Image.fromarray(depth_visualized)
            depth_pred_visualized = pipe.image_processor.visualize_depth(
                depth_pred, val_min=0, val_max=max_distance
            )[0]
            visualized = Image.fromarray(
                utils.make_grid(
                    np.stack(
                        [
                            np.asarray(im)
                            for im in [img, depth_visualized, depth_pred_visualized]
                        ],
                        axis=0,
                    ),
                    rows=1,
                    cols=3,
                    resize=(
                        (output_size[0], output_size[1])
                        if output_size is not None
                        else None
                    ),
                    # NOTE: Resize depth map with nearest neighbor interpolation
                    interpolation=[
                        cv2.INTER_LINEAR,
                        cv2.INTER_NEAREST,
                        cv2.INTER_LINEAR,
                    ],
                )
            )
            save_dir = (out_dir / "vis" / img_path.relative_to(img_dir)).parent
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
                logger.info(
                    f"Created directory for saving visualization outputs at {save_dir}"
                )
            visualized_path = save_dir / f"{img_path.stem}_vis.jpg"
            visualized.save(visualized_path)
            logger.info(f"Saved visualized outputs at {visualized_path}")
    logger.success(f"Finished processing {len(img_paths):,} image-depth pairs")


if __name__ == "__main__":
    main()
