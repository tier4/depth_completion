import sys
from pathlib import Path

import click
import cv2
import numpy as np
import torch
from diffusers import DDIMScheduler
from loguru import logger
from PIL import Image

from marigold_dc import MarigoldDepthCompletionPipeline
from utils import CommaSeparated, get_img_paths, is_empty_img, make_grid, to_depth

DEPTH_CKPT = "prs-eth/marigold-depth-v1-0"
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
    "-n",
    "--steps",
    type=click.IntRange(min=1),
    default=50,
    help="Number of denoising steps.",
    show_default=True,
)
@click.option(
    "-r",
    "--resolution",
    type=click.IntRange(min=1),
    default=768,
    help="Input resolution. Input images will be resized to resolution x resolution.",
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
    type=CommaSeparated(int, n=2),
    default=None,
    show_default=True,
)
def main(
    img_dir: Path,
    depth_dir: Path,
    out_dir: Path,
    steps: int,
    resolution: int,
    max_distance: float,
    output_size: list[int] | None,
) -> None:
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.critical("CUDA must be available to run this script.")
        sys.exit(1)

    # Get paths of input images
    input_img_paths = get_img_paths(img_dir)

    # Get paths of input depth maps
    input_depth_paths: list[Path] = []
    for path in input_img_paths:
        depth_path = depth_dir / path.relative_to(img_dir).with_suffix(".png")
        if not depth_path.exists():
            logger.warning(f"No depth map found for image {path} (skipping)")
            continue
        input_depth_paths.append(depth_path)
    assert len(input_depth_paths) == len(input_img_paths)
    logger.info(f"Found {len(input_depth_paths):,} input image-depth pairs")

    # Create output directory if it doesn't exist
    if img_dir.is_dir() and not out_dir.exists():
        out_dir.mkdir(parents=True)
        logger.info(f"Created output directory: {out_dir}")

    # Initialize pipeline
    pipe = MarigoldDepthCompletionPipeline.from_pretrained(
        DEPTH_CKPT, prediction_type="depth"
    ).to(torch.device("cuda"))
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    # Inference
    logger.info("Starting inference")
    for i, (img_path, depth_path) in enumerate(
        zip(input_img_paths, input_depth_paths, strict=False)
    ):
        logger.info(
            f"[{i+1:,} / {len(input_img_paths):,}] "
            f"Processing {img_path} and {depth_path}"
        )
        img = Image.open(img_path).convert("RGB")
        if is_empty_img(img):
            logger.warning(f"Empty input image found: {img_path} (skipping)")
            continue
        depth_img = Image.open(depth_path).convert("RGB")
        if is_empty_img(depth_img):
            logger.warning(f"Empty input depth map found: {depth_path} (skipping)")
            continue
        depth = to_depth(depth_img, max_distance=max_distance)
        preds = pipe(
            image=img,
            sparse_depth=depth,
            num_inference_steps=steps,
            processing_resolution=resolution,
        )
        out_img = pipe.image_processor.visualize_depth(
            preds, val_min=0, val_max=max_distance
        )[0]
        depth_img_cnv = pipe.image_processor.visualize_depth(
            depth, val_min=0, val_max=max_distance
        )[0]
        depth_map_vis = np.array(depth_img_cnv)
        depth_map_vis[depth < EPSILON] = 0
        depth_img_cnv = Image.fromarray(depth_map_vis)
        grid_img = Image.fromarray(
            make_grid(
                np.stack(
                    [np.asarray(im) for im in [img, depth_img_cnv, out_img]], axis=0
                ),
                rows=1,
                cols=3,
                resize=(
                    (output_size[0], output_size[1])
                    if output_size is not None
                    else None
                ),
                # NOTE: Resize depth map with nearest neighbor interpolation
                interpolation=[cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_LINEAR],
            )
        )
        save_dir = (out_dir / img_path.relative_to(img_dir)).parent
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        # Save output depth array
        save_path = save_dir / f"{img_path.stem}.npy"
        np.save(save_path, preds)
        logger.info(f"Saved depth array at {save_path}")

        # Save visualization output
        save_path = save_dir / f"{img_path.stem}_vis.jpg"
        grid_img.save(save_path)
        logger.info(f"Saved visualization of output depth map at {save_path}")


if __name__ == "__main__":
    main()
