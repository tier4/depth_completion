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
@click.argument("input_img", type=click.Path(exists=True, path_type=Path))
@click.argument("input_depth", type=click.Path(exists=True, path_type=Path))
@click.argument("output_depth", type=click.Path(exists=False, path_type=Path))
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
    input_img: Path,
    input_depth: Path,
    output_depth: Path,
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
    input_img_paths = get_img_paths(input_img) if input_img.is_dir() else [input_img]

    # Get paths of input depth maps
    input_depth_paths = (
        get_img_paths(input_depth) if input_depth.is_dir() else [input_depth]
    )

    # Create mapping of image stem to path
    img_stem_to_path = {}
    img_stems_to_skip = []
    for path in input_img_paths:
        if path.stem in img_stem_to_path:
            logger.warning(f"Duplicate image filename found: {path.stem} (skipping)")
            img_stems_to_skip.append(path.stem)
        else:
            img_stem_to_path[path.stem] = path
    for key in img_stems_to_skip:
        img_stem_to_path.pop(key)

    # Create mapping of depth stem to path
    depth_stem_to_path = {}
    depth_stems_to_skip = []
    for path in input_depth_paths:
        if path.stem in depth_stem_to_path:
            logger.warning(f"Duplicate depth filename found: {path.stem} (skipping)")
            depth_stems_to_skip.append(path.stem)
        else:
            depth_stem_to_path[path.stem] = path
    for key in depth_stems_to_skip:
        depth_stem_to_path.pop(key)

    # Create output directory if it doesn't exist
    if input_img.is_dir() and not output_depth.exists():
        output_depth.mkdir(parents=True)
        logger.info(f"Created output directory: {output_depth}")

    # Create list of corresponding image-depth pairs
    input_pairs: list[tuple[Path, Path]] = []
    for stem in img_stem_to_path:
        if stem in depth_stem_to_path:
            input_pairs.append((img_stem_to_path[stem], depth_stem_to_path[stem]))
        else:
            logger.warning(f"No depth map found for image {img_stem_to_path[stem]}")
    if len(input_pairs) == 0:
        logger.critical("No input image-depth pairs found")
        sys.exit(1)
    else:
        logger.info(f"Found {len(input_pairs):,} input image-depth pairs")

    # Initialize pipeline
    pipe = MarigoldDepthCompletionPipeline.from_pretrained(
        DEPTH_CKPT, prediction_type="depth"
    ).to(torch.device("cuda"))
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    # Inference
    logger.info("Starting inference")
    for i, (img_path, depth_path) in enumerate(input_pairs):
        logger.info(
            f"[{i+1:,} / {len(input_pairs):,}] Processing {img_path} and {depth_path}"
        )
        img = Image.open(img_path).convert("RGB")
        if is_empty_img(img):
            logger.warning(f"Empty input image found: {img_path} (skipping)")
            continue
        depth = to_depth(
            Image.open(depth_path).convert("RGB"), max_distance=max_distance
        )
        preds = pipe(
            image=img,
            sparse_depth=depth,
            depth_range=(0, max_distance),
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
        if output_depth.is_dir():
            save_path = output_depth / f"{img_path.stem}_vis.jpg"
        else:
            save_path = output_depth
        grid_img.save(save_path)
        logger.info(f"Saved visualization of output depth map at {save_path}")


if __name__ == "__main__":
    main()
