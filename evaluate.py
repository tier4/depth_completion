import sys
from pathlib import Path

import click
import torch
from loguru import logger

from utils import get_img_paths

DEPTH_CKPT = "prs-eth/marigold-depth-v1-0"


@click.command()
@click.argument("input_img", type=click.Path(exists=True, path_type=Path))
@click.argument("input_depth", type=click.Path(exists=True, path_type=Path))
@click.argument("output_depth", type=click.Path(exists=False, path_type=Path))
@click.option(
    "-n", "--steps", type=int, default=50, help="Number of denoising steps.", show_default=True
)
@click.option(
    "-r", "--resolution", type=int, default=768, help="Input resolution.", show_default=True
)
def main(
    input_img: Path, input_depth: Path, output_depth: Path, steps: int, resolution: int
) -> None:
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.critical("CUDA must be available to run this script.")
        sys.exit(1)

    # Get paths of input images
    if input_img.is_dir():
        # Search for all images in the directory
        input_img_paths = get_img_paths(input_img)
    else:
        input_img_paths = [input_img]

    # Get paths of input depth maps
    if input_depth.is_dir():
        input_depth_paths = get_img_paths(input_depth)
    else:
        input_depth_paths = [input_depth]

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
        logger.info(f"Found {len(input_pairs)} input image-depth pairs")


if __name__ == "__main__":
    main()
