import sys
from pathlib import Path
from typing import Literal

import click
from loguru import logger

import utils


@click.group(help="Utility to convert data into a specific format.")
def cli() -> None:
    pass


@cli.command(help="Convert depth images to depth arrays.")
@click.argument("data_format", type=click.Choice(["comlops"]), default="comlops")
@click.argument("depth_img_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("out_dir", type=click.Path(path_type=Path))
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
    "-c",
    "--compress",
    type=click.Choice(["npz", "bl2", "none"]),
    default="bl2",
    help="Specify the compression format for the output depth maps. If none, saves uncompressed.",
    show_default=True,
)
def depth_img2array(
    data_format: Literal["comlops"],
    depth_img_dir: Path,
    out_dir: Path,
    log: Path,
    log_level: Literal[
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    ],
    compress: Literal["npz", "bl2", "none"],
) -> None:
    # Set log level
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.debug(f"Set log level to {log_level}")

    # Configure logger if log path is provided
    if log is not None:
        if not log.parent.exists():
            log.parent.mkdir(parents=True)
        logger.add(log, rotation="10 MB", level=log_level)
        logger.info(f"Saving logs to {log}")

    # Get paths of input depth images
    depth_img_paths = utils.get_img_paths(depth_img_dir)

    # Sort input image paths by filename
    depth_img_paths.sort(key=lambda x: x.name)
    logger.info(f"Found {len(depth_img_paths):,} input depth images in {depth_img_dir}")

    # Create output root directory if it does not exist
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
        logger.info(f"Created output root directory at {out_dir}")

    # Set max distance
    if data_format == "comlops":
        max_distance = 120.0
    else:
        logger.critical("Invalid format: {format}")
        sys.exit(1)

    # Conversion loop
    for idx, depth_img_path in enumerate(depth_img_paths):
        # Load depth image
        depth_img, is_valid = utils.load_img(depth_img_path, "RGB")
        if not is_valid:
            logger.warning(
                f"Empty input depth image found: {depth_img_path} (skipping)"
            )
            continue

        # Convert depth image to depth array
        depth_array = utils.to_depth_map(depth_img, max_distance=max_distance)

        # Save depth array
        save_dir = out_dir / (depth_img_path.relative_to(depth_img_dir)).parent
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
            logger.info(f"Created output directory at {save_dir}")
        save_path = save_dir / f"{depth_img_path.stem}"
        if compress != "none":
            save_path = save_path.with_suffix(f".{compress}")
        else:
            save_path = save_path.with_suffix(".npy")
        utils.save_array(
            depth_array, save_path, compress=compress if compress != "none" else None
        )
        logger.info(
            f"[{idx+1:,}/{len(depth_img_paths):,}] Saved depth array at {save_path}"
        )
    logger.success(f"Finished converting {len(depth_img_paths):,} depth images")


if __name__ == "__main__":
    cli()
