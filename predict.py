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
from marigold_dc import (
    MARIGOLD_CKPT_LCM,
    MARIGOLD_CKPT_ORIGINAL,
    VAE_CKPT_LIGHT,
    MarigoldDepthCompletionPipeline,
)
from utils import NPARRAY_EXTENSIONS

torch.set_float32_matmul_precision("high")  # NOTE: Optimize fp32 arithmetic


@click.command(
    help="Predict dense depth maps from sparse depth maps and camera images."
)
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
    "--seg-dir",
    type=click.Path(exists=True, path_type=Path, file_okay=False, dir_okay=True),
    default=None,
    help="Path to directory containing segmentation maps to use for depth completion.",
    show_default=True,
)
@click.option(
    "--marigold",
    type=click.Choice(["original", "lcm"]),
    default="original",
    help="Marigold model to use for depth completion. "
    "original - The original Marigold model. "
    "lcm - The LCM-based Marigold model.",
    show_default=True,
)
@click.option(
    "--vae",
    type=click.Choice(["original", "light"]),
    default="original",
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
    "-v",
    "--vis",
    type=bool,
    default=True,
    show_default=True,
    help="Whether to save visualization of output depth maps.",
)
@click.option(
    "--save-depth-map",
    type=bool,
    default=True,
    help="Whether to save the inferenced depth maps. "
    "Output format can be specified with --compress.",
    show_default=True,
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
    type=click.Choice(["fp16", "bf16", "fp32"]),
    default="fp32",
    help="Data precision for inference.",
    show_default=True,
)
@click.option(
    "-c",
    "--compress",
    type=click.Choice(["npz", "bl2", "none"]),
    default="bl2",
    help="Specify the compression format for the output depth maps. "
    "If none, saves uncompressed. "
    "This option is ignored if --save-depth-map=False",
    show_default=True,
)
@click.option(
    "--use-compile",
    type=bool,
    default=True,
    help="Whether to compile the inference pipeline using torch.compile.",
    show_default=True,
)
@click.option(
    "--elemwise-scaling",
    type=bool,
    default=False,
    help="Whether to use element-wise scaling for depth completion.",
    show_default=True,
)
@click.option(
    "--postprocess",
    type=bool,
    default=True,
    help="Whether to postprocess the predicted depth maps.",
    show_default=True,
)
def main(
    img_dir: Path,
    depth_dir: Path,
    out_dir: Path,
    seg_dir: Path | None,
    marigold: Literal["original", "lcm"],
    vae: Literal["original", "light"],
    steps: int,
    res: int,
    max_distance: float,
    save_depth_map: bool,
    vis: bool,
    log: Path | None,
    log_level: Literal[
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    ],
    dtype: Literal["bf16", "fp32"],
    compress: Literal["npz", "bl2", "none"],
    postprocess: bool,
    use_compile: bool,
    elemwise_scaling: bool,
) -> None:
    # Set log level
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    # Configure logger if log path is provided
    if log is not None:
        if not log.parent.exists():
            log.parent.mkdir(parents=True)
        logger.add(log, rotation="100 MB", level=log_level)
        logger.info(f"Saving logs to {log}")

    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.critical("CUDA must be available to run this script.")
        sys.exit(1)

    # Load segmentation mapping data (if provided)
    if seg_dir is not None:
        seg_meta_path = seg_dir / "map.csv"
        if not seg_meta_path.exists():
            logger.error(f"Segmentation mapping file not found at {seg_meta_path}")
            seg_dir = None
        else:
            seg_meta = utils.load_csv(
                seg_meta_path,
                columns={"id": int, "name": str, "r": int, "g": int, "b": int},
            )
            try:
                seg_sky_id = seg_meta["id"][seg_meta["name"].index("sky")]
            except ValueError:
                logger.warning("class=sky not found in segmentation map")
                seg_sky_id = None

    # Get paths of input images
    img_paths_all = utils.get_img_paths(img_dir)

    # Sort input image paths by filename
    img_paths_all.sort(key=lambda x: x.name)

    # Get paths of input depth images and optionally segmentation maps
    depth_paths: list[Path] = []
    img_paths: list[Path] = []
    seg_paths: list[Path] = []
    for path in img_paths_all:
        depth_path_candidates = list(
            filter(
                lambda p: p.exists(),
                [
                    depth_dir / path.relative_to(img_dir).with_suffix(ext)
                    for ext in NPARRAY_EXTENSIONS
                ],
            )
        )
        if len(depth_path_candidates) == 0:
            logger.warning(f"No depth map found for {path} (skipping)")
            continue
        if seg_dir is not None:
            seg_path = seg_dir / path.relative_to(img_dir).with_suffix(".npy")
            if not seg_path.exists():
                logger.warning(f"No segmentation map found for {path} (skipping)")
                continue
        depth_paths.append(depth_path_candidates[0])
        img_paths.append(path)
        if seg_dir is not None:
            seg_paths.append(seg_path)
    if len(img_paths) == 0:
        logger.error("No valid input pairs found")
        sys.exit(1)
    logger.info(f"Found {len(depth_paths):,} input pairs")

    # Create output directory if it doesn't exist
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
        logger.info(f"Created output directory at {out_dir}")

    # Initialize pipeline
    # NOTE: Do not use float16 as it will make nans in predictions
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
    if marigold == "original":
        marigold_ckpt = MARIGOLD_CKPT_ORIGINAL
    else:  # marigold == "lcm"
        marigold_ckpt = MARIGOLD_CKPT_LCM
    pipe = MarigoldDepthCompletionPipeline.from_pretrained(
        marigold_ckpt,
        prediction_type="depth",
        torch_dtype=torch_dtype,
    ).to("cuda")
    if vae == "light":
        pipe.vae = AutoencoderTiny.from_pretrained(
            VAE_CKPT_LIGHT, torch_dtype=torch_dtype
        ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    if use_compile:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)
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

        # Load segmentation map
        seg: np.ndarray | None = None
        if seg_dir is not None:
            seg = utils.load_array(seg_paths[i])
            if seg.shape != depth.shape:
                logger.warning(
                    f"Segmentation map shape {seg.shape} does not match "
                    f"depth map shape {depth.shape}. "
                    f"Segmentation map will be ignored."
                )
                seg = None

        # Add segmentation hints to depth map
        if seg is not None:
            # Set sky pixels to max distance
            if seg_sky_id is not None:
                depth[(seg == seg_sky_id) & (depth <= 0)] = max_distance

        # Run inference
        start_time = time.time()
        depth_pred = pipe(
            image=img,
            sparse_depth=depth,
            num_inference_steps=steps,
            processing_resolution=res,
            elemwise_scaling=elemwise_scaling,
        )
        duration_pred = time.time() - start_time
        logger.info(f"Inference time: {duration_pred:.2f} seconds")
        if utils.has_nan(depth_pred):
            logger.error("NaN values found in inferenced depth map (skipping)")
            continue

        # Postprocess
        if postprocess:
            if seg_dir is not None:
                if seg_sky_id is not None:
                    depth_pred[seg == seg_sky_id] = max_distance

        # Save inferenced depth map
        if save_depth_map:
            save_dir = (out_dir / "depth" / img_path.relative_to(img_dir)).parent
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
                logger.info(
                    f"Created output directory for saving depth maps at {save_dir}"
                )
            if compress != "none":
                depth_pred_path = save_dir / img_path.with_suffix(f".{compress}").name
            else:
                depth_pred_path = save_dir / img_path.with_suffix(".npy").name
            utils.save_array(
                depth_pred,
                depth_pred_path,
                compress=compress if compress != "none" else None,
            )
            logger.info(f"Saved inferenced depth map at {depth_pred_path}")

        # Save visualization of inferenced depth map
        if vis:
            depth_visualized = pipe.image_processor.visualize_depth(
                depth, val_min=0, val_max=max_distance
            )[0]
            depth_visualized = np.array(depth_visualized)
            depth_visualized[depth <= 0] = 0
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
                )
            )
            save_dir = (out_dir / "vis" / img_path.relative_to(img_dir)).parent
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
                logger.info(
                    f"Created directory for saving visualization outputs at {save_dir}"
                )
            visualized_path = save_dir / f"{img_path.stem}_vis.png"
            visualized.save(visualized_path)
            logger.info(f"Saved visualized outputs at {visualized_path}")
    logger.success(f"Finished processing {len(img_paths):,} input pairs")


if __name__ == "__main__":
    main()
