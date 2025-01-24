import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Literal

import click
import cv2
import numpy as np
import torch
from diffusers import AutoencoderTiny, DDIMScheduler
from loguru import logger
from PIL import Image

from marigold_dc import MarigoldDepthCompletionPipeline
from utils import (
    CommaSeparated,
    get_img_paths,
    has_nan,
    load_img,
    mae,
    make_grid,
    reduce,
    rmse,
    to_depth_map,
)

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
    "--bin-size",
    type=click.FloatRange(min=0, min_open=True),
    default=10.0,
    help="Bin size [m] of input sparse depth maps for error evaluation.",
    show_default=True,
)
@click.option(
    "-os",
    "--output-size",
    type=CommaSeparated(int, n=2),
    default=None,
    show_default=True,
)
@click.option(
    "-v",
    "--visualize",
    type=bool,
    default=True,
    show_default=True,
    help="Whether to save visualization of output depth maps.",
)
@click.option(
    "--log",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to save logs. If not set, logs will only be shown in stdout.",
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
def main(
    img_dir: Path,
    depth_dir: Path,
    out_dir: Path,
    vae: Literal["original", "light"],
    steps: int,
    resolution: int,
    max_distance: float,
    bin_size: float,
    output_size: list[int] | None,
    visualize: bool,
    log: Path | None,
    dtype: Literal["bf16", "fp32"],
) -> None:
    # Configure logger if log path is provided
    if log is not None:
        if not log.parent.exists():
            log.parent.mkdir(parents=True)
        logger.add(log, rotation="10 MB")
        logger.info(f"Saving logs to {log}")

    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.critical("CUDA must be available to run this script.")
        sys.exit(1)

    # Check if bin size is less than max distance
    if bin_size > max_distance:
        logger.critical(
            "Bin size must be less than max distance for error evaluation. "
            f"Got bin_size={bin_size} and max_distance={max_distance}."
        )
        sys.exit(1)

    # Get paths of input images
    img_paths = get_img_paths(img_dir)

    # Sort input image paths by filename
    img_paths.sort(key=lambda x: x.name)

    # Get paths of input depth images
    depth_img_paths: list[Path] = []
    for path in img_paths:
        depth_path = depth_dir / path.relative_to(img_dir).with_suffix(".png")
        if not depth_path.exists():
            logger.warning(f"No depth map found for image {path} (skipping)")
            continue
        depth_img_paths.append(depth_path)
    assert len(depth_img_paths) == len(img_paths)
    logger.info(f"Found {len(depth_img_paths):,} input image-depth pairs")

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
        pipe.vae = AutoencoderTiny.from_pretrained(VAE_CKPT_LIGHT, torch_dtype=torch_dtype).to(
            "cuda"
        )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    logger.info(f"Initialized inference pipeline (dtype={dtype}, vae={vae})")

    # Evaluation loop
    results: dict[str, Any] = {}
    num_bins = math.ceil(max_distance / bin_size)
    for i, (img_path, depth_path) in enumerate(zip(img_paths, depth_img_paths, strict=True)):
        logger.info(f"[{i+1:,} / {len(img_paths):,}] " f"Processing {img_path} and {depth_path}")

        # Load camera image
        img, is_valid = load_img(img_path, "RGB")
        if not is_valid:
            logger.warning(f"Empty input image found: {img_path} (skipping)")
            continue

        # Load depth image
        depth_img, is_valid = load_img(depth_path, "RGB")
        if not is_valid:
            logger.warning(f"Empty input depth map found: {depth_path} (skipping)")
            continue

        # Convert depth image to depth map
        depth_map = to_depth_map(depth_img, max_distance=max_distance)
        depth_mask = depth_map > EPSILON

        # Run inference
        start_time = time.time()
        depth_map_pred = pipe(
            image=img,
            sparse_depth=depth_map,
            num_inference_steps=steps,
            processing_resolution=resolution,
        )
        duration_pred = time.time() - start_time
        if has_nan(depth_map_pred):
            logger.warning("NaN values found in depth map prediction (skipping)")
            continue

        # Save predicted depth map
        save_dir = (out_dir / "depth" / img_path.relative_to(img_dir)).parent
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
            logger.info(f"Created output directory for saving depth maps at {save_dir}")
        depth_map_pred_path = save_dir / f"{img_path.stem}.npy"
        np.save(depth_map_pred_path, depth_map_pred)
        logger.info(f"Saved predicted depth map at {depth_map_pred_path}")

        # Save visualization of predicted depth map
        if visualize:
            depth_img_pred = pipe.image_processor.visualize_depth(
                depth_map_pred, val_min=0, val_max=max_distance
            )[0]
            depth_img_ = pipe.image_processor.visualize_depth(
                depth_map, val_min=0, val_max=max_distance
            )[0]
            depth_img_ = np.array(depth_img_)
            depth_img_[~depth_mask] = 0
            depth_img_ = Image.fromarray(depth_img_)
            vis_img = Image.fromarray(
                make_grid(
                    np.stack(
                        [np.asarray(im) for im in [img, depth_img_, depth_img_pred]],
                        axis=0,
                    ),
                    rows=1,
                    cols=3,
                    resize=((output_size[0], output_size[1]) if output_size is not None else None),
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
                logger.info(f"Created directory for saving visualization outputs at {save_dir}")
            vis_img_path = save_dir / f"{img_path.stem}_vis.jpg"
            vis_img.save(vis_img_path)
            logger.info(f"Saved visualization outputs at {vis_img_path}")

        # Save evaluation results for this input
        result: dict[str, Any] = {
            "error": {
                "all": {
                    "mae": mae(depth_map_pred, depth_map, mask=depth_mask),
                    "rmse": rmse(depth_map_pred, depth_map, mask=depth_mask),
                }
            },
            "duration": {
                "inference": duration_pred,
            },
        }
        logger.info(f"Inference duration: {duration_pred:.3f} [s]")
        logger.info(f"Depth error (all): {result['error']['all']}")
        result["error"]["binned"] = []
        for bin_idx in range(num_bins):
            bin_start = bin_idx * bin_size
            bin_end = min(bin_start + bin_size, max_distance)
            if bin_idx == num_bins - 1:
                bin_mask = (depth_map >= bin_start) & (depth_map <= max_distance)
            elif bin_idx == 0:
                bin_mask = (depth_map >= EPSILON) & (depth_map < bin_end)
            else:
                bin_mask = (depth_map >= bin_start) & (depth_map < bin_end)
            is_empty = bin_mask.sum() == 0
            if is_empty:
                logger.warning(f"Empty bin found: {bin_start} - {bin_end} (skipping)")
            result["error"]["binned"].append(
                {
                    "bin_start": bin_start,
                    "bin_end": bin_end,
                    "mae": mae(depth_map_pred, depth_map, mask=bin_mask) if not is_empty else None,
                    "rmse": (
                        rmse(depth_map_pred, depth_map, mask=bin_mask) if not is_empty else None
                    ),
                }
            )
            logger.info(
                f"Depth error ({bin_start}-{bin_end}): {result['error']['binned'][bin_idx]}"
            )
        result_key = str(depth_map_pred_path.relative_to(out_dir))
        results[result_key] = result

    if len(results.keys()) == 0:
        logger.warning("No valid evaluation results found")
        sys.exit(1)

    # Save metric values for all inputs
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved evaluation results for all inputs at {results_path}")

    # Calc final metrics
    results_final: dict[str, Any] = {}
    reduce_methods = ["mean", "std", "median", "min", "max"]
    mask_types = ["all", "binned"]
    metric_types = ["error", "duration"]
    metric_names = ["mae", "rmse"]
    for metric_type in metric_types:
        results_final[metric_type] = {}
        if metric_type == "error":
            for mask_type in mask_types:
                if mask_type == "all":
                    results_final[metric_type][mask_type] = {}
                    for metric_name in metric_names:
                        values = [v[metric_type][mask_type][metric_name] for v in results.values()]
                        results_final[metric_type][mask_type][metric_name] = {
                            method: reduce(np.array(values), method) if len(values) > 0 else None
                            for method in reduce_methods
                        }
                elif mask_type == "binned":
                    results_final[metric_type][mask_type] = []
                    for bin_idx in range(num_bins):
                        bin_start = bin_idx * bin_size
                        bin_end = min(bin_start + bin_size, max_distance)
                        results_final[metric_type][mask_type].append(
                            {"bin_start": bin_start, "bin_end": bin_end}
                        )
                        for metric_name in metric_names:
                            values = [
                                v[metric_type][mask_type][bin_idx][metric_name]
                                for v in results.values()
                                if v[metric_type][mask_type][bin_idx][metric_name] is not None
                            ]
                            results_final[metric_type][mask_type][bin_idx][metric_name] = {
                                method: (
                                    reduce(np.array(values), method) if len(values) > 0 else None
                                )
                                for method in reduce_methods
                            }
        elif metric_type == "duration":
            for duration_type in result[metric_type]:
                values = [v[metric_type][duration_type] for v in results.values()]
                results_final[metric_type][duration_type] = {
                    method: reduce(np.array(values), method) for method in reduce_methods
                }
    results_final_path = out_dir / "results_final.json"
    with open(results_final_path, "w") as f:
        json.dump(results_final, f, indent=2)
    logger.info(f"Saved final evaluation results at {results_final_path}")


if __name__ == "__main__":
    main()
