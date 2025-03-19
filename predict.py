import sys
import time
from pathlib import Path

import click
import numpy as np
import torch
from diffusers import AutoencoderTiny, DDIMScheduler
from loguru import logger
from PIL import Image

import utils
from marigold_dc import (
    MARIGOLD_CKPT_LCM,
    MARIGOLD_CKPT_ORIGINAL,
    SUPPORTED_LOSS_FUNCS,
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
    "--model",
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
    "-v",
    "--vis",
    type=bool,
    default=True,
    show_default=True,
    help="Whether to save visualization of output depth maps.",
)
@click.option(
    "--save-depth",
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
    "-p",
    "--precision",
    type=click.Choice(["fp16", "bf16", "fp32"]),
    default="bf16",
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
    "This option is ignored if --save-depth=False",
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
    "--interp-mode",
    type=click.Choice(["bilinear", "nearest"]),
    default="nearest",
    help="Interpolation mode for depth completion.",
    show_default=True,
)
@click.option(
    "--loss-funcs",
    type=utils.CommaSeparated(str),
    default="l1,l2",
    help="Comma-separated list of loss functions to use for depth completion. "
    "Available options: l1, l2, edge, smooth",
    show_default=True,
)
@click.option(
    "--predict-normed",
    type=bool,
    default=False,
    help="Whether to predict depth maps in normalized [0, 1] range.",
    show_default=True,
)
@click.option(
    "--overlay-sparse",
    type=bool,
    default=False,
    help="Whether to overlay sparse depth maps on visualization of "
    "inferenced depth maps. Saved inferenced depth maps are not affected.",
    show_default=True,
)
@click.option(
    "--aa",
    type=bool,
    default=False,
    help="Whether to enable anti-aliasing during depth completion.",
    show_default=True,
)
@click.option(
    "--opt",
    type=click.Choice(["adam", "adamw", "sgd"]),
    default="adam",
    help="Optimizer to use for depth completion.",
    show_default=True,
)
@click.option(
    "--lr-latent",
    type=click.FloatRange(min=0, min_open=True),
    default=0.05,
    help="Learning rate for latent variable.",
    show_default=True,
)
@click.option(
    "--lr-scaling",
    type=click.FloatRange(min=0, min_open=True),
    default=0.005,
    show_default=True,
    help="Learning rate for scale and shift parameters.",
)
def main(
    img_dir: Path,
    depth_dir: Path,
    out_dir: Path,
    seg_dir: Path | None,
    model: str,
    vae: str,
    steps: int,
    res: int,
    max_distance: float,
    save_depth: bool,
    vis: bool,
    log: Path | None,
    log_level: str,
    precision: str,
    compress: str,
    use_compile: bool,
    elemwise_scaling: bool,
    interp_mode: str,
    loss_funcs: list[str],
    predict_normed: bool,
    overlay_sparse: bool,
    aa: bool,
    opt: str,
    lr_latent: float,
    lr_scaling: float,
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

    # Check loss functions
    loss_funcs_ = []
    for loss_func in loss_funcs:
        if loss_func not in SUPPORTED_LOSS_FUNCS:
            logger.error(f"Invalid loss function (skipped): {loss_func}")
        else:
            loss_funcs_.append(loss_func)
    loss_funcs = loss_funcs_

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
            seg_ids: dict[str, int] = {
                seg_meta["name"][i]: seg_meta["id"][i]
                for i in range(len(seg_meta["name"]))
            }

    # Get paths of input images
    img_paths_all = utils.get_img_paths(img_dir)
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
            logger.warning(f"No depth map found for image {path} (skipped)")
            continue
        if seg_dir is not None:
            seg_path_candidates = list(
                filter(
                    lambda p: p.exists(),
                    [
                        seg_dir / path.relative_to(img_dir).with_suffix(ext)
                        for ext in NPARRAY_EXTENSIONS
                    ],
                )
            )
            if len(seg_path_candidates) == 0:
                logger.warning(f"No segmentation map found for image{path} (skipped)")
                continue
            seg_paths.append(seg_path_candidates[0])
        depth_paths.append(depth_path_candidates[0])
        img_paths.append(path)
    if len(img_paths) == 0:
        logger.critical("No valid input found")
        sys.exit(1)
    logger.info(f"Found {len(depth_paths):,} input pairs")

    # Create output directory if it doesn't exist
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    # Select computation precision
    # NOTE: fp16 may produce nans in prediction outputs
    if precision == "fp32":
        dtype = torch.float32
    elif precision == "fp16":
        dtype = torch.float16
        logger.warning(
            "fp16 precision tends to produce nans in prediction outputs. "
            "We strongly recommend using bf16 precision instead"
        )
    else:  # precision == "bf16"
        dtype = torch.bfloat16

    # Select marigold checkpoint
    if model == "original":
        marigold_ckpt = MARIGOLD_CKPT_ORIGINAL
    else:  # marigold == "lcm"
        marigold_ckpt = MARIGOLD_CKPT_LCM
    pipeline_args = {
        "prediction_type": "depth",
        "torch_dtype": dtype,
    }
    if precision == "fp16":
        # NOTE: Need to set variant to "fp16" for fp16 inference.
        # https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage#speeding-up-inference
        pipeline_args["variant"] = "fp16"
    pipe = MarigoldDepthCompletionPipeline.from_pretrained(
        marigold_ckpt,
        **pipeline_args,
    ).to("cuda")

    # Select vae checkpoint
    if vae == "light":
        pipe.vae = AutoencoderTiny.from_pretrained(
            VAE_CKPT_LIGHT,
            torch_dtype=dtype,
        ).to("cuda")

    # Set scheduler
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    # Compile model for faster inference
    if use_compile:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)
    logger.info(
        f"Initialized inference pipeline "
        f"(dtype={precision}, vae={vae}, model={model}, loss_funcs={loss_funcs})"
    )

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
            logger.error(f"Empty input image found: {img_path} (skipped)")
            continue

        # Load depth map
        depth = utils.load_array(depth_path)
        if predict_normed:
            depth /= max_distance

        # Load segmentation map
        seg: np.ndarray | None = None
        if seg_dir is not None:
            seg = utils.load_array(seg_paths[i])
            if seg.shape != depth.shape:
                logger.error(
                    f"Shape of segmentation map {seg_paths[i]} does not match "
                    f"shape of depth map {depth_path}. "
                    f"Segmentation map will be ignored."
                )
                seg = None

        # Add guides to depth map using segmentation map
        if seg is not None:
            # Set sky pixels to max distance
            depth_src = depth.copy()
            if "sky" in seg_ids:
                depth[(seg == seg_ids["sky"]) & (depth_src <= 0)] = max_distance
            # Complete missing depth values using segmentation map for specific classes
            for class_name in [
                "ego_vehicle",
                "road",
                "crosswalk",
                "striped_road_marking",
            ]:
                if class_name not in seg_ids:
                    continue
                seg_mask = seg == seg_ids[class_name]
                seg_mask_with_sparse_depth = (depth_src > 0) & seg_mask

                # Calculate sum of depth values for each row where mask is True
                row_sums = np.sum(depth_src * seg_mask_with_sparse_depth, axis=-1)

                # Count number of valid depth points in each row
                row_counts = np.sum(seg_mask_with_sparse_depth, axis=-1)

                # Avoid division by zero by setting counts of 0 to 1
                safe_counts = np.maximum(row_counts, 1)

                # Calculate average of non-zero depth values for each row
                completion_depth_values = row_sums / safe_counts

                # Only use average values where we had at least one valid depth point
                completion_depth_values = np.where(
                    row_counts > 0, completion_depth_values, 0
                )

                # Update depth map
                for y in range(completion_depth_values.shape[0]):
                    if completion_depth_values[y] > 0:
                        depth[y, seg_mask[y]] = completion_depth_values[y]

        # Run inference
        start = time.time()
        depth_pred: np.ndarray = pipe(
            img,
            depth,
            steps=steps,
            resolution=res,
            elemwise_scaling=elemwise_scaling,
            interp_mode=interp_mode,
            loss_funcs=loss_funcs,
            aa=aa,
            opt=opt,
            lr=(lr_latent, lr_scaling),
        )
        end = time.time()
        logger.info(f"Inference time: {end - start:.3f} [s]")
        if utils.has_nan(depth_pred):
            logger.error("NaN values found in inferenced depth map (skipped)")
            continue
        if predict_normed:
            depth_pred *= max_distance
            depth *= max_distance

        # Save inferenced depth map
        if save_depth:
            save_dir = (out_dir / "depth" / img_path.relative_to(img_dir)).parent
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            if compress != "none":
                save_path = save_dir / img_path.with_suffix(f".{compress}").name
            else:
                save_path = save_dir / img_path.with_suffix(".npy").name
            utils.save_array(
                depth_pred,
                save_path,
                compress=compress if compress != "none" else None,
            )
            logger.info(f"Saved inferenced depth map at {save_path}")

        # Save visualization of inferenced depth map
        if vis:
            depth_vis = pipe.image_processor.visualize_depth(
                depth, val_min=0, val_max=max_distance
            )[0]
            depth_vis = np.array(depth_vis)
            depth_vis[depth <= 0] = 0
            # Overlay sparse depth map on inferenced depth map
            if overlay_sparse:
                mask = depth > 0
                depth_pred[mask] = depth[mask]
            depth_pred_vis = pipe.image_processor.visualize_depth(
                depth_pred, val_min=0, val_max=max_distance
            )[0]
            out = Image.fromarray(
                utils.make_grid(
                    np.stack(
                        [img, depth_vis, depth_pred_vis],
                        axis=0,
                    ),
                    rows=1,
                    cols=3,
                )
            )
            save_dir = (out_dir / "vis" / img_path.relative_to(img_dir)).parent
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            save_path = save_dir / f"{img_path.stem}_vis.jpg"
            out.save(save_path)
            logger.info(f"Saved visualized outputs at {save_path}")
    logger.success(f"Finished processing {len(img_paths):,} input pairs")


if __name__ == "__main__":
    main()
