import sys
import time
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
import tqdm
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
from utils import (
    DENSE_DIR_NAME,
    IMAGE_DIR_NAME,
    NPARRAY_EXTENSIONS,
    SPARSE_DIR_NAME,
    VIS_DIR_NAME,
    find_dataset_dirs,
)

torch.backends.cudnn.benchmark = True  # NOTE: Optimize convolution algorithms
torch.set_float32_matmul_precision("high")  # NOTE: Optimize fp32 arithmetic


@click.command(
    help="Predict dense depth maps from sparse depth maps and camera images."
)
@click.argument(
    "src_root",
    type=click.Path(exists=True, path_type=Path, file_okay=False, dir_okay=True),
)
@click.argument("dst_root", type=click.Path(exists=False, path_type=Path))
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
    help="Input resolution. Input images will be resized so that the size "
    "of the longer side is ${res} keeping aspect ratio.",
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
    help="Whether to save visualization of inferenced dense depth maps.",
)
@click.option(
    "-vr",
    "--vis-range",
    type=click.Choice(["abs", "rel"]),
    default="abs",
    help="Range of visualization of inferenced dense depth maps. "
    "abs - [0, --max-distance]. "
    "rel - [min(dense.min(), sparse.min()), max(dense.max(), sparse.max())].",
    show_default=True,
)
@click.option(
    "--save-dense",
    type=bool,
    default=True,
    help="Whether to save the inferenced dense depth maps. "
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
    "This option is ignored if --save-dense=False",
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
    "--overlay-sparse",
    type=bool,
    default=False,
    help="Whether to overlay sparse depth maps on visualization of "
    "dense depth maps. Saved dense depth maps are not affected.",
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
    default="sgd",
    help="Optimizer to use for depth completion.",
    show_default=True,
)
@click.option(
    "--lr-latent",
    type=click.FloatRange(min=0, min_open=True),
    default=0.1,
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
@click.option(
    "-bs",
    "--batch-size",
    type=click.IntRange(min=1),
    default=1,
    help="Batch size for inference.",
    show_default=True,
)
def main(
    src_root: Path,
    dst_root: Path,
    model: str,
    vae: str,
    steps: int,
    res: int,
    max_distance: float,
    save_dense: bool,
    vis: bool,
    log: Path | None,
    log_level: str,
    precision: str,
    compress: str,
    use_compile: bool,
    interp_mode: str,
    loss_funcs: list[str],
    overlay_sparse: bool,
    aa: bool,
    opt: str,
    lr_latent: float,
    lr_scaling: float,
    vis_range: str,
    batch_size: int,
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
        logger.warning(
            "torch.compile is enabled, which takes a long time for "
            "the first inference path to compile. This is normal and expected"
        )
    logger.info(
        f"Initialized inference pipeline "
        f"(dtype={precision}, vae={vae}, model={model}, "
        f"loss_funcs={loss_funcs}, batch_size={batch_size})"
    )

    # Check loss functions
    loss_funcs_ = []
    for loss_func in loss_funcs:
        if loss_func not in SUPPORTED_LOSS_FUNCS:
            logger.error(f"Invalid loss function (skipped): {loss_func}")
        else:
            loss_funcs_.append(loss_func)
    loss_funcs = loss_funcs_

    # Find dataset directories
    dataset_dirs = find_dataset_dirs(src_root)
    if len(dataset_dirs) == 0:
        logger.critical(f"No dataset directories found in {src_root}")
        sys.exit(1)
    logger.info(f"{len(dataset_dirs):,} dataset directories found")

    # Collect image and sparse map paths
    img_paths_all: dict[str, list[Path]] = {}
    sparse_paths_all: dict[str, list[Path]] = {}
    for dataset_dir in dataset_dirs:
        img_dir = dataset_dir / IMAGE_DIR_NAME
        img_paths = utils.get_img_paths(img_dir)
        img_paths.sort(key=lambda path: path.name)
        sparse_dir = dataset_dir / SPARSE_DIR_NAME
        key = dataset_dir.name
        img_paths_all[key] = []
        sparse_paths_all[key] = []
        for path in img_paths:
            sparse_path_candids = list(
                filter(
                    lambda p: p.exists(),
                    [
                        sparse_dir / path.relative_to(img_dir).with_suffix(ext)
                        for ext in NPARRAY_EXTENSIONS
                    ],
                )
            )
            if len(sparse_path_candids) == 0:
                logger.warning(f"No sparse map found for image {path} (skipped)")
                continue
            sparse_path = sparse_path_candids[0]
            img_paths_all[key].append(path)
            sparse_paths_all[key].append(sparse_path)
        logger.info(f"Found {len(img_paths_all[key]):,} input pairs for dataset {key}")

    # Create output root if it doesn't exist
    if not dst_root.exists():
        dst_root.mkdir(parents=True)

    # Evaluation loop
    for dataset_dir in dataset_dirs:
        logger.info(f"Processing dataset {dataset_dir.name}")
        img_paths = img_paths_all[dataset_dir.name]
        sparse_paths = sparse_paths_all[dataset_dir.name]
        img_dir = dataset_dir / IMAGE_DIR_NAME
        dst_dir = dst_root / (dataset_dir.relative_to(src_root))
        dense_dir = dst_dir / DENSE_DIR_NAME
        vis_dir = dst_dir / VIS_DIR_NAME
        progbar = tqdm.tqdm(
            total=len(img_paths),
            dynamic_ncols=True,
        )
        postfix: dict[str, Any] = {}
        for idx in range(0, len(img_paths), batch_size):
            batch_img_paths = img_paths[idx : idx + batch_size]
            batch_sparse_paths = sparse_paths[idx : idx + batch_size]
            bs = len(batch_img_paths)

            # Load images and arrays
            imgs: list[np.ndarray] = []
            sparses: list[np.ndarray] = []
            batch_img_paths_: list[Path] = []
            batch_sparse_paths_: list[Path] = []
            time_disk = 0.0
            stime = time.time()
            for j in range(len(batch_img_paths)):
                # Load image
                img, is_valid = utils.load_img(batch_img_paths[j], "RGB")
                if not is_valid:
                    logger.error(
                        f"Empty input image found: {batch_img_paths[j]} (skipped)"
                    )
                    continue
                imgs.append(img)
                batch_img_paths_.append(batch_img_paths[j])
                # Load sparse map
                sparse = utils.load_array(batch_sparse_paths[j])
                sparses.append(sparse)
                batch_sparse_paths_.append(batch_sparse_paths[j])
            batch_img_paths = batch_img_paths_
            batch_sparse_paths = batch_sparse_paths_

            # Skip to the next batch if all images are invalid
            if len(imgs) == 0:
                progbar.update(bs)
                continue

            # Make batch arrays
            batch_imgs = np.stack(imgs, axis=0)
            batch_sparses = np.stack(sparses, axis=0) / max_distance
            time_disk += time.time() - stime

            # Run inference
            stime = time.time()
            batch_denses: np.ndarray = pipe(
                np.expand_dims(batch_imgs, axis=1),
                np.expand_dims(batch_sparses, axis=1),
                steps=steps,
                resolution=res,
                interp_mode=interp_mode,
                loss_funcs=loss_funcs,
                aa=aa,
                opt=opt,
                lr=(lr_latent, lr_scaling),
            )[
                :, 0
            ]  # [N, H, W]
            batch_denses *= max_distance
            batch_sparses *= max_distance
            postfix["time/infer"] = time.time() - stime

            # Iterate over each dense maps
            time_vis = 0.0
            for dense, sparse, img_path, img in zip(
                batch_denses, batch_sparses, batch_img_paths, batch_imgs, strict=True
            ):
                if utils.has_nan(dense):
                    logger.error("NaN values found in dense depth map (skipped)")
                    continue

                # Save dense depth map
                if save_dense:
                    stime = time.time()
                    save_dir = (dense_dir / img_path.relative_to(img_dir)).parent
                    if not save_dir.exists():
                        save_dir.mkdir(parents=True)
                    if compress != "none":
                        save_path = save_dir / img_path.with_suffix(f".{compress}").name
                    else:
                        save_path = save_dir / img_path.with_suffix(".npy").name
                    utils.save_array(
                        dense,
                        save_path,
                        compress=compress if compress != "none" else None,
                    )
                    time_disk += time.time() - stime

                # Save visualization of dense depth map
                if vis:
                    if vis_range == "abs":
                        vis_min, vis_max = 0, max_distance
                    else:
                        vis_min, vis_max = min(dense.min(), sparse.min()), max(
                            dense.max(), sparse.max()
                        )
                    stime = time.time()
                    sparse_vis = pipe.image_processor.visualize_depth(
                        sparse, val_min=vis_min, val_max=vis_max
                    )[0]
                    sparse_vis = np.array(sparse_vis)
                    sparse_vis[sparse <= 0] = 0
                    # Overlay sparse depth map on visualization of dense depth map
                    if overlay_sparse:
                        mask = sparse > 0
                        dense[mask] = sparse[mask]
                    dense_vis = pipe.image_processor.visualize_depth(
                        dense, val_min=vis_min, val_max=vis_max
                    )[0]
                    out = Image.fromarray(
                        utils.make_grid(
                            np.stack(
                                [img, sparse_vis, dense_vis],
                                axis=0,
                            ),
                            rows=1,
                            cols=3,
                        )
                    )
                    time_vis += time.time() - stime
                    stime = time.time()
                    save_dir = (vis_dir / img_path.relative_to(img_dir)).parent
                    if not save_dir.exists():
                        save_dir.mkdir(parents=True)
                    save_path = save_dir / f"{img_path.stem}_vis.jpg"
                    out.save(save_path)
                    time_disk += time.time() - stime
            postfix["time/disk"] = time_disk
            postfix["time/vis"] = time_vis
            progbar.set_postfix(postfix)
            progbar.update(bs)
        progbar.close()
        logger.success(f"Finished processing dataset {dataset_dir}")
    logger.success("Finished all predictions")


if __name__ == "__main__":
    main()
