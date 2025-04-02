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

torch.backends.cudnn.benchmark = True  # NOTE: Optimize convolution algorithms
torch.set_float32_matmul_precision("high")  # NOTE: Optimize fp32 arithmetic


@click.command(
    help="Predict dense depth maps from sparse depth maps and camera images."
)
@click.argument(
    "img_dir",
    type=click.Path(exists=True, path_type=Path, file_okay=False, dir_okay=True),
)
@click.argument(
    "sparse_dir",
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
    "--max-depth",
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
    "abs - [0, --max-depth]. "
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
    "--normed",
    type=bool,
    default=False,
    help="Whether to predict dense depth maps in normalized [0, 1] range.",
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
@click.option(
    "-bs",
    "--batch-size",
    type=click.IntRange(min=1),
    default=1,
    help="Batch size for inference.",
    show_default=True,
)
def main(
    img_dir: Path,
    sparse_dir: Path,
    out_dir: Path,
    seg_dir: Path | None,
    model: str,
    vae: str,
    steps: int,
    res: int,
    max_depth: float,
    save_dense: bool,
    vis: bool,
    log: Path | None,
    log_level: str,
    precision: str,
    compress: str,
    use_compile: bool,
    elemwise_scaling: bool,
    interp_mode: str,
    loss_funcs: list[str],
    normed: bool,
    overlay_sparse: bool,
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

    ############################################################
    # Model initialization
    ############################################################
    # Check loss functions
    loss_funcs_ = []
    for loss_func in loss_funcs:
        if loss_func not in SUPPORTED_LOSS_FUNCS:
            logger.error(f"Invalid loss function (skipped): {loss_func}")
        else:
            loss_funcs_.append(loss_func)
    loss_funcs = loss_funcs_

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

    ############################################################
    # Data loading
    ############################################################
    # Load segmentation meta file (if provided)
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

    # Find paths of input images
    img_paths_all = utils.find_img_paths(img_dir)
    img_paths_all.sort(key=lambda x: x.name)

    # Find paths of input sparse depth maps and paired images
    # and optionally segmentation maps
    sparse_paths: list[Path] = []
    img_paths: list[Path] = []
    seg_paths: list[Path] = []
    for path in img_paths_all:
        sparse_path = utils.find_file_with_exts(
            (sparse_dir / path.relative_to(img_dir)).with_suffix(".npy"),
            utils.NPARRAY_EXTENSIONS,
        )
        if sparse_path is None:
            logger.warning(f"No depth map found for image {path} (skipped)")
            continue
        img_paths.append(path)
        sparse_paths.append(sparse_path)
        if seg_dir is not None:
            seg_path = utils.find_file_with_exts(
                (seg_dir / path.relative_to(img_dir)).with_suffix(".npy"),
                utils.NPARRAY_EXTENSIONS,
            )
            if seg_path is None:
                logger.warning(f"No segmentation map found for image {path} (skipped)")
                continue
            seg_paths.append(seg_path)
    if len(img_paths) == 0:
        logger.critical("No valid input pairs found")
        sys.exit(1)
    logger.info(f"Found {len(sparse_paths):,} input pairs")

    # Create output directory if it doesn't exist
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    ############################################################
    # Evaluation loop
    ############################################################
    progbar = tqdm.tqdm(total=len(img_paths), dynamic_ncols=True)
    postfix: dict[str, Any] = {}
    for i in range(0, len(img_paths), batch_size):
        batch_img_paths = img_paths[i : i + batch_size]
        batch_sparse_paths = sparse_paths[i : i + batch_size]
        batch_seg_paths = seg_paths[i : i + batch_size] if seg_dir is not None else []
        bs = len(batch_img_paths)

        ############################################################
        # Data loading
        ############################################################
        # Load images
        stime_disk = time.time()
        ret = utils.load_imgs(
            batch_img_paths, mode="RGB", num_threads=len(batch_img_paths)
        )
        if sum([r[1] for r in ret]) == 0:
            # Skip to the next batch if all images are invalid
            progbar.update(bs)
            continue
        batch_img_paths = [batch_img_paths[j] for j in range(len(ret)) if ret[j][1]]
        batch_imgs = np.stack([r[0] for r in ret if r[1]], axis=0)

        # Load sparse depth maps
        batch_sparse_paths = [
            batch_sparse_paths[j] for j in range(len(ret)) if ret[j][1]
        ]
        batch_sparses = np.stack(
            utils.load_arrays(batch_sparse_paths, num_threads=len(batch_sparse_paths)),
            axis=0,
        )

        # Load segmentation maps
        if seg_dir is not None:
            batch_seg_paths = [batch_seg_paths[j] for j in range(len(ret)) if ret[j][1]]
            batch_segs = np.stack(
                utils.load_arrays(batch_seg_paths, num_threads=len(batch_seg_paths)),
                axis=0,
            )
            batch_segs = batch_segs
        time_disk = time.time() - stime_disk

        # Normalize
        if normed:
            batch_sparses /= max_depth

        ############################################################
        # Add guides to sparse depth map
        ############################################################
        # Add guides to sparse depth map using segmentation map
        # TODO: Eliminate for loop over batch dimension
        if seg_dir is not None:
            # TODO: Implement adding guides to sparse depth map feature
            pass
        postfix["time/guides"] = 0.0

        ############################################################
        # Run inference
        ############################################################
        stime_disk = time.time()
        batch_denses: np.ndarray = pipe(
            np.expand_dims(batch_imgs, axis=1),
            np.expand_dims(batch_sparses, axis=1),
            steps=steps,
            resolution=res,
            elemwise_scaling=elemwise_scaling,
            interp_mode=interp_mode,
            loss_funcs=loss_funcs,
            opt=opt,
            lr=(lr_latent, lr_scaling),
        )[
            :, 0
        ]  # [N, H, W]
        if normed:
            batch_denses *= max_depth
            batch_sparses *= max_depth
        postfix["time/infer"] = time.time() - stime_disk

        ############################################################
        # Save results
        ############################################################
        time_vis = 0.0
        for dense, sparse, img, img_path in zip(
            batch_denses, batch_sparses, batch_imgs, batch_img_paths, strict=True
        ):
            if utils.has_nan(dense):
                logger.error("NaN values found in dense depth map (skipped)")
                continue

            # Save dense depth map
            if save_dense:
                stime_disk = time.time()
                save_dir = (out_dir / "dense" / img_path.relative_to(img_dir)).parent
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
                time_disk += time.time() - stime_disk

            # Save visualization of inferenced depth map
            if vis:
                if vis_range == "abs":
                    vis_min, vis_max = 0, max_depth
                else:
                    vis_min, vis_max = min(dense.min(), sparse.min()), max(
                        dense.max(), sparse.max()
                    )
                stime_disk = time.time()
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
                time_vis += time.time() - stime_disk
                stime_disk = time.time()
                save_dir = (out_dir / "vis" / img_path.relative_to(img_dir)).parent
                if not save_dir.exists():
                    save_dir.mkdir(parents=True)
                save_path = save_dir / f"{img_path.stem}_vis.jpg"
                out.save(save_path)
                time_disk += time.time() - stime_disk
        postfix["time/disk"] = time_disk
        postfix["time/vis"] = time_vis
        progbar.set_postfix(postfix)
        progbar.update(bs)
    logger.success(f"Finished processing {len(img_paths):,} input pairs")


if __name__ == "__main__":
    main()
