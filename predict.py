import sys
import time
from pathlib import Path
from typing import Any, cast

import click
import torch
import tqdm
from diffusers import AutoencoderTiny, DDIMScheduler
from loguru import logger

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
    help="Input resolution. Input images will be resized so "
    "that the longer side has length ${res}.",
    show_default=True,
)
@click.option(
    "--norm",
    type=click.Choice(["const", "minmax", "percentile"]),
    default="const",
    help="Normalization method for input sparse depth maps. "
    "const - Normalize with --min-depth and --max-depth. "
    "minmax - Normalize with min and max depth values of input sparse depth maps. "
    "percentile - Normalize with depth values at specified percentiles (e.g., 0.05). "
    "NOTE: even when minmax or percentile, "
    "depth values are clamped to [--min-depth, --max-depth].",
    show_default=True,
)
@click.option(
    "--percentile",
    type=utils.CommaSeparated(float),
    default="0.01,0.99",
    help="Percentile values for determining depth range from input sparse depth maps. "
    "Format: min_percentile,max_percentile. Values should be in the range [0, 1]. "
    "For example, 0.01,0.99 means the depth range is determined by "
    "the 1st and 99th percentiles of the sparse depth values. "
    "Only used when --norm=percentile.",
    show_default=True,
)
@click.option(
    "--max-sparse-depth",
    type=click.FloatRange(min=0, min_open=True),
    default=120.0,
    help="Max absolute distance [m] of input sparse depth maps."
    "Used to decode range images to depth maps.",
    show_default=True,
)
@click.option(
    "--max-depth",
    type=click.FloatRange(min=0, min_open=True),
    default=120.0,
    help="Max absolute distance [m] of output dense depth maps.",
    show_default=True,
)
@click.option(
    "--min-depth",
    type=click.FloatRange(min=0),
    default=0.0,
    help="Min absolute distance [m] of output dense depth maps.",
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
    "--vis-res",
    type=click.Tuple([int, int]),
    default=(512, -1),
    help="Resolution (height, width)of visualization of inferenced dense depth maps. "
    "This option is valid when --vis=True. "
    "If one side is -1, the other side will be scaled to preserve the aspect ratio "
    "of the input images.",
    show_default=True,
)
@click.option(
    "-vo",
    "--vis-order",
    type=utils.CommaSeparated(str),
    default="image,sparse,dense",
    help="Order of visualization of inputs and outputs. "
    "This option is valid when --vis=True. "
    "Available options: image,sparse,dense",
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
    type=click.Choice(["bf16", "fp32"]),
    default="bf16",
    help="Data precision for inference.",
    show_default=True,
)
@click.option(
    "-c",
    "--compress",
    type=click.Choice(["npz", "bl2", "npy"]),
    default="bl2",
    help="Specify the compression format for the output depth maps. "
    "If npy, saves uncompressed. "
    "This option is ignored if --save-dense=False",
    show_default=True,
)
@click.option(
    "--compile-graph",
    type=bool,
    default=False,
    help="Whether to compile the inference pipeline using torch.compile.",
    show_default=True,
)
@click.option(
    "--compile-mode",
    type=click.Choice(["max-autotune", "reduce-overhead", "default"]),
    default="reduce-overhead",
    help="Compilation mode for torch.compile.",
    show_default=True,
)
@click.option(
    "--interp-mode",
    type=click.Choice(["bilinear", "nearest"]),
    default="bilinear",
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
    "--opt",
    type=click.Choice(["adam", "sgd", "adagrad"]),
    default="adam",
    help="Optimizer to use for depth completion. "
    "adam - Adam optimizer. "
    "sgd - Stochastic gradient descent optimizer. "
    "adagrad - Adagrad optimizer. ",
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
    "--kld",
    type=bool,
    default=False,
    help="Whether to apply KL divergence penalty to keep "
    "the distribution of prediction latents close to N(0,1).",
    show_default=True,
)
@click.option(
    "--kld-mode",
    type=click.Choice(["simple", "strict"]),
    default="simple",
    help="KL divergence mode. "
    "simple - Uses a simplified penalty "
    "based on squared L2 norm of latents. "
    "strict - Computes proper forward KL divergence between "
    "latent distribution and N(0,1). ",
    show_default=True,
)
@click.option(
    "--kld-weight",
    type=click.FloatRange(min=0, min_open=True),
    default=0.1,
    help="Weight for the KL divergence penalty term.",
    show_default=True,
)
@click.option(
    "-bs",
    "--batch-size",
    type=click.IntRange(min=1),
    default=1,
    help="Batch size for inference.",
    show_default=True,
)
@click.option(
    "--use-prev-latent",
    type=bool,
    default=False,
    help="Whether to use previous latent variables as a prior.",
    show_default=True,
)
@click.option(
    "--beta",
    type=click.FloatRange(min=0, min_open=True),
    default=0.9,
    help="Weight for the latent consistency term. " "Used when --use-prev-latent=True",
    show_default=True,
)
@click.option(
    "--use-segmask",
    type=bool,
    default=False,
    help="Whether to use segmentation masks for depth completion.",
    show_default=True,
)
@click.option(
    "--affine-invariant",
    type=bool,
    default=True,
    help="Whether to inference affine invariant depth completion.",
    show_default=True,
)
@click.option(
    "--closed-form",
    type=bool,
    default=False,
    help="Whether to use closed-form solution for affine transformation parameters."
    "If False, affine transformation parameters are inferred by a neural network. "
    "This option is valid when --affine-invariant=True.",
    show_default=True,
)
@click.option(
    "--projection",
    type=click.Choice(["linear", "log", "log10"]),
    default="linear",
    help="Projection method for depth values. "
    "linear - Linear projection. "
    "log - Logarithmic projection. "
    "log10 - Logarithmic base 10 projection. ",
    show_default=True,
)
@click.option(
    "--inv",
    type=bool,
    default=False,
    help="Whether to use inverse projection.",
    show_default=True,
)
def main(
    src_root: Path,
    dst_root: Path,
    model: str,
    vae: str,
    steps: int,
    res: int,
    norm: str,
    max_sparse_depth: float,
    max_depth: float,
    min_depth: float,
    percentile: list[float],
    save_dense: bool,
    vis: bool,
    vis_res: tuple[int, int],
    vis_order: list[str],
    log: Path | None,
    log_level: str,
    precision: str,
    compress: str,
    compile_graph: bool,
    compile_mode: str,
    interp_mode: str,
    loss_funcs: list[str],
    opt: str,
    lr_latent: float,
    lr_scaling: float,
    use_prev_latent: bool,
    beta: float,
    batch_size: int,
    kld: bool,
    kld_weight: float,
    kld_mode: str,
    use_segmask: bool,
    affine_invariant: bool,
    closed_form: bool,
    projection: str,
    inv: bool,
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

    # Check --vis-order args
    if vis:
        vis_order_ = []
        for view in vis_order:
            if view not in ["image", "sparse", "dense"]:
                logger.error(f"Invalid order (skipped): {view}")
                continue
            vis_order_.append(view)
        if len(vis_order_) == 0:
            logger.critical("No valid visualization order specified")
            sys.exit(1)
        vis_order = vis_order_

    # Check loss functions
    loss_funcs_ = []
    for loss_func in loss_funcs:
        if loss_func not in SUPPORTED_LOSS_FUNCS:
            logger.error(f"Invalid loss function (skipped): {loss_func}")
        else:
            loss_funcs_.append(loss_func)
    loss_funcs = loss_funcs_

    # NOTE:
    # Force batch_size =1 when use_prev_latent=True
    if use_prev_latent and batch_size > 1:
        logger.warning(
            "Currently, batch_size is forced to 1 when use_prev_latent=True. "
            "This will be fixed in the future"
        )
        batch_size = 1

    # NOTE:
    # Force norm = "minmax" when projection = "log" or "log10" and norm = "const"
    if (projection in ["log", "log10"] or inv) and norm == "const":
        logger.error(
            "norm=const is not allowed when projection=log or log10. "
            "Falling back to norm=minmax"
        )
        norm = "minmax"

    ############################################################
    # Model initialization
    ############################################################
    # Select computation precision
    if precision == "fp32":
        dtype = torch.float32
    else:  # precision == "bf16"
        dtype = torch.bfloat16

    # Select marigold checkpoint
    if model == "original":
        marigold_ckpt = MARIGOLD_CKPT_ORIGINAL
    else:  # marigold == "lcm"
        logger.warning("LCM-based Marigold model is experimental and unstable")
        marigold_ckpt = MARIGOLD_CKPT_LCM
    pipeline_args = {
        "prediction_type": "depth",
        "torch_dtype": dtype,
    }
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
    if compile_graph:
        pipe.unet = torch.compile(pipe.unet, mode=compile_mode, fullgraph=True)
        pipe.vae = torch.compile(pipe.vae, mode=compile_mode, fullgraph=True)
        logger.warning(
            "torch.compile is enabled, which takes a long time for "
            "the first inference path to compile. This is normal and expected"
        )

    ############################################################
    # Data loading
    ############################################################
    # Find dataset directories
    dataset_dirs = utils.find_dataset_dirs(src_root)
    if len(dataset_dirs) == 0:
        logger.critical(f"No dataset directories found at {src_root}")
        sys.exit(1)
    logger.info(f"Found {len(dataset_dirs):,} dataset directories")

    # Find images and sparse depth maps by dataset directory
    img_paths_all: dict[str, list[Path]] = {}
    sparse_paths_all: dict[str, list[Path]] = {}
    segmask_paths_all: dict[str, list[Path]] = {}
    segmaps: dict[str, dict[str, Any]] = {}
    for dataset_dir in dataset_dirs:

        # Load segmentation mapping file
        is_segmask_enabled = use_segmask
        segmask_dir = dataset_dir / utils.DATASET_DIR_NAME_SEGMASK
        if not segmask_dir.exists():
            logger.error(
                f"No segmentation directory found at {segmask_dir}. "
                f"Segmentation masks will not used for {dataset_dir.name}"
            )
            is_segmask_enabled = False
        else:
            segmap_path = segmask_dir / "map.csv"
            if not segmap_path.exists():
                logger.error(
                    f"No segmentation mapping file found at {segmap_path}. "
                    f"Segmentation masks will not be used for {dataset_dir.name}"
                )
                is_segmask_enabled = False
            else:
                segmaps[dataset_dir.name] = utils.load_segmap(segmap_path)

        # Find paths of input images
        img_dir = dataset_dir / utils.DATASET_DIR_NAME_IMAGE
        img_paths = utils.find_img_paths(img_dir)
        img_paths.sort(key=lambda x: x.name)

        # Find paths of input sparse depth maps and paired images
        # and optionally segmentation masks
        sparse_dir = dataset_dir / utils.DATASET_DIR_NAME_SPARSE
        sparse_paths_all[dataset_dir.name] = []
        segmask_paths_all[dataset_dir.name] = []
        img_paths_all[dataset_dir.name] = []
        for path in img_paths:
            sparse_path = sparse_dir / path.relative_to(img_dir).with_suffix(".png")
            if not sparse_path.exists():
                logger.warning(f"No sparse depth map found for image {path} (skipped)")
                continue
            segmask_path = segmask_dir / path.relative_to(img_dir).with_suffix(".png")
            if is_segmask_enabled:
                if not segmask_path.exists():
                    logger.warning(
                        f"No segmentation mask found for image {path} (skipped)"
                    )
                    continue
            segmask_paths_all[dataset_dir.name].append(segmask_path)
            sparse_paths_all[dataset_dir.name].append(sparse_path)
            img_paths_all[dataset_dir.name].append(path)
        n = len(img_paths_all[dataset_dir.name])
        if n == 0:
            logger.critical("No valid input pairs found")
            sys.exit(1)
        logger.info(f"Found {n:,} input pairs for {dataset_dir.name}")

    # Create output directory if it doesn't exist
    if not dst_root.exists():
        dst_root.mkdir(parents=True)

    ############################################################
    # Predict dense depth maps
    ############################################################
    for dataset_idx, dataset_dir in enumerate(dataset_dirs):
        out_dir = dst_root / (dataset_dir.relative_to(src_root))
        img_dir = dataset_dir / utils.DATASET_DIR_NAME_IMAGE
        sparse_dir = dataset_dir / utils.DATASET_DIR_NAME_SPARSE
        img_paths = img_paths_all[dataset_dir.name]
        sparse_paths = sparse_paths_all[dataset_dir.name]
        segmask_paths = segmask_paths_all[dataset_dir.name]
        is_segmask_enabled = len(segmask_paths) > 0
        progbar = tqdm.tqdm(
            total=len(img_paths),
            dynamic_ncols=True,
            desc=f"{dataset_idx + 1}/{len(dataset_dirs)} - {dataset_dir.name}",
        )
        postfix: dict[str, Any] = {}
        batch_pred_latents_prev: torch.Tensor | None = None
        pipe.pred_latents_ema = None
        for i in range(0, len(img_paths), batch_size):
            batch_img_paths = img_paths[i : i + batch_size]
            batch_sparse_paths = sparse_paths[i : i + batch_size]
            batch_segmask_paths = segmask_paths[i : i + batch_size]
            progbar_n = len(batch_img_paths)

            ############################################################
            # Data loading
            ############################################################
            time_io = 0.0
            stime_io = time.time()

            # Load images
            imgs_list = utils.load_img_tensors(
                batch_img_paths,
                mode="RGB",
                num_threads=len(batch_img_paths),
            )

            # Load sparse depth maps
            sparses_list = utils.load_img_tensors(
                batch_sparse_paths,
                mode="RGB",
                num_threads=len(batch_sparse_paths),
            )

            # Load segmentation masks if provided
            segmasks_list: list[torch.Tensor | None] = []
            if is_segmask_enabled:
                segmasks_list = utils.load_img_tensors(
                    batch_segmask_paths,
                    mode="RGB",
                    num_threads=len(batch_segmask_paths),
                )

            # Get flags indicating successful loading
            flags: list[bool] = []
            for j in range(len(imgs_list)):
                ok = imgs_list[j] is not None and sparses_list[j] is not None
                if is_segmask_enabled:
                    ok = ok and segmasks_list[j] is not None
                flags.append(ok)
            if not any(flags):
                logger.error(f"All images in batch {i+1} failed to load (skipped)")
                progbar.update(progbar_n)
                continue

            # Filter out input pairs missing any of the required data
            batch_img_paths = utils.filterout(batch_img_paths, flags)
            batch_sparse_paths = utils.filterout(batch_sparse_paths, flags)
            imgs_list = utils.filterout(imgs_list, flags)
            sparses_list = utils.filterout(sparses_list, flags)
            if is_segmask_enabled:
                batch_segmask_paths = utils.filterout(batch_segmask_paths, flags)
                segmasks_list = utils.filterout(segmasks_list, flags)

            # Create batch tensors and transfer to GPU
            batch_imgs = torch.stack(imgs_list).cuda(non_blocking=True)
            batch_sparses = torch.stack(sparses_list).cuda(non_blocking=True)
            batch_sparses = utils.to_depth(batch_sparses, max_distance=max_sparse_depth)
            if is_segmask_enabled:
                segmap = segmaps[dataset_dir.name]
                batch_segmasks = torch.stack(segmasks_list).cuda(non_blocking=True)  # type: ignore
                batch_segmasks = utils.to_segmask(batch_segmasks, segmap["color"])
            time_io += time.time() - stime_io

            ############################################################
            # Run inference
            ############################################################
            stime_infer = time.time()
            batch_denses, batch_pred_latents = pipe(
                batch_imgs,
                batch_sparses,
                max_depth,
                min_depth=min_depth,
                projection=projection,
                inv=inv,
                norm=norm,
                percentile=percentile,
                pred_latents_prev=batch_pred_latents_prev,
                beta=beta,
                steps=steps,
                resolution=res,
                interp_mode=interp_mode,
                loss_funcs=loss_funcs,
                opt=opt,
                lr=(lr_latent, lr_scaling),
                kld=kld,
                kld_mode=kld_mode,
                kld_weight=kld_weight,
                affine_invariant=affine_invariant,
                closed_form=closed_form,
            )
            batch_denses = cast(torch.Tensor, batch_denses)
            batch_pred_latents = cast(torch.Tensor, batch_pred_latents)
            if use_prev_latent:
                # Set as previous latent variables
                batch_pred_latents_prev = batch_pred_latents
            postfix["time/infer"] = time.time() - stime_infer

            ############################################################
            # Save results
            ############################################################
            time_vis = 0.0
            for dense, sparse, sparse_path, img, img_path in zip(
                batch_denses,
                batch_sparses,
                batch_sparse_paths,
                batch_imgs,
                batch_img_paths,
                strict=True,
            ):
                # Check if dense depth map has NaN values
                if utils.has_nan(dense):
                    logger.error("NaN values found in dense depth map (skipped)")
                    continue

                # Save dense depth map
                if save_dense:
                    stime_io = time.time()
                    save_dir = (
                        out_dir
                        / utils.RESULT_DIR_NAME_DENSE
                        / sparse_path.relative_to(sparse_dir)
                    ).parent
                    if not save_dir.exists():
                        save_dir.mkdir(parents=True)
                    save_path = save_dir / sparse_path.with_suffix(f".{compress}").name
                    utils.save_tensor(dense, save_path, compress=compress)
                    time_io += time.time() - stime_io

                # Save visualization of dense depth map
                if vis:
                    # Create grid image of visualization of inputs and outputs
                    stime_vis = time.time()
                    mask = (sparse <= 0.0).repeat(img.shape[0], 1, 1)  # [C, H, W]
                    to_vis: list[torch.Tensor] = []
                    for order in vis_order:
                        if order == "image":
                            to_vis.append(img)
                        elif order == "sparse":
                            sparse_vis = utils.visualize_depth(
                                sparse[torch.newaxis],
                                min_depth=min_depth,
                                max_depth=max_depth,
                            ).squeeze(0)
                            sparse_vis[mask] = 0
                            to_vis.append(sparse_vis)
                        elif order == "dense":
                            dense_vis = utils.visualize_depth(
                                dense[torch.newaxis],
                                min_depth=min_depth,
                                max_depth=max_depth,
                            ).squeeze(0)
                            to_vis.append(dense_vis)
                    grid_img = utils.make_grid(to_vis, resize=vis_res)
                    time_vis += time.time() - stime_vis

                    # Save grid image
                    stime_io = time.time()
                    save_dir = (
                        out_dir
                        / utils.RESULT_DIR_NAME_VIS
                        / img_path.relative_to(img_dir)
                    ).parent
                    save_path = save_dir / f"{img_path.stem}_vis.jpg"
                    utils.save_img_tensor(grid_img, save_path)
                    time_io += time.time() - stime_io

            postfix["time/io"] = time_io
            postfix["time/vis"] = time_vis
            progbar.set_postfix(postfix)
            progbar.update(progbar_n)
        progbar.close()
        logger.success(f"Finished processing {dataset_dir.name}")
    logger.success(f"Finished processing all {len(dataset_dirs):,} datasets")


if __name__ == "__main__":
    main()
