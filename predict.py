import sys
import time
from pathlib import Path
from typing import Any

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
    type=click.Choice(["adam", "sgd"]),
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
    "--kl-penalty",
    type=bool,
    default=False,
    help="Whether to apply KL divergence penalty to keep "
    "the distribution of prediction latents close to N(0,1).",
    show_default=True,
)
@click.option(
    "--kl-weight",
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
    help="Weight for the latent consistency term. "
    "Higher values give more weight to new latents, "
    "while lower values preserve more information from previous frames.",
    show_default=True,
)
@click.option(
    "--use-seg",
    type=bool,
    default=False,
    help="Whether to use segmentation maps for depth completion.",
    show_default=True,
)
def main(
    src_root: Path,
    dst_root: Path,
    model: str,
    vae: str,
    steps: int,
    res: int,
    max_depth: float,
    save_dense: bool,
    vis: bool,
    vis_res: tuple[int, int],
    vis_order: list[str],
    log: Path | None,
    log_level: str,
    precision: str,
    compress: str,
    compile_graph: bool,
    interp_mode: str,
    loss_funcs: list[str],
    opt: str,
    lr_latent: float,
    lr_scaling: float,
    use_prev_latent: bool,
    beta: float,
    batch_size: int,
    kl_penalty: bool,
    kl_weight: float,
    use_seg: bool,
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
    # Find dataset directories
    dataset_dirs = utils.find_dataset_dirs(src_root)
    if len(dataset_dirs) == 0:
        logger.critical(f"No dataset directories found at {src_root}")
        sys.exit(1)
    logger.info(f"Found {len(dataset_dirs):,} dataset directories")

    # Find images and sparse depth maps by dataset directory
    img_paths_all: dict[str, list[Path]] = {}
    sparse_paths_all: dict[str, list[Path]] = {}
    seg_paths_all: dict[str, list[Path]] = {}
    for dataset_dir in dataset_dirs:
        # Check if segmentation directory exists
        seg_dir = dataset_dir / utils.DATASET_DIR_NAME_SEG
        has_seg_dir = seg_dir.exists()

        # Find paths of input images
        img_dir = dataset_dir / utils.DATASET_DIR_NAME_IMAGE
        img_paths = utils.find_img_paths(img_dir)
        img_paths.sort(key=lambda x: x.name)

        # Find paths of input sparse depth maps and paired images
        # and optionally segmentation maps
        sparse_dir = dataset_dir / utils.DATASET_DIR_NAME_SPARSE
        sparse_paths: list[Path] = []
        img_paths_: list[Path] = []
        seg_paths: list[Path] = []
        for path in img_paths:
            sparse_path = utils.find_file_with_exts(
                (sparse_dir / path.relative_to(img_dir)).with_suffix(".npy"),
                utils.NPARRAY_EXTS,
            )
            if sparse_path is None:
                logger.warning(f"No sparse depth map found for image {path} (skipped)")
                continue
            if use_seg and has_seg_dir:
                seg_path = utils.find_file_with_exts(
                    (seg_dir / path.relative_to(img_dir)).with_suffix(".npy"),
                    utils.NPARRAY_EXTS,
                )
                if seg_path is None:
                    logger.warning(
                        f"No segmentation map found for image {path} (skipped)"
                    )
                    continue
                seg_paths.append(seg_path)
            sparse_paths.append(sparse_path)
            img_paths_.append(path)
        if len(img_paths_) == 0:
            logger.critical("No valid input pairs found")
            sys.exit(1)
        img_paths = img_paths_
        seg_paths_all[dataset_dir.name] = seg_paths
        img_paths_all[dataset_dir.name] = img_paths
        sparse_paths_all[dataset_dir.name] = sparse_paths
        logger.info(f"Found {len(img_paths):,} input pairs for {dataset_dir.name}")

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
        seg_paths = seg_paths_all[dataset_dir.name]
        has_seg = len(seg_paths) > 0
        progbar = tqdm.tqdm(
            total=len(img_paths),
            dynamic_ncols=True,
            desc=f"{dataset_idx + 1}/{len(dataset_dirs)} - {dataset_dir.name}",
        )
        postfix: dict[str, Any] = {}
        batch_pred_latents_prev: torch.Tensor | None = None
        for i in range(0, len(img_paths), batch_size):
            batch_img_paths = img_paths[i : i + batch_size]
            batch_sparse_paths = sparse_paths[i : i + batch_size]
            batch_seg_paths = seg_paths[i : i + batch_size]
            bs = len(batch_img_paths)

            ############################################################
            # Data loading
            ############################################################
            # Load images
            time_io = 0.0
            stime_io = time.time()
            ret = utils.load_img_tensors(
                batch_img_paths,
                mode="RGB",
                num_threads=len(batch_img_paths),
            )

            # Check if all images failed to load
            if all(img is None for img in ret):
                logger.warning(
                    f"All images in batch {i//batch_size + 1} failed to load (skipped)"
                )
                progbar.update(bs)
                continue

            # Process loaded images
            batch_img_paths = [
                batch_img_paths[j] for j in range(len(ret)) if ret[j] is not None
            ]
            batch_imgs = torch.stack(
                [img for img in ret if img is not None], dim=0
            ).cuda(non_blocking=True)

            # Load sparse depth maps
            batch_sparse_paths = [
                batch_sparse_paths[j] for j in range(len(ret)) if ret[j] is not None
            ]
            batch_sparses = (
                torch.stack(
                    utils.load_tensors(
                        batch_sparse_paths,
                        num_threads=len(batch_sparse_paths),
                    ),
                    dim=0,
                )
                .unsqueeze(1)
                .cuda(non_blocking=True)
            )  # [N, 1, H, W]

            # Load segmentation maps if provided
            if use_seg and has_seg:
                batch_seg_paths = [
                    batch_seg_paths[j] for j in range(len(ret)) if ret[j] is not None
                ]
                batch_segs = torch.stack(
                    utils.load_tensors(
                        batch_seg_paths,
                        num_threads=len(batch_seg_paths),
                    ),
                    dim=0,
                ).cuda(
                    non_blocking=True
                )  # [N, H, W]
                batch_segs = batch_segs.unsqueeze(1)  # [N, 1, H, W]
            time_io += time.time() - stime_io

            ############################################################
            # Run inference
            ############################################################
            stime_infer = time.time()
            batch_denses, batch_pred_latents = pipe(
                batch_imgs,
                batch_sparses,
                pred_latents_prev=batch_pred_latents_prev,
                max_depth=max_depth,
                steps=steps,
                resolution=res,
                interp_mode=interp_mode,
                loss_funcs=loss_funcs,
                opt=opt,
                lr=(lr_latent, lr_scaling),
                kl_penalty=kl_penalty,
                kl_weight=kl_weight,
                beta=beta,
            )
            assert isinstance(batch_denses, torch.Tensor)
            assert isinstance(batch_pred_latents, torch.Tensor)
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
                    sparse_mask = (sparse <= 0.0).repeat(
                        img.shape[0], 1, 1
                    )  # [C, H, W]
                    to_vis: list[torch.Tensor] = []
                    for order in vis_order:
                        if order == "image":
                            to_vis.append(img)
                        elif order == "sparse":
                            sparse_vis = utils.visualize_depth(
                                sparse[torch.newaxis],
                                max_depth=max_depth,
                            ).squeeze(0)
                            sparse_vis[sparse_mask] = 0
                            to_vis.append(sparse_vis)
                        elif order == "dense":
                            dense_vis = utils.visualize_depth(
                                dense[torch.newaxis],
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
                    utils.save_img(grid_img, save_path)
                    time_io += time.time() - stime_io

            postfix["time/io"] = time_io
            postfix["time/vis"] = time_vis
            progbar.set_postfix(postfix)
            progbar.update(bs)
        progbar.close()
        logger.success(f"Finished processing {dataset_dir.name}")
    logger.success(f"Finished processing all {len(dataset_dirs):,} datasets")


if __name__ == "__main__":
    main()
