import json
import sys
from pathlib import Path
from typing import Any, Literal

import click
import numpy as np
import torch
import tqdm
from loguru import logger

import utils

METRICS = ["mae", "rmse"]
Metric = Literal["mae", "rmse"]

torch.backends.cudnn.benchmark = True  # NOTE: Optimize convolution algorithms
torch.set_float32_matmul_precision("high")  # NOTE: Optimize fp32 arithmetic


@click.command(help="Analyze results of depth completion.")
@click.argument(
    "dataset_root",
    type=click.Path(exists=True, path_type=Path, file_okay=False, dir_okay=True),
)
@click.argument(
    "result_root",
    type=click.Path(exists=True, path_type=Path, file_okay=False, dir_okay=True),
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
    "--metrics",
    type=utils.CommaSeparated(str),
    default="mae,rmse",
    help="Comma-separated list of metrics to compute. " "Available options: mae, rmse",
    show_default=True,
)
@click.option(
    "--calc-binned-scores",
    type=bool,
    default=False,
    help="Whether to compute binned scores.",
    show_default=True,
)
@click.option(
    "--bin-size",
    type=click.FloatRange(min=0, min_open=True),
    default=10.0,
    help="Bin size in meters.",
    show_default=True,
)
@click.option(
    "--max-sparse-depth",
    type=click.FloatRange(min=0, min_open=True),
    default=120.0,
    help="Maximum distance in meters of sparse depth maps.",
    show_default=True,
)
@click.option(
    "--max-depth",
    type=click.FloatRange(min=0, min_open=True),
    default=120.0,
    help="Maximum distance in meters of dense depth maps.",
    show_default=True,
)
@click.option(
    "--min-depth",
    type=click.FloatRange(min=0),
    default=0.0,
    help="Minimum distance in meters of dense depth maps.",
    show_default=True,
)
@click.option(
    "-bs",
    "--batch-size",
    type=click.IntRange(min=1),
    default=32,
    help="Batch size for loading sparse & dense depth maps.",
    show_default=True,
)
@click.option(
    "-nt",
    "--num-threads",
    type=click.IntRange(min=1),
    default=8,
    help="Number of threads for loading sparse & dense depth maps.",
    show_default=True,
)
@click.option(
    "--cuda",
    type=bool,
    default=True,
    help="Whether to use CUDA for faster processing.",
    show_default=True,
)
def main(
    dataset_root: Path,
    result_root: Path,
    metrics: list[Metric],
    calc_binned_scores: bool,
    log: Path | None,
    log_level: Literal[
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    ],
    bin_size: float,
    max_sparse_depth: float,
    max_depth: float,
    min_depth: float,
    batch_size: int,
    num_threads: int,
    cuda: bool,
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

    # Check cuda availability
    if cuda and not torch.cuda.is_available():
        logger.warning("CUDA is not available. Using CPU instead.")
        cuda = False

    # Check metrics
    metrics_: list[Metric] = []
    for metric in metrics:
        if metric not in METRICS:
            logger.error(f"Invalid metric: {metric} (skipped)")
        else:
            metrics_.append(metric)
    if len(metrics_) == 0:
        logger.critical("No valid metrics provided")
        sys.exit(1)
    metrics = metrics_

    # Find dataset directories
    dataset_dirs = utils.find_dataset_dirs(dataset_root)
    if len(dataset_dirs) == 0:
        logger.critical("No dataset directories found")
        sys.exit(1)
    logger.info(f"Found {len(dataset_dirs):,} datasets")

    # Evaluation
    bin_ranges = utils.calc_bins(min_depth, max_depth, bin_size)
    scores_overall_all: dict[Metric, list[float]] = {metric: [] for metric in metrics}
    scores_binned_all: list[dict[Metric, list[float]]] = [
        {metric: [] for metric in metrics} for _ in range(len(bin_ranges))
    ]
    for dataset_idx, dataset_dir in enumerate(dataset_dirs):
        result_dir = result_root / (dataset_dir.relative_to(dataset_root))
        if not result_dir.exists():
            logger.warning(
                f"No result directory found for {dataset_dir.name}. "
                "Skip this dataset"
            )
            continue
        sparse_dir = dataset_dir / utils.DATASET_DIR_NAME_SPARSE
        dense_dir = result_dir / utils.RESULT_DIR_NAME_DENSE

        # Find paths to sparse and dense depth maps
        sparse_paths: list[Path] = []
        dense_paths: list[Path] = []
        cache: set[str] = set()
        for path in sparse_dir.rglob("*"):
            if path.suffix != ".png":
                continue
            stem = path.stem
            if stem in cache:
                continue
            cache.add(stem)
            sparse_path = path
            dense_path = utils.find_file_with_exts(
                dense_dir / sparse_path.relative_to(sparse_dir),
                utils.NPARRAY_EXTS,
            )
            if dense_path is None:
                logger.warning(f"No dense depth map found for {sparse_path} (skipped)")
                continue
            sparse_paths.append(sparse_path)
            dense_paths.append(dense_path)
        if len(sparse_paths) == 0:
            logger.warning(
                f"No dense & sparse depth map pairs found for {dataset_dir.name}. "
                "Skip this dataset"
            )
            continue
        logger.info(
            f"Found {len(sparse_paths):,} pairs of sparse & dense "
            f"depth maps for {dataset_dir.name}"
        )

        # Compute overall metrics
        scores_overall: dict[Metric, list[float]] = {metric: [] for metric in metrics}
        scores_binned: list[dict[Metric, list[float]]] = [
            {metric: [] for metric in metrics} for _ in range(len(bin_ranges))
        ]
        progbar = tqdm.tqdm(
            total=len(sparse_paths),
            desc=f"{dataset_idx + 1}/{len(dataset_dirs)} - {dataset_dir.name}",
            dynamic_ncols=True,
        )
        for i in range(0, len(sparse_paths), batch_size):
            batch_sparse_paths = sparse_paths[i : i + batch_size]
            batch_dense_paths = dense_paths[i : i + batch_size]

            # Load sparse depth maps
            batch_sparses = utils.to_depth(
                torch.stack(
                    utils.load_img_tensors(
                        batch_sparse_paths, mode="RGB", num_threads=num_threads
                    )  # type: ignore
                ),
                max_distance=max_sparse_depth,
            )
            batch_denses = torch.stack(
                utils.load_tensors(batch_dense_paths, num_threads=num_threads),
            )
            if cuda:
                batch_sparses = batch_sparses.cuda(non_blocking=True)
                batch_denses = batch_denses.cuda(non_blocking=True)
            mask = batch_sparses > 0
            batch_sparses = batch_sparses.clamp(min=min_depth, max=max_depth)
            batch_denses = batch_denses.clamp(min=min_depth, max=max_depth)

            # Compute overall metrics
            for metric in metrics:
                if metric == "mae":
                    score = utils.mae(batch_denses, batch_sparses, mask=mask)
                else:
                    score = utils.rmse(batch_denses, batch_sparses, mask=mask)
                scores_overall[metric].append(score)
                scores_overall_all[metric].append(score)

            # Compute bin-wise metrics
            if calc_binned_scores:
                for bin_idx, bin_range in enumerate(bin_ranges):
                    lower, upper = bin_range
                    mask_binned = (
                        mask & (batch_sparses >= lower) & (batch_sparses <= upper)
                    )
                    if not torch.any(mask_binned):
                        continue
                    for metric in metrics:
                        if metric == "mae":
                            score = utils.mae(
                                batch_denses, batch_sparses, mask=mask_binned
                            )
                        else:
                            score = utils.rmse(
                                batch_denses, batch_sparses, mask=mask_binned
                            )
                        scores_binned[bin_idx][metric].append(score)
                        scores_binned_all[bin_idx][metric].append(score)
            progbar.update(len(batch_sparse_paths))
        progbar.close()

        # Print overall scores
        logger.info(f"[{dataset_dir.name}]:")
        logger.info(f"  {min_depth:.1f} <= x <= {max_depth:.1f}:")
        results: dict[str, Any] = {"overall": {}}
        for metric in metrics:
            score = float(np.mean(scores_overall[metric]))
            results["overall"][metric] = score
            logger.info(f"    {metric}: {score:.2f}")

        # Print binned scores
        if calc_binned_scores:
            logger.info(f"[{dataset_dir.name}]:")
            results["binned"] = []
            for bin_idx, bin_range in enumerate(bin_ranges):
                lower, upper = bin_range
                result: dict[str, Any] = {"range": (lower, upper), "metrics": {}}
                logger.info(f"  {lower:.1f} <= x <= {upper:.1f}:")
                for metric in metrics:
                    score = float(np.mean(scores_binned[bin_idx][metric]))
                    result["metrics"][metric] = score
                    logger.info(f"    {metric}: {score:.2f}")
                results["binned"].append(result)

        # Save results
        save_path = result_dir / "results.json"
        with save_path.open("w") as f:
            json.dump(results, f, indent=2)
        logger.success(f"Saved results to {save_path}")

    # Calculate scores for all datasets
    logger.info("[All]:")
    logger.info(f"  {min_depth:.1f} <= x <= {max_depth:.1f}:")
    results_all: dict[str, Any] = {"overall": {}, "binned": []}
    for metric in metrics:
        score = float(np.mean(scores_overall_all[metric]))
        results_all["overall"][metric] = score
        logger.info(f"    {metric}: {score:.2f}")
    if calc_binned_scores:
        logger.info("[All]:")
        for bin_idx, bin_range in enumerate(bin_ranges):
            lower, upper = bin_range
            result = {"range": bin_range, "metrics": {}}
            logger.info(f"  {lower:.1f} <= x <= {upper:.1f}:")
            for metric in metrics:
                score = float(np.mean(scores_binned_all[bin_idx][metric]))
                result["metrics"][metric] = score
                logger.info(f"    {metric}: {score:.2f}")
            results_all["binned"].append(result)

    # Save results for all datasets
    save_path = result_root / "results_all.json"
    with save_path.open("w") as f:
        json.dump(results_all, f, indent=2)
    logger.success(f"Saved results for all datasets to {save_path}")


if __name__ == "__main__":
    main()
