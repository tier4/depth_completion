import sys
from pathlib import Path
from typing import Literal

import click
import numpy as np
import tqdm
from loguru import logger

import utils

METRICS = ["mae", "rmse"]
Metric = Literal["mae", "rmse"]


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
    "--max-depth",
    type=click.FloatRange(min=0, min_open=True),
    default=120.0,
    help="Maximum distance in meters of depth maps.",
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
    max_depth: float,
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

    # Evaluation
    for dataset_dir in dataset_dirs:
        result_dir = result_root / (dataset_dir.relative_to(dataset_root))
        if not result_dir.exists():
            logger.warning(
                f"No result directory found for {dataset_dir.name}. "
                "Skip this dataset"
            )
            continue
        sparse_dir = dataset_dir / utils.DATASET_DIR_NAME_SPARSE
        dense_dir = result_dir / utils.RESULT_DIR_NAME_DENSE

        # Load sparse and dense depth maps
        sparse_paths: list[Path] = []
        dense_paths: list[Path] = []
        cache: set[str] = set()
        for path in sparse_dir.rglob("*"):
            if not utils.is_array_path(path):
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
        if len(sparse_paths) != len(dense_paths):
            logger.critical(
                f"Number of sparse and dense depth maps "
                f"mismatch: {len(sparse_paths)} != {len(dense_paths)}"
                f" for {dataset_dir.name}"
            )
            sys.exit(1)
        elif len(sparse_paths) == 0:
            logger.warning(
                f"No dense & sparse depth map pairs found for {dataset_dir.name}. "
                "Skip this dataset"
            )
            continue
        logger.info(
            f"Found {len(sparse_paths):,} pairs of sparse & dense "
            f"depth maps for {dataset_dir.name}"
        )

        # Compute overall metrics for each pair
        scores_overall: dict[Metric, list[float]] = {metric: [] for metric in metrics}
        progbar = tqdm.tqdm(
            total=len(sparse_paths),
            desc="Computing overall metrics...",
            dynamic_ncols=True,
        )
        for sparse_path, dense_path in zip(sparse_paths, dense_paths, strict=True):
            sparse_map = utils.load_array(sparse_path)
            dense_map = utils.load_array(dense_path)

            # Compute overall metrics
            for metric in metrics:
                mask = (sparse_map > 0) & (sparse_map <= max_depth)
                if metric == "mae":
                    loss = utils.mae(dense_map, sparse_map, mask=mask)
                else:
                    loss = utils.rmse(dense_map, sparse_map, mask=mask)
                scores_overall[metric].append(loss)
            progbar.update(1)
        progbar.close()

        # Print overall scores
        logger.info("Overall scores:")
        logger.info(f"  0.0 < x <= {max_depth:.1f} [m]:")
        for metric in metrics:
            logger.info(f"    {metric}: {np.mean(scores_overall[metric]):.2f}")

        # Compute bin-wise metrics if requested
        if calc_binned_scores:
            # Calculate bin boundaries
            scores_binned: list[dict[Metric, list[float]]] = []
            lowers: list[float] = [0.0]
            while lowers[-1] < max_depth:  # NOTE: Calc lower bounds of bins
                lowers.append(lowers[-1] + bin_size)
            lowers.pop()  # NOTE: Remove last lower bound
            for _ in lowers:
                scores_binned.append({metric: [] for metric in metrics})

            progbar = tqdm.tqdm(
                total=len(sparse_paths),
                desc="Computing binned metrics...",
                dynamic_ncols=True,
            )
            for sparse_path, dense_path in zip(sparse_paths, dense_paths, strict=True):
                sparse_map = utils.load_array(sparse_path)
                dense_map = utils.load_array(dense_path)

                # Compute bin-wise metrics
                mask_base = (sparse_map > 0) & (sparse_map <= max_depth)
                for bin_idx, lower in enumerate(lowers):
                    upper = min(lower + bin_size, max_depth)
                    if bin_idx == len(lowers) - 1:
                        mask = mask_base & (lower <= sparse_map)
                    else:
                        mask = mask_base & (lower <= sparse_map) & (sparse_map < upper)
                    if not np.any(mask):
                        continue
                    for metric in metrics:
                        if metric == "mae":
                            loss = utils.mae(dense_map, sparse_map, mask=mask)
                        else:
                            loss = utils.rmse(dense_map, sparse_map, mask=mask)
                        scores_binned[bin_idx][metric].append(loss)
                progbar.update(1)
            progbar.close()

            # Print binned scores
            logger.info("Binned scores:")
            for bin_idx, lower in enumerate(lowers):
                upper = min(lower + bin_size, max_depth)
                if bin_idx == 0:
                    if len(lowers) == 1:
                        logger.info(f"  {lower:.1f} < x <= {upper:.1f} [m]:")
                    else:
                        logger.info(f"  {lower:.1f} < x < {upper:.1f} [m]:")
                elif bin_idx == len(lowers) - 1:
                    logger.info(f"  {lower:.1f} <= x <= {upper:.1f} [m]:")
                else:
                    logger.info(f"  {lower:.1f} <= x < {upper:.1f} [m]:")
                for metric in metrics:
                    logger.info(
                        f"    {metric}: {np.mean(scores_binned[bin_idx][metric]):.2f}"
                    )


if __name__ == "__main__":
    main()
