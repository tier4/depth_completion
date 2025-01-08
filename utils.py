from pathlib import Path

import cv2
import imagesize
import numpy as np
from PIL import Image


def is_empty_img(img: Image.Image) -> bool:
    """Check if a PIL Image is empty (all values are 0).

    Args:
        img (Image.Image): Input PIL Image

    Returns:
        bool: True if image is empty (all values are 0), False otherwise
    """
    return not np.any(np.array(img))


def make_grid(
    imgs: np.ndarray,
    rows: int | None = None,
    cols: int | None = None,
    resize: tuple[int, int] | None = None,
) -> np.ndarray:
    """Create a grid of images from a numpy array of shape (N,H,W,C).

    Args:
        imgs (np.ndarray): Array of images with shape (N,H,W,C) where:
            N is number of images, H is height, W is width, C is channels
        rows (int, optional): Number of rows in grid.
            If None, will be calculated from cols.
            If both None, will try to create a square grid.
        cols (int, optional): Number of columns in grid.
            If None, will be calculated from rows.
            If both None, will try to create a square grid.
        resize (tuple[int, int], optional): Target size (height, width) of output.
            If None, no resizing is performed.
            If -1 is specified for either dimension, aspect ratio is preserved.
            If both dimensions are -1, no resizing is performed.

    Returns:
        np.ndarray: Grid image as numpy array

    Raises:
        ValueError: If images array is empty or not 4-dimensional
    """
    if imgs.size == 0 or len(imgs.shape) != 4:
        raise ValueError("Images must be non-empty 4D array (N,H,W,C)")

    n, h, w = imgs.shape[:3]

    # Calculate grid dimensions
    if rows is None and cols is None:
        cols = int(np.ceil(np.sqrt(n)))
    if rows is None:
        rows = int(np.ceil(n / cols))
    if cols is None:
        cols = int(np.ceil(n / rows))

    # Create and fill grid
    grid = np.zeros((h * rows, w * cols) + imgs.shape[3:], dtype=imgs.dtype)
    for idx in range(n):
        i, j = idx // cols, idx % cols
        grid[i * h : (i + 1) * h, j * w : (j + 1) * w] = imgs[idx]

    # Handle resizing
    if resize is not None:
        th, tw = resize
        if th != -1 or tw != -1:
            if th == -1:
                th = int(tw * grid.shape[0] / grid.shape[1])
            elif tw == -1:
                tw = int(th * grid.shape[1] / grid.shape[0])
            grid = cv2.resize(grid, (tw, th), interpolation=cv2.INTER_LINEAR)

    return grid


def to_depth(
    img: Image.Image, dtype: str = "float32", max_distance: float = 120.0
) -> np.ndarray:
    """Convert a PIL Image to a depth map.

    Args:
        img (Image.Image): Input PIL image
        dtype (str, optional): Data type for output array. Defaults to "float32".
        max_distance (float, optional): Maximum depth value in meters.
                                        Defaults to 120.0.

    Returns:
        np.ndarray: Depth map as numpy array with values ranging from 0 to max_distance,
                   where 0 represents the closest depth and max_distance the farthest.

    Raises:
        ValueError: If input image is not in RGB format
    """
    if img.mode != "RGB":
        raise ValueError(f"Input image must be RGB format, got {img.mode}")
    return max_distance * np.array(img, dtype=dtype)[..., 0] / 255.0


def is_img_file(path: Path) -> bool:
    """Check if a path points to a valid image file.

    Args:
        path (Path): Path to check

    Returns:
        bool: True if path points to a valid image file that can be opened,
              False otherwise
    """
    return path.is_file() and imagesize.get(path) != (-1, -1)


def get_img_paths(root: Path) -> list[Path]:
    """Get all image file paths under the given root directory.

    Args:
        root (Path): Root directory to search for images

    Returns:
        list[Path]: List of paths to image files
    """
    return [path for path in root.rglob("*") if is_img_file(path)]
