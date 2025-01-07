from pathlib import Path

import imagesize
import numpy as np
from PIL import Image


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
