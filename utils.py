from pathlib import Path

import imagesize


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
