import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import blosc2
import click
import cv2
import imagesize
import numpy as np
from PIL import Image

NPARRAY_EXTENSIONS = [".npy", ".npz", ".bl2"]

CAMERA_CATEGORIES = [
    "CAM_FRONT_WIDE",
    "CAM_FRONT_NARROW",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK_WIDE",
    "CAM_BACK_NARROW",
]

IMAGE_DIR_NAME = "image"
SPARSE_DIR_NAME = "sparse"
VIS_DIR_NAME = "vis"
DENSE_DIR_NAME = "dense"


def crop_center(
    image: np.ndarray, height_ratio: float, width_ratio: float
) -> np.ndarray:
    """Perform center crop on an image array.

    Args:
        image (np.ndarray): Input image array with shape (N, H, W) or (N, H, W, C).
        height_ratio (float): Ratio of height to keep (0.0 to 1.0).
        width_ratio (float): Ratio of width to keep (0.0 to 1.0).

    Returns:
        np.ndarray: Center cropped image with the same number of dimensions as input.

    Raises:
        ValueError: If height_ratio or width_ratio is not between 0 and 1.
        ValueError: If image dimensions are not valid (must be 3D or 4D).

    Examples:
        >>> # Crop a batch of RGB images to 80% of height and 60% of width
        >>> cropped = crop_center(images, 0.8, 0.6)
    """
    if not (0.0 < height_ratio <= 1.0) or not (0.0 < width_ratio <= 1.0):
        raise ValueError("Height and width ratios must be between 0 and 1")

    if image.ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D array, got {image.ndim}D")

    # Get original dimensions
    if image.ndim == 3:  # (N, H, W)
        _, h, w = image.shape
    else:  # (N, H, W, C)
        _, h, w, _ = image.shape

    # Calculate new dimensions
    new_h = int(h * height_ratio)
    new_w = int(w * width_ratio)

    # Calculate start indices for cropping
    start_h = (h - new_h) // 2
    start_w = (w - new_w) // 2

    # Perform the crop
    if image.ndim == 3:
        return image[:, start_h : start_h + new_h, start_w : start_w + new_w]
    else:
        return image[:, start_h : start_h + new_h, start_w : start_w + new_w, :]


def load_csv(path: Path, columns: dict[str, type]) -> dict[str, list[Any]]:
    """Load a CSV file from disk with column selection and type conversion.

    Args:
        path (Path): Path to the CSV file.
        columns (dict[str, type]): Dictionary mapping column names
            to their desired types.

    Returns:
        dict[str, list[Any]]: A dictionary mapping column names to lists of values.

    Raises:
        ValueError: If any of the required columns are missing from the CSV file.

    Examples:
        >>> # Get specific columns with type conversion
        >>> data = load_csv(
        ...     Path("data.csv"),
        ...     columns={"id": int, "value": float, "name": str}
        ... )
        >>> ids = data["id"]  # List of integers
        >>> values = data["value"]  # List of floats
    """  # noqa: E501

    with open(path, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)

        header = data[0]
        rows = data[1:]

        # Check if all required columns exist in the CSV
        missing_columns = [col for col in columns if col not in header]
        if missing_columns:
            missing_cols_str = ", ".join(missing_columns)
            raise ValueError(
                f"Missing required columns in CSV file: {missing_cols_str}"
            )

        # Create a mapping from column name to index
        col_indices = {col: header.index(col) for col in columns}

        # Initialize result dictionary
        result: dict[str, list[Any]] = {col: [] for col in columns}

        # Extract and convert values
        for row in rows:
            for col, idx in col_indices.items():
                if idx < len(row):
                    value = row[idx]
                    value = columns[col](value)  # Data type conversion
                    result[col].append(value)

        return result


def is_array_path(path: Path) -> bool:
    """Check if a path points to a numpy array file.

    Args:
        path (Path): Path to check

    Returns:
        bool: True if path points to a numpy array file, False otherwise
    """
    return path.is_file() and path.suffix in NPARRAY_EXTENSIONS


def load_array(path: Path) -> np.ndarray:
    """Load a numpy array from disk.

    Args:
        path (Path): Path to the numpy array file. Supports .npy (uncompressed numpy),
            .npz (numpy compressed), and .bl2 (blosc2 compressed) formats.

    Returns:
        np.ndarray: The loaded numpy array. For depth maps, this typically contains
            distance values in meters, with zeros or negative values indicating
            invalid/missing measurements.

    Raises:
        ValueError: If the file extension is not one of the supported formats.

    Examples:
        >>> arr = load_array(Path("array.npy"))  # Load uncompressed array
        >>> arr = load_array(Path("array.npz"))  # Load npz compressed array
        >>> arr = load_array(Path("array.bl2"))   # Load blosc2 compressed array
        >>> depth_map = load_array(Path("depth.npy"))  # Load depth map
    """  # noqa: E501
    if not is_array_path(path):
        raise ValueError(
            f"Invalid extension: {path.suffix} (must be one of {NPARRAY_EXTENSIONS}"
        )
    if path.suffix == ".bl2":
        return blosc2.load_array(str(path))
    elif path.suffix == ".npz":
        return np.load(path)["arr_0"]
    return np.load(path)


def save_array(
    x: np.ndarray,
    path: Path,
    compress: str | None = None,
) -> None:
    """Save a numpy array to disk with optional compression.

    Args:
        x (np.ndarray): The numpy array to save
        path (Path): Path where the array should be saved
        compress (str | None, optional): The compression format to use.
            "npz" uses numpy's compressed format, "bl2" uses blosc2 compression.
            If None, saves uncompressed. Defaults to None.

    Raises:
        ValueError: If the file extension doesn't match the compression format:
            - .npy for uncompressed
            - .npz for npz compression
            - .bl2 for blosc2 compression

    Examples:
        >>> arr = np.random.rand(100, 100)
        >>> save_array(arr, Path("array.npy"))  # Save uncompressed
        >>> save_array(arr, Path("array.npz"), compress="npz")  # Save with npz compression
        >>> save_array(arr, Path("array.bl2"), compress="bl2")  # Save with blosc2 compression
    """  # noqa: E501
    # Check extension of given path
    if compress is None and path.suffix != ".npy":
        raise ValueError(f"Invalid extension: {path.suffix} (must be .npy)")
    elif compress == "npz" and path.suffix != ".npz":
        raise ValueError(f"Invalid extension: {path.suffix} (must be .npz)")
    elif compress == "bl2" and path.suffix != ".bl2":
        raise ValueError(f"Invalid extension: {path.suffix} (must be .bl2)")
    # Compress if requested
    if compress == "npz":
        np.savez_compressed(path, x)
    elif compress == "bl2":
        blosc2.save_array(x, str(path), mode="w")
    else:
        np.save(path, x)


def infer_camera_category(img_path: Path) -> str | None:
    """Infer the camera category from an image file path.

    Checks if any of the predefined camera categories (e.g. CAM_FRONT_WIDE, CAM_BACK_LEFT etc.)
    appear in the filename. Returns the first matching category found.

    Args:
        img_path (Path): Path to the image file

    Returns:
        str | None: The inferred camera category if found in the filename, None otherwise
    """  # noqa: E501
    for category in CAMERA_CATEGORIES:
        if category in img_path.name:
            return category
    return None


def mae(preds: np.ndarray, depth: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Calculate the mean absolute error between two depth maps.

    Args:
        preds (np.ndarray): Predicted depth map
        depth (np.ndarray): Ground truth depth map
        mask (np.ndarray | None, optional): Mask to apply to the depth maps.
    Returns:
        float: Mean absolute error between the two depth maps
    """  # noqa: E501
    if mask is not None:
        preds = preds[mask]
        depth = depth[mask]
    return float(np.mean(np.abs(preds - depth)))


def rmse(preds: np.ndarray, depth: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Calculate the root mean squared error between two depth maps.

    Args:
        preds (np.ndarray): Predicted depth map
        depth (np.ndarray): Ground truth depth map
        mask (np.ndarray | None, optional): Mask to apply to the depth maps.

    Returns:
        float: Root mean squared error between the two depth maps
    """  # noqa: E501
    if mask is not None:
        preds = preds[mask]
        depth = depth[mask]
    return float(np.sqrt(np.mean((preds - depth) ** 2)))


class CommaSeparated(click.ParamType):
    """A Click parameter type that parses comma-separated values into a list.

    This class extends Click's ParamType to handle comma-separated input strings,
    converting them into a list of values of a specified type. It can optionally
    enforce a specific number of values.

    Args:
        type_ (type): The type to convert each comma-separated value to. Defaults to str.
        n (int | None): If specified, enforces exactly this many comma-separated values.
            Must be None or a positive integer. Defaults to None.

    Raises:
        ValueError: If n is not None and not a positive integer.

    Examples:
        Basic usage with strings:
            @click.command()
            @click.option("--names", type=CommaSeparated())
            def cmd(names):
                # --names "alice,bob,charlie" -> ["alice", "bob", "charlie"]
                pass

        With integers and fixed length:
            @click.command()
            @click.option("--coords", type=CommaSeparated(int, n=2))
            def cmd(coords):
                # --coords "10,20" -> [10, 20]
                # --coords "1,2,3" -> Error: not exactly 2 values
                pass

        With floats:
            @click.command()
            @click.option("--weights", type=CommaSeparated(float))
            def cmd(weights):
                # --weights "0.1,0.2,0.7" -> [0.1, 0.2, 0.7]
                pass
    """  # noqa: E501

    name = "comma_separated"

    def __init__(self, type_: type = str, n: int | None = None) -> None:
        if n is not None and n <= 0:
            raise ValueError("n must be None or a positive integer")
        self.type = type_
        self.n = n

    def convert(
        self,
        value: str | None,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> list[Any] | None:
        if value is None:
            return None
        value = value.strip()
        if value == "":
            return []
        items = value.split(",")
        if self.n is not None and len(items) != self.n:
            self.fail(
                f"{value} does not contain exactly {self.n} comma separated values",
                param,
                ctx,
            )
        try:
            return [self.type(item) for item in items]
        except ValueError:
            self.fail(
                f"{value} is not a valid comma separated list of {self.type.__name__}",
                param,
                ctx,
            )


def is_empty_img(img: Image.Image) -> bool:
    """Check if a PIL Image is empty (all values are 0).

    Args:
        img (Image.Image): Input PIL Image

    Returns:
        bool: True if image is empty (all values are 0), False otherwise
    """  # noqa: E501
    return not np.any(np.array(img))


def load_img(path: Path, mode: str | None = None) -> tuple[np.ndarray, bool]:
    """Load an image from a file path and check if it's empty.

    Opens an image file using PIL and optionally converts it to a specific color mode.
    Also checks if the image is empty (all values are 0).
    Returns the image as a numpy array.

    Args:
        path (Path): Path to the image file to load
        mode (str | None, optional): PIL color mode to convert image
            to (e.g. 'RGB', 'L'). If None, keeps original mode.
            Defaults to None.

    Returns:
        tuple[np.ndarray, bool]: A tuple containing:
            - The loaded image as a numpy array
            - A boolean indicating if the image is non-empty (True) or empty (False)

    Example:
        >>> # Load RGB image
        >>> img, is_valid = load_img(Path("image.jpg"), mode="RGB")
        >>> # Load grayscale image
        >>> img, is_valid = load_img(Path("depth.png"), mode="L")
    """  # noqa: E501
    img_pil: Image.Image = Image.open(path)
    if mode is not None:
        img_pil = img_pil.convert(mode)
    img = np.array(img_pil)
    if not np.any(img):
        return img, False
    return img, True


def make_grid(
    imgs: np.ndarray,
    rows: int | None = None,
    cols: int | None = None,
    resize: tuple[int, int] | None = None,
    interpolation: int | list[int] = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Create a grid of images from a numpy array.

    Takes a batch of images and arranges them in a grid pattern. Can optionally resize
    the final grid output.

    Args:
        imgs (np.ndarray): Array of images with shape (N,H,W,C) where:
            N is number of images
            H is height of each image
            W is width of each image
            C is number of channels per image
        rows (int | None, optional): Number of rows in output grid. If None:
            - Will be calculated from cols if cols is specified
            - Will create a square-ish grid if cols is also None
        cols (int | None, optional): Number of columns in output grid. If None:
            - Will be calculated from rows if rows is specified
            - Will create a square-ish grid if rows is also None
        resize (tuple[int, int] | None, optional): Target (height, width) to resize final grid to.
            - If None: No resizing is performed
            - If either dimension is -1: That dimension is calculated to preserve aspect ratio
            - If both dimensions are -1: No resizing is performed
        interpolation (cv2.InterpolationFlags | list[cv2.InterpolationFlags], optional):
            OpenCV interpolation method(s) for resizing. Can be either:
            - A single interpolation flag to use for all images
            - A list of flags matching the number of input images
            Defaults to cv2.INTER_LINEAR.

    Returns:
        np.ndarray: Grid image with shape (grid_height, grid_width, C) containing all input
            images arranged in a grid pattern.

    Raises:
        ValueError: If imgs is empty or not a 4D array
        ValueError: If a list of interpolation methods is provided but length doesn't match
            number of input images

    Example:
        >>> # Create 2x2 grid from 4 images
        >>> grid = make_grid(images, rows=2, cols=2)
        >>> # Create auto-sized grid, resized to 512x512
        >>> grid = make_grid(images, resize=(512,512))
        >>> # Create grid with different interpolation per image
        >>> grid = make_grid(images, interpolation=[cv2.INTER_LINEAR, cv2.INTER_NEAREST])
    """  # noqa: E501
    if imgs.size == 0 or len(imgs.shape) != 4:
        raise ValueError("Images must be non-empty 4D array (N,H,W,C)")

    n = imgs.shape[0]
    if isinstance(interpolation, Sequence) and len(interpolation) != n:
        raise ValueError(
            f"Interpolation list length ({len(interpolation)}) "
            "must match number of images ({n})"
        )

    # Calculate grid dimensions
    if rows is None and cols is None:
        cols = int(np.ceil(np.sqrt(n)))
    if rows is None:
        rows = int(np.ceil(n / cols))
    if cols is None:
        cols = int(np.ceil(n / rows))

    h, w = imgs.shape[1:3]
    grid_h, grid_w = h * rows, w * cols

    # Calculate target size for the grid
    if resize is not None:
        th, tw = resize
        if th != -1 or tw != -1:
            methods = (
                interpolation
                if isinstance(interpolation, Sequence)
                else [interpolation] * n
            )
            target_h = th if th != -1 else int(tw * grid_h / grid_w)
            target_w = tw if tw != -1 else int(th * grid_w / grid_h)
            # Calculate individual image size based on grid target size
            h = target_h // rows
            w = target_w // cols
            # Resize all images to the new size
            imgs = np.array(
                [
                    cv2.resize(img, (w, h), interpolation=method)
                    for img, method in zip(imgs, methods, strict=True)
                ]
            )

    # Create and fill grid
    grid = np.zeros((h * rows, w * cols) + imgs.shape[3:], dtype=imgs.dtype)
    for idx in range(n):
        i, j = idx // cols, idx % cols
        grid[i * h : (i + 1) * h, j * w : (j + 1) * w] = imgs[idx]

    return grid


def has_nan(x: np.ndarray) -> bool:
    """Check if a numpy array contains any NaN values.

    Args:
        x (np.ndarray): Input array to check

    Returns:
        bool: True if array contains NaN values, False otherwise
    """  # noqa: E501
    return np.isnan(x).any()


def is_dataset_dir(path: Path) -> bool:
    """Check if a path points to a valid dataset directory.

    A valid dataset directory must be a directory that contains both
    an 'image' subdirectory and a 'sparse' subdirectory.

    Args:
        path (Path): Path to check

    Returns:
        bool: True if path is a valid dataset directory, False otherwise
    """
    if not path.is_dir():
        return False
    return (path / IMAGE_DIR_NAME).exists() and (path / SPARSE_DIR_NAME).exists()


def find_dataset_dirs(root: Path) -> list[Path]:
    """Find all valid dataset directories under the given root directory.

    A valid dataset directory is one that passes the is_dataset_dir() check.
    This function recursively searches through all subdirectories.

    Args:
        root (Path): Root directory to search for dataset directories

    Returns:
        list[Path]: List of paths to valid dataset directories
    """
    if is_dataset_dir(root):
        return [root]
    ret = []
    for path in root.rglob("*"):
        if is_dataset_dir(path):
            ret.append(path)
    return ret


def to_depth_map(
    img: np.ndarray, dtype: str = "float32", max_distance: float = 120.0
) -> np.ndarray:
    """Convert an RGB image array to a depth map.

    Args:
        img (np.ndarray): Input image array in RGB format
        dtype (str, optional): Data type for output array. Defaults to "float32".
        max_distance (float, optional): Maximum depth value in meters.
                                        Defaults to 120.0.

    Returns:
        np.ndarray: Depth map as numpy array with values ranging from 0 to max_distance,
                   where 0 represents the closest depth and max_distance the farthest.
                   The function uses only the red channel (index 0) of the RGB image.
    """  # noqa: E501
    return max_distance * np.array(img, dtype=dtype)[..., 0] / 255.0


def is_img_file(path: Path) -> bool:
    """Check if a path points to a valid image file.

    Args:
        path (Path): Path to check

    Returns:
        bool: True if path points to a valid image file that can be opened,
              False otherwise
    """  # noqa: E501
    return path.is_file() and imagesize.get(path) != (-1, -1)


def get_img_paths(root: Path) -> list[Path]:
    """Get all image file paths under the given root directory.

    Args:
        root (Path): Root directory to search for images

    Returns:
        list[Path]: List of paths to image files
    """  # noqa: E501
    return [path for path in root.rglob("*") if is_img_file(path)]
