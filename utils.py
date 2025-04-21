import concurrent.futures
import csv
from pathlib import Path
from typing import Any

import blosc2
import click
import cv2
import imagesize
import numpy as np
import torch
import torchvision
from diffusers.pipelines.marigold.marigold_image_processing import (
    MarigoldImageProcessor,
)

NPARRAY_EXTS = [".npy", ".npz", ".bl2"]
IMAGE_EXTS = [".jpg", ".png"]

DATASET_DIR_NAME_SPARSE = "sparse"
DATASET_DIR_NAME_IMAGE = "image"
DATASET_DIR_NAME_SEGMASK = "segmask"
RESULT_DIR_NAME_DENSE = "dense"
RESULT_DIR_NAME_VIS = "vis"

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


def calc_bins(
    lower_bound: float, upper_bound: float, bin_size: float
) -> list[tuple[float, float]]:
    """Calculate bin ranges from lower bound to upper bound with specified bin size.

    This function divides a range into bins of equal size. Each bin is represented
    as a tuple of (lower, upper) bounds. The last bin may be smaller than bin_size
    if the range is not evenly divisible.

    Args:
        lower_bound (float): The starting value for the first bin
        upper_bound (float): The maximum value (inclusive) for the last bin
        bin_size (float): The size of each bin

    Returns:
        list[tuple[float, float]]: List of (lower, upper) bound tuples for each bin

    Raises:
        ValueError: If lower_bound is greater than or equal to upper_bound
    """
    if lower_bound >= upper_bound:
        raise ValueError(
            f"Lower bound {lower_bound} must be less than upper bound {upper_bound}"
        )
    bins: list[tuple[float, float]] = []
    while lower_bound < upper_bound:
        bins.append((lower_bound, min(lower_bound + bin_size, upper_bound)))
        lower_bound += bin_size
    return bins


def is_dataset_dir(path: Path) -> bool:
    """Check if a path points to a valid dataset directory.

    A valid dataset directory must contain both 'image' and 'sparse' subdirectories.

    Args:
        path (Path): Path to check

    Returns:
        bool: True if path is a directory containing both image and sparse subdirectories,
              False otherwise
    """  # noqa: E501
    sparse_dir = path / DATASET_DIR_NAME_SPARSE
    img_dir = path / DATASET_DIR_NAME_IMAGE
    return path.is_dir() and sparse_dir.is_dir() and img_dir.is_dir()


def find_dataset_dirs(root: Path) -> list[Path]:
    """Find all valid dataset directories recursively under the given root directory.

    This function first checks if the root itself is a valid dataset directory.
    If so, it returns just the root. Otherwise, it recursively searches for all
    directories that contain both 'image' and 'sparse' subdirectories.

    Args:
        root (Path): Root directory to search for dataset directories

    Returns:
        list[Path]: List of paths to valid dataset directories (containing both
                   image and sparse subdirectories)
    """  # noqa: E501
    if is_dataset_dir(root):
        return [root]
    ret = [path for path in root.rglob("*") if is_dataset_dir(path)]
    return ret


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
    return path.is_file() and path.suffix in NPARRAY_EXTS


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
            f"Invalid extension: {path.suffix} (must be one of {NPARRAY_EXTS}"
        )
    if path.suffix == ".bl2":
        return blosc2.load_array(str(path))
    elif path.suffix == ".npz":
        return np.load(path)["arr_0"]
    return np.load(path)


def load_tensor(path: Path) -> torch.Tensor:
    """Load a numpy array from disk and convert it to a PyTorch tensor.

    Args:
        path (Path): Path to the numpy array file. Supports .npy (uncompressed numpy),
            .npz (numpy compressed), and .bl2 (blosc2 compressed) formats.

    Returns:
        torch.Tensor: The loaded array as a PyTorch tensor.

    Example:
        >>> # Load array as tensor
        >>> tensor = load_tensor(Path("array.npy"))
    """
    array = load_array(path)
    tensor = torch.from_numpy(array)
    return tensor


def visualize_depth(
    depth_maps: torch.Tensor,
    max_depth: float,
    min_depth: float = 0.0,
    color_map: str = "Spectral",
) -> torch.Tensor:
    """Visualize depth maps by converting them to colored representations.

    This function takes a batch of depth maps and converts them to RGB visualizations
    using a specified color map. The depth values are normalized to the range [0, 1]
    based on the provided min and max depth values.

    Args:
        depth_maps (torch.Tensor): Batch of depth maps with shape [N,1,H,W],
            where N is the batch size, and H, W are the height and width.
        max_depth (float): Maximum depth value for normalization.
        min_depth (float, optional): Minimum depth value for normalization.
            Defaults to 0.0.
        color_map (str, optional): Color map to use for visualization.
            Defaults to "Spectral".

    Returns:
        torch.Tensor: Batch of visualized depth maps as RGB images with shape [N,3,H,W].

    Raises:
        ValueError: If min_depth is greater than or equal to max_depth.
        ValueError: If depth_maps does not have the expected shape [N,1,H,W].

    Example:
        >>> depth_tensor = torch.randn(4, 1, 480, 640)  # 4 depth maps
        >>> rgb_tensor = visualize_depth(depth_tensor, 10.0)
        >>> # rgb_tensor has shape [4, 3, 480, 640]
    """  # noqa: E501
    if min_depth >= max_depth:
        raise ValueError(f"Invalid values range: [{min_depth}, {max_depth}].")
    if depth_maps.ndim != 4 or depth_maps.shape[1] != 1:
        raise ValueError(
            f"Input depth maps must have shape [N,1,H,W], got {depth_maps.shape}"
        )

    # Normalize depth maps to [0, 1]
    depth_maps = (depth_maps - min_depth) / (max_depth - min_depth)

    # Visualize each depth map and stack results
    visualized = torch.stack(
        [
            MarigoldImageProcessor.colormap(depth_map[0], cmap=color_map, bytes=True)
            for depth_map in depth_maps
        ],
        dim=0,
    )

    # Convert from [N,H,W,3] to [N,3,H,W]
    visualized = visualized.permute(0, 3, 1, 2)

    return visualized


def load_tensors(paths: list[Path], num_threads: int = 1) -> list[torch.Tensor]:
    """Load multiple numpy arrays from file paths in parallel using multithreading.

    Opens multiple numpy array files in parallel, converts them to PyTorch tensors.
    Returns a list of loaded tensors.

    Args:
        paths (list[Path]): List of paths to the numpy array files to load.
            Supports .npy (uncompressed numpy), .npz (numpy compressed),
            and .bl2 (blosc2 compressed) formats.
        num_threads (int, optional): Number of worker threads to use.
            Defaults to 1.

    Returns:
        list[torch.Tensor]: A list of loaded PyTorch tensors in the same order as the input paths.

    Example:
        >>> # Load multiple arrays as tensors in parallel
        >>> array_paths = [Path("array1.npy"), Path("array2.npz"), Path("array3.bl2")]
        >>> tensors = load_tensors(array_paths, num_threads=8)
    """  # noqa: E501
    if num_threads == 1:
        return [load_tensor(path) for path in paths]

    # Define worker function that calls load_tensor
    def worker(path: Path) -> torch.Tensor:
        return load_tensor(path)

    # Use ThreadPoolExecutor for parallel loading with executor.map to preserve order
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # map preserves the order of the input paths in the results
        results = list(executor.map(worker, paths))

    return results


def load_arrays(paths: list[Path], num_threads: int = 1) -> list[np.ndarray]:
    """Load multiple numpy arrays from file paths in parallel using multithreading.

    Opens multiple numpy array files in parallel and returns a list of loaded arrays.

    Args:
        paths (list[Path]): List of paths to the numpy array files to load.
            Supports .npy (uncompressed numpy), .npz (numpy compressed),
            and .bl2 (blosc2 compressed) formats.
        num_threads (int, optional): Number of worker threads to use.
            Defaults to 1.

    Returns:
        list[np.ndarray]: A list of loaded numpy arrays in the same order as the input paths.

    Example:
        >>> # Load multiple arrays in parallel
        >>> array_paths = [Path("array1.npy"), Path("array2.npz"), Path("array3.bl2")]
        >>> arrays = load_arrays(array_paths, num_threads=8)
        >>> for arr in arrays:
        ...     # Process array
        ...     pass
    """  # noqa: E501

    if not paths:
        return []

    # Define worker function that calls load_array
    def worker(path: Path) -> np.ndarray:
        return load_array(path)

    # Use sequential loading when num_threads=1
    if num_threads == 1:
        return [load_array(path) for path in paths]

    # Use ThreadPoolExecutor for parallel loading with executor.map to preserve order
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # map preserves the order of the input paths in the results
        results = list(executor.map(worker, paths))

    return results


def save_img(img: torch.Tensor, path: Path) -> None:
    """Save a PyTorch tensor image to disk as an image file.

    This function handles the conversion of tensor data to image format and saves it to disk.
    It automatically creates any necessary parent directories and handles different tensor
    data types appropriately.

    Args:
        img (torch.Tensor): The image tensor to save with shape [C, H, W].
            For uint8 tensors, values should be in range [0, 255].
            For float32 tensors, values should be in range [0.0, 1.0].
        path (Path): Path where the image should be saved. The file extension
            determines the output format (e.g., .png, .jpg).

    Returns:
        None

    Raises:
        ValueError: If the tensor has an unsupported dtype or if float32 values
            are outside the valid range [0.0, 1.0].

    Example:
        >>> # Save an RGB image tensor
        >>> img_tensor = torch.rand(3, 256, 256)  # [C, H, W] format in range [0, 1]
        >>> save_img(img_tensor, Path("output.png"))
        >>>
        >>> # Save a grayscale image
        >>> gray_tensor = torch.rand(1, 512, 512)  # Single channel
        >>> save_img(gray_tensor, Path("grayscale.jpg"))
    """  # noqa: E501
    # Ensure the directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure img is on CPU and convert to appropriate format
    img = img.detach().cpu()

    if img.dtype == torch.uint8:
        img = img.float() / 255.0
    elif img.dtype == torch.float32:
        if img.max() > 1.0 or img.min() < 0.0:
            raise ValueError(
                "Image tensor must be in the range [0, 1] if dtype is float32"
            )
    else:
        raise ValueError(f"Unsupported image type: {img.dtype}")

    # Use torchvision to save the image
    torchvision.utils.save_image(img, path)


def save_tensor(
    x: torch.Tensor,
    path: Path,
    compress: str | None = None,
) -> None:
    """Save a PyTorch tensor to disk with optional compression.

    Args:
        x (torch.Tensor): The PyTorch tensor to save
        path (Path): Path where the tensor should be saved
        compress (str | None, optional): The compression format to use.
            "npz" uses numpy's compressed format, "bl2" uses blosc2 compression,
            "npy" saves as uncompressed numpy format.
            If None, saves uncompressed as .npy. Defaults to None.

    Raises:
        ValueError: If the file extension doesn't match the compression format:
            - .npy for uncompressed or when compress="npy"
            - .npz for npz compression
            - .bl2 for blosc2 compression

    Examples:
        >>> tensor = torch.rand(100, 100)
        >>> save_tensor(tensor, Path("tensor.npy"))  # Save uncompressed
        >>> save_tensor(tensor, Path("tensor.npz"), compress="npz")  # Save with npz compression
        >>> save_tensor(tensor, Path("tensor.bl2"), compress="bl2")  # Save with blosc2 compression
        >>> save_tensor(tensor, Path("tensor.npy"), compress="npy")  # Explicitly save as .npy
    """  # noqa: E501
    # Convert tensor to numpy array
    x_np = x.detach().cpu().numpy()

    # Use save_array to handle the actual saving with compression
    save_array(x_np, path, compress=compress)


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
            "npz" uses numpy's compressed format, "bl2" uses blosc2 compression,
            "npy" saves as uncompressed numpy format.
            If None, saves uncompressed as .npy. Defaults to None.

    Raises:
        ValueError: If the file extension doesn't match the compression format:
            - .npy for uncompressed or when compress="npy"
            - .npz for npz compression
            - .bl2 for blosc2 compression

    Examples:
        >>> arr = np.random.rand(100, 100)
        >>> save_array(arr, Path("array.npy"))  # Save uncompressed
        >>> save_array(arr, Path("array.npz"), compress="npz")  # Save with npz compression
        >>> save_array(arr, Path("array.bl2"), compress="bl2")  # Save with blosc2 compression
        >>> save_array(arr, Path("array.npy"), compress="npy")  # Explicitly save as .npy
    """  # noqa: E501
    # Check extension of given path
    if compress is None and path.suffix != ".npy":
        raise ValueError(f"Invalid extension: {path.suffix} (must be .npy)")
    elif compress == "npz" and path.suffix != ".npz":
        raise ValueError(f"Invalid extension: {path.suffix} (must be .npz)")
    elif compress == "bl2" and path.suffix != ".bl2":
        raise ValueError(f"Invalid extension: {path.suffix} (must be .bl2)")
    elif compress == "npy" and path.suffix != ".npy":
        raise ValueError(f"Invalid extension: {path.suffix} (must be .npy)")
    # Compress if requested
    if compress == "npz":
        np.savez_compressed(path, x)
    elif compress == "bl2":
        blosc2.save_array(x, str(path), mode="w")
    elif compress == "npy":
        np.save(path, x)
    else:
        np.save(path, x)


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


def load_img_tensor(path: Path, mode: str | None = None) -> torch.Tensor | None:
    """Load an image from a file path as a PyTorch tensor.

    Opens an image file using OpenCV, optionally converts it to a specific color mode,
    and returns it as a PyTorch tensor. Returns None if the image is empty or couldn't be loaded.

    Args:
        path (Path): Path to the image file to load
        mode (str | None, optional): Color mode to convert image
            to (e.g. 'RGB', 'BGR', 'L'). If None, automatically determines mode
            based on image channels (RGB for 3 channels, L for 1 channel).
            Defaults to None.

    Returns:
        torch.Tensor | None: The loaded image as a PyTorch tensor with shape [C, H, W],
            or None if the image is empty or couldn't be loaded.

    Example:
        >>> # Load RGB image as tensor
        >>> img = load_img_tensor(Path("image.jpg"), mode="RGB")
        >>> if img is not None:
        ...     # Process valid tensor
        ...     pass
        >>> # Load BGR image (OpenCV default) as tensor
        >>> img = load_img_tensor(Path("image.jpg"), mode="BGR")
        >>> # Load grayscale image as tensor
        >>> img = load_img_tensor(Path("depth.png"), mode="L")
        >>> # Auto-detect mode based on channels
        >>> img = load_img_tensor(Path("image.jpg"))
    """  # noqa: E501
    img_array = load_img_array(path, mode)
    if img_array is None:
        return None

    # Convert numpy array to PyTorch tensor
    if img_array.ndim == 2:  # Grayscale image
        tensor = torch.from_numpy(img_array).unsqueeze(0)  # [1, H, W]
    else:  # Color image
        # Move channel dimension from last to first: [H, W, C] -> [C, H, W]
        tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    return tensor


def load_img_array(path: Path, mode: str | None = None) -> np.ndarray | None:
    """Load an image from a file path.

    Opens an image file using OpenCV and optionally converts it to a specific color mode.
    Returns the image as a numpy array, or None if the image is empty or couldn't be loaded.

    Args:
        path (Path): Path to the image file to load
        mode (str | None, optional): Color mode to convert image
            to (e.g. 'RGB', 'BGR', 'L'). If None, automatically determines mode
            based on image channels (RGB for 3 channels, L for 1 channel).
            Defaults to None.

    Returns:
        np.ndarray | None: The loaded image as a numpy array, or None if the image
            is empty or couldn't be loaded.

    Example:
        >>> # Load RGB image
        >>> img = load_img(Path("image.jpg"), mode="RGB")
        >>> if img is not None:
        ...     # Process valid image
        ...     pass
        >>> # Load BGR image (OpenCV default)
        >>> img = load_img(Path("image.jpg"), mode="BGR")
        >>> # Load grayscale image
        >>> img = load_img(Path("depth.png"), mode="L")
        >>> # Auto-detect mode based on channels
        >>> img = load_img(Path("image.jpg"))
    """  # noqa: E501
    if not is_img_file(path):
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        # Return None if image couldn't be loaded
        return None

    # Determine mode automatically if not specified
    if mode is None:
        if img.ndim == 3 and img.shape[2] == 3:
            mode = "RGB"
        elif img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            mode = "L"
        # Keep as BGR for other cases

    # Convert color mode if needed
    if mode == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif mode == "L":
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Add dimension to keep consistent shape for grayscale
        if img.ndim == 2:
            img = img[..., np.newaxis]

    # Check if image is empty (all zeros)
    if not np.any(img):
        return None
    return img


def load_img_arrays(
    paths: list[Path], mode: str | None = None, num_threads: int = 1
) -> list[np.ndarray | None]:
    """Load multiple images from file paths in parallel using multithreading.

    Opens multiple image files using OpenCV in parallel and optionally converts them
    to a specific color mode. Returns a list of loaded images as numpy arrays or None
    for images that couldn't be loaded.

    Args:
        paths (list[Path]): List of paths to the image files to load
        mode (str | None, optional): Color mode to convert images
            to (e.g. 'RGB', 'BGR', 'L'). If None, automatically determines mode
            based on image channels (RGB for 3 channels, L for 1 channel).
            Defaults to None.
        num_threads (int, optional): Number of worker threads to use.
            Defaults to 1.

    Returns:
        list[np.ndarray | None]: A list of loaded images as numpy arrays, or None for
            images that couldn't be loaded.

    Example:
        >>> # Load multiple RGB images in parallel
        >>> image_paths = [Path("image1.jpg"), Path("image2.jpg"), Path("image3.jpg")]
        >>> results = load_imgs(image_paths, mode="RGB", num_threads=8)
        >>> for img in results:
        ...     if img is not None:
        ...         # Process valid image
        ...         pass
    """  # noqa: E501
    if not paths:
        return []

    # For single-threaded execution, use simple loop instead of ThreadPoolExecutor
    if num_threads == 1:
        return [load_img_array(path, mode) for path in paths]

    # Define worker function that calls load_img
    def worker(path: Path) -> np.ndarray | None:
        return load_img_array(path, mode)

    # Use ThreadPoolExecutor for parallel loading with executor.map to preserve order
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # map preserves the order of the input paths in the results
        results = list(executor.map(worker, paths))

    return results


def load_img_tensors(
    paths: list[Path],
    mode: str | None = None,
    num_threads: int = 1,
) -> list[torch.Tensor | None]:
    """Load multiple images from file paths as PyTorch tensors in parallel using multithreading.

    Opens multiple image files using OpenCV in parallel, optionally converts them
    to a specific color mode, and returns them as PyTorch tensors. Returns a list of
    loaded images as tensors or None for images that couldn't be loaded.

    Args:
        paths (list[Path]): List of paths to the image files to load
        mode (str | None, optional): Color mode to convert images
            to (e.g. 'RGB', 'BGR', 'L'). If None, automatically determines mode
            based on image channels (RGB for 3 channels, L for 1 channel).
            Defaults to None.
        num_threads (int, optional): Number of worker threads to use.
            Defaults to 1.

    Returns:
        list[torch.Tensor | None]: A list of loaded images as PyTorch tensors with shape [C, H, W],
            or None for images that couldn't be loaded.

    Example:
        >>> # Load multiple RGB images as tensors in parallel
        >>> image_paths = [Path("image1.jpg"), Path("image2.jpg"), Path("image3.jpg")]
        >>> results = load_img_tensors(image_paths, mode="RGB", num_threads=8)
        >>> for tensor in results:
        ...     if tensor is not None:
        ...         # Process valid tensor
        ...         pass
    """  # noqa: E501
    if not paths:
        return []

    # For single-threaded execution, use simple loop instead of ThreadPoolExecutor
    if num_threads == 1:
        return [load_img_tensor(path, mode) for path in paths]

    # Define worker function that calls load_img_tensor
    def worker(path: Path) -> torch.Tensor | None:
        return load_img_tensor(path, mode)

    # Use ThreadPoolExecutor for parallel loading with executor.map to preserve order
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # map preserves the order of the input paths in the results
        results = list(executor.map(worker, paths))

    return results


def make_grid(
    imgs: torch.Tensor | list[torch.Tensor],
    nrow: int | None = None,
    resize: tuple[int, int] | None = None,
    interpolation: str = "bilinear",
    antialias: bool = False,
) -> torch.Tensor:
    """Create a grid of images using torchvision.utils.make_grid.

    Args:
        imgs (torch.Tensor | list[torch.Tensor]): Tensor of images with shape (N,C,H,W)
            or list of tensors with shape (C,H,W)
        nrow (int | None): Number of images in each row of the grid. If None, all images
            will be placed in a single row. Defaults to None.
        resize (tuple[int, int] | None): Target (height, width) to resize final grid to.
            If None: No resizing is performed
            If either dimension is -1: That dimension is calculated to preserve aspect ratio
        interpolation (str): Interpolation mode for resizing. Valid values are:
            "nearest", "bilinear", "bicubic", "lanczos". Defaults to "bilinear".
        antialias (bool): Whether to use antialiasing when resizing. This can improve
            quality but may be slower. Defaults to False.

    Returns:
        torch.Tensor: Grid image with shape (C, grid_height, grid_width)

    Raises:
        ValueError: If an empty list of images is provided, if images have incorrect shape,
            or if an unsupported interpolation mode is specified.

    Example:
        >>> # Create a grid from a batch of images
        >>> batch = torch.rand(8, 3, 64, 64)  # 8 RGB images
        >>> grid = make_grid(batch, nrow=4)  # 2x4 grid
        >>>
        >>> # Create a grid from individual images and resize
        >>> images = [torch.rand(3, 64, 64) for _ in range(5)]
        >>> grid = make_grid(images, resize=(256, -1))  # Resize height to 256, width auto
    """  # noqa: E501

    # Convert list of tensors to a single tensor if needed
    if isinstance(imgs, list):
        if not imgs:
            raise ValueError("Empty list of images provided")
        # Check if all tensors have shape (C,H,W)
        for img in imgs:
            if not isinstance(img, torch.Tensor) or img.dim() != 3:
                raise ValueError("Each image in the list must be a 3D tensor (C,H,W)")
        # Stack tensors to create a batch
        imgs = torch.stack(imgs)

    if imgs.dim() != 4:
        raise ValueError("Images must be 4D tensor (N,C,H,W)")

    # Create grid using torchvision
    if nrow is None:
        nrow = len(imgs)
    grid = torchvision.utils.make_grid(imgs, nrow=nrow)

    # Resize if needed
    if resize is not None:
        th, tw = resize
        if th != -1 or tw != -1:
            # Get current dimensions
            _, h, w = grid.shape

            # Preserve aspect ratio if either dimension is -1
            target_h = th if th != -1 else int(tw * h / w)
            target_w = tw if tw != -1 else int(th * w / h)

            # Get interpolation mode (only accept lowercase)
            interpolation = interpolation.lower()
            if interpolation == "nearest":
                mode = torchvision.transforms.InterpolationMode.NEAREST
            elif interpolation == "bilinear":
                mode = torchvision.transforms.InterpolationMode.BILINEAR
            elif interpolation == "bicubic":
                mode = torchvision.transforms.InterpolationMode.BICUBIC
            elif interpolation == "lanczos":
                mode = torchvision.transforms.InterpolationMode.LANCZOS
            else:
                raise ValueError(
                    f"Unsupported interpolation mode: {interpolation}. "
                    "Supported modes are: 'nearest', 'bilinear', 'bicubic', 'lanczos'."
                )

            # Resize directly without creating a transform object
            grid = torchvision.transforms.functional.resize(
                grid.unsqueeze(0),
                (target_h, target_w),
                interpolation=mode,
                antialias=antialias,
            ).squeeze(0)

    return grid


def has_nan(x: np.ndarray | torch.Tensor) -> bool:
    """Check if an array contains any NaN values.

    Args:
        x (np.ndarray | torch.Tensor): Input array to check

    Returns:
        bool: True if array contains NaN values, False otherwise
    """  # noqa: E501
    if isinstance(x, torch.Tensor):
        return bool(torch.isnan(x).any().item())
    else:
        return np.isnan(x).any()


def to_depth(
    imgs: torch.Tensor, dtype: torch.dtype = torch.float32, max_distance: float = 120.0
) -> torch.Tensor:
    """Convert a normalized depth image tensor to a depth map.

    Given an input tensor where pixel intensity in the first channel encodes
    depth in the range [0, 255], this function rescales and casts values to the
    specified floating-point dtype and maps them linearly to [0, max_distance].

    Args:
        imgs (torch.Tensor): Input image tensor with shape (..., C) or (..., 1)
            where the first channel contains depth-encoded intensities in [0, 255].
        dtype (torch.dtype, optional): Desired output data type (e.g., torch.float32).
            Defaults to torch.float32.
        max_distance (float, optional): Maximum depth value corresponding to
            intensity=255. Defaults to 120.0.

    Returns:
        torch.Tensor: Depth map tensor with the same spatial dimensions as the input,
            values scaled to [0, max_distance] and cast to `dtype`.
    """
    return max_distance * imgs.to(dtype)[..., 0] / 255.0


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


def find_img_paths(root: Path) -> list[Path]:
    """Find all valid image file paths recursively under the given root directory.

    This function searches for all files in the root directory and its subdirectories,
    and filters them to include only valid image files that can be opened.

    Args:
        root (Path): Root directory to search for images

    Returns:
        list[Path]: List of paths to valid image files that can be opened
    """  # noqa: E501
    return [path for path in root.rglob("*") if is_img_file(path)]


def find_file_with_exts(path: Path, exts: list[str] | None = None) -> Path | None:
    """Find a file with the given path or with alternative extensions.

    This function first checks if the exact path exists and is a file. If not,
    it tries to find a file with the same stem but with one of the provided
    alternative extensions.

    Args:
        path (Path): The original path to check
        exts (list[str] | None, optional): List of alternative extensions to try.
                                          Defaults to None.

    Returns:
        Path | None: The path to the found file, or None if no matching file exists

    Examples:
        >>> # Check if 'data.npy' exists, or try 'data.npz' and 'data.bl2'
        >>> file_path = find_file_with_exts(Path('data.npy'), ['.npz', '.bl2'])
    """
    if path.exists() and path.is_file():
        return path

    # Try each extension
    if exts is not None:
        for ext in exts:
            alt_path = path.with_suffix(ext)
            if alt_path.exists() and alt_path.is_file():
                return alt_path

    return None
