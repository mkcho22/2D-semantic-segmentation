"""Exact-name pairing and the original PIL/ImageNet preprocessing."""

from pathlib import Path
import warnings

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .classes import MEAN, STD, NUM_CLASSES

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def pair_files(root_dir: str | Path, split: str, skip_missing: bool = False):
    split_dir = Path(root_dir) / split
    directories = (split_dir / "RGB", split_dir / "Label")
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {directory}")
    images, labels = [
        {p.name: p for p in directory.iterdir()
         if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS}
        for directory in directories
    ]
    missing_labels = sorted(images.keys() - labels.keys())
    missing_images = sorted(labels.keys() - images.keys())
    names = sorted(images.keys() & labels.keys())
    report = {
        "split": split, "rgb_count": len(images), "label_count": len(labels),
        "paired_count": len(names), "missing_labels": missing_labels,
        "missing_images": missing_images, "skip_missing": skip_missing,
    }
    if missing_labels or missing_images:
        message = (f"{split}: unmatched files; missing labels={missing_labels}; "
                   f"missing RGB={missing_images}")
        if not skip_missing:
            raise ValueError(message + ". Restore the files or explicitly use --skip-missing.")
        warnings.warn(message + f". Using {len(names)} exact-name pairs.", stacklevel=2)
    if not names:
        raise ValueError(f"No matched image/label pairs in {split_dir}")
    return [(images[name], labels[name]) for name in names], report


def load_label(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    """Read raw class IDs from grayscale/palette or the RGB red channel.

    Label 255 is remapped to class 0, retaining the historical project's policy.
    It is not treated as an ignored pixel. RGB colorized masks are not supported.
    """
    with Image.open(path) as image:
        array = np.array(image)
    if array.ndim == 3:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D class-ID mask: {path}")
    array = array.astype(np.int64)
    array[array == 255] = 0
    invalid = np.unique(array[(array < 0) | (array >= NUM_CLASSES)])
    if invalid.size:
        raise ValueError(f"Invalid class IDs {invalid.tolist()} in {path}; expected 0–12 or 255")
    if size is not None:
        array = np.array(Image.fromarray(array.astype(np.uint8)).resize(
            size, Image.Resampling.NEAREST), dtype=np.int64)
    return array


def load_image(path: str | Path, size: tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.array(image.convert("RGB").resize(size, Image.Resampling.BILINEAR), copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div(255)
    return (tensor - tensor.new_tensor(MEAN)[:, None, None]) / tensor.new_tensor(STD)[:, None, None]


class CarlaDataset(Dataset):
    def __init__(self, root_dir, split="Training", height=600, width=800, skip_missing=False):
        self.pairs, self.report = pair_files(root_dir, split, skip_missing)
        self.size = (width, height)
        if min(self.size) < 16:
            raise ValueError("Image dimensions must be at least 16")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, label_path = self.pairs[index]
        return load_image(image_path, self.size), torch.from_numpy(load_label(label_path, self.size))
