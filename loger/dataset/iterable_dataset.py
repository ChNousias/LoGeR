from typing import Tuple, List, Iterator

import math
import torch
import numpy as np

from PIL import Image
from torchvision import transforms

from torch.utils.data import IterableDataset, get_worker_info


def estimate_resize(
    img: Image.Image,
    PIXEL_LIMIT: int = 255000,
    target_w: int | None = None,
    target_h: int | None = None,
    verbose: bool = True,
):
    """
    Estimate target width and height from (irst) image.
    """
    if target_w is None and target_h is None:
        W_orig, H_orig = img.size

        scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
        W_target, H_target = W_orig * scale, H_orig * scale
        k, m = round(W_target / 14), round(H_target / 14)
        while (k * 14) * (m * 14) > PIXEL_LIMIT:
            if k / m > W_target / H_target:
                k -= 1
            else:
                m -= 1
        TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    else:
        TARGET_W, TARGET_H = target_w, target_h

    if verbose:
        print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    return TARGET_W, TARGET_H


def get_window_indices(window_size: int, overlap_size: int, N: int) -> Tuple[int, int]:
    """
    Get window indices.
    """
    windows = []

    if window_size <= 0 or window_size >= N:
        windows.append((0, N))
        eff_overlap = 0
        eff_window_size = N
    else:
        step = max(window_size - overlap_size, 1)

        for start_idx in range(0, N, step):
            end_idx = min(start_idx + window_size, N)

            if end_idx - start_idx >= overlap_size or (end_idx == N and start_idx < N):
                windows.append((start_idx, end_idx))
            if end_idx == N:
                break

        eff_overlap = overlap_size
        eff_window_size = window_size

    return windows, eff_window_size, eff_overlap


def validate_image_paths(image_paths: List[str]) -> List[str]:
    """
    Validate collected image paths.
    """
    print("Validating input image frames.")
    validated_paths = []

    for img_path in image_paths:
        try:
            _ = Image.open(img_path).convert("RGB")

            validated_paths.append(img_path)
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")

    print("Valid image frames found: {}".format(len(validated_paths)))

    return validated_paths


def load_images_from_paths(image_paths: List[str]) -> List[Image.Image]:
    """
    Load images from paths.
    """
    pil_images = []
    for img_path in image_paths:
        try:
            pil_images.append(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")

    return pil_images


def pil_images_to_tensors(
    pil_images: List[Image.Image], target_w: int, target_h: int
) -> torch.Tensor:
    """
    Convert PIL images to torch tensors.
    """
    if not pil_images:
        print("No images found or loaded.")
        return torch.empty(0)

    to_tensor_transform = transforms.ToTensor()

    tensor_list = []

    for pil_img in pil_images:
        try:
            resized_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        return torch.empty(0)

    return torch.stack(tensor_list, dim=0)


def pil_images_to_numpy_array(
    pil_images: List[Image.Image], target_w: int, target_h: int
) -> np.ndarray:
    """
    Convert PIL images to numpy arrays. The returned numpy arrays follow
    the (N x H x W x C) in contrast to tensor arrays returned as (N x C x H x W)
    """
    if not pil_images:
        print("No images found or loaded.")
        return np.empty(0)

    # transforms.ToTensor
    # Convert a PIL Image or ndarray to tensor and scale the values accordingly.
    # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to
    # a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if the
    # PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA,
    # CMYK, 1) or if the numpy.ndarray has dtype = np.uint8
    # In the other cases, tensors are returned without scaling.

    array_list = []

    for pil_img in pil_images:
        try:
            resized_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

            img_array = np.array(resized_img, dtype="float32")

            np.divide(img_array, 255.0, out=img_array)

            array_list.append(img_array)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not array_list:
        return np.empty(0)

    return np.stack(array_list, axis=0)


class ImageWindowIterableDataset(IterableDataset):
    """
    Initializes a torch `IterableDataset` class for training.
    """

    def __init__(
        self,
        paths: List[str],
        window_size: int = 32,
        overlap_size: int = 3,
        target_w: int | None = None,
        target_h: int | None = None,
    ) -> None:
        """
        Initializes the Pytorch dataset class.

        Parameters
        ----------
        paths : List[str]
            Image paths.
        window_size : int
            Window size for non-causal inference (-1 for full sequence).
        overlap_size : int
            Overlap size for sliding window inference.
        target_w : int
            Target width size in pixels.
        target_h : int
            Target height size in pixels.
        """
        super().__init__()

        self.window_size = window_size
        self.overlap_size = overlap_size
        self.target_w = target_w
        self.target_h = target_h

        self.paths = validate_image_paths(paths)
        self.N = len(self.paths)

        try:
            _img = Image.open(self.paths[0]).convert("RGB")

            self.target_w, self.target_h = estimate_resize(
                img=_img, target_w=target_w, target_h=target_h
            )

            self.windows, self.window_size, self.overlap_size = get_window_indices(
                self.window_size, self.overlap_size, self.N
            )
        except IndexError:
            print("No valid image frames are given.")

    def init_img_path_stream(self, paths: List[str]) -> Iterator[torch.tensor]:
        """
        Create image stream.
        """
        for s_idx, e_idx in self.windows:
            yield paths[s_idx:e_idx]

    def img_path_processor(self, paths: List[str]) -> torch.Tensor:
        """
        Preprocess operation of a list of image paths.
        """
        imgs = load_images_from_paths(paths)

        return pil_images_to_tensors(imgs, self.target_w, self.target_h)

    def __iter__(self):
        """ """
        worker_info = get_worker_info()
        img_path_iterator = self.init_img_path_stream(self.paths)

        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Basic worker sharding: each worker processes every Nth record
        sharded_iterator = (
            record
            for i, record in enumerate(img_path_iterator)
            if i % num_workers == worker_id
        )

        # Apply processing within the worker's iterator chain
        processed_iterator = map(self.img_path_processor, sharded_iterator)
        return processed_iterator

    def __len__(self):
        # This refers to the length of the iterator, not the image paths.
        return len(self.windows)

    def load_all_images_as_np_array(self) -> np.ndarray:
        """
        Load images as numpy arrays
        """
        imgs = load_images_from_paths(self.paths)

        return pil_images_to_numpy_array(imgs, self.target_w, self.target_h)
