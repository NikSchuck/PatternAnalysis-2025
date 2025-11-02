import os
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

def _strip_multi_suffix(path):
    """ strips the file suffix """
    name = path.name
    for suf in ['.nii.png', '.png']:
        if name.endswith(suf):
            return name[: -len(suf)]
    return path.stem  # fallback

def _match_mask_name(case_name):
    """ matches e.g. 'case_001_slice_0' with 'seg_001_slice_0' """
    return 'seg_' + case_name[len('case_'):]

def _find_images(folder):
    """ collects images """
    extensions = ['*.nii.png', '*.png']
    files = []
    for ext in extensions:
        files.extend(sorted(folder.glob(ext)))
    return files

# load dataset

class Oasis2DDataset(Dataset):
    """ initialises the data set of single channel greyscale images """
    def __init__(self, base_dir, split, normalize, dtype_img=np.float32):
        super().__init__()
        self.base = Path(base_dir)
        split = split.lower()
        if split == 'val':
            split = 'validate'
        #if split not in {'train', 'validate', 'test'}:
        #    raise ValueError("split must be 'train', 'validate', or 'test'")

        # map split to tehir folder names
        self.img_dir = self.base / f'keras_png_slices_{split}'
        self.msk_dir = self.base / f'keras_png_slices_seg_{split}'

        # inform of missing files or incorrect directory
        if not self.img_dir.is_dir() or not self.msk_dir.is_dir():
            raise FileNotFoundError(f"Expected folders missing: {self.img_dir} or {self.msk_dir}")

        # inform whether there are files in dir folders or for example, if file extension is wrong
        imgs = _find_images(self.img_dir)
        if len(imgs) == 0:
            raise ValueError(f"No images found in {self.img_dir}. Check path and extensions.")

        # match pairs by file extension mapping
        pairs = []
        missing = 0
        for ip in imgs:
            stem_case = _strip_multi_suffix(ip)
            seg_name = _match_mask_name(stem_case)

            # try both .nii.png and .png
            cand1 = self.msk_dir / f"{seg_name}.nii.png" # tries either extension
            cand2 = self.msk_dir / f"{seg_name}.png"
            mp = cand1 if cand1.exists() else cand2 if cand2.exists() else None
            if mp is None:
                missing += 1
                continue
            pairs.append((ip, mp))

        if len(pairs) == 0:
            raise ValueError(
                "No (image, mask) pairs found."
                f"Checked {len(imgs)} images, {missing} masks missing.")

        self.pairs = pairs
        self.normalize = normalize
        self.dtype_img = dtype_img

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path) -> np.ndarray:
        """ load as grayscale float32 in [0,1] then normalize """
        arr = np.array(Image.open(path).convert('L'), dtype=self.dtype_img)
        if arr.max() > 0:
            arr = arr / 255.0
        if self.normalize == 'zscore':
            m, s = float(arr.mean()), float(arr.std())
            if s > 0:
                arr = (arr - m) / s
            else:
                arr = arr - m
        return arr[None, ...]  # (1,H,W)

    def _load_mask(self, path) -> np.ndarray:
        """ OASIS masks are single-channel integer png's """
        arr = np.array(Image.open(path), dtype=np.int64)
        if arr.ndim == 3:
            # if stored as RGB, just in case for translating to new dataset
            arr = arr[..., 0].astype(np.int64)
        return arr

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """ gets images and masks, converts to tensors (with paths) """
        ip, mp = self.pairs[idx]
        img = self._load_image(ip)   # (1,H,W) float32
        msk = self._load_mask(mp)    # (H,W)   int64
        return {
            "image": torch.from_numpy(img).float(),
            "mask": torch.from_numpy(msk).long(),
            "image_path": str(ip),
            "mask_path": str(mp),
        }

def build_loaders(base_dir, batch_size=8, num_workers=0, pin_memory=False, normalize='zscore'):

    ds_train = Oasis2DDataset(base_dir, 'train', normalize=normalize)
    ds_val   = Oasis2DDataset(base_dir, 'validate', normalize=normalize)
    ds_test  = Oasis2DDataset(base_dir, 'test', normalize=normalize)

    def _make(dl_dataset, shuffle):
        return DataLoader(
            dl_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return {
        "train": _make(ds_train, shuffle=True),
        "validate": _make(ds_val, shuffle=False),
        "test": _make(ds_test, shuffle=False),
    }

if __name__ == "__main__":
    base = r"C:\Users\nschu\OneDrive\Desktop\COMP3710\Final Report\OASIS Dataset"
    loaders = build_loaders(base, batch_size=4)
    print("Train size:", len(loaders["train"].dataset))
    print("Val size:", len(loaders["validate"].dataset))
    print("Test size:", len(loaders["test"].dataset))

    batch = next(iter(loaders["train"]))
    x, y = batch["image"], batch["mask"]
    print("Batch image:", x.shape, x.dtype, x.min().item(), x.max().item())
    print("Batch mask: ", y.shape, y.dtype, int(y.min()), int(y.max()))
