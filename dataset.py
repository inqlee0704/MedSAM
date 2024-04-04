# %%
import random
from os.path import join, isfile, basename
import numpy as np
import torch
from torch.utils.data import Dataset

import cv2
from pathlib import Path
from file_io import rle_decode_multivalue
import pickle


class EncodedDataset(Dataset):
    def __init__(self, data_root, image_size=256, bbox_shift=5, data_aug=True):
        self.data_root = data_root
        self.gt_path = Path(join(data_root, "gts"))
        self.img_path = Path(join(data_root, "imgs"))
        self.gt_path_files = self.gt_path.glob("*.*")
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if isfile(self.img_path / f"{file.stem}.png")
        ]
        self.image_size = image_size
        self.target_length = image_size
        self.bbox_shift = bbox_shift
        self.data_aug = data_aug

    def __len__(self):
        return len(self.gt_path_files)

    def _load_gt(self, gt_path):
        gt_path = Path(gt_path)
        if gt_path.suffix == ".pkl":
            with open(gt_path, "rb") as f:
                rle_encode = pickle.load(f)
            gt = rle_decode_multivalue(rle_encode, [self.image_size, self.image_size])
        else:
            gt = np.load(gt_path, "r", allow_pickle=True)
        return gt

    def __getitem__(self, index):
        img_name = basename(self.gt_path_files[index])
        assert img_name == basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        img_path = self.img_path / f"{self.gt_path_files[index].stem}.png"
        img_3c = cv2.imread(img_path.as_posix())
        # img_3c = np.load(join(self.img_path, img_name), 'r', allow_pickle=True) # (H, W, 3)
        img_resize = self.resize_longest_side(img_3c)
        # Resizing
        img_resize = (img_resize - img_resize.min()) / np.clip(
            img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3
        img_padded = self.pad_image(img_resize)  # (256, 256, 3)
        # convert the shape to (3, H, W)
        img_padded = np.transpose(img_padded, (2, 0, 1))  # (3, 256, 256)
        assert (
            np.max(img_padded) <= 1.0 and np.min(img_padded) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = self._load_gt(self.gt_path_files[index])
        gt = cv2.resize(
            gt,
            (img_resize.shape[1], img_resize.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)
        gt = self.pad_image(gt)  # (256, 256)
        label_ids = np.unique(gt)[1:]
        try:
            gt2D = np.uint8(
                gt == random.choice(label_ids.tolist())
            )  # only one label, (256, 256)
        except:
            print(img_name, "label_ids.tolist()", label_ids.tolist())
            gt2D = np.uint8(gt == np.max(gt))  # only one label, (256, 256)
        # add data augmentation: random fliplr and random flipud
        if self.data_aug:
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-1))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
                # print('DA with flip left right')
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-2))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
                # print('DA with flip upside down')
        gt2D = np.uint8(gt2D > 0)
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        return {
            "image": torch.tensor(img_padded).float(),
            "gt2D": torch.tensor(gt2D[None, :, :]).long(),
            "bboxes": torch.tensor(bboxes[None, None, ...]).float(),  # (B, 1, 4)
            "image_name": img_name,
            "new_size": torch.tensor(
                np.array([img_resize.shape[0], img_resize.shape[1]])
            ).long(),
            "original_size": torch.tensor(
                np.array([img_3c.shape[0], img_3c.shape[1]])
            ).long(),
        }

    def resize_longest_side(self, image):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        long_side_length = self.target_length
        oldh, oldw = image.shape[0], image.shape[1]
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww, newh = int(neww + 0.5), int(newh + 0.5)
        target_size = (neww, newh)

        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    def pad_image(self, image):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        # Pad
        h, w = image.shape[0], image.shape[1]
        padh = self.image_size - h
        padw = self.image_size - w
        if len(image.shape) == 3:  ## Pad image
            image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
        else:  ## Pad gt mask
            image_padded = np.pad(image, ((0, padh), (0, padw)))

        return image_padded


if __name__ == "__main__":
    dataset = EncodedDataset("/Datas/competition/cvpr/train")
    for d in dataset:
        d
    pass
