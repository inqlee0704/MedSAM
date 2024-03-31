import argparse
import os
import random
from copy import deepcopy
from datetime import datetime
from glob import glob
from os import listdir, makedirs
from os.path import basename, exists, isdir, isfile, join
from time import time

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from matplotlib import pyplot as plt
from sklearn.model_selection import GroupKFold, KFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
from tqdm.auto import tqdm

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT


class NpyDataset(Dataset):
    def __init__(self, data_root, image_size=256, bbox_shift=5, data_aug=True):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(glob(join(self.gt_path, "*.npy"), recursive=True))
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if isfile(join(self.img_path, basename(file)))
        ]
        self.image_size = image_size
        self.target_length = image_size
        self.bbox_shift = bbox_shift
        self.data_aug = data_aug

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = basename(self.gt_path_files[index])
        assert img_name == basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        img_3c = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (H, W, 3)
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
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
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


def main():
    data_root = "./data"
    img_dir = join(data_root, "imgs")
    mask_dir = join(data_root, "gts")
    gt_path_list = sorted(glob(join(mask_dir, "*.npy"), recursive=True))
    gt_path_list = [
        file for file in gt_path_list if isfile(join(img_dir, basename(file)))
    ]
    tr_dataset = NpyDataset(data_root, data_aug=True)
    modality_subcls_list = [
        x.split("/")[-1].split("_")[0] + "_" + x.split("/")[-1].split("_")[1]
        for x in tr_dataset.gt_path_files
    ]
    modality_subcls_list, modality_subcls_count = np.unique(
        modality_subcls_list, return_counts=True
    )
    subcls_dict = dict(zip(modality_subcls_list, modality_subcls_count))

    df_list = []
    for i, subcls in enumerate(subcls_dict.keys()):
        gt_subcls_path_list = [x for x in gt_path_list if subcls in x]
        img_subcls_path_list = [
            x.replace("/gts/", "/imgs/") for x in gt_path_list if subcls in x
        ]
        filename_list = [x.split("/")[-1] for x in gt_path_list if subcls in x]
        if subcls.startswith("CT") or ("MR" in subcls):
            pid_list = [x.rsplit("-", 1)[0] for x in filename_list]
            df = pd.DataFrame(
                {
                    "filename": filename_list,
                    "img_path": img_subcls_path_list,
                    "gt_path": gt_subcls_path_list,
                    "patient_id": pid_list,
                }
            )
            n_split = min(10, len(np.unique(pid_list)))
            skf = GroupKFold(n_splits=n_split)  # choose between 10% or just 1 paitent
            for fold, (train_idx, val_idx) in enumerate(
                skf.split(df, groups=df["patient_id"])
            ):
                df.loc[val_idx, "fold"] = int(fold)
            df_list.append(df)
        else:  # 2D images (no group kofld needed)
            df = pd.DataFrame(
                {
                    "filename": filename_list,
                    "img_path": img_subcls_path_list,
                    "gt_path": gt_subcls_path_list,
                }
            )
            skf = KFold(n_splits=10)
            for fold, (train_idx, val_idx) in enumerate(skf.split(df)):
                df.loc[val_idx, "fold"] = int(fold)
            df_list.append(df)

    df_train_all = pd.DataFrame([])
    df_valid_all = pd.DataFrame([])
    for df in df_list:
        df_train = df[df["fold"] != 0].reset_index(drop=True)
        df_valid = df[df["fold"] == 0].reset_index(drop=True)

        df_train_all = pd.concat([df_train_all, df_train.iloc[:, :3]])
        df_train_all.reset_index(drop=True, inplace=True)
        df_valid_all = pd.concat([df_valid_all, df_valid.iloc[:, :3]])
        df_valid_all.reset_index(drop=True, inplace=True)
    print(len(df_train_all))
    print(len(df_valid_all))
    df_train_all.to_csv(join(data_root, "train_list.csv"), index=False)
    df_valid_all.to_csv(join(data_root, "valid_list.csv"), index=False)


if __name__ == "__main__":
    main()
