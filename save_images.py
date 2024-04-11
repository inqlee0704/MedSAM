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
import monai
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
import pandas as pd

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.45])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.45])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


def cal_iou(result, reference):

    intersection = torch.count_nonzero(
        torch.logical_and(result, reference), dim=[i for i in range(1, result.ndim)]
    )
    union = torch.count_nonzero(
        torch.logical_or(result, reference), dim=[i for i in range(1, result.ndim)]
    )

    iou = intersection.float() / union.float()

    return iou.unsqueeze(1)


# %%
class NpyDataset(Dataset):
    def __init__(
        self, data_root, image_size=256, bbox_shift=5, data_aug=True, mode="train"
    ):
        self.mode = mode
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.train_csv = pd.read_csv(join(data_root, "train_list.csv"))
        self.valid_csv = pd.read_csv(join(data_root, "valid_list.csv"))
        if mode == "train":
            self.gt_path_files = self.train_csv["gt_path"].values
        elif mode == "valid":
            self.gt_path_files = self.valid_csv["gt_path"].values
        else:  # use all data
            self.gt_path_files = sorted(
                glob(join(self.gt_path, "*.npy"), recursive=True)
            )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if isfile(join(self.img_path, basename(file)))
        ]
        self.gt_path_files = [x for x in self.gt_path_files if "Microscopy" in x]

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


if __name__ == "__main__":
    work_dir = "./workdir"
    img_save_dir = join(work_dir, "images_micro")
    os.makedirs(img_save_dir, exist_ok=True)
    data_root = "./data"
    medsam_lite_checkpoint = "lite_medsam.pth"
    num_epochs = 10
    batch_size = 4
    num_workers = 4
    device = "cuda:0"
    bbox_shift = 5
    lr = 5e-5
    weight_decay = 0.01
    iou_loss_weight = 1.0
    seg_loss_weight = 1.0
    ce_loss_weight = 1.0
    do_sancheck = True
    checkpoint = "workdir/medsam_lite_latest.pth"
    makedirs(work_dir, exist_ok=True)
    makedirs(img_save_dir, exist_ok=True)

    torch.cuda.empty_cache()
    os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
    os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

    tr_dataset = NpyDataset(data_root, data_aug=False, mode="train")
    tr_dataloader = DataLoader(tr_dataset, batch_size=4, shuffle=True)
    # val_dataset = NpyDataset(data_root, data_aug=False, mode='valid')
    # val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # for step, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
    for step, batch in tqdm(enumerate(tr_dataloader), total=len(tr_dataloader)):
        _, axs = plt.subplots(2, 2, figsize=(10, 10))
        idx = 0
        # idx = random.randint(0, 9)

        image = batch["image"]
        gt = batch["gt2D"]
        bboxes = batch["bboxes"]
        names_temp = batch["image_name"]

        axs[0, 0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
        show_mask(gt[idx].cpu().squeeze().numpy(), axs[0, 0])
        show_box(bboxes[idx].numpy().squeeze(), axs[0, 0])
        axs[0, 0].axis("off")
        # set title
        axs[0, 0].set_title(names_temp[idx])

        idx = 1
        # idx = random.randint(10, 19)
        axs[0, 1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
        show_mask(gt[idx].cpu().squeeze().numpy(), axs[0, 1])
        show_box(bboxes[idx].numpy().squeeze(), axs[0, 1])
        axs[0, 1].axis("off")
        # set title
        axs[0, 1].set_title(names_temp[idx])

        idx = 2
        # idx = random.randint(20, 29)
        axs[1, 0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
        show_mask(gt[idx].cpu().squeeze().numpy(), axs[1, 0])
        show_box(bboxes[idx].numpy().squeeze(), axs[1, 0])
        axs[1, 0].axis("off")
        # set title
        axs[1, 0].set_title(names_temp[idx])

        idx = 3
        # idx = random.randint(30, 39)
        axs[1, 1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
        show_mask(gt[idx].cpu().squeeze().numpy(), axs[1, 1])
        show_box(bboxes[idx].numpy().squeeze(), axs[1, 1])
        axs[1, 1].axis("off")
        # set title
        axs[1, 1].set_title(names_temp[idx])

        plt.subplots_adjust(wspace=0.01, hspace=0)
        plt.savefig(
            join(img_save_dir, names_temp[0].replace(".npy", ".png")),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
