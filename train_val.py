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
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

from dataset.npydataset import NpyDataset
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
from dataset.compressed import EncodedDataset

torch.cuda.empty_cache()
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


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


class MedSAM_Lite(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, boxes):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )  # (B, 1, 256, 256)

        return low_res_masks, iou_predictions

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing
        """
        # Crop
        masks = masks[:, :, : new_size[0], : new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks


def main():
    medsam_lite_image_encoder = TinyViT(
        img_size=256,
        in_chans=3,
        embed_dims=[
            64,  ## (64, 256, 256)
            128,  ## (128, 128, 128)
            160,  ## (160, 64, 64)
            320,  ## (320, 64, 64)
        ],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8,
    )

    medsam_lite_prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16,
    )

    medsam_lite_mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )

    medsam_lite_model = MedSAM_Lite(
        image_encoder=medsam_lite_image_encoder,
        mask_decoder=medsam_lite_mask_decoder,
        prompt_encoder=medsam_lite_prompt_encoder,
    )

    if medsam_lite_checkpoint is not None:
        if isfile(medsam_lite_checkpoint):
            print(f"Finetuning with pretrained weights {medsam_lite_checkpoint}")
            medsam_lite_ckpt = torch.load(medsam_lite_checkpoint, map_location="cpu")
            medsam_lite_model.load_state_dict(medsam_lite_ckpt, strict=True)
        else:
            print(
                f"Pretained weights {medsam_lite_checkpoint} not found, training from scratch"
            )

    medsam_lite_model = medsam_lite_model.to(device)
    medsam_lite_model.train()
    print(f"MedSAM Lite size: {sum(p.numel() for p in medsam_lite_model.parameters())}")
    optimizer = optim.AdamW(
        medsam_lite_model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=weight_decay,
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=5, cooldown=0
    )
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    iou_loss = nn.MSELoss(reduction="mean")
    train_dataset = EncodedDataset(data_root, data_aug=True, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = EncodedDataset(data_root, data_aug=False, mode="valid")
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    checkpoint = "workdir/temp.pth"
    if checkpoint and isfile(checkpoint):
        print(f"Resuming from checkpoint {checkpoint}")
        checkpoint = torch.load(checkpoint)
        medsam_lite_model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["loss"]
        print(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        start_epoch = 0
        best_loss = 1e10

    train_loss_list = []
    valid_loss_list = []

    for epoch in range(start_epoch + 1, num_epochs):
        train_dataset.reload()  # sample per modality

        epoch_start_time = time()
        train_epoch_loss = 0
        valid_epoch_loss = 0
        pbar = tqdm(train_loader)
        for step, batch in enumerate(pbar):
            image = batch["image"]
            gt2D = batch["gt2D"]
            boxes = batch["bboxes"]
            optimizer.zero_grad()
            image, gt2D, boxes = image.to(device), gt2D.to(device), boxes.to(device)
            logits_pred, iou_pred = medsam_lite_model(image, boxes)
            l_seg = seg_loss(logits_pred, gt2D)
            l_ce = ce_loss(logits_pred, gt2D.float())
            # mask_loss = l_seg + l_ce
            mask_loss = seg_loss_weight * l_seg + ce_loss_weight * l_ce
            iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
            l_iou = iou_loss(iou_pred, iou_gt)
            # loss = mask_loss + l_iou
            loss = mask_loss + iou_loss_weight * l_iou
            train_epoch_loss = loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(
                f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}"
            )
        epoch_end_time = time()
        train_epoch_loss = train_epoch_loss / len(train_loader)

        train_loss_list.append(train_epoch_loss)
        lr_scheduler.step(train_epoch_loss)

        # valid
        with torch.no_grad():
            pbar = tqdm(valid_loader)
            for step, batch in enumerate(pbar):
                image = batch["image"]
                gt2D = batch["gt2D"]
                boxes = batch["bboxes"]
                image, gt2D, boxes = image.to(device), gt2D.to(device), boxes.to(device)
                logits_pred, iou_pred = medsam_lite_model(image, boxes)
                l_seg = seg_loss(logits_pred, gt2D)
                l_ce = ce_loss(logits_pred, gt2D.float())
                mask_loss = seg_loss_weight * l_seg + ce_loss_weight * l_ce
                iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
                l_iou = iou_loss(iou_pred, iou_gt)
                # loss = mask_loss + l_iou
                loss = mask_loss + iou_loss_weight * l_iou
                valid_epoch_loss += loss.item()
                pbar.set_description(
                    f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}"
                )
        valid_epoch_loss = valid_epoch_loss / len(valid_loader)

        model_weights = medsam_lite_model.state_dict()
        checkpoint = {
            "model": model_weights,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "loss": valid_epoch_loss,
            "best_loss": best_loss,
        }

        torch.save(checkpoint, join(work_dir, "medsam_lite_latest.pth"))
        if valid_epoch_loss < best_loss:
            print(f"New best loss: {best_loss:.4f} -> {valid_epoch_loss:.4f}")
            best_loss = valid_epoch_loss
            checkpoint["best_loss"] = best_loss
            torch.save(checkpoint, join(work_dir, "medsam_lite_best.pth"))

        plt.plot(train_loss_list)
        plt.plot(valid_loss_list)
        plt.title("Dice + Binary Cross Entropy + IoU Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(work_dir, "train_loss.png"))
        plt.close()
        wandb.log(
            {
                "train/loss": train_epoch_loss,
                "valid/loss": valid_epoch_loss,
            }
        )


if __name__ == "__main__":
    work_dir = "./workdir"
    data_root = "D:\\Datas\\competition\\cvpr2024\\train"
    medsam_lite_checkpoint = "lite_medsam.pth"
    num_epochs = 10
    batch_size = 8
    num_workers = 8
    device = "cuda:0"
    bbox_shift = 5
    lr = 5e-5
    weight_decay = 0.01
    iou_loss_weight = 1.0
    seg_loss_weight = 1.0
    ce_loss_weight = 1.0
    do_sancheck = True
    checkpoint = "workdir/temp.pth"
    # checkpoint = "workdir/medsam_lite_latest.pth"
    debug = False
    makedirs(work_dir, exist_ok=True)
    wandb.init(project="medsam")
    main()
