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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

from callback.checkpoint import model_checkpoint
from dataset.encoded_dataset import EncodedDataset
from dataset.encoded_dataset import _modality_list as modality_list
from dataset.npy_dataset import NpyDataset
from evaluation.iou import cal_iou
from evaluation.SurfaceDice import (
    compute_dice_coefficient,
    compute_surface_dice_at_tolerance,
    compute_surface_distances,
)
from loss.default_loss import DefaultLoss
from loss.dice_loss import DiceLoss
from models.medsam_lite import MedSAM_Lite
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT

torch.cuda.empty_cache()
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


def main(loss_fn, image_encoder_cfg, prompt_encoder_cfg, mask_decoder_cfg):
    valid_batch_size = batch_size * 2
    start_epoch = 0
    best_loss = 1e10
    # * Network

    medsam_lite_image_encoder = TinyViT(**image_encoder_cfg)

    medsam_lite_prompt_encoder = PromptEncoder(**prompt_encoder_cfg)

    medsam_lite_mask_decoder = MaskDecoder(**mask_decoder_cfg)

    medsam_lite_model = MedSAM_Lite(
        image_encoder=medsam_lite_image_encoder,
        mask_decoder=medsam_lite_mask_decoder,
        prompt_encoder=medsam_lite_prompt_encoder,
    )

    medsam_lite_model = medsam_lite_model.to(device)
    medsam_lite_model.train()
    print(f"MedSAM Lite size: {sum(p.numel() for p in medsam_lite_model.parameters())}")
    # * Optimizer
    optimizer = optim.AdamW(
        medsam_lite_model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=weight_decay,
    )

    pretrained_checkpoint = "lite_medsam.pth"
    if pretrained_checkpoint and isfile(pretrained_checkpoint):
        print(f"Finetuning with pretrained weights {pretrained_checkpoint}")
        medsam_lite_ckpt = torch.load(pretrained_checkpoint, map_location="cpu")
        medsam_lite_model.load_state_dict(medsam_lite_ckpt, strict=True)

    checkpoint = "workdir/medsam_lite_best.pth"
    # checkpoint = "workdir/medsam_lite_latest.pth"
    if checkpoint and isfile(checkpoint):
        print(f"Resuming from checkpoint {checkpoint}")
        checkpoint = torch.load(checkpoint)
        medsam_lite_model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        # start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["loss"]
        # print(f"Loaded checkpoint from epoch {start_epoch}")

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=5, cooldown=0
    )

    top_k = 3
    train_dataset = EncodedDataset(
        data_root, data_aug=True, mode="train", sample=1000, modality=modality_list
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # * Dataloader
    total_modality_list = modality_list
    for epoch in tqdm(range(start_epoch + 1, num_epochs, 2)):
        valid_metrics, best_loss = valid_model(
            loss_fn, valid_batch_size, medsam_lite_model, optimizer, epoch, best_loss
        )
        exclude_modalites = extract_exclude_modalities(valid_metrics, top_k)

        train_dataset.modality_list = total_modality_list
        train_dataset.reload()  # sample per modality
        train_metrics = train_model(
            train_loader,
            loss_fn,
            medsam_lite_model,
            optimizer,
            lr_scheduler,
            epoch,
        )
        metrics = {}
        metrics.update(valid_metrics)
        metrics.update(train_metrics)
        wandb.log(metrics)

        partial_modality_list = deepcopy(total_modality_list)
        partial_modality_list = list(
            filter(lambda x: x not in exclude_modalites, partial_modality_list)
        )

        train_dataset.modality_list = partial_modality_list
        train_dataset.reload()  # sample per modality
        train_metrics = train_model(
            train_loader,
            loss_fn,
            medsam_lite_model,
            optimizer,
            lr_scheduler,
            epoch + 1,
        )
        metrics = {}
        metrics.update(valid_metrics)
        metrics.update(train_metrics)
        wandb.log(metrics)


def extract_exclude_modalities(valid_metrics, top_k):
    modalities = []
    losses = []
    for k, v in valid_metrics.items():
        if "dice loss" in k:
            m = k.split("/")[0]
            modalities.append(m)
            losses.append(np.mean(v))
    exclude_indices = np.argsort(losses)[:top_k]
    exclude_modalites = [modalities[i] for i in exclude_indices]
    return exclude_modalites


def valid_model(
    loss_fn, valid_batch_size, medsam_lite_model, optimizer, epoch, best_loss
):
    valid_partial = {}

    valid_partial_count = {m: 0 for m in modality_list}
    medsam_lite_model.eval()

    spacing = [1.0, 1.0, 1.0]
    valid_epoch_loss = 0
    valid_count = 0

    for m in modality_list:
        valid_dataset = EncodedDataset(
            data_root, data_aug=False, mode="valid", modality=m
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=valid_batch_size, shuffle=False
        )
        # valid
        valid_partial[f"{m}/iou"] = []
        valid_partial[f"{m}/dcs"] = []
        valid_partial[f"{m}/nsd"] = []
        valid_partial[f"{m}/loss"] = []
        valid_partial[f"{m}/dice loss"] = []

        with torch.no_grad():
            pbar = tqdm(valid_loader)
            for step, batch in enumerate(pbar):
                image = batch["image"]
                gt2Ds = batch["gt2D"]
                boxes = batch["bboxes"]
                filenames = batch["image_name"]
                image, gt2Ds, boxes = (
                    image.to(device),
                    gt2Ds.to(device),
                    boxes.to(device),
                )
                logits_preds, iou_preds = medsam_lite_model(image, boxes)
                pred_masks = torch.sigmoid(logits_preds) > 0.5
                gt_masks = gt2Ds.bool()
                # ious = cal_iou(pred_mask, gt_mask)
                tolerance_mm = 2
                for fn, pred_mask, gt_mask, logits_pred, iou_pred, gt2D in zip(
                    filenames, pred_masks, gt_masks, logits_preds, iou_preds, gt2Ds
                ):
                    # surface_distances = compute_surface_distances(
                    # gt_mask, pred_mask, spacing
                    # )
                    surface_distances = compute_surface_distances(
                        gt_mask.to("cpu").numpy(),
                        pred_mask.to("cpu").numpy(),
                        spacing,
                    )
                    nsd = compute_surface_dice_at_tolerance(
                        surface_distances, tolerance_mm
                    )
                    dcs = compute_dice_coefficient(gt_mask, pred_mask)

                    loss = loss_fn(gt2D[None], logits_pred[None], iou_pred[None])
                    iou = cal_iou(pred_mask, gt_mask).flatten()
                    dice_loss = DiceLoss()(logits_pred[None], gt2D[None])
                    valid_epoch_loss += loss
                    valid_partial[f"{m}/loss"].append(loss.item())
                    valid_partial[f"{m}/dice loss"].append(dice_loss.item())
                    valid_partial[f"{m}/iou"].append(iou.item())
                    valid_partial[f"{m}/nsd"].append(nsd.item())
                    valid_partial[f"{m}/dcs"].append(dcs.item())
                    valid_partial_count[m] += 1
                    valid_count += 1
                modality_loss = np.mean(valid_partial[f"{m}/loss"])
                pbar.set_description(
                    f"Epoch {epoch} - {m} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {modality_loss:.4f}"
                )
            indices = np.argsort(valid_partial[f"{m}/loss"])[-3:]
            imgs = []
            losses = []
            valid_dataset
            for i in indices:
                item = valid_dataset[i]
                image = item["image"][None]
                gt2Ds = item["gt2D"][None]
                boxes = item["bboxes"][None]
                image, gt2Ds, boxes = (
                    image.to(device),
                    gt2Ds.to(device),
                    boxes.to(device),
                )
                logits_preds, iou_preds = medsam_lite_model(image, boxes)
                pred_masks = torch.sigmoid(logits_preds) > 0.5
                gt_masks = gt2Ds.bool()
                image = (
                    (image[0].permute(1, 2, 0) * 255)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                pred_mask = ((pred_masks.squeeze()).detach().cpu().numpy()).astype(
                    np.uint8
                )
                gt2D = ((gt_masks.squeeze()).detach().cpu().numpy()).astype(np.uint8)
                pred_mask = cv2.cvtColor(
                    (pred_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB
                )
                pred_mask[:, :, 1] = 0
                pred_mask[:, :, 2] = 0
                gt2D = cv2.cvtColor((gt2D * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                gt2D[:, :, 0] = 0
                gt2D[:, :, 1] = 0
                mask_combined = cv2.addWeighted(pred_mask, 0.5, gt2D, 0.5, 0)
                img_combined = cv2.addWeighted(image, 0.6, mask_combined, 0.4, 0)
                ys, xs = np.where(pred_mask[:, :, 0])

                if len(xs) != 0 and len(ys) != 0:
                    cv2.rectangle(
                        img_combined,
                        (xs.min(), ys.min()),
                        (xs.max(), ys.max()),
                        (255, 0, 0),
                        1,
                    )

                ys, xs = np.where(gt2D[:, :, 2])
                if len(xs) != 0 and len(ys) != 0:
                    cv2.rectangle(
                        img_combined,
                        (xs.min(), ys.min()),
                        (xs.max(), ys.max()),
                        (0, 0, 255),
                        1,
                    )

                losses.append(valid_partial[f"{m}/dice loss"][i])
                imgs.append(img_combined)
            wandb.log(
                {
                    f"{m} valid": [
                        wandb.Image(img, caption=f"loss: {l}")
                        for img, l in zip(imgs, losses)
                    ]
                }
            )
    valid_epoch_loss = valid_epoch_loss / valid_count

    metrics = {
        "valid/loss": valid_epoch_loss,
    }
    for k, v in valid_partial.items():
        valid_partial[k] = np.mean(v)
    metrics.update(valid_partial)

    best_loss = model_checkpoint(
        medsam_lite_model, work_dir, optimizer, best_loss, epoch, valid_epoch_loss
    )
    return metrics, best_loss


def train_model(
    train_loader, loss_fn, medsam_lite_model, optimizer, lr_scheduler, epoch
):

    medsam_lite_model.train()
    train_epoch_loss = 0
    pbar = tqdm(train_loader)
    for step, batch in enumerate(pbar):
        image = batch["image"]
        gt2Ds = batch["gt2D"]
        boxes = batch["bboxes"]
        optimizer.zero_grad()
        image, gt2Ds, boxes = image.to(device), gt2Ds.to(device), boxes.to(device)
        logits_preds, iou_preds = medsam_lite_model(image, boxes)
        loss = loss_fn(gt2Ds, logits_preds, iou_preds)
        train_epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        pbar.set_description(
            f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}"
        )
    train_epoch_loss = train_epoch_loss / len(train_loader)

    lr_scheduler.step(train_epoch_loss)
    train_metrics = {
        "train/loss": train_epoch_loss,
    }
    return train_metrics


if __name__ == "__main__":
    work_dir = "./workdir"
    # data_root = "D:\\Datas\\competition\\cvpr2024\\train"
    data_root = [
        "D:\\Datas\\competition\\cvpr2024\\train",
        "D:\\Datas\\competition\\cvpr2024\\nuclei",
    ]
    # data_root = "/data1/inqlee0704/medsam/train/compressed"
    medsam_lite_checkpoint = "lite_medsam.pth"
    num_epochs = 100
    batch_size = 8
    num_workers = 4
    device = "cuda"
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
    image_encoder_cfg = dict(
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

    prompt_encoder_cfg = dict(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16,
    )
    mask_decoder_cfg = dict(
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
    loss_fn = DefaultLoss(seg_loss_weight, ce_loss_weight, iou_loss_weight)
    main(
        loss_fn,
        image_encoder_cfg=image_encoder_cfg,
        prompt_encoder_cfg=prompt_encoder_cfg,
        mask_decoder_cfg=mask_decoder_cfg,
    )
