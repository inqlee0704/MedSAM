import argparse
import os
import random
from copy import deepcopy
from datetime import datetime
from glob import glob
from os import listdir, makedirs
from os.path import basename, exists, isdir, isfile, join
from time import time
from evaluation.iou import cal_iou
from callback.checkpoint import model_checkpoint
from loss.default_loss import DefaultLoss

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
from models.medsam_lite import MedSAM_Lite

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


def main(loss_fn, image_encoder_cfg, prompt_encoder_cfg, mask_decoder_cfg):

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

    checkpoint = "workdir/temp.pth"
    if checkpoint and isfile(checkpoint):
        print(f"Resuming from checkpoint {checkpoint}")
        checkpoint = torch.load(checkpoint)
        medsam_lite_model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["loss"]
        print(f"Loaded checkpoint from epoch {start_epoch}")

    # * Optimizer
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

    # * Dataloader
    train_dataset = EncodedDataset(data_root, data_aug=True, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = EncodedDataset(data_root, data_aug=False, mode="valid")
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

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
            loss = loss_fn(gt2D, logits_pred, iou_pred)
            train_epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            pbar.set_description(
                f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}"
            )
            # break
        epoch_end_time = time()
        train_epoch_loss = train_epoch_loss / len(train_loader)

        train_loss_list.append(train_epoch_loss)
        lr_scheduler.step(train_epoch_loss)

        # valid
        valid_partial_iou = {f"{m}/iou": 0 for m in valid_dataset.modality_list}
        valid_partial_count = {m: 0 for m in valid_dataset.modality_list}
        with torch.no_grad():

            pbar = tqdm(valid_loader)
            # valid_dataset.total_data_indices
            for step, batch in enumerate(pbar):
                image = batch["image"]
                gt2D = batch["gt2D"]
                boxes = batch["bboxes"]
                filenames = batch["image_name"]
                image, gt2D, boxes = image.to(device), gt2D.to(device), boxes.to(device)
                logits_pred, iou_pred = medsam_lite_model(image, boxes)
                ious = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
                for fn, iou in zip(filenames, ious):
                    m = fn.split("_")[0]
                    valid_partial_iou[f"{m}/iou"] += iou
                    valid_partial_count[m] += 1
                # cal_iou()
                loss = loss_fn(gt2D, logits_pred, iou_pred)
                valid_epoch_loss += loss.item()
                pbar.set_description(
                    f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}"
                )
        valid_epoch_loss = valid_epoch_loss / len(valid_loader)
        for m, c in valid_partial_count.items():
            valid_partial_iou[f"{m}/iou"] /= c
        metrics = {
            "train/loss": train_epoch_loss,
            "valid/loss": valid_epoch_loss,
        }
        metrics.update(valid_partial_iou)

        wandb.log(metrics)

        best_loss = model_checkpoint(
            medsam_lite_model, work_dir, optimizer, best_loss, epoch, valid_epoch_loss
        )
        # model_checkpoint(checkpoint, work_dir, best_loss, valid_epoch_loss)


if __name__ == "__main__":
    work_dir = "./workdir"
    data_root = "D:\\Datas\\competition\\cvpr2024\\train"
    medsam_lite_checkpoint = "lite_medsam.pth"
    num_epochs = 1000
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
