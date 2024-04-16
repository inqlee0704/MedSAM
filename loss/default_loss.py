import torch.nn as nn
import monai
from evaluation.iou import cal_iou
import torch


class DefaultLoss:
    def __init__(self, seg_loss_weight, ce_loss_weight, iou_loss_weight):
        self.seg_loss_weight = seg_loss_weight
        self.ce_loss_weight = ce_loss_weight
        self.iou_loss_weight = iou_loss_weight
        self.seg_loss = monai.losses.DiceLoss(
            sigmoid=True, squared_pred=True, reduction="mean"
        )
        self.ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.iou_loss = nn.MSELoss(reduction="mean")

    def __call__(self, gt2D, logits_pred, iou_pred):
        l_seg = self.seg_loss(logits_pred, gt2D)
        l_ce = self.ce_loss(logits_pred, gt2D.float())
        # mask_loss = l_seg + l_ce
        mask_loss = self.seg_loss_weight * l_seg + self.ce_loss_weight * l_ce
        iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
        l_iou = self.iou_loss(iou_pred, iou_gt)
        # loss = mask_loss + l_iou
        loss = mask_loss + self.iou_loss_weight * l_iou
        return loss
