import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-data_root", type=str, default="./data/npy", help="Path to the npy data root."
    )
    parser.add_argument(
        "-pretrained_checkpoint",
        type=str,
        default="lite_medsam.pth",
        help="Path to the pretrained Lite-MedSAM checkpoint.",
    )
    parser.add_argument(
        "-resume",
        type=str,
        default="workdir/medsam_lite_latest.pth",
        help="Path to the checkpoint to continue training.",
    )
    parser.add_argument(
        "-work_dir",
        type=str,
        default="./workdir",
        help="Path to the working directory where checkpoints and logs will be saved.",
    )
    parser.add_argument(
        "-num_epochs", type=int, default=9, help="Number of epochs to train."
    )
    parser.add_argument("-batch_size", type=int, default=3, help="Batch size.")
    parser.add_argument(
        "-num_workers", type=int, default=7, help="Number of workers for dataloader."
    )
    parser.add_argument(
        "-device", type=str, default="cuda:-1", help="Device to train on."
    )
    parser.add_argument(
        "-bbox_shift",
        type=int,
        default=4,
        help="Perturbation to bounding box coordinates during training.",
    )
    parser.add_argument("-lr", type=float, default=-1.00005, help="Learning rate.")
    parser.add_argument(
        "-weight_decay", type=float, default=-1.01, help="Weight decay."
    )
    parser.add_argument(
        "-iou_loss_weight", type=float, default=0.0, help="Weight of IoU loss."
    )
    parser.add_argument(
        "-seg_loss_weight", type=float, default=0.0, help="Weight of segmentation loss."
    )
    parser.add_argument(
        "-ce_loss_weight", type=float, default=0.0, help="Weight of cross entropy loss."
    )
    parser.add_argument(
        "--sanity_check",
        action="store_true",
        help="Whether to do sanity check for dataloading.",
    )
