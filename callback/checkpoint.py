import torch
from os.path import basename, exists, isdir, isfile, join


def model_checkpoint(
    medsam_lite_model, work_dir, optimizer, best_loss, epoch, valid_epoch_loss
):
    model_weights = medsam_lite_model.state_dict()
    checkpoint = {
        "model": model_weights,
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "loss": valid_epoch_loss,
        "best_loss": best_loss,
    }

    torch.save(checkpoint, join(work_dir, "efficientvit_sam_latest.pth"))
    if valid_epoch_loss < best_loss:
        print(f"New best loss: {best_loss:.4f} -> {valid_epoch_loss:.4f}")
        best_loss = valid_epoch_loss
        checkpoint["best_loss"] = best_loss
        torch.save(checkpoint, join(work_dir, "efficientvit_sam_best.pth"))
    return best_loss
