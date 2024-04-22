import json
import os
import os.path as osp
import pickle

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_util
from tqdm.auto import tqdm

_DATA_DIR = "/data1/inqlee0704/medsam/train/75_nuclei-segmentation"

DATASETS = {
    "nuclei_stage1_train": {
        "IM_DIR": _DATA_DIR + "/Nuclei/stage_1_train",
        "ANN_FN": _DATA_DIR + "/Nuclei/annotations/stage1_train.json",
    },
    "nuclei_stage_1_local_train_split": {
        "IM_DIR": _DATA_DIR + "/Nuclei/stage_1_train",
        "ANN_FN": _DATA_DIR + "/Nuclei/annotations/stage_1_local_train_split.json",
    },
    "nuclei_stage_1_local_val_split": {
        "IM_DIR": _DATA_DIR + "/Nuclei/stage_1_train",
        "ANN_FN": _DATA_DIR + "/Nuclei/annotations/stage_1_local_val_split.json",
    },
    "nuclei_stage_1_test": {
        "IM_DIR": _DATA_DIR + "/Nuclei/stage_1_test",
        "ANN_FN": _DATA_DIR + "/Nuclei/annotations/stage_1_test.json",
    },
    "nuclei_stage_2_test": {
        "IM_DIR": _DATA_DIR + "/Nuclei/stage_2_test",
        "ANN_FN": _DATA_DIR + "/Nuclei/annotations/stage_2_test.json",
    },
    "cluster_nuclei": {
        "IM_DIR": _DATA_DIR + "/Nuclei/cluster_nuclei",
        "ANN_FN": _DATA_DIR + "/Nuclei/annotations/cluster_nuclei.json",
    },
    "BBBC007": {
        "IM_DIR": _DATA_DIR + "/Nuclei/BBBC007",
        "ANN_FN": _DATA_DIR + "/Nuclei/annotations/BBBC007.json",
    },
    "BBBC006": {
        "IM_DIR": _DATA_DIR + "/Nuclei/BBBC006",
        "ANN_FN": _DATA_DIR + "/Nuclei/annotations/BBBC006.json",
    },
    "BBBC018": {
        "IM_DIR": _DATA_DIR + "/Nuclei/BBBC018",
        "ANN_FN": _DATA_DIR + "/Nuclei/annotations/BBBC018.json",
    },
    "BBBC020": {
        "IM_DIR": _DATA_DIR + "/Nuclei/BBBC020",
        "ANN_FN": _DATA_DIR + "/Nuclei/annotations/BBBC020.json",
    },
    "nucleisegmentationbenchmark": {
        "IM_DIR": _DATA_DIR + "/Nuclei/nucleisegmentationbenchmark",
        "ANN_FN": _DATA_DIR + "/Nuclei/annotations/nucleisegmentationbenchmark.json",
    },
    "2009_ISBI_2DNuclei": {
        "IM_DIR": _DATA_DIR + "/Nuclei/2009_ISBI_2DNuclei",
        "ANN_FN": _DATA_DIR + "/Nuclei/annotations/2009_ISBI_2DNuclei.json",
    },
    "nuclei_partial_annotations": {
        "IM_DIR": _DATA_DIR + "/Nuclei/nuclei_partial_annotations",
        "ANN_FN": _DATA_DIR + "/Nuclei/annotations/nuclei_partial_annotations.json",
    },
    "TNBC_NucleiSegmentation": {
        "IM_DIR": _DATA_DIR + "/Nuclei/TNBC_NucleiSegmentation",
        "ANN_FN": _DATA_DIR + "/Nuclei/annotations/TNBC_NucleiSegmentation.json",
    },
}


def get_masks(im_metadata, COCO):
    image_annotations = []
    for annotation in COCO["annotations"]:
        if annotation["image_id"] == im_metadata["id"]:
            image_annotations.append(annotation)

    segments = [annotation["segmentation"] for annotation in image_annotations]
    masks = mask_util.decode(segments)
    return masks


def resize_longest_side(image, target_length=256):
    """
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    long_side_length = target_length
    oldh, oldw = image.shape[0], image.shape[1]
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def pad_image(image):
    """
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    h, w = image.shape[0], image.shape[1]
    image_size = max(h, w)
    padh = image_size - h
    padw = image_size - w
    if len(image.shape) == 3:  ## Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else:  ## Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))

    return image_padded


def rle_encode_multivalue(mask):

    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.diff(pixels) != 0  # 값이 변경되는 지점 찾기
    indexes = np.where(changes)[0] + 1  # 변경 지점의 인덱스
    runs = [
        (indexes[i], indexes[i + 1] - indexes[i], pixels[indexes[i]])
        for i in range(len(indexes) - 1)
    ]

    return runs


def main():
    new_img_dir = osp.join(_DATA_DIR, "imgs")
    new_gt_dir = osp.join(_DATA_DIR, "gts")
    os.makedirs(new_img_dir, exist_ok=True)
    os.makedirs(new_gt_dir, exist_ok=True)

    for key, value in DATASETS.items():
        print(key)
        if key == "nuclei_stage_2_test":
            continue
        img_dir = value["IM_DIR"]
        anno_path = value["ANN_FN"]
        COCO = json.load(open(anno_path))
        pbar = tqdm(COCO["images"], total=len(COCO["images"]))
        for img_meta in pbar:
            new_img_path = osp.join(
                new_img_dir, img_meta["file_name"].replace(".jpg", ".png")
            )
            new_gt_path = osp.join(
                new_gt_dir, img_meta["file_name"].replace(".jpg", ".pkl")
            )

            img_path = osp.join(img_dir, img_meta["file_name"])
            img = np.array(Image.open(img_path))
            mask = get_masks(img_meta, COCO)
            new_mask = np.zeros(mask.shape[:2])
            for i in range(mask.shape[2]):
                new_mask[mask[:, :, i] > 0] = i + 1
            img = resize_longest_side(img)
            new_mask = resize_longest_side(new_mask)
            new_mask = cv2.resize(
                new_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            img = pad_image(img)
            img = (img - img.min()) / np.clip(
                img.max() - img.min(), a_min=1e-8, a_max=None
            )
            img = (img * 255).astype(np.uint8)
            cv2.imwrite(new_img_path, img)

            new_mask = pad_image(new_mask)
            new_mask = np.array(new_mask, dtype=np.uint16)
            encoded = rle_encode_multivalue(new_mask)
            if len(pickle.dumps(encoded)) < 256 * 256 * 2:
                with open(new_gt_path, "wb") as f:
                    pickle.dump(encoded, f)
            else:
                np.save(new_gt_path.replace(".pkl", "npy"), new_mask)


if __name__ == "__main__":
    main()
