import numpy as np


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


def rle_decode_multivalue(runs, shape):

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length, value in runs:
        img[start - 1 : start + length - 1] = value
    return img.reshape(shape)
