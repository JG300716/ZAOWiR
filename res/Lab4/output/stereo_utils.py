import numpy as np
import cv2
import re


def read_pfm(filename):
    with open(filename, 'rb') as f:
        header = f.readline().decode().rstrip()
        color = header == 'PF'

        dims = f.readline().decode()
        while dims.startswith('#'):
            dims = f.readline().decode()

        width, height = map(int, dims.split())
        scale = float(f.readline().decode().strip())
        endian = '<' if scale < 0 else '>'

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)

        return np.flipud(data)


def read_calibration(filename):
    with open(filename, 'r') as f:
        text = f.read()

    cam0 = re.search(r'cam0=\[([^\]]+)\]', text).group(1)
    fx = float(cam0.split()[0])

    baseline = float(re.search(r'baseline=([0-9.]+)', text).group(1))
    doffs = float(re.search(r'doffs=([0-9.]+)', text).group(1))

    return fx, baseline, doffs


def disparity_to_depth(disparity, f, b, doffs):
    disparity = disparity.astype(np.float32)

    depth = np.zeros_like(disparity)
    valid = disparity > 0

    depth[valid] = (b * f) / (disparity[valid] + doffs)
    depth[~valid] = 0

    return depth


def normalize_depth(depth):
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    return depth_norm.astype(np.uint8)


def depth_to_rgb24(depth, max_depth):
    depth = np.nan_to_num(depth, nan=0.0, posinf=max_depth)
    depth_clipped = np.clip(depth, 0, max_depth)

    normalized = depth_clipped / max_depth

    max_val = (256**3) - 1  # 16777215
    values = (normalized * max_val).astype(np.uint32)

    B = ((values >> 16) & 0xFF).astype(np.uint8)
    G = ((values >> 8) & 0xFF).astype(np.uint8)
    R = (values & 0xFF).astype(np.uint8)

    return np.dstack((B, G, R))


def rgb24_to_depth(bgr, max_depth):
    B = bgr[:, :, 0].astype(np.uint32)
    G = bgr[:, :, 1].astype(np.uint32)
    R = bgr[:, :, 2].astype(np.uint32)

    value = R + (G << 8) + (B << 16)
    normalized = value / ((256**3) - 1)

    return normalized * max_depth