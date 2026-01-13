import pathlib

import cv2
import numpy as np
from stereo_utils import (
    read_pfm,
    read_calibration,
    disparity_to_depth,
    normalize_depth,
    depth_to_rgb24,
    rgb24_to_depth
)


def task1_reference_depth(disp_file, calib_file, out_name):
    disp = read_pfm(disp_file)
    f, b, doffs = read_calibration(calib_file)

    depth_mm = disparity_to_depth(disp, f, b, doffs)
    depth_m = depth_mm / 1000.0
    depth_8bit = normalize_depth(depth_m)

    cv2.imwrite(out_name, depth_8bit)
    print(f"✔ Zadanie 1 zapisane: {out_name}")


def task2_stereo_depth(imgL_path, imgR_path, calib_file, out_name):
    imgL = cv2.imread(imgL_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(imgR_path, cv2.IMREAD_GRAYSCALE)

    if imgL is None or imgR is None:
        raise IOError("Nie można wczytać obrazów stereo")

    stereo = cv2.StereoSGBM.create(
        minDisparity=0,
        numDisparities=288,
        blockSize=7,
        P1=8 * 7 * 7,
        P2=32 * 7 * 7,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    f, b, doffs = read_calibration(calib_file)
    depth_mm = disparity_to_depth(disparity, f, b, doffs)
    depth_m = depth_mm / 1000.0
    depth_8bit = normalize_depth(depth_m)

    cv2.imwrite(out_name, depth_8bit)
    print(f"✔ Zadanie 2 zapisane: {out_name}")


def task3_depth_rgb24(disparity, calib_file, out_name, max_depth=5):
    f, b, doffs = read_calibration(calib_file)

    depth_mm = disparity_to_depth(disparity, f, b, doffs)
    depth_m = depth_mm / 1000.0
    depth_rgb = depth_to_rgb24(depth_m, max_depth=max_depth)

    valid = depth_m > 0
    print(depth_m[valid].min(), depth_m[valid].max())

    cv2.imwrite(out_name, depth_rgb)
    print(f"✔ Zadanie 3 zapisane: {out_name}")


def task4_depth_to_disparity(depth_rgb24_file, out_name, max_depth=1000, baseline=0.1, hfov_deg=60):
    depth_rgb = cv2.imread(str(depth_rgb24_file))
    if depth_rgb is None:
        raise ValueError(f"Nie można wczytać obrazu: {depth_rgb24_file}")

    depth_m = rgb24_to_depth(depth_rgb, max_depth=max_depth)

    height, width = depth_m.shape

    hfov_rad = np.deg2rad(hfov_deg)
    focal_length_px = (width / 2.0) / np.tan(hfov_rad / 2.0)

    disparity = np.zeros_like(depth_m, dtype=np.float32)
    valid = depth_m > 0
    disparity[valid] = (focal_length_px * baseline) / depth_m[valid]

    disp_min = disparity[valid].min()
    disp_max = disparity[valid].max()

    disparity_normalized = np.zeros_like(disparity, dtype=np.float32)
    disparity_normalized[valid] = (disparity[valid] - disp_min) / (disp_max - disp_min) * 255

    disparity_8bit = disparity_normalized.astype(np.uint8)

    print(f"Focal length: {focal_length_px:.2f} px")
    print(f"Image size: {width}x{height}")
    print(f"Depth range: {depth_m[depth_m > 0].min():.2f} - {depth_m.max():.2f} m")
    print(f"Disparity range: {disparity_normalized.min():.2f} - {disparity_normalized.max():.2f} px")

    # Zapisz
    cv2.imwrite(str(out_name), disparity_8bit)

    print(f"✔ Zadanie 4 zapisane: {out_name}")

    return disparity_8bit


if __name__ == "__main__":
    calib_file = pathlib.Path("res/Lab4/Z1/calib.txt")
    disp_map_file = pathlib.Path("res/Lab4/Z1/disp0.pfm")
    imgL_file = pathlib.Path("res/Lab4/Z1/im0.png")
    imgR_file = pathlib.Path("res/Lab4/Z1/im1.png")

    out_path = pathlib.Path("res/Lab4/output")
    out_path.mkdir(exist_ok=True)
    pathlib.Path.cwd().joinpath(out_path)

    """
    task1_reference_depth(
        disp_file=disp_map_file,
        calib_file=calib_file,
        out_name=out_path / "task1_depth.png"
    )
    """
    
    """
    task2_stereo_depth(
        imgL_path=imgL_file,
        imgR_path=imgR_file,
        calib_file=calib_file,
        out_name=out_path / "task2_depth.png"
    )
    """

    disp = read_pfm(disp_map_file)
    task3_depth_rgb24(
        disparity=disp,
        calib_file=calib_file,
        out_name=out_path / "task3_depth_rgb24.png",
        max_depth=1000
    )

    task4_depth_to_disparity(
        depth_rgb24_file=pathlib.Path("res/Lab4/Z4/depth.png"),
        out_name=out_path / "task4_disparity.png",
        max_depth=1000,
        baseline=0.1,
        hfov_deg=60
    )
