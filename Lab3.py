import cv2
import numpy as np
import sys
import argparse
from pathlib import Path


def CustomDisparity(left_img, right_img, num_disparities, block_size):

    disparity_map = np.zeros(left_img.shape, dtype=np.float32)
    for y in range(left_img.shape[0] - block_size):
        for x in range(left_img.shape[1] - block_size):
            left_block = left_img[y:y + block_size, x:x + block_size]

            best_offset = 0
            min_sad = float('inf')

            for offset in range(num_disparities):
                if x - offset < 0:
                    continue

                right_block = right_img[y:y + block_size, x - offset:x - offset + block_size]
                sad = np.sum(np.abs(left_block.astype(np.int32) - right_block.astype(np.int32)))

                if sad < min_sad:
                    min_sad = sad
                    best_offset = offset

            disparity_map[y, x] = best_offset
    return disparity_map


def main():
    left_img = cv2.imread(args.left_image, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(args.right_image, cv2.IMREAD_GRAYSCALE)

    if left_img is None:
        print(f"[ERROR] Could not load left image: {args.left_image}")
        return
    if right_img is None:
        print(f"[ERROR] Could not load right image: {args.right_image}")
        return

    num_disparities = args.num_disparities + (16 - args.num_disparities % 16)
    block_size = args.block_size + (args.block_size % 2 == 0)

    if args.method == "BM":
        stereo = cv2.StereoBM.create(numDisparities=num_disparities, blockSize=block_size)
        disparity_map = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
    elif args.method == "SGBM":
        stereo = cv2.StereoSGBM.create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=16 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2
        )
        disparity_map = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
    elif args.method == "CUSTOM":
        disparity_map = CustomDisparity(left_img, right_img, num_disparities, block_size)
    else:
        print(f"[ERROR] Unknown method: {args.method}")
        return

    disparity_map = cv2.medianBlur(disparity_map, 5)
    disp_vis = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)

    cv2.imshow("Disparity Map", disp_vis)
    cv2.waitKey(0)
    if args.save:
        save_path = Path(args.method + "_disparity.png")
        cv2.imwrite(save_path.as_posix(), disp_vis)
        print(f"[INFO] Disparity map saved to {save_path.as_posix()}")


def parse_args(argv):
    p = argparse.ArgumentParser(description="DisparityMap")
    p.add_argument("--method", choices=["BM", "SGBM", "CUSTOM"], default="BM", help="Method for disparity computation")
    p.add_argument("--block_size", type=int, default=5, help="Block size for matching")
    p.add_argument("--num_disparities", type=int, default=16 * 4, help="Number of disparities")
    p.add_argument("--left_image", required=True, help="left camera image")
    p.add_argument("--right_image", required=True, help="right camera image")
    p.add_argument("--save", action="store_true", help="Save disparity map")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main()
