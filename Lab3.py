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


def compare_disparity(path, ref_path):
    path = Path(path)
    files = list(path.glob("*_disparity.png"))
    if not files:
        print("[ERROR] No *_disparity.png files found")
        return

    ref = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED)
    if ref is None:
        print("[ERROR] Could not load reference disparity map")
        return

    ref = ref.astype(np.float32)

    valid_mask = ref > 0
    ref_real = ref / 4.0  # skala 0.25–63.75

    print(f"[INFO] Loaded reference disparity: {ref_path}")
    print(f"[INFO] Valid GT pixels: {np.sum(valid_mask)}")

    for file in files:
        print("\n=====================================")
        print(f"[INFO] Comparing file: {file.name}")

        disp = cv2.imread(file.as_posix(), cv2.IMREAD_GRAYSCALE)
        if disp is None:
            print("[ERROR] Could not load:", file)
            continue

        disp = disp.astype(np.float32)

        if disp.shape != ref.shape:
            disp = cv2.resize(disp, (ref.shape[1], ref.shape[0]))

        error = np.abs(disp - ref_real)
        error_masked = error[valid_mask]

        mae = np.mean(error_masked)
        rmse = np.sqrt(np.mean(error_masked ** 2))
        bad_px = np.mean(error_masked > 1.0) * 100.0  # ≥1 px błędu

        print(f"MAE  = {mae:.4f} px")
        print(f"RMSE = {rmse:.4f} px")
        print(f"Bad pixels (err>1.0) = {bad_px:.2f}%")

        # === Wizualizacja mapy błędów ===
        error_vis = np.zeros_like(error, dtype=np.uint8)
        error_vis[valid_mask] = np.clip(error[valid_mask] * 4, 0, 255)

        heatmap = cv2.applyColorMap(error_vis.astype(np.uint8), cv2.COLORMAP_JET)

        cv2.imshow(f"Error Map - {file.name}", heatmap)
        cv2.waitKey(200)

        save_path = file.with_name(file.stem + "_error.png")
        cv2.imwrite(save_path.as_posix(), heatmap)
        print(f"[INFO] Error map saved to {save_path.as_posix()}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

    # generowanie dysparycji
    p.add_argument("--method", choices=["BM", "SGBM", "CUSTOM"], default="BM",
                   help="Method for disparity computation")
    p.add_argument("--block_size", type=int, default=5, help="Block size for matching")
    p.add_argument("--num_disparities", type=int, default=16 * 4, help="Number of disparities")
    p.add_argument("--left_image", help="Left camera image")
    p.add_argument("--right_image", help="Right camera image")
    p.add_argument("--save", action="store_true", help="Save disparity map")

    # porównanie map dysparycji
    p.add_argument("--compare", action="store_true",
                   help="Activate disparity comparison mode")
    p.add_argument("--path", type=str,
                   help="Path to folder containing *_disparity files")
    p.add_argument("--ref_path", type=str,
                   help="Path to reference disparity map (GT)")

    args = p.parse_args(argv)

    # walidacja wymagań
    if args.compare:
        if args.path is None or args.ref_path is None:
            p.error("--compare requires --path and --ref_path")

    else:
        # generowanie wymaga obrazów
        if args.left_image is None or args.right_image is None:
            p.error("Generating disparity requires --left_image and --right_image")

    return args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    if args.compare:
        compare_disparity(args.path, args.ref_path)
    else:
        main()
