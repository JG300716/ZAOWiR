import cv2
import numpy as np
import json
from pathlib import Path
import argparse
import sys

DEBUG = False


def print_dbg(msg):
    if DEBUG:
        print_dbg(f"[DEBUG] {msg}")


def load_image_unicode(path: Path):
    try:
        with path.open("rb") as f:
            data = f.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print_dbg(f"[ERROR] load_image_unicode {path}: {e}")
        return None


def imwrite_unicode(path: Path, img):
    ext = path.suffix
    ret, enc = cv2.imencode(ext, img)
    if not ret:
        return False
    try:
        with path.open("wb") as f:
            f.write(enc.tobytes())
        return True
    except Exception:
        return False


def load_intrinsics(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    mtx = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)
    return mtx, dist


def load_stereo(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    R = np.array(data["rotation_matrix"], dtype=np.float64)
    T = np.array(data["translation_vector"], dtype=np.float64).reshape(3, 1)
    return R, T


# -------- main pipeline ----------
def collect_image_pairs(left_folder: Path, right_folder: Path):
    # assume images correspond by filename or by sorted order; we try to match names present in both folders
    left_files = sorted(
        [p for p in left_folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")])
    right_files = sorted(
        [p for p in right_folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")])

    # try match by filename
    left_names = {p.name: p for p in left_files}
    pairs = []
    for r in right_files:
        ln = r.name
        if ln in left_names:
            pairs.append((left_names[ln], r))

    if not pairs:
        # fallback: pair by index up to min length
        mn = min(len(left_files), len(right_files))
        pairs = list(zip(left_files[:mn], right_files[:mn]))

    return pairs


def rectify_and_show(left_folder: Path, right_folder: Path,
                     left_json: Path, right_json: Path, stereo_json: Path,
                     save: bool = False, alpha: float = 0.0, remap_interpolation = cv2.INTER_LINEAR):
    print_dbg("[INFO] Loading intrinsics...")
    mtxL, distL = load_intrinsics(left_json)
    mtxR, distR = load_intrinsics(right_json)
    R, T = load_stereo(stereo_json)
    print_dbg("[INFO] Loaded intrinsics and stereo parameters.")

    pairs = collect_image_pairs(left_folder, right_folder)
    if not pairs:
        print("[ERROR] No image pairs found.")
        return

    # read a sample to get image size
    sample = load_image_unicode(pairs[0][0])
    if sample is None:
        print_dbg("[ERROR] Failed to load sample image.")
        return
    h, w = sample.shape[:2]
    print_dbg(f"[INFO] Image size: {w}x{h}")

    # stereoRectify (Bouguet-style) -> R1,R2,P1,P2,Q
    flags = cv2.CALIB_ZERO_DISPARITY  # keep principal points on same row
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        mtxL, distL, mtxR, distR, (w, h),
        R, T, flags=flags, alpha=alpha
    )
    print_dbg("[INFO] stereoRectify done.")
    # create maps
    map1x, map1y = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, (w, h), cv2.CV_32FC1)
    print_dbg("[INFO] initUndistortRectifyMap done.")

    # prepare output folders
    out_left = left_folder.parent / "rectified_left"
    out_right = right_folder.parent / "rectified_right"
    if save:
        out_left.mkdir(parents=True, exist_ok=True)
        out_right.mkdir(parents=True, exist_ok=True)

    # loop pairs
    win_name = "Rectified pair (L | R). Press any key -> next, q -> quit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    for idx, (pL, pR) in enumerate(pairs, start=1):
        imgL = load_image_unicode(pL)
        imgR = load_image_unicode(pR)
        if imgL is None or imgR is None:
            print_dbg(f"[WARN] failed load pair {pL.name}, {pR.name}")
            continue

        rectL = cv2.remap(imgL, map1x, map1y, interpolation=remap_interpolation)
        rectR = cv2.remap(imgR, map2x, map2y, interpolation=remap_interpolation)

        # optionally draw horizontal lines to inspect rectification
        visL = rectL.copy()
        visR = rectR.copy()
        # stack side by side
        side_by_side = np.hstack((visL, visR))

        # draw several guide horizontal lines
        n_lines = 10
        step = int(h / (n_lines + 1))
        color = (0, 255, 0)
        for k in range(1, n_lines + 1):
            y = k * step
            cv2.line(side_by_side, (0, y), (2 * w - 1, y), color, 1)

        cv2.imshow(win_name, side_by_side)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

        if save:
            outL = out_left / pL.name
            outR = out_right / pR.name
            okL = imwrite_unicode(outL, rectL)
            okR = imwrite_unicode(outR, rectR)
            print_dbg(
                f"[SAVE] {pL.name} -> {outL} : {'OK' if okL else 'FAIL'}; {pR.name} -> {outR} : {'OK' if okR else 'FAIL'}")

    cv2.destroyAllWindows()


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Stereo rectification viewer using stereoRectify/initUndistortRectifyMap/remap")
    p.add_argument("--left_folder", required=True, help="Folder with left camera images")
    p.add_argument("--right_folder", required=True, help="Folder with right camera images")
    p.add_argument("--left_json", required=True, help="Left camera intrinsics JSON")
    p.add_argument("--right_json", required=True, help="Right camera intrinsics JSON")
    p.add_argument("--stereo_json", required=True, help="Stereo calibration JSON (contains R and T)")
    p.add_argument("--save", action="store_true",
                   help="Save rectified images to ../rectified_left and ../rectified_right")
    p.add_argument("--alpha", type=float, default=0.0,
                   help="alpha parameter passed to stereoRectify (0..1; 0=crop, 1=keep all)")
    p.add_argument("--time", action="store_true", help="Measure and print processing time")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])

    if args.time:
        interpolations = [cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4, cv2.INTER_CUBIC, cv2.INTER_AREA]
        for interp in interpolations:
            print(f"[TIME] Measuring time for interpolation method: {interp}")
            import time
            start_time = time.time()
            rectify_and_show(
                Path(args.left_folder), Path(args.right_folder),
                Path(args.left_json), Path(args.right_json), Path(args.stereo_json),
                save=False, alpha=args.alpha, remap_interpolation=interp
            )
            end_time = time.time()
            print(f"[TIME] Processing time with interpolation {interp}: {end_time - start_time:.2f} seconds")
    else:
        rectify_and_show(
            Path(args.left_folder), Path(args.right_folder),
            Path(args.left_json), Path(args.right_json), Path(args.stereo_json),
            save=args.save, alpha=args.alpha
        )


if __name__ == "__main__":
    main()
