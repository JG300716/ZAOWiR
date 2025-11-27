import json
from pathlib import Path
import time
import cv2
import numpy as np
import sys
import argparse

DEBUG = True


def print_dbg(msg: str):
    if DEBUG:
        print(msg)


def load_image_unicode(p: Path):
    try:
        with p.open("rb") as f:
            data = f.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[ERROR] load_image_unicode {p}: {e}")
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


def collect_pairs(left_folder: Path, right_folder: Path):
    left_files = sorted(
        [p for p in left_folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")])
    right_files = sorted(
        [p for p in right_folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")])
    # match by filename first
    left_map = {p.name: p for p in left_files}
    pairs = [(left_map[r.name], r) for r in right_files if r.name in left_map]
    if not pairs:
        # fallback: pair by index up to min length
        mn = min(len(left_files), len(right_files))
        pairs = list(zip(left_files[:mn], right_files[:mn]))

    return pairs


# ---- map of interpolation names to cv2 flags ----
INTERPOLATIONS = {
    "INTER_NEAREST": cv2.INTER_NEAREST,
    "INTER_LINEAR": cv2.INTER_LINEAR,
    "INTER_CUBIC": cv2.INTER_CUBIC,
    "INTER_AREA": cv2.INTER_AREA,
    "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
}


# ---- main benchmarking routine ----
def benchmark_remap(left_folder: Path, right_folder: Path,
                    left_json: Path, right_json: Path, stereo_json: Path,
                    param_list, repeats=3, show=False, alpha=0.0):
    # load intrinsics and stereo
    mtxL, distL = load_intrinsics(left_json)
    mtxR, distR = load_intrinsics(right_json)
    R, T = load_stereo(stereo_json)
    print_dbg("[INFO] Loaded intrinsics and stereo parameters.")

    pairs = collect_pairs(left_folder, right_folder)
    if not pairs:
        print("[ERROR] No pairs found. Check folders.")
        return

    # read sample image to get size
    sample = load_image_unicode(pairs[0][0])
    if sample is None:
        print("[ERROR] Failed to load sample image.")
        return
    h, w = sample.shape[:2]
    print_dbg(f"[INFO] Image size: {w}x{h}")

    # compute stereoRectify and maps once
    flags = cv2.CALIB_ZERO_DISPARITY
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(mtxL, distL, mtxR, distR, (w, h), R, T, flags=flags, alpha=alpha)
    map1x, map1y = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, (w, h), cv2.CV_32FC1)
    print_dbg("[INFO] initUndistortRectifyMap done.")

    # restrict pairs to reasonable count to keep benchmark short if many images
    max_pairs = min(len(pairs), 50)  # limit to 50 pairs for speed; change if you want
    pairs = pairs[:max_pairs]
    n_images = len(pairs)
    print(f"[INFO] Benchmarking on {n_images} image pairs, repeats={repeats}, methods={param_list}")

    results = []

    # warmup (run a quick remap to fill caches)
    left_img = load_image_unicode(pairs[0][0])
    right_img = load_image_unicode(pairs[0][1])
    if left_img is None or right_img is None:
        print("[ERROR] Cannot load warmup images.")
        return
    _ = cv2.remap(left_img, map1x, map1y, interpolation=cv2.INTER_LINEAR)
    _ = cv2.remap(right_img, map2x, map2y, interpolation=cv2.INTER_LINEAR)

    for name in param_list:
        if name not in INTERPOLATIONS:
            print(f"[WARN] Unknown interpolation: {name}, skipping.")
            continue
        flag = INTERPOLATIONS[name]
        total_time = 0.0
        total_calls = 0

        # measure repeats * n_images remaps for average
        for rep in range(repeats):
            t0 = time.perf_counter()
            for (pL, pR) in pairs:
                imgL = load_image_unicode(pL)
                imgR = load_image_unicode(pR)
                if imgL is None or imgR is None:
                    continue
                # do remap both images
                _ = cv2.remap(imgL, map1x, map1y, interpolation=flag)
                _ = cv2.remap(imgR, map2x, map2y, interpolation=flag)
                total_calls += 2
            t1 = time.perf_counter()
            total_time += (t1 - t0)
        avg_time_per_call = (total_time / total_calls) * 1000.0 if total_calls else float('inf')  # ms
        avg_time_per_pair = (total_time / (repeats * n_images)) * 1000.0 if n_images else float(
            'inf')  # ms per pair (both imgs)
        results.append((name, total_time, total_calls, avg_time_per_call, avg_time_per_pair))

        # optional show one example for visual judgement
        if show:
            rectL = cv2.remap(left_img, map1x, map1y, interpolation=flag)
            rectR = cv2.remap(right_img, map2x, map2y, interpolation=flag)
            vis = np.hstack((rectL, rectR))
            cv2.putText(vis, f"Interp: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow(f"Rectified - {name}", vis)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow(f"Rectified - {name}")
            if key == ord('q'):
                show = False  # stop showing further

    # print nicely
    print("\n=== Benchmark results ===")
    print(f"Images tested: {n_images}, repeats: {repeats}")
    print(f"{'Method':20s} {'Total(s)':>10s} {'Calls':>8s} {'Avg ms/call':>15s} {'Avg ms/pair':>15s}")
    for name, total_time, calls, avg_call, avg_pair in results:
        print(f"{name:20s} {total_time:10.4f} {calls:8d} {avg_call:15.4f} {avg_pair:15.4f}")

    print("\n[INFO] Done. No images were saved (as requested).")


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
                     left_json: Path, right_json: Path, stereo_json: Path, time: bool = False,
                     save: bool = False, alpha: float = 0.0, remap_interpolation=cv2.INTER_LINEAR):
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
    if not time:
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

        if not time:
            cv2.imshow(win_name, side_by_side)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

        if save:
            outL = out_left / pL.name
            outR = out_right / pR.name
            okL = imwrite_unicode(outL, rectL)
            okR = imwrite_unicode(outR, rectR)
            # 🔹 Save also the side-by-side rectified image with horizontal lines
            combined_name = f"{pL.stem}_rectified_with_lines.png"
            out_combined = out_left / combined_name
            okCombined = imwrite_unicode(out_combined, side_by_side)
        
            print_dbg(
                f"[SAVE] {pL.name} -> {outL} : {'OK' if okL else 'FAIL'}; "
                f"{pR.name} -> {outR} : {'OK' if okR else 'FAIL'}; "
                f"Combined -> {out_combined} : {'OK' if okCombined else 'FAIL'}")

    cv2.destroyAllWindows()


def parse_args(argv):
    p = argparse.ArgumentParser(description="Benchmark cv2.remap() interpolation methods")
    p.add_argument("--left_folder", required=True, help="Left images folder")
    p.add_argument("--right_folder", required=True, help="Right images folder")
    p = argparse.ArgumentParser(
        description="Stereo rectification viewer using stereoRectify/initUndistortRectifyMap/remap")
    p.add_argument("--left_folder", required=True, help="Folder with left camera images")
    p.add_argument("--right_folder", required=True, help="Folder with right camera images")
    p.add_argument("--left_json", required=True, help="Left camera intrinsics JSON")
    p.add_argument("--right_json", required=True, help="Right camera intrinsics JSON")
    p.add_argument("--stereo_json", required=True, help="Stereo calibration JSON (contains R and T)")
    p.add_argument("--param", default="all",
                   help='Which interpolation(s) to test: "all" or comma list e.g. "INTER_NEAREST,INTER_CUBIC"')
    p.add_argument("--repeats", type=int, default=3, help="Number of repeats for timing")
    p.add_argument("--show", action="store_true", help="Show one example image per interpolation for visual check")
    p.add_argument("--alpha", type=float, default=0.0, help="alpha passed to stereoRectify")
    p.add_argument("--save", action="store_true",
                   help="Save rectified images to ../rectified_left and ../rectified_right")
    p.add_argument("--time", action="store_true", help="Measure and print processing time")
    p.add_argument("--benchmark", action="store_true", help="Run benchmark of remap performance")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    left_folder = Path(args.left_folder)
    right_folder = Path(args.right_folder)
    left_json = Path(args.left_json)
    right_json = Path(args.right_json)
    stereo_json = Path(args.stereo_json)

    if args.param.strip().lower() == "all":
        param_list = list(INTERPOLATIONS.keys())
    if args.benchmark:
        benchmark_remap(left_folder, right_folder, left_json, right_json, stereo_json,
                        param_list, repeats=args.repeats, show=args.show, alpha=args.alpha)
        return
    if args.time:
        interpolations = [cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4, cv2.INTER_CUBIC, cv2.INTER_AREA]
        for interp in interpolations:
            print(f"[TIME] Measuring time for interpolation method: {interp}")
            import time
            start_time = time.time()
            rectify_and_show(
                Path(args.left_folder), Path(args.right_folder),
                Path(args.left_json), Path(args.right_json), Path(args.stereo_json), args.time,
                save=False, alpha=args.alpha, remap_interpolation=interp
            )
            end_time = time.time()
            print(f"[TIME] Processing time with interpolation {interp}: {end_time - start_time:.2f} seconds")
    else:
        rectify_and_show(
            Path(args.left_folder), Path(args.right_folder),
            Path(args.left_json), Path(args.right_json), Path(args.stereo_json), args.time,
            save=args.save, alpha=args.alpha
        )


if __name__ == "__main__":
    main()
