import cv2
import numpy as np
import json
from pathlib import Path
import argparse


def load_image_unicode(path: Path):
    try:
        with path.open("rb") as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[BŁĄD] Nie udało się wczytać obrazu {path}: {e}")
        return None


def load_intrinsics(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mtx = np.array(data["camera_matrix"])
    dist = np.array(data["distortion_coefficients"])
    return mtx, dist


def find_corners_in_folder(folder_path: Path, pattern_size, square_size):
    image_files = sorted(
        [f for f in folder_path.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")]
    )
    objpoints, imgpoints, gray_shape = [], [], None

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    for img_path in image_files:
        img = load_image_unicode(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            print(f"[OK] {img_path.name}")
        else:
            print(f"[--] Brak wzorca na {img_path.name}")

    return objpoints, imgpoints, gray_shape


def stereo_calibrate(
        left_folder: Path,
        right_folder: Path,
        left_json: Path,
        right_json: Path,
        pattern_size,
        square_size,
        save_json: bool = False
):

    mtx_left, dist_left = load_intrinsics(left_json)
    mtx_right, dist_right = load_intrinsics(right_json)

    objpoints_L, imgpoints_L, gray_shape_L = find_corners_in_folder(left_folder, pattern_size, square_size)
    objpoints_R, imgpoints_R, gray_shape_R = find_corners_in_folder(right_folder, pattern_size, square_size)

    pairs = min(len(objpoints_L), len(objpoints_R))
    objpoints = objpoints_L[:pairs]
    imgpoints_left = imgpoints_L[:pairs]
    imgpoints_right = imgpoints_R[:pairs]
    print(f"[INFO] Zestawiono {pairs} par obrazów.")

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_left, dist_left, mtx_right, dist_right,
        gray_shape_L, criteria=criteria, flags=flags
    )

    print(f"\n[WYNIK] RMS error: {ret:.6f}")
    print("Macierz rotacji (R):\n", R)
    print("Wektor translacji (T):\n", T)

    # === Zadanie 1.2: Współczynniki dystorsji ===
    print("\n[INFO] Współczynniki dystorsji:")
    print("Lewa kamera [k1, k2, p1, p2, k3]:", dist_left.ravel())
    print("Prawa kamera [k1, k2, p1, p2, k3]:", dist_right.ravel())
    print("\nOpis współczynników:")
    print("  k1, k2, k3 – współczynniki dystorsji radialnej (zniekształcenie typu beczka/poduszka).")
    print("  p1, p2 – współczynniki dystorsji tangencjalnej (powstałe przy przekrzywieniu soczewki).")

    if save_json:
        output_path = left_folder.parent / "stereo_calibration.json"
        data = {
            "rms_error": float(ret),
            "rotation_matrix": R.tolist(),
            "translation_vector": T.tolist(),
            "essential_matrix": E.tolist(),
            "fundamental_matrix": F.tolist(),
            "distortion_left": dist_left.ravel().tolist(),
            "distortion_right": dist_right.ravel().tolist(),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"[ZAPIS] Zapisano wyniki stereo do: {output_path}")


def compute_baseline(T, obj_unit="mm"):
    T = np.asarray(T).reshape(3,)
    baseline_orig = np.linalg.norm(T)
    if obj_unit == "mm":
        baseline_m = baseline_orig / 1000.0

    print(f"Baseline (orig unit = {obj_unit}): {baseline_orig:.6f} {obj_unit}")
    print(f"Baseline (meters): {baseline_m:.6f} m")
    # także składowa X (często dominująca)
    print(f"T vector: {T}")
    print(f"|T_x| = {abs(T[0]):.6f} {obj_unit}")

    return baseline_orig, baseline_m


def parse_args():
    parser = argparse.ArgumentParser(description="Stereo kalibracja kamer z obrazów szachownicy.")
    parser.add_argument("-l", "--left", required=True, help="Folder ze zdjęciami z lewej kamery.")
    parser.add_argument("-r", "--right", required=True, help="Folder ze zdjęciami z prawej kamery.")
    parser.add_argument("-w", "--width", type=int, required=True, help="Liczba pól wewnętrznych w poziomie.")
    parser.add_argument("-H", "--height", type=int, required=True, help="Liczba pól wewnętrznych w pionie.")
    parser.add_argument("-s", "--size", type=float, required=True, help="Rozmiar jednego pola [mm].")
    parser.add_argument("-j", "--json", action="store_true", help="Zapisz wynik kalibracji do stereo_calibration.json.")
    parser.add_argument("--left_json", required=True, help="Plik JSON z kalibracją lewej kamery.")
    parser.add_argument("--right_json", required=True, help="Plik JSON z kalibracją prawej kamery.")
    parser.add_argument("--compute_baseline", required=False, help="Plik JSON z kalibracją stereo kamery.")
    return parser.parse_args()


def main():
    args = parse_args()

    # tryb obliczania baseline
    if args.compute_baseline:
        with open(args.compute_baseline, "r", encoding="utf-8") as f:
            data = json.load(f)
        T = np.array(data["translation_vector"])
        compute_baseline(T, obj_unit="mm")
        return

    # tryb stereo kalibracji
    stereo_calibrate(
        Path(args.left),
        Path(args.right),
        Path(args.left_json),
        Path(args.right_json),
        (args.width, args.height),
        args.size,
        args.json
    )


if __name__ == "__main__":
    main()
