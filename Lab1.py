import cv2
import argparse
from pathlib import Path
import numpy as np
import json
import os
import math


def load_image_unicode(path: Path):
    try:
        with path.open("rb") as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[ERROR] Nie udało się wczytać pliku {path}: {e}")
        return None


def find_chessboard_in_folder(folder_path: Path, pattern_size, square_size):
    image_files = sorted(
        [f for f in folder_path.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")]
    )
    if not image_files:
        print("[INFO] Brak obrazów w folderze.")
        return [], [], None, []

    objpoints = []
    imgpoints = []
    detected_images = []
    used_files = []

    objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    first_gray_shape = None
    for idx, image_path in enumerate(image_files, start=1):
        image = load_image_unicode(image_path)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if first_gray_shape is None:
            first_gray_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if found:
            criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_subpix)
            objpoints.append(objp)
            detected_images.append(idx)
            used_files.append(image_path.name)

    print("=== PODSUMOWANIE ===")
    print("Ilość zdjęć z wykrytą tablicą kalibracyjną:", len(detected_images), "/", len(image_files))
    return objpoints, imgpoints, first_gray_shape, used_files


def describe_distortion(dist):
    """Opisuje współczynniki dystorsji"""
    coeffs = dist.ravel().tolist()
    names = ["k1", "k2", "p1", "p2", "k3"]
    print("\n=== Współczynniki dystorsji ===")
    for name, val in zip(names, coeffs):
        print(f"{name}: {val:.6f}")
    print("\nOpis:")
    print("k1, k2, k3 – współczynniki dystorsji radialnej (zniekształcenie 'beczkowe' lub 'poduszkowe')")
    print("p1, p2 – współczynniki dystorsji tangencjalnej (przesunięcie punktów spowodowane nachyleniem soczewek)")
    return dict(zip(names, coeffs))


def mean_error_full(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    total_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
        total_error += error ** 2
        total_points += len(objpoints[i])
    mean_err = np.sqrt(total_error / total_points)
    print(f"[Full] Średni błąd reprojekcji: {mean_err:.6f}")
    return mean_err


def save_calibration_to_json(folder_path: Path, mtx, dist, used_files, side="camera"):
    out_path = folder_path.parent / f"calibration_{side}.json"
    data = {
        "camera_side": side,
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.ravel().tolist(),
        "distortion_coefficients_named": describe_distortion(dist),
        "used_images": used_files
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[INFO] Wyniki zapisane do: {out_path}")


def load_calibration_from_json(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)
    mtx = np.array(data["camera_matrix"], dtype=np.float32)
    dist = np.array(data["distortion_coefficients"], dtype=np.float32)
    print(f"[INFO] Parametry wczytane z {json_path}")
    return mtx, dist


def compute_fov_from_calibration(mtx, image_shape):
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    width, height = image_shape
    fov_x = 2 * math.degrees(math.atan(width / (2 * fx)))
    fov_y = 2 * math.degrees(math.atan(height / (2 * fy)))
    print(f"[INFO] FOV poziomy: {fov_x:.2f}°, pionowy: {fov_y:.2f}°")
    return fov_x, fov_y


def undistort_single_images(left_path: Path, right_path: Path, mtx, dist):
    output_dir = Path("undistorted_single")
    output_dir.mkdir(exist_ok=True)

    for path, name in [(left_path, "left"), (right_path, "right")]:
        img = load_image_unicode(path)
        if img is None:
            print(f"[ERROR] Nie udało się wczytać {path}")
            continue
        undistorted = cv2.undistort(img, mtx, dist, None)
        out_path = output_dir / f"{name}_undistorted.png"
        cv2.imwrite(str(out_path), undistorted)
        print(f"[INFO] Zapisano wynik dla {name}: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Kalibracja i korekcja obrazu tablicy szachownicy.")
    parser.add_argument("-i", "--input", required=False, help="Ścieżka do folderu z obrazami.")
    parser.add_argument("-w", "--width", type=int, required=False, help="Liczba pól w poziomie.")
    parser.add_argument("-H", "--height", type=int, required=False, help="Liczba pól w pionie.")
    parser.add_argument("-s", "--size", type=float, required=False, help="Rozmiar 1 pola [mm].")
    parser.add_argument("-json", "--save_json", action="store_true", help="Zapisuje wyniki kalibracji do calibration.json")
    parser.add_argument("-load_json", type=str, help="Ścieżka do pliku calibration.json do korekcji obrazów")
    parser.add_argument("-FOV", action="store_true", help="Oblicza FOV z pliku calibration.json (wymaga -load_json)")
    parser.add_argument("--undistort_single", nargs=2, metavar=("LEFT_IMG", "RIGHT_IMG"),
                        help="Usuwa dystorsję z pojedynczych klatek (lewa i prawa kamera)")

    args = parser.parse_args()

    if args.load_json:
        mtx, dist = load_calibration_from_json(Path(args.load_json))

        if args.FOV:
            folder_path = Path(args.input) if args.input else Path(".")
            # Filtruj tylko pliki graficzne
            image_files = [f for f in folder_path.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp", ".tiff"]]
        
            if not image_files:
                print("[ERROR] Nie znaleziono żadnych plików graficznych w folderze:", folder_path)
            else:
                sample_img = image_files[0]
                img = load_image_unicode(sample_img)
                if img is not None:
                    h, w = img.shape[:2]
                    compute_fov_from_calibration(mtx, (w, h))
                else:
                    print("[ERROR] Nie udało się wczytać próbki obrazu do obliczenia FOV:", sample_img)

        if args.undistort_single:
            left, right = args.undistort_single
            undistort_single_images(Path(left), Path(right), mtx, dist)

    elif args.width and args.height and args.size and args.input:
        folder_path = Path(args.input)
        pattern_size = (args.width, args.height)
        square_size = args.size
        objpoints, imgpoints, gray_shape, used_files = find_chessboard_in_folder(folder_path, pattern_size, square_size)
        if objpoints and imgpoints:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
            print("Camera matrix:\n", mtx)
            print("Distortion coefficients:\n", dist.ravel())
            mean_error_full(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
            if args.save_json:
                # Ustal stronę kamery na podstawie nazwy folderu
                side = "left" if "left" in folder_path.name.lower() else "right"
                save_calibration_to_json(folder_path, mtx, dist, used_files, side)
    else:
        print("[ERROR] Niepoprawne argumenty. Podaj dane kalibracji lub plik JSON.")


if __name__ == "__main__":
    main()
