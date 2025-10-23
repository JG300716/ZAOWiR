import cv2
import argparse
from pathlib import Path
import numpy as np
import json
import os


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
        return [], [], None

    objpoints = []
    imgpoints = []
    detected_images = []

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

    print("=== PODSUMOWANIE ===")
    print("Ilosc zdjęć z wykrytą tablicą kalibracyjną: ", len(detected_images), "/", len(image_files))
    return objpoints, imgpoints, first_gray_shape


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


def save_calibration_to_json(folder_path: Path, mtx, dist):
    out_path = folder_path.parent / "calibration.json"
    data = {
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.ravel().tolist()
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


def undistort_images(folder_path: Path, mtx, dist):
    output_dir = folder_path.parent / "undistort"
    output_dir.mkdir(exist_ok=True)
    for image_path in folder_path.iterdir():
        if image_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
            continue
        img = load_image_unicode(image_path)
        if img is None:
            continue
        undistorted = cv2.undistort(img, mtx, dist, None)
        out_path = output_dir / image_path.name
        cv2.imwrite(str(out_path), undistorted)
        success, encoded_img = cv2.imencode(".png", undistorted)
        if success:
            with open(out_path, "wb") as f:
                f.write(encoded_img.tobytes())
    print(f"[INFO] Poprawione obrazy zapisane w: {output_dir}")


def remap_images(folder_path: Path, mtx, dist):
    output_dir = folder_path.parent / "undistort_with_remap"
    output_dir.mkdir(exist_ok=True)
    sample_img = next(folder_path.iterdir())
    img_sample = load_image_unicode(sample_img)
    h, w = img_sample.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (w, h), cv2.CV_32FC1)

    for image_path in folder_path.iterdir():
        if image_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
            continue
        #else:
        #   print("Znaleziono plik:", image_path)
        img = load_image_unicode(image_path)
        if img is None:
            print(f"[UWAGA] Nie udało się wczytać {image_path}")
            continue
        #else:
        #   print(f"[OK] Udało się wczytać {image_path}")
        undistorted = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR)
        out_path = output_dir / image_path.name
        success, encoded_img = cv2.imencode(".png", undistorted)
        if success:
            with open(out_path, "wb") as f:
                f.write(encoded_img.tobytes())
        #else:
        #   print(f"[BŁĄD] Nie udało się zakodować obrazu {out_path}")
    print(f"[INFO] Poprawione obrazy (remap) zapisane w: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Kalibracja i korekcja obrazu tablicy szachownicy.")
    parser.add_argument("-i", "--input", required=True, help="Ścieżka do folderu z obrazami.")
    parser.add_argument("-w", "--width", type=int, required=False, help="Liczba pól w poziomie.")
    parser.add_argument("-H", "--height", type=int, required=False, help="Liczba pól w pionie.")
    parser.add_argument("-s", "--size", type=float, required=False, help="Rozmiar 1 pola [mm].")
    parser.add_argument("-json", "--save_json", action="store_true",
                        help="Zapisuje wyniki kalibracji do calibration.json")
    parser.add_argument("-load_json", type=str, help="Ścieżka do pliku calibration.json do korekcji obrazów")
    args = parser.parse_args()

    folder_path = Path(args.input)

    if args.load_json:
        mtx, dist = load_calibration_from_json(Path(args.load_json))
        undistort_images(folder_path, mtx, dist)
        remap_images(folder_path, mtx, dist)
    elif args.width and args.height and args.size:
        pattern_size = (args.width, args.height)
        square_size = args.size
        objpoints, imgpoints, gray_shape = find_chessboard_in_folder(folder_path, pattern_size, square_size)
        if objpoints and imgpoints:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
            print("Camera matrix:\n", mtx)
            print("Distortion coefficients:\n", dist.ravel())
            mean_error_full(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
            if args.save_json:
                save_calibration_to_json(folder_path, mtx, dist)
    else:
        print("[ERROR] Podaj albo parametry kalibracji (width, height, size) albo plik JSON (-load_json)")


if __name__ == "__main__":
    main()
