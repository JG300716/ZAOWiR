import numpy as np
import cv2 as cv
import argparse
import time
from collections import deque


class OpticalFlowAnalyzer:
    """Klasa do analizy przepływu optycznego"""

    def __init__(self):
        # Parametry dla detekcji narożników Shi-Tomasi
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

        # Parametry dla Lucas-Kanade
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Parametry dla Farneback
        self.farneback_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Parametry dla detekcji ruchu
        self.motion_threshold = 2.0  # Próg prędkości do detekcji ruchu
        self.min_area = 500  # Minimalny obszar obiektu w pikselach

        # Kolory do wizualizacji
        self.colors = np.random.randint(0, 255, (100, 3))

        # Historia czasów przetwarzania
        self.processing_times = deque(maxlen=30)

    def sparse_optical_flow(self, video_path, output_path=None):
        cap = cv.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Błąd: nie można otworzyć pliku {video_path}")
            return

        # Odczyt pierwszej klatki
        ret, old_frame = cap.read()
        if not ret:
            print("Błąd: nie można odczytać pierwszej klatki")
            return

        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params)

        if p0 is None:
            print("Błąd: nie znaleziono punktów charakterystycznych")
            return

        # Maska do rysowania śladów
        mask = np.zeros_like(old_frame)

        # Konfiguracja zapisu wideo
        if output_path:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv.CAP_PROP_FPS))
            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Obliczanie przepływu optycznego
            p1, st, err = cv.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **self.lk_params
            )

            if p1 is not None and st is not None:
                # Wybór dobrych punktów
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # Rysowanie śladów
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = int(new[0]), int(new[1])
                    c, d = int(old[0]), int(old[1])
                    mask = cv.line(mask, (a, b), (c, d),
                                   self.colors[i % len(self.colors)].tolist(), 2)
                    frame = cv.circle(frame, (a, b), 5,
                                      self.colors[i % len(self.colors)].tolist(), -1)

                img = cv.add(frame, mask)

                # Dodanie informacji na ekranie
                cv.putText(img, f'Frame: {frame_count}', (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv.putText(img, f'Points: {len(good_new)}', (10, 70),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv.imshow('Sparse Optical Flow (Lucas-Kanade)', img)

                if output_path:
                    out.write(img)

                # Aktualizacja
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)

                # Ponowne wykrycie punktów jeśli ich jest za mało
                if len(p0) < 10:
                    p0 = cv.goodFeaturesToTrack(old_gray, mask=None,
                                                **self.feature_params)
                    mask = np.zeros_like(old_frame)

            frame_count += 1

            key = cv.waitKey(30) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('r'):  # Reset
                p0 = cv.goodFeaturesToTrack(old_gray, mask=None,
                                            **self.feature_params)
                mask = np.zeros_like(old_frame)

        cap.release()
        if output_path:
            out.release()
        cv.destroyAllWindows()
        print(f"Przetworzono {frame_count} klatek")

    def dense_optical_flow(self, video_path, output_path=None):
        cap = cv.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Błąd: nie można otworzyć pliku {video_path}")
            return

        ret, frame1 = cap.read()
        if not ret:
            print("Błąd: nie można odczytać pierwszej klatki")
            return

        prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255

        # Konfiguracja zapisu wideo
        if output_path:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv.CAP_PROP_FPS))
            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        while True:
            ret, frame2 = cap.read()
            if not ret:
                break

            next_frame = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

            # Obliczanie gęstego przepływu optycznego
            flow = cv.calcOpticalFlowFarneback(
                prvs, next_frame, None,
                **self.farneback_params
            )

            # Konwersja do współrzędnych biegunowych
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

            # Wizualizacja w przestrzeni HSV
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

            # Tworzenie połączonego widoku
            combined = np.hstack([frame2, bgr])

            # Dodanie informacji
            cv.putText(combined, f'Frame: {frame_count}', (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.putText(combined, f'Avg Flow: {np.mean(mag):.2f}', (10, 70),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv.imshow('Dense Optical Flow (Farneback)', combined)

            if output_path:
                out.write(bgr)

            prvs = next_frame
            frame_count += 1

            key = cv.waitKey(30) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('s'):  # Zapis klatki
                cv.imwrite('flow_frame.png', frame2)
                cv.imwrite('flow_visualization.png', bgr)
                print("Zapisano klatki")

        cap.release()
        if output_path:
            out.release()
        cv.destroyAllWindows()
        print(f"Przetworzono {frame_count} klatek")

    def detect_moving_objects(self, video_path, output_path=None):
        cap = cv.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Błąd: nie można otworzyć pliku {video_path}")
            return

        ret, frame1 = cap.read()
        if not ret:
            return

        prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

        # Konfiguracja zapisu wideo
        if output_path:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv.CAP_PROP_FPS))
            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Obliczanie przepływu optycznego
            flow = cv.calcOpticalFlowFarneback(
                prvs, gray, None,
                **self.farneback_params
            )

            # Obliczanie wielkości i kąta przepływu
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

            # Utworzenie maski dla ruchomych obszarów
            motion_mask = (mag > self.motion_threshold).astype(np.uint8) * 255

            # Operacje morfologiczne dla poprawy detekcji
            kernel = np.ones((5, 5), np.uint8)
            motion_mask = cv.morphologyEx(motion_mask, cv.MORPH_CLOSE, kernel)
            motion_mask = cv.morphologyEx(motion_mask, cv.MORPH_OPEN, kernel)

            # Znajdowanie konturów
            contours, _ = cv.findContours(
                motion_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )

            result_frame = frame.copy()

            # Analiza każdego konturu
            for contour in contours:
                area = cv.contourArea(contour)

                if area > self.min_area:
                    # Bounding box
                    x, y, w, h = cv.boundingRect(contour)

                    # Obliczanie średniego przepływu w obszarze obiektu
                    roi_flow = flow[y:y+h, x:x+w]
                    roi_mag = mag[y:y+h, x:x+w]
                    roi_ang = ang[y:y+h, x:x+w]

                    avg_flow_x = np.mean(roi_flow[..., 0])
                    avg_flow_y = np.mean(roi_flow[..., 1])
                    avg_speed = np.mean(roi_mag)
                    avg_angle = np.mean(roi_ang) * 180 / np.pi

                    # Określenie kierunku ruchu
                    direction = self._get_direction(avg_flow_x, avg_flow_y)

                    # Rysowanie bounding box
                    cv.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Rysowanie strzałki kierunku
                    center_x, center_y = x + w//2, y + h//2
                    arrow_end_x = int(center_x + avg_flow_x * 5)
                    arrow_end_y = int(center_y + avg_flow_y * 5)
                    cv.arrowedLine(result_frame, (center_x, center_y),
                                   (arrow_end_x, arrow_end_y), (0, 0, 255), 2)

                    # Dodanie tekstu z informacjami
                    info_text = f"{direction} {avg_speed:.1f}px/f"
                    cv.putText(result_frame, info_text, (x, y-10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Informacje ogólne
            cv.putText(result_frame, f'Frame: {frame_count}', (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv.putText(result_frame, f'Objects: {len([c for c in contours if cv.contourArea(c) > self.min_area])}',
                       (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv.imshow('Moving Object Detection', result_frame)
            cv.imshow('Motion Mask', motion_mask)

            if output_path:
                out.write(result_frame)

            prvs = gray
            frame_count += 1

            key = cv.waitKey(30) & 0xFF
            if key == 27:  # ESC
                break

        cap.release()
        if output_path:
            out.release()
        cv.destroyAllWindows()
        print(f"Przetworzono {frame_count} klatek")

    def realtime_detection(self, camera_id=0, filter_type='all', speed_range=(0, 100)):
        print(f"Tryb filtrowania: {filter_type}")
        print(f"Zakres prędkości: {speed_range}")
        print("Sterowanie:")
        print("  ESC - wyjście")
        print("  'h' - filtr poziomy")
        print("  'v' - filtr pionowy")
        print("  'f' - tylko szybkie obiekty")
        print("  's' - tylko wolne obiekty")
        print("  'a' - wszystkie obiekty")
        print("  '+' - zwiększ próg")
        print("  '-' - zmniejsz próg")

        cap = cv.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"Błąd: nie można otworzyć kamery {camera_id}")
            return

        # Ustawienie rozdzielczości dla lepszej wydajności
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

        ret, frame1 = cap.read()
        if not ret:
            return

        prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

        current_filter = filter_type
        current_threshold = self.motion_threshold
        min_speed, max_speed = speed_range

        frame_count = 0

        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Obliczanie przepływu optycznego
            flow = cv.calcOpticalFlowFarneback(
                prvs, gray, None,
                **self.farneback_params
            )

            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

            # Maska ruchu
            motion_mask = (mag > current_threshold).astype(np.uint8) * 255

            kernel = np.ones((5, 5), np.uint8)
            motion_mask = cv.morphologyEx(motion_mask, cv.MORPH_CLOSE, kernel)
            motion_mask = cv.morphologyEx(motion_mask, cv.MORPH_OPEN, kernel)

            contours, _ = cv.findContours(
                motion_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )

            result_frame = frame.copy()
            detected_objects = 0

            for contour in contours:
                area = cv.contourArea(contour)

                if area > self.min_area:
                    x, y, w, h = cv.boundingRect(contour)

                    roi_flow = flow[y:y+h, x:x+w]
                    roi_mag = mag[y:y+h, x:x+w]

                    avg_flow_x = np.mean(roi_flow[..., 0])
                    avg_flow_y = np.mean(roi_flow[..., 1])
                    avg_speed = np.mean(roi_mag)

                    # Filtrowanie według wybranego typu
                    should_display = self._should_display_object(
                        current_filter, avg_flow_x, avg_flow_y,
                        avg_speed, min_speed, max_speed
                    )

                    if should_display:
                        detected_objects += 1
                        direction = self._get_direction(avg_flow_x, avg_flow_y)

                        cv.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                        center_x, center_y = x + w//2, y + h//2
                        arrow_end_x = int(center_x + avg_flow_x * 5)
                        arrow_end_y = int(center_y + avg_flow_y * 5)
                        cv.arrowedLine(result_frame, (center_x, center_y),
                                       (arrow_end_x, arrow_end_y), (0, 0, 255), 2)

                        info_text = f"{direction} {avg_speed:.1f}px/f"
                        cv.putText(result_frame, info_text, (x, y-10),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Obliczenie czasu przetwarzania
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            avg_processing_time = np.mean(self.processing_times)
            fps = 1000 / avg_processing_time if avg_processing_time > 0 else 0

            # Wyświetlanie informacji
            info_y = 30
            cv.putText(result_frame, f'FPS: {fps:.1f}', (10, info_y),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += 35
            cv.putText(result_frame, f'Time: {avg_processing_time:.1f}ms', (10, info_y),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += 35
            cv.putText(result_frame, f'Objects: {detected_objects}', (10, info_y),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += 35
            cv.putText(result_frame, f'Filter: {current_filter}', (10, info_y),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += 35
            cv.putText(result_frame, f'Threshold: {current_threshold:.1f}', (10, info_y),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv.imshow('Real-time Motion Detection', result_frame)

            prvs = gray
            frame_count += 1

            # Co 30 klatek wyświetl statystyki
            if frame_count % 30 == 0:
                print(f"Średni czas przetwarzania: {avg_processing_time:.2f}ms, FPS: {fps:.1f}")

            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('h'):
                current_filter = 'horizontal'
                print("Filtr: ruch poziomy")
            elif key == ord('v'):
                current_filter = 'vertical'
                print("Filtr: ruch pionowy")
            elif key == ord('f'):
                current_filter = 'fast'
                print("Filtr: szybkie obiekty")
            elif key == ord('s'):
                current_filter = 'slow'
                print("Filtr: wolne obiekty")
            elif key == ord('a'):
                current_filter = 'all'
                print("Filtr: wszystkie obiekty")
            elif key == ord('+') or key == ord('='):
                current_threshold += 0.5
                print(f"Próg: {current_threshold:.1f}")
            elif key == ord('-') or key == ord('_'):
                current_threshold = max(0.5, current_threshold - 0.5)
                print(f"Próg: {current_threshold:.1f}")

        cap.release()
        cv.destroyAllWindows()

        # Podsumowanie
        if self.processing_times:
            print(f"\n=== STATYSTYKI ===")
            print(f"Średni czas przetwarzania: {np.mean(self.processing_times):.2f}ms")
            print(f"Minimalny czas: {np.min(self.processing_times):.2f}ms")
            print(f"Maksymalny czas: {np.max(self.processing_times):.2f}ms")
            print(f"Średnie FPS: {1000 / np.mean(self.processing_times):.1f}")
            print(f"Przetworzono klatek: {frame_count}")

    def _get_direction(self, flow_x, flow_y):
        """Określa kierunek ruchu na podstawie przepływu"""
        angle = np.arctan2(flow_y, flow_x) * 180 / np.pi

        if -22.5 <= angle < 22.5:
            return "E"
        elif 22.5 <= angle < 67.5:
            return "SE"
        elif 67.5 <= angle < 112.5:
            return "S"
        elif 112.5 <= angle < 157.5:
            return "SW"
        elif 157.5 <= angle or angle < -157.5:
            return "W"
        elif -157.5 <= angle < -112.5:
            return "NW"
        elif -112.5 <= angle < -67.5:
            return "N"
        elif -67.5 <= angle < -22.5:
            return "NE"
        return "?"

    def _should_display_object(self, filter_type, flow_x, flow_y, speed,
                               min_speed, max_speed):
        """Określa czy obiekt powinien być wyświetlony według filtra"""
        if filter_type == 'all':
            return min_speed <= speed <= max_speed
        elif filter_type == 'horizontal':
            return abs(flow_x) > abs(flow_y) and min_speed <= speed <= max_speed
        elif filter_type == 'vertical':
            return abs(flow_y) > abs(flow_x) and min_speed <= speed <= max_speed
        elif filter_type == 'fast':
            return speed > 5.0
        elif filter_type == 'slow':
            return 1.0 < speed <= 5.0
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Aplikacja do analizy przepływu optycznego',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:

  Zadanie 1 - Rzadki przepływ optyczny:
    python optical_flow_app.py --task sparse --input video.mp4
  
  Zadanie 2 - Gęsty przepływ optyczny:
    python optical_flow_app.py --task dense --input video.mp4
  
  Zadanie 3 - Detekcja ruchomych obiektów:
    python optical_flow_app.py --task detect --input video.mp4
  
  Zadanie 4 - Analiza w czasie rzeczywistym:
    python optical_flow_app.py --task realtime --camera 0
    python optical_flow_app.py --task realtime --camera 0 --filter horizontal
        """
    )

    parser.add_argument('--task', type=str, required=True,
                        choices=['sparse', 'dense', 'detect', 'realtime'],
                        help='Rodzaj zadania do wykonania')

    parser.add_argument('--input', type=str,
                        help='Ścieżka do pliku wideo (dla zadań 1-3)')

    parser.add_argument('--output', type=str,
                        help='Ścieżka do zapisu wyniku (opcjonalne)')

    parser.add_argument('--camera', type=int, default=0,
                        help='ID kamery dla zadania 4 (domyślnie: 0)')

    parser.add_argument('--filter', type=str, default='all',
                        choices=['all', 'horizontal', 'vertical', 'fast', 'slow'],
                        help='Typ filtra dla zadania 4')

    parser.add_argument('--min-speed', type=float, default=0,
                        help='Minimalna prędkość dla zadania 4')

    parser.add_argument('--max-speed', type=float, default=100,
                        help='Maksymalna prędkość dla zadania 4')

    parser.add_argument('--threshold', type=float, default=2.0,
                        help='Próg detekcji ruchu (domyślnie: 2.0)')

    parser.add_argument('--min-area', type=int, default=500,
                        help='Minimalny obszar obiektu w pikselach (domyślnie: 500)')

    args = parser.parse_args()

    # Walidacja argumentów
    if args.task in ['sparse', 'dense', 'detect'] and not args.input:
        parser.error(f"Zadanie '{args.task}' wymaga argumentu --input")

    # Utworzenie analizatora
    analyzer = OpticalFlowAnalyzer()
    analyzer.motion_threshold = args.threshold
    analyzer.min_area = args.min_area

    # Wykonanie odpowiedniego zadania
    if args.task == 'sparse':
        analyzer.sparse_optical_flow(args.input, args.output)

    elif args.task == 'dense':
        analyzer.dense_optical_flow(args.input, args.output)

    elif args.task == 'detect':
        analyzer.detect_moving_objects(args.input, args.output)

    elif args.task == 'realtime':
        analyzer.realtime_detection(
            args.camera,
            args.filter,
            (args.min_speed, args.max_speed)
        )


if __name__ == '__main__':
    main()