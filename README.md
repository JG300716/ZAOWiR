# Zaawansowana Analiza Obrazu, Wideo i Ruchu
## Jakub Golder
## 📷 Camera Calibration and Stereo Vision Tool

---

## 🆘 Pomoc

Aby wyświetlić pełną listę dostępnych parametrów:
```bash
python main.py -h
```

---

## 🔬 Lab 1 - Camera Calibration

### 📝 Tworzenie pliku kalibracyjnego

Generuje plik `calibration.json` na podstawie zdjęć szachownicy kalibracyjnej.

#### Parametry:

| Parametr | Opis |
|----------|------|
| `--input` | Ścieżka do katalogu ze zdjęciami |
| `--width` | Szerokość szachownicy (liczba narożników wewnętrznych) |
| `--height` | Wysokość szachownicy (liczba narożników wewnętrznych) |
| `--size` | Rozmiar pojedynczego kwadratu [mm] |
| `-json` | Flaga zapisująca wyniki do pliku JSON |

#### Przykład użycia:
```bash
python main.py \
  --input /path/to/pictures \
  --width 10 \
  --height 7 \
  --size 28.67 \
  -json
```

---

### 🖼️ Tworzenie zdjęć bez zniekształceń

Koryguje zniekształcenia optyczne na podstawie wcześniej utworzonej kalibracji.

#### Parametry:

| Parametr | Opis |
|----------|------|
| `--input` | Ścieżka do katalogu ze zdjęciami do korekcji |
| `-load_json` | Ścieżka do pliku `calibration.json` |

#### Przykład użycia:
```bash
python main.py \
  --input /path/to/pictures \
  -load_json /path/to/calibration.json
```

---

## 🎥 Lab 2 - Stereo Vision

### 🔧 Tworzenie kalibracji stereo

Generuje kalibrację dla pary kamer stereo wraz z obliczeniem linii bazowej (baseline).

#### Parametry:

| Parametr | Opis |
|----------|------|
| `--left` | Ścieżka do katalogu ze zdjęciami z lewej kamery |
| `--right` | Ścieżka do katalogu ze zdjęciami z prawej kamery |
| `--width` | Szerokość szachownicy (liczba narożników) |
| `--height` | Wysokość szachownicy (liczba narożników) |
| `--size` | Rozmiar pojedynczego kwadratu [mm] |
| `--json` | Flaga zapisująca wyniki do JSON |
| `--left_json` | Ścieżka do pliku kalibracji lewej kamery |
| `--right_json` | Ścieżka do pliku kalibracji prawej kamery |
| `--compute_baseline` | Ścieżka wyjściowa dla pliku kalibracji stereo |

#### Przykład użycia:
```bash
python main.py \
  --left /path/to/left/images \
  --right /path/to/right/images \
  --width 10 \
  --height 7 \
  --size 28.67 \
  --json \
  --left_json /path/to/left_calibration.json \
  --right_json /path/to/right_calibration.json \
  --compute_baseline /path/to/stereo_calibration.json
```

---

### 📊 Przetwarzanie obrazów stereo

Wyświetla linie epipolarne, obrazy rektyfikowane i mapę dysparacji. Zapisuje wyniki w formacie PNG.

#### Parametry:

| Parametr | Opis |
|----------|------|
| `--left_folder` | Ścieżka do katalogu z lewymi obrazami |
| `--right_folder` | Ścieżka do katalogu z prawymi obrazami |
| `--left_json` | Ścieżka do kalibracji lewej kamery |
| `--right_json` | Ścieżka do kalibracji prawej kamery |
| `--stereo_json` | Ścieżka do kalibracji stereo |
| `--save` | Katalog wyjściowy dla zapisanych wyników |

#### Przykład użycia:
```bash
python main.py \
  --left_folder /path/to/left/images \
  --right_folder /path/to/right/images \
  --left_json /path/to/left_calibration.json \
  --right_json /path/to/right_calibration.json \
  --stereo_json /path/to/stereo_calibration.json \
  --save /path/to/output
```

---

### ⚡ Benchmark algorytmów interpolacji stereo

Uruchamia testy wydajnościowe dla różnych algorytmów interpolacji używanych w przetwarzaniu stereo.

#### Parametry podstawowe:
Wszystkie parametry z sekcji **Przetwarzanie obrazów stereo** plus:

#### Parametry dodatkowe:

| Parametr | Opis |
|----------|------|
| `--benchmark` | Włącza tryb benchmarku |
| `--repeats` | Liczba powtórzeń testu (domyślnie: 10) |
| `--param` | Wybór algorytmów: `"all"` lub lista oddzielona przecinkami |
| `--show` | Wyświetla wyniki podczas testu |

#### Dostępne algorytmy interpolacji:
- `INTER_NEAREST` - interpolacja metodą najbliższego sąsiada
- `INTER_LINEAR` - interpolacja dwuliniowa
- `INTER_CUBIC` - interpolacja bicubic
- `INTER_AREA` - resampling używający relacji obszarów pikseli
- `INTER_LANCZOS4` - interpolacja Lanczos przez okno 8×8

#### Przykład użycia:
```bash
python main.py \
  --left_folder /path/to/left/images \
  --right_folder /path/to/right/images \
  --left_json /path/to/left_calibration.json \
  --right_json /path/to/right_calibration.json \
  --stereo_json /path/to/stereo_calibration.json \
  --benchmark \
  --repeats 10 \
  --param "all" \
  --show
```

#### Przykład testowania wybranych algorytmów:
```bash
python main.py \
  --left_folder /path/to/left/images \
  --right_folder /path/to/right/images \
  --left_json /path/to/left_calibration.json \
  --right_json /path/to/right_calibration.json \
  --stereo_json /path/to/stereo_calibration.json \
  --benchmark \
  --repeats 20 \
  --param "INTER_NEAREST,INTER_CUBIC,INTER_LANCZOS4"
```

---

## 📋 Uwagi

> **💡 Wskazówki:**
> - Wszystkie ścieżki mogą być względne lub bezwzględne
> - Obsługiwane formaty obrazów: JPG, PNG, BMP, TIFF (wszystkie standardowe formaty OpenCV)
> - Pliki JSON zawierają kompletne parametry kalibracji kamer (macierz kamery, współczynniki zniekształceń, etc.)
> - Przy kalibracji stereo wymagane są wcześniej utworzone pliki kalibracji dla obu kamer
> - Rozmiar szachownicy podawany jest jako liczba **narożników wewnętrznych**, nie pól

---

## 📁 Struktura plików wyjściowych

### Kalibracja pojedynczej kamery:
```
calibration.json          # Parametry kalibracji kamery
undistorted/             # Katalog ze skorygowanymi obrazami (opcjonalnie)
```

### Kalibracja stereo:
```
stereo_calibration.json   # Parametry kalibracji stereo
output/
  ├── epipolar/          # Obrazy z liniami epipolarnymi
  ├── rectified/         # Rektyfikowane pary obrazów
  └── disparity/         # Mapy dysparacji
```

---

## 🤝 Wymagania

- Python 3.7+
- OpenCV (cv2)
- NumPy
- JSON (standardowa biblioteka)

---