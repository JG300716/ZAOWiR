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

### 📝 Kalibracja Kamer

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

### 🔧 Kalibracja systemu kamer stereo

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
## 🎥 Lab 3 - Stereo Vision
### 🖥️ Odtwarzanie trójwymiarowej sceny na podstawie wielu perspektyw

Narzędzie pozwala generować mapy dysparycji metodami **BM**, **SGBM** oraz **CUSTOM**, a także porównywać wygenerowane mapy z referencyjną mapą GT oraz wizualizować błędy za pomocą kolorowych map cieplnych.

---

### 🎛️ Parametry

| Parametr | Opis |
|----------|------|
| `--method` | Wybór metody dysparycji: `BM`, `SGBM`, `CUSTOM` |
| `--block_size` | Rozmiar bloku dopasowania (wartość parzysta jest automatycznie poprawiana do nieparzystej) |
| `--num_disparities` | Liczba dysparycji (zaokrąglana do wielokrotności 16) |
| `--left_image` | Obraz z lewej kamery |
| `--right_image` | Obraz z prawej kamery |
| `--save` | Zapisuje obliczoną mapę dysparycji do pliku PNG |
| `--compare` | Aktywuje tryb porównania map dysparycji |
| `--path` | Folder zawierający pliki `*_disparity.png` wygenerowane wcześniej |
| `--ref_path` | Ścieżka do referencyjnej mapy GT (skalowanej ×4, 0 = brak danych) |

---



### 📊 Porównywanie map dysparycji z ground truth

Funkcja porównująca mapy dysparycji automatycznie:

- wczytuje wszystkie pliki `*_disparity.png` z podanego folderu,
- ładuje i skaluje mapę referencyjną GT (zakodowaną ×4, 0 = brak danych),
- oblicza metryki jakości:
    - **MAE** – średni błąd bezwzględny,
    - **RMSE** – pierwiastek z błędu średniokwadratowego,
    - **Bad pixels** – procent pikseli, gdzie błąd > 1.0 px,
- generuje kolorową mapę błędów (JET colormap),
- zapisuje wyniki w formie: `*_error.png`.

#### Parametry:
| Parametr | Opis |
|----------|------|
| `--compare` | Aktywuje tryb porównania map dysparycji |
| `--path` | Folder zawierający pliki `*_disparity.png` |
| `--ref_path` | Mapa referencyjna GT zakodowana ×4 |

---
#### Przykład:
Oblicza mapę dysparycji za pomocą wybranego algorytmu i opcjonalnie zapisuje ją do pliku.
```bash
python main.py 
  --method SGBM 
  --left_image data/left.png 
  --right_image data/right.png 
  --save
```
Następnie porównuje wszystkie zapisane mapy dysparycji w katalogu `results/` z referencyjną mapą GT.
```bash
python main.py \
  --compare \
  --path results/ \
  --ref_path GT/disp_gt.png
```

---
## 🎥 Lab 4
### 🖥️ Mapy głębi i chmury punktów

---

---
## 🎥 Lab 5
### 🖥️ Przepływ optyczny

---

## 📖 Opis

Aplikacja do analizy przepływu optycznego implementująca metody Lucas-Kanade (rzadki przepływ) i Farneback (gęsty przepływ). Umożliwia wykrywanie i śledzenie ruchomych obiektów w sekwencjach wideo oraz analizę w czasie rzeczywistym z kamery.

---

## 🚀 Użycie

```bash
# Zadanie 1 - Rzadki przepływ optyczny (Lucas-Kanade)
python optical_flow_app.py --task sparse --input video.mp4

# Zadanie 2 - Gęsty przepływ optyczny (Farneback)
python optical_flow_app.py --task dense --input video.mp4

# Zadanie 3 - Detekcja ruchomych obiektów
python optical_flow_app.py --task detect --input video.mp4

# Zadanie 4 - Analiza w czasie rzeczywistym z kamery
python optical_flow_app.py --task realtime --camera 0
```

---

## ⚙️ Parametry

### Podstawowe

| Parametr | Typ | Opis | Domyślnie |
|----------|-----|------|-----------|
| `--task` | string | Rodzaj zadania: `sparse`, `dense`, `detect`, `realtime` | **wymagany** |
| `--input` | string | Ścieżka do pliku wideo (zadania 1-3) | - |
| `--output` | string | Ścieżka do zapisu wyniku | - |
| `--camera` | int | ID kamery (zadanie 4) | `0` |

### Detekcja ruchu

| Parametr | Typ | Opis | Domyślnie |
|----------|-----|------|-----------|
| `--threshold` | float | Próg prędkości do detekcji ruchu | `2.0` |
| `--min-area` | int | Minimalny obszar obiektu [px²] | `500` |
| `--min-speed` | float | Minimalna prędkość obiektu | `0` |
| `--max-speed` | float | Maksymalna prędkość obiektu | `100` |

### Filtry

| Parametr | Wartości | Opis |
|----------|----------|------|
| `--filter` | `all` | Wszystkie obiekty |
| | `horizontal` | Tylko ruch poziomy |
| | `vertical` | Tylko ruch pionowy |
| | `fast` | Szybkie obiekty (>5 px/frame) |
| | `slow` | Wolne obiekty (1-5 px/frame) |

---

## 🎮 Sterowanie

### Filter Sparse
- `ESC` - zakończenie
- `r` - reset punktów śledzenia

### Filter (Dense)
- `ESC` - zakończenie
- `s` - zapis bieżącej klatki

### Filter (Detect)
- `ESC` - zakończenie

### Filter (Realtime)
- `ESC` - zakończenie
- `h` - filtr poziomy
- `v` - filtr pionowy
- `f` - tylko szybkie obiekty
- `s` - tylko wolne obiekty
- `a` - wszystkie obiekty
- `+` / `=` - zwiększ próg
- `-` / `_` - zmniejsz próg

---

## 📊 Przykłady

### Z zapisem wyniku
```bash
python optical_flow_app.py --task sparse --input video.mp4 --output result.mp4
```

### Detekcja z dostosowanymi parametrami
```bash
python optical_flow_app.py --task detect --input video.mp4 \
    --threshold 3.0 --min-area 1000
```

### Monitoring ruchu poziomego
```bash
python optical_flow_app.py --task realtime --filter horizontal \
    --min-speed 3.0
```

### Wykrywanie szybkich obiektów
```bash
python optical_flow_app.py --task realtime --filter fast \
    --threshold 5.0 --min-area 1500
```
---

## 📈 Wyświetlane informacje

- Numer klatki
- Liczba punktów / średni przepływ
- Kierunek ruchu (N, NE, E, SE, S, SW, W, NW)
- Prędkość [px/frame]
- **FPS** - klatki na sekundę
- **Time** - czas przetwarzania [ms]
- **Objects** - liczba wykrytych obiektów
- **Filter** - aktywny filtr
- **Threshold** - próg detekcji

---

## 🛠️ Dostrajanie wydajności

### Wysoka czułość (więcej detekcji)
```bash
--threshold 1.5 --min-area 300
```

### Niska czułość (mniej fałszywych detekcji)
```bash
--threshold 4.0 --min-area 1500
```

### Optymalizacja szybkości
```bash
--threshold 3.0 --min-area 1000 --filter horizontal
```

---

## 🔍 Algorytmy

- **Lucas-Kanade** - lokalna metoda różniczkowa dla rzadkiego przepływu
- **Farneback** - metoda bazująca na aproksymacji wielomianowej dla gęstego przepływu
- **Shi-Tomasi** - detekcja punktów charakterystycznych (narożników)

---

# 🤝 Wymagania

- Python 3.7+
- OpenCV (cv2)
- NumPy
- JSON (standardowa biblioteka)

---