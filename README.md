# Raport Techniczny — Projekt BlindIQA: Klasyfikacja Skali Downscalingu

---

## 1. Opis projektu

Projekt dotyczy budowy systemu **ślepej oceny jakości obrazu** (Blind Image Quality Assessment — IQA) ukierunkowanego na automatyczne wykrywanie i klasyfikację degradacji wynikającej z downscalingu. Głównym celem jest stworzenie klasyfikatora CNN zdolnego do rozróżnienia, z jakim współczynnikiem skalowania obraz został zdegradowany — bez znajomości oryginału.

Dane wejściowe stanowią zdjęcia RAW w formacie `.NEF` (1000 zdjęć), które poddawane są kontrolowanej degradacji (downscaling) i następnie rekonstrukcji (upscaling) za pomocą różnych algorytmów. Każda para algorytmów tworzy oddzielną klasę artefaktów wizualnych, co pozwala na systematyczną analizę wpływu metody przetwarzania na percepowaną jakość obrazu.

**Środowisko obliczeniowe:** serwer `komp4` z GPU, TensorFlow + PyTorch, dysk dodatkowy EXT4.

Architektura referencyjna CNN pochodzi z pracy: *Frackiewicz et al., Sensors 2025*.

---

## 2. Struktura projektu i przepływ danych

Projekt składa się z dwóch głównych notebooków:

```
pipeline_v2_komp4.ipynb   →   generacja datasetu, zapis manifest.jsonl + K_out.zip
model_v2_komp4.ipynb      →   trening klasyfikatora CNN, tuning Optuna, ewaluacja
```

**Przepływ ogólny:**

```
Zdjęcia RAW (.NEF)
       │
       ▼
[pipeline] Smart Crop → Degradacja (downscale) → Rekonstrukcja (upscale) → Patch 256×256
       │
       ▼
Dataset (manifest.jsonl + pliki PNG) w folderze K_out/
       │
       ▼
[model] Lokalna normalizacja MSCN → tf.data pipeline → Trening CNN → Ewaluacja
```

---

## 3. Etapy przetwarzania danych (Pipeline)

### 3.1. Pobranie i przygotowanie danych wejściowych

Zdjęcia w formacie `.NEF` pobierane są z Google Drive za pomocą narzędzia `gdown`. Automatyczny strażnik ZIP-ów rozpakowuje archiwa i raportuje liczbę wykrytych plików. Oryginalne pliki przechowywane są w katalogu `1K/`.

### 3.2. Instalacja zależności i inicjalizacja modeli AI

Instalowane są pakiety: `basicsr`, `facexlib`, `gfpgan`, `lpips`, `pytorch-msssim`, `scikit-image`, `rawpy`. Repozytorium Real-ESRGAN klonowane jest z GitHuba i instalowane w trybie deweloperskim. Zastosowany jest **patch zgodności** dla `basicsr` z nowszymi wersjami `torchvision` (≥ 0.17), które usunęły moduł `functional_tensor`.

Wagi modeli SR pobierane są z oficjalnych repozytoriów:

- `RealESRGAN_x4plus.pth` — z GitHub xinntao/Real-ESRGAN
- `RRDB_ESRGAN_x4.pth` — z HuggingFace ai-forever/Real-ESRGAN

### 3.3. Konfiguracja pipeline'u

Parametry konfiguracyjne zdefiniowane są w jednym miejscu (`KROK 3`) i zapisywane do `pipeline_config.json`:

| Parametr | Wartość domyślna | Opis |
|---|---|---|
| `N` | 1000 | Liczba obrazów wejściowych |
| `SCALE_FACTORS` | [1..10] | Współczynniki skalowania |
| `CROP_SIZE` | 256 px | Docelowy rozmiar patcha |
| `N_CANDIDATES` | 16 | Liczba kandydatów w smart crop |
| `ENTROPY_MIN` | 4.5 bitu | Minimalny próg entropii Shannona |
| `NOISE_VAR_MAX` | 2500 | Maksymalna wariancja Laplacjanu |
| `SPLIT_RATIO` | 70/15/15 | Podział train/val/test |
| `SPLIT_SEED` | 42 | Ziarno losowości |
| `EXCLUDED_PAIR_FROM_TEST` | 4 | Para wykluczona ze zbioru testowego |

### 3.4. Deterministyczny podział na zbiory

Podział plików źródłowych odbywa się **przed** przetwarzaniem — eliminuje to przecieki danych (data leakage), gwarantując, że ten sam obraz nie pojawi się jednocześnie w zbiorze treningowym i testowym.

- Pliki zawierające słowa kluczowe (`szachownica`, `checker`, `ai_gen`, `render`, `anime`, itp.) trafiają do zbioru `stress_test` (materiały syntetyczne i niezgodne z dystrybucją treningową).
- Pozostałe pliki tasowane są z ustaloną losowością (`SPLIT_SEED=42`) i dzielone zgodnie z `SPLIT_RATIO`.

Wynik zapisywany jest do `splits.json`.

### 3.5. Smart Crop — inteligentny wybór patcha

Funkcja `smart_crop` losuje `N_CANDIDATES` (domyślnie 16) obszarów o rozmiarze 320×320 pikseli z każdego obrazu i wybiera ten o najwyższej **entropii Shannona**, spełniający jednocześnie dwa warunki jakości:

- **Entropia ≥ ENTROPY_MIN (4.5 bitu)** — wycina nudne, jednorodne tła
- **Wariancja Laplacjanu ≤ NOISE_VAR_MAX (2500)** — odrzuca obszary zbyt szumowe lub rozmyte

Jeśli żaden kandydat nie spełnia warunków, awaryjnie wycina środek obrazu. Po procesie degradacji stosowany jest **center crop** do 256×256 px.

Podejście to gwarantuje, że do datasetu trafiają fragmenty obrazu o wysokiej zawartości informacyjnej, a nie jednolite obszary nieba czy ściany.

### 3.6. Główna pętla przetwarzania

Dla każdego obrazu wejściowego pipeline wykonuje:

1. Wczytanie pliku NEF przez `rawpy` i konwersję do RGB
2. Przeskalowanie do maksymalnie 1024×1024 px (thumbnail)
3. Smart Crop — wybór optymalnego patcha 320×320 px
4. Dla każdego `sf` ∈ `SCALE_FACTORS` × każdej pary algorytmów:
   - downscale całego obrazu → low-res
   - upscale low-res → high-res 320×320 px
   - center crop do 256×256 px
   - zapis PNG + wpis JSONL w manifeście

Manifest (`manifest.jsonl`) zawiera dla każdej próbki pełne metadane: ID próbki, ID obrazu źródłowego, split, ID pary algorytmów, współczynnik skalowania, parametry cropu (entropia, wariancja szumu, współrzędne), oraz ścieżkę względną pliku.

**Checkpoint:** pipeline jest idempotentny — zapisuje przetworzone pliki do manifestu w trybie `append`, co pozwala na wznowienie przerwanego przetwarzania bez powtarzania pracy.

**Zarządzanie pamięcią:** po każdym obrazie bazowym zwalniany jest RAM (`gc.collect()`) i VRAM (`torch.cuda.empty_cache()`).

---

## 4. Opis par algorytmów

Zdefiniowano **7 par** (degradacja → rekonstrukcja), z których para 0 stanowi punkt odniesienia (kontrolę):

| Para ID | Downscale | Upscale | Cel badawczy / Typ artefaktów |
|:---:|---|---|---|
| **0** | Identity (CTRL) | Identity (CTRL) | Punkt odniesienia — brak zmian |
| **1** | Nearest Neighbor | Nearest Neighbor | Maksymalna pikseloza, ostre schodkowanie (aliasing) |
| **2** | Nearest Neighbor | Bicubic | Próba wygładzenia pikselozy metodą klasyczną |
| **3** | Box Filter | Bicubic | Ekstremalne rozmycie, utrata detali, miękkie krawędzie |
| **4** | Bicubic | Bicubic | Standardowa, zrównoważona utrata wysokich częstotliwości (baseline) |
| **5** | Lanczos | Bicubic | Najwyższa wierność downscalingu klasycznego + klasyczny upscale |
| **6** | Bicubic | Real-ESRGAN | Halucynacje AI — jak sieć generatywna „dodaje" nieistniejące detale |

**Uzasadnienie doboru par:**

Pary 1–3 badają klasyczne artefakty interpolacyjne w skrajnych wariantach. Pary 4–5 stanowią reprezentatywne dla branży przypadki użycia narzędzi desktopowych. Para 6 testuje zachowanie modelu generatywnego (ESRGAN), który może produkować realistycznie wyglądające, lecz nieistniejące tekstury — zjawisko szczególnie istotne z perspektywy oceny jakości w aplikacjach krytycznych.

Para o ID równym `EXCLUDED_PAIR_FROM_TEST` (domyślnie para 4) jest wykluczona ze zbioru testowego, co pozwala ocenić zdolność generalizacji modelu na niewidzianej kombinacji algorytmów.

**Modele AI (ESRGAN):** działają natywnie w trybie ×4. Dla współczynników skali innych niż 4 wyniki są dodatkowo skalowane metodą Lanczos do docelowego rozmiaru.

---

## 5. Budowa modelu CNN

### 5.1. Preprocessing — lokalna normalizacja MSCN

Przed podaniem obrazu do sieci stosowana jest **lokalna normalizacja** wzorowana na modelu Mean Subtracted Contrast Normalized (MSCN), zgodna z równaniami 1–3 z pracy referencyjnej:

$$\hat{I}(x,y) = \frac{I(x,y) - \mu(x,y)}{\sigma(x,y) + C}$$

gdzie $\mu$ i $\sigma$ liczone są w oknie 3×3, a $C = 1/255$ zapobiega dzieleniu przez zero. Wariancja wyznaczana jest z tożsamości: $\sigma^2 = E[I^2] - \mu^2$.

Normalizacja uniezależnia reprezentację obrazu od globalnej jasności i kontrastu, podkreślając lokalne wzorce tekstury i krawędzi — cechy kluczowe dla detekcji artefaktów.

W pipeline `tf.data` normalizacja zaimplementowana jest w TensorFlow (operacje `avg_pool2d`) dla wydajności GPU.

### 5.2. Architektura sieci

Klasyfikator `cnn_downscale_classifier` oparty jest na **6 blokach konwolucyjnych** (Conv→BN→MaxPool), identycznych z architekturą z Tabeli 1 pracy Frackiewicz et al., z jedną modyfikacją: głowica zmieniona z regresji (1 neuron, aktywacja liniowa) na **klasyfikację wieloklasową** (softmax, N_CLASSES = 10).

| Blok | Filtry Conv (3×3) | Rozmiar mapy cech po MaxPool 2×2 |
|---|---|---|
| 1 | 32 | 128×128 |
| 2 | 64 | 64×64 |
| 3 | 128 | 32×32 |
| 4 | 256 | 16×16 |
| 5 | 256 | 8×8 |
| 6 | 256 | 4×4 |
| Flatten | — | 4096 |
| Dense (ReLU) | n_neurons (Optuna) | — |
| Dropout | rate (Optuna) | — |
| **Dense (Softmax)** | **N_CLASSES = 10** | — |

**Klasy:** CTRL, 2×, 3×, 4×, 5×, 6×, 7×, 8×, 9×, 10×

**Optymalizator:** Adam z krokiem uczenia strojonym przez Optuna.  
**Funkcja straty:** `sparse_categorical_crossentropy`.

### 5.3. Pipeline danych (tf.data)

Zamiast ładować cały dataset do RAM (potencjalnie ~11 GB), zaimplementowany jest leniwy (`lazy`) pipeline `tf.data.Dataset` wczytujący obrazy z dysku na bieżąco — w pamięci przebywa tylko aktualny batch. Zastosowano:

- `num_parallel_calls=AUTOTUNE` — równoległe wczytywanie i preprocessing
- `prefetch(AUTOTUNE)` — nakładanie wczytywania kolejnego batcha na czas treningu
- `shuffle` z ustalonym seedem dla powtarzalności

### 5.4. Tuning hiperparametrów — Optuna + TPE

Do optymalizacji trzech hiperparametrów stosowana jest **bayesowska optymalizacja** z próbnikiem TPE (Tree-structured Parzen Estimator), zgodnie z zakresami z Tabeli 2 pracy:

| Hiperparametr | Zakres |
|---|---|
| `n_neurons` (głowica Dense) | 500 – 1000 |
| `eta` (learning rate) | 1e-5 – 1e-3 (log-uniform) |
| `dropout_rate` | 0.0 – 0.8 |

**Strategia tuningu:**
- Liczba triali: `N_OPTUNA_TRIALS = 10`
- Epoki per trial: `EPOCHS_TUNING = 25`
- Metryka optymalizowana: `val_accuracy` (maksymalizacja)
- **Przycinanie (pruning):** słabe triale ucinane `MedianPruner` (3 triale startowe, 5 kroków warmup)
- **Early stopping** w każdym trialu: patience = 8 epok

Najlepsze parametry zapisywane są do `best_params.json`.

### 5.5. Finalny trening

Model finalny trenowany jest z najlepszymi hiperparametrami przez maksymalnie `EPOCHS_FINAL = 80` epok, z callbackami:

- `EarlyStopping` (patience=8, monitor=`val_accuracy`)
- `ReduceLROnPlateau` (factor=0.5, patience=4, min_lr=1e-6)

Model zapisywany jest w formacie `.keras` do katalogu `K_out/`.

### 5.6. Ewaluacja

Ewaluacja obejmuje:

- **Dokładność ogólna** (accuracy) na zbiorach `val` i `test`
- **Raport klasyfikacji per klasa** (precision, recall, F1-score)
- **Macierz pomyłek** (znormalizowana, wizualizacja heatmap)
- **Krzywe uczenia** (train vs. val accuracy po epokach)
- **Accuracy per kombinacja** (para algorytmów × współczynnik skali) — pozwala wykryć systematyczne błędy klasyfikatora dla konkretnych typów degradacji

---

## 6. Dodatkowe aspekty techniczne

### 6.1. Infrastruktura i zarządzanie pamięcią GPU

Skonfigurowane dynamiczne przydzielanie pamięci GPU (`set_memory_growth = True`), co zapobiega zajmowaniu całości VRAM przez TensorFlow przy starcie. Modele ESRGAN (PyTorch) działają w trybie half-precision (`half=True`) gdy dostępne jest CUDA.

### 6.2. Odtwarzalność

Seed `42` ustawiony jest globalnie dla: NumPy (`np.random.seed`), TensorFlow (`tf.random.set_seed`), Optuna (`TPESampler(seed=42)`), oraz podziału danych (`random.Random(SPLIT_SEED)`). Każdy obraz źródłowy posiada własny deterministyczny RNG do cropu, bazujący na SHA-1 nazwy pliku.

### 6.3. Stress test

Wydzielony zbiór `stress_test` zawiera materiały syntetyczne, rendery i rysunki, które celowo odbiegają od dystrybucji treningowej. Pozwala to na ocenę zachowania modelu w warunkach dystrybucji wychodzących poza domain treningowy (out-of-distribution robustness).

### 6.4. Inferencja na nowych obrazach

Finalna funkcja `predict_downscale(image_path)` przyjmuje dowolny obraz, stosuje lokalną normalizację MSCN i zwraca:
- przewidywaną klasę skali downscalingu
- indeks klasy
- rozkład prawdopodobieństwa po wszystkich klasach

---

## 7. Podsumowanie techniczne

| Aspekt | Wartość / Opis |
|---|---|
| Liczba obrazów wejściowych | 1000 (.NEF) |
| Liczba par algorytmów | 7 (w tym 1 CTRL) |
| Liczba współczynników skali | 10 (1× – 10×) |
| Łączna liczba próbek w datasecie | ~70 000 (1000 × 10 × 7) |
| Rozmiar patcha | 256×256 px (RGB, PNG) |
| Podział danych | 70% train / 15% val / 15% test |
| Liczba klas klasyfikatora | 10 |
| Architektura CNN | 6× (Conv→BN→MaxPool) + Dense + Dropout + Softmax |
| Preprocessing | Lokalna normalizacja MSCN (okno 3×3) |
| Tuning hiperparametrów | Optuna + TPE (10 triali, pruning median) |
| Epoki treningu finalnego | maks. 80 (early stopping patience=8) |
| Framework DL | TensorFlow / Keras |
| Framework SR (AI) | PyTorch + Real-ESRGAN |
