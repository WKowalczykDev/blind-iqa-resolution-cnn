## Stworzono pełną funkcjonalność pipeline`u
Pipeline jest gotowy na całe DB

Zapisywany jest plik manifest.jsonl ktory zawiera giga duzo informacji co jak gdzie sie dzieje

Po przerobieniu wszystkich danych będziemy mieli ich wystarczająco żeby wyuczyć model od zera

colab_walk do chodzenia po colabie - colab ma wlasny dysk i to nie jest ten sam ktory wywolujemy na drive.

Dlatego kolejnosc jest taka, ze wywolujemy wewnatrz colaba - batchujemy - pozniej wszystko przekazujemy na dysk pod zipem

jest podzial na zbiory treningowe walidacyjne i testowe - dynamicznie mozna dobierac proporcje

są 4 pary:
Para 1: Nearest Neighbor → Nearest Neighbor 
Para 2: Box Filter → Bicubic 
Para 3: Bicubic → Real-ESRGAN 
Para 4: Lanczos → ESRGAN (classic)

Wszystkie metryki ktore sa wykorzystywane, zeby pozniej zostaly zapisane do modelu. Niektore robione na oryginale, niektóre robione na juz zmienionych zdjeciacg:

## KROK 6b — Metryki jakości, cechy spektralne FFT i cechy NR

  

Inicjalizuje model LPIPS (AlexNet, ładowany raz na GPU) i definiuje funkcje metryk.

  

**Metryki full-reference** (każdy przetworzony patch vs. oryginał):

- `PSNR` — szczytowy SNR [dB]; dla oryginału = ∞

- `SSIM` — strukturalne podobieństwo [0, 1]

- `MS-SSIM` — wieloskalowy SSIM [0, 1]

- `LPIPS` — perceptualna odległość (AlexNet) [0, ~1]; niżej = lepiej

- `pseudo_mos` — łączona ocena: `0.3·PSNR_norm + 0.3·SSIM + 0.4·(1 − LPIPS/0.7)`, zakres [0, 1]

  

**Cechy FFT** (widmo mocy 2D → cechy skalarne):

- `hf_energy_ratio` — udział energii wysokich częstotliwości (r > W/4)

- `mf_energy_ratio` — udział energii średnich częstotliwości (W/8 ≤ r ≤ W/4)

- `spectrum_entropy` — entropia widma

- `radial_profile_decile` — profil radialny w 10 pierścieniach (znormalizowany)

  

**Cechy no-reference** — `patch_stats(img)` na obrazie RGB/gray → 3 cechy skalarne:

- `entropy_bits` — entropia Shannona histogramu 256-binowego [0, 8]; wyższa = więcej informacji

- `noise_var_laplacian` — wariancja filtru Laplace'a; proxy ostrości/szumu

- `mean_gradient` — średnia magnituda gradientu Sobela; proxy gęstości krawędzi

  

Używanie `patch_stats`:

- na `orig_256` → `crop.mean_gradient` (diagnostyka) + `patch_stats_result` dla oryginału

- na `result_256` → `patch_stats_result` (wchodzi do modelu jako NR input)

- na `low_res` → `patch_stats_low_res` (diagnostyka, porównanie przed/po upscalingu)

Jest zaimplementowane inteligentne wycinanie na podstawie entropii wycinkow - dziala niezle - dobrze widac przy algorytmach upscaling z wykorzystaniem modeli


Jest dynamicznie ustalone na 3 stopnie downscalingu x2 x4 x8