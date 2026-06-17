**Metryki — serce pipeline'u v0.5**

Wszystko kręci się wokół `manifest.jsonl`. Każdy sample ma kompletny profil jakościowy, który pozwoli modelowi nie tylko uczyć się rekonstrukcji, ale też rozumieć *co* zostało zepsute i *jak bardzo*.

---

### 1. Full-Reference (FR) — benchmark vs oryginał 256×256

Mierzone wyłącznie dla `type=processed`, porównują wynik z `orig_256`.

| Metryka | Zakres | Co mierzy | Uwagi |
|---|---|---|---|
| **PSNR** | [0, ∞] dB | Błąd średniokwadratowy w skali logarytmicznej. Wyżej = mniejsza różnica piksel-piksel. | Dla oryginału = ∞. Słabo koreluje z percepcją, ale świetnie wykrywa "duchowate" artefakty i przesunięcia. |
| **SSIM** | [0, 1] | Podobieństwo strukturalne (luminancja, kontrast, korelacja). | Lepszy niż PSNR przy rozmyciu, gorszy przy szumie. |
| **MS-SSIM** | [0, 1] | Wieloskalowy SSIM — sprawdza zgodność na 5 rozdzielczościach. | Bardziej zgodny z ludzkim wzrokiem niż zwykły SSIM. |
| **LPIPS** (AlexNet) | [0, ~1] | Odległość w przestrzeni cech głębokiej sieci (AlexNet). **Niżej = lepiej.** | Najbliższy percepcji. Wykrywa halucynacje AI, które wyglądają "ładnie" ale są niezgodne z oryginałem. |
| **pseudo_MOS** | [0, 1] | Syntetyczna ocena jakości: `0.3·norm(PSNR) + 0.3·SSIM + 0.4·(1−LPIPS/0.7)` | Łączy pikselową i perceptualną wiedzę w jeden skalar. Target do ewentualnej regresji jakościowej modelu. |

**Kluczowa obserwacja:** LPIPS jest tutaj nadrzędny. Real-ESRGAN osiąga wysoki SSIM, ale LPIPS wykryje, czy model nie wymyślił tekstury zamiast ją odtworzyć.

---

### 2. Cechy spektralne (FFT) — widmo mocy 2D

Transformata Fouriera na obrazie w skali szarości. Pozwala zobaczyć utratę informacji w dziedzinie częstotliwości, co metryki pikselowe przegapiają.

| Cecha | Definicja | Po co |
|---|---|---|
| `hf_energy_ratio` | Energia w pierscieniu r > W/4 / całkowita energia | Ile wysokich częstotliwości (detale, krawędzie) przetrwało pipeline. |
| `mf_energy_ratio` | Energia W/8 ≤ r ≤ W/4 / całkowita energia | Średnie częstotliwości — tekstury, wzory. |
| `spectrum_entropy` | Entropia rozkładu mocy w widmie | Czy widmo jest "bogate" (różnorodne częstotliwości) czy "wypalone" (po agresywnym downsamplingu). |
| `radial_profile_decile` | Średnia moc w 10 pierścieniach od środka widma, znormalizowana | Pełny profil degradacji częstotliwościowej. |

**Delta FFT (`fft_delta`):**
- `hf_loss_ratio` = `(hf_orig − hf_result) / hf_orig` — procentowa utrata detali. Dla NN→NN rośnie monotonicznie ze skalą (sanity check).
- `spectrum_entropy_diff` — czy upscaling "wymyślił" nowe częstotliwości (halucynacje AI) czy tylko odtworzył stare.

---

### 3. Cechy No-Reference (NR) — model widzi tylko zdegradowany obraz

**To najważniejsza nowość v0.5.** W realnym zastosowaniu model nie ma dostępu do oryginału. Te cechy pozwalają mu oszacować jakość wejścia i ewentualnie dostosować strategię rekonstrukcji.

| Cecha | Jak liczona | Co mówi |
|---|---|---|
| `entropy_bits` | Entropia Shannona histogramu 256-binowego obrazu szarości | Ile informacji zawiera obraz. Wysoka = dużo detali. Niska = rozmycie / płaskie obszary. Zakres: [0, 8.5]. |
| `noise_var_laplacian` | Wariancja odpowiedzi filtru Laplace'a na obraz | Proxy ostrości i szumu. Wysoka = ostry obraz lub dużo szumu. Niska = mocno rozmyty. |
| `mean_gradient` | Średnia magnituda gradientu Sobela (x+y) | Gęstość krawędzi. Wysoka = dużo konturów i tekstur. |

**Gdzie są mierzone:**
- **`orig_256`** → `crop.mean_gradient` (diagnostyka wyboru patcha) + `patch_stats_result` (baseline oryginału)
- **`result_256`** → `patch_stats_result` — **to widzi model jako NR input**
- **`low_res`** → `patch_stats_low_res` — diagnostyka: porównanie przed/po upscaling (np. czy Real-ESRGAN faktycznie podniósł entropię)

---

### 4. Mapowanie: co jest mierzone na czym

| Obiekt | FR | FFT | NR |
|---|---|---|---|
| `orig_256` | — (baseline) | `fft_orig` | `patch_stats_result` |
| `low_res` (po downscale) | — | `fft_low_res` | `patch_stats_low_res` |
| `result_256` (po upscale) | PSNR/SSIM/MS-SSIM/LPIPS/pseudo_MOS | `fft_result` + `fft_delta` | `patch_stats_result` |

---

### 5. Co model dostaje do pracy

Z `manifest.jsonl` można zbudować dataloader, który podaje modelowi:

**Wejście (X):**
- Obraz `result_256` (lub `low_res` jeśli trenujemy end-to-end SR)
- Cechy NR: `entropy_bits`, `noise_var_laplacian`, `mean_gradient` z `patch_stats_result`
- Metadata: `scale_factor`, `pair_id` (typ degradacji)

**Targety (y) — opcjonalnie wielozadaniowo:**
- Rekonstrukcja pikselowa: `orig_256`
- Regresja jakości: `pseudo_MOS` lub `lpips_alex`
- Detekcja artefaktów: klasa z `pair_group`

**W skrócie:** manifest nie jest tylko logiem — to strukturyzowany dataset z bogatymi cechami, które pozwalają nauczyć model nie tylko "powiększać", ale też *rozumieć* jakość tego co robi.