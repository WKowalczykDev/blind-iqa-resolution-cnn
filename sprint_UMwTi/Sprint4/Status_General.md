**Pipeline SR-Dataset v0.5 — changelog & status**

---

### 🔧 Architektura
- **Przepływ:** Google Drive → Colab (batch) → ZIP → Google Drive
- Colab ma własny dysk `/content/` — przetwarzanie lokalne, ewakuacja dopiero na końcu jako ZIP
- Deterministyczny split **przed** przetwarzaniem (seed=42) — zero przecieków train/val/test

---

### 📦 Output
```
1K_out/
├── train/ | val/ | test/ | stress_test/
├── manifest.jsonl   # główny rejestr wszystkich sampli + metryki
├── splits.json      # mapowanie plików → zbiory
└── pipeline_config.json
```

---

### ✂️ Smart Crop (nowe)
- Losuje 16 kandydatów 320×320
- Wybiera patch z max entropią Shannona (min 4.5 bitów) + filtr szumu Laplaciana (max 2500)
- Fallback do centrum jeśli żaden nie przejdzie
- Efekt widoczny szczególnie przy AI upscalingu

---

### 🔄 4 pary algorytmów × 3 skale
| Para | Down → Up | Klasa artefaktów |
|:---:|---|---|
| 1 | Nearest → Nearest | aliasing, blokowatość |
| 2 | Box → Bicubic | rozmycie, utrata HF |
| 3 | Bicubic → **Real-ESRGAN** | halucynacje AI |
| 4 | Lanczos → **ESRGAN classic** | wierność vs perceptual |

**Skale:** ×2, ×4, ×8  
**Liczba sampli/obraz:** 1 oryginał + 4 pary × 3 skale = **13**

---

### 📊 Metryki (zapisane do manifest.jsonl)

**Full-Reference** (vs oryginał):
- PSNR, SSIM, MS-SSIM, LPIPS (AlexNet), **pseudo_MOS** (`0.3·PSNR_norm + 0.3·SSIM + 0.4·(1−LPIPS/0.7)`)

**Spektralne (FFT):**
- `hf_energy_ratio`, `mf_energy_ratio`, `spectrum_entropy`, `radial_profile_decile`
- `fft_delta.hf_loss_ratio` — ilościowa utrata detali

**No-Reference (NR) — nowe:**
- `entropy_bits` — entropia histogramu
- `noise_var_laplacian` — ostrość/szum
- `mean_gradient` — gęstość krawędzi (Sobel)

Mierzone na: `orig_256` (baseline), `result_256` (NR input do modelu), `low_res` (diagnostyka).

---

### 🧪 Stress Test
Pliki z nazwami zawierającymi: `szachownica`, `checker`, `ai_gen`, `render`, `anime`, `rysunek`, `drawing` → osobny zbiór `stress_test`

---

### ✅ Sanity Checks v0.5
- Brak przecieków source_id między splitami
- `pseudo_MOS` ∈ [0,1], PSNR oryginału = ∞
- Monotoniczność `hf_loss_ratio` ze skalą (NN→NN)
- Kompletność `patch_stats` + zakresy wartości
- Liczba linii manifestu zgodna z oczekiwaną

---

### ⚙️ Konfig (KROK 3 — jedno miejsce edycji)
- `N` = 10 (test) / 1000 (full)
- `SCALE_FACTORS = [2, 4, 8]`
- `SPLIT_RATIO = (0.70, 0.15, 0.15)`
- Progi crop: `ENTROPY_MIN = 4.5`, `NOISE_VAR_MAX = 2500`

---

**Status:** Pipeline gotowy na pełne DB. Po przerobieniu wszystkich danych będzie wystarczająco materiału do nauki modelu od zera.