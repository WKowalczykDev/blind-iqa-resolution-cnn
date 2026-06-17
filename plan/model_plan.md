# Specyfikacja techniczna modelu IQA — wersja końcowa

## 1. Inwentaryzacja Twoich danych

Z Twojego `manifest.jsonl` mam:

**Cechy NR (input modelu — dostępne na inference)** — z `fft_result`:
- `hf_energy_ratio` (1)
- `mf_energy_ratio` (1)
- `spectrum_entropy` (1)
- `radial_profile_decile` (10)

→ **13 cech tabelarycznych** + `spectral_energy_total` POMIJAMY (zależy od jasności sceny, nie od jakości).

**Targety (etykiety — z `quality_metrics`)**:
- `pseudo_mos` (główny target)
- `psnr`, `ssim`, `ms_ssim`, `lpips_alex` (auxiliary heads)

**⚠️ Ostrzeżenie**: `crop.entropy_bits` i `crop.noise_var_laplacian` są policzone na ORYGINALE (przed degradacją). NIE używaj ich jako inputu — to leak z ground truth. Model NR widzi tylko zdegradowany obraz.

**💡 Warto dodać** (tani upgrade, +3 cechy NR): policz `entropy_bits`, `noise_var_laplacian`, `mean_gradient` na obrazie wynikowym (`result_256`) i dodaj do manifestu jako `patch_stats_result`. Możesz to zrobić post-processem bez ponownego puszczania pipeline'u — wystarczy iteracja po PNG-ach z folderu `train/val/test`.

---

## 2. Architektura modelu — `HybridIQA-CNN-1024`

### 2.1 Branch obrazowy (CNN backbone)

| # | Layer | Output Shape | Info | Stride/Padding |
|---|---|---|---|---|
| 1 | Input | 256 × 256 × 3 | RGB, normalizacja [0,1] | — |
| 2 | Conv2D | 256 × 256 × 32 | 32 kerneli 3×3 | 1 / same |
| 3 | BatchNorm2D | 256 × 256 × 32 | — | — |
| 4 | ReLU | 256 × 256 × 32 | — | — |
| 5 | MaxPool2D | 128 × 128 × 32 | 2×2 | 2 / valid |
| 6 | Conv2D | 128 × 128 × 64 | 64 kerneli 3×3 | 1 / same |
| 7 | BatchNorm2D | 128 × 128 × 64 | — | — |
| 8 | ReLU | 128 × 128 × 64 | — | — |
| 9 | MaxPool2D | 64 × 64 × 64 | 2×2 | 2 / valid |
| 10 | Conv2D | 64 × 64 × 128 | 128 kerneli 3×3 | 1 / same |
| 11 | BatchNorm2D | 64 × 64 × 128 | — | — |
| 12 | ReLU | 64 × 64 × 128 | — | — |
| 13 | MaxPool2D | 32 × 32 × 128 | 2×2 | 2 / valid |
| 14 | Conv2D | 32 × 32 × 256 | 256 kerneli 3×3 | 1 / same |
| 15 | BatchNorm2D | 32 × 32 × 256 | — | — |
| 16 | ReLU | 32 × 32 × 256 | — | — |
| 17 | MaxPool2D | 16 × 16 × 256 | 2×2 | 2 / valid |
| 18 | Conv2D | 16 × 16 × 256 | 256 kerneli 3×3 | 1 / same |
| 19 | BatchNorm2D | 16 × 16 × 256 | — | — |
| 20 | ReLU | 16 × 16 × 256 | — | — |
| 21 | MaxPool2D | 8 × 8 × 256 | 2×2 | 2 / valid |
| 22 | AdaptiveAvgPool2D | 2 × 2 × 256 | output_size=(2,2) | — |
| 23 | Flatten | 1024 | — | — |
| 24 | Dense | 512 | FC | — |
| 25 | ReLU | 512 | — | — |
| 26 | Dropout | 512 | rate = 0.5 | — |

→ wyjście: **embedding 512-d**

### 2.2 Branch cech tabelarycznych (Feature MLP)

| # | Layer | Output Shape | Info |
|---|---|---|---|
| 1 | Input | 13 | NR cechy z `fft_result` (po standaryzacji) |
| 2 | Dense | 64 | FC |
| 3 | ReLU | 64 | — |
| 4 | Dense | 32 | FC |
| 5 | ReLU | 32 | — |

→ wyjście: **embedding 32-d**

### 2.3 Fusion + multi-task head

| # | Layer | Output Shape | Info |
|---|---|---|---|
| 1 | Concat | 544 | [512 (CNN) ‖ 32 (MLP)] |
| 2 | Dense | 128 | FC |
| 3 | ReLU | 128 | — |
| 4 | Dropout | 128 | rate = 0.3 |
| 5 | Dense | 5 | multi-task: pseudo_mos, psnr_n, ssim, ms_ssim, lpips_n |
| 6 | Sigmoid | 5 | wszystkie targety w [0,1] |

**Liczba parametrów**: ~1.6M. Zmieści się na T4 z marginesem.

---

## 3. Preprocessing

### 3.1 Obraz
```python
img = PIL.Image.open(path).convert('RGB')          # 256×256×3
img = np.array(img).astype(np.float32) / 255.0     # [0,1]
img = torch.from_numpy(img).permute(2, 0, 1)       # CHW
# BEZ normalizacji ImageNet — uczymy od zera
```

### 3.2 Cechy tabelaryczne — KRYTYCZNE
Wartości w `radial_profile_decile` różnią się o rzędy wielkości (pierwszy bin ≈ 0.998, dziesiąty ≈ 1e-7). Bez tego model nauczy się tylko pierwszej cechy.

```python
# Krok 1: log1p na cechach o szerokim zakresie
for k in ['hf_energy_ratio', 'mf_energy_ratio',
          'spectrum_entropy'] + radial_decile_keys:
    feats[k] = np.log1p(feats[k])

# Krok 2: standaryzacja per-cecha (mean/std liczone TYLKO na trainie)
feats = (feats - train_mean) / (train_std + 1e-8)
```

Statystyki `train_mean`, `train_std` zapisz do `feature_scaler.json` razem z modelem.

### 3.3 Augmentacja
Wyłącznie:
- `RandomHorizontalFlip(p=0.5)`
- `RandomVerticalFlip(p=0.5)`

**ZAKAZANE**: rotacje o losowy kąt, RandomCrop, ColorJitter, GaussianBlur. Każda z nich zmienia/nakłada nowe degradacje na sygnał, który sieć ma rozpoznawać.

---

## 4. Funkcja straty

Multi-task weighted MSE:

```python
def loss_fn(pred, target):
    # weights: [pseudo_mos, psnr_n, ssim, ms_ssim, lpips_n]
    w = torch.tensor([1.0, 0.3, 0.3, 0.3, 0.5], device=pred.device)
    per_sample = F.mse_loss(pred, target, reduction='none')   # (B, 5)
    return (per_sample * w).mean()
```

Wagi tłumaczę: `pseudo_mos` ma być głównym celem (waga 1.0), LPIPS jest najbardziej "perceptual" więc dostaje 0.5, reszta to regularyzacja przez auxiliary signal (0.3).

Po 10 epokach możesz sprawdzić, czy `psnr_n` head konwerguje — jeśli tak, jego waga przestaje być potrzebna, i można ją zmniejszyć do 0.1.

---

## 5. Training protocol

```python
# Optymalizator
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

# LR schedule
scheduler = CosineAnnealingLR(optimizer, T_max=50)

# Hiperparametry
BATCH_SIZE   = 16        # T4 16GB, mieści się; jeśli OOM → 8
EPOCHS       = 50
GRAD_CLIP    = 1.0       # clip_grad_norm_, na wypadek wybuchu
EARLY_STOP   = 10        # patience na walidacji (PLCC + SROCC) / 2
```

### 5.1 Ewaluacja — NIE używaj MSE jako kryterium walidacyjnego

Standard w IQA to:
- **PLCC** (Pearson Linear Correlation Coefficient) — czy wartości się zgadzają
- **SROCC** (Spearman Rank Order Correlation) — czy ranking się zgadza
- **KROCC** (Kendall) — bonus, ale wystarczy PLCC+SROCC

```python
from scipy.stats import pearsonr, spearmanr

def validate(model, loader):
    preds, gts = [], []
    model.eval()
    with torch.no_grad():
        for img, feat, tgt in loader:
            p = model(img.cuda(), feat.cuda())[:, 0]   # tylko pseudo_mos
            preds.extend(p.cpu().numpy())
            gts.extend(tgt[:, 0].numpy())
    return {
        'plcc':  pearsonr(preds, gts)[0],
        'srocc': spearmanr(preds, gts)[0],
        'mse':   ((np.array(preds) - np.array(gts))**2).mean(),
    }

# Checkpoint na max((plcc + srocc) / 2)
```

**Oczekiwane wyniki przy 96k danych**:
- PLCC walidacyjny > 0.92, SROCC > 0.91 → SOTA-ish
- PLCC > 0.85, SROCC > 0.83 → solidny baseline
- PLCC < 0.75 → coś jest źle (sprawdź standaryzację cech)

---

## 6. Wymagana ilość danych

### 6.1 Twoja sytuacja

```
8000 RAW × 4 pary × 3 scale factors = 96 000 sampli przetworzonych
+ 8000 oryginałów (ich nie używamy do treningu — patrz niżej)
```

Po splicie 70/15/15 **per source image**:
| Split | Sources | Samples |
|---|---:|---:|
| train | ~5 600 | ~67 200 |
| val | ~1 200 | ~14 400 |
| test | ~1 200 | ~14 400 |

### 6.2 Progi (wg papera CNN-IQA który podlinkował Kakalko + literatura)

| Liczba sampli (train) | Spodziewany poziom | Komentarz |
|---|---|---|
| < 5 000 | nie działa from scratch | trzeba transfer learning z ImageNet |
| 5–15 000 | słaby baseline | PLCC ~0.7–0.8 |
| 15–40 000 | solidny | PLCC ~0.85–0.92, **próg "papierowy"** |
| 40 000+ | SOTA-territory | PLCC > 0.92 |

**Twoje ~67k** = komfortowo w SOTA-territory. Nie potrzebujecie augmentacji ani transfer learningu.

### 6.3 Minimalna konfiguracja jeśli ucinasz

Jeśli z jakichś powodów (czas, Drive, koszt) musisz ograniczyć:
- **Minimum**: 2000 source images (24k sampli) → ~16 800 train. Słaby baseline ale działa.
- **Sweet spot**: 4000 source images (48k sampli) → ~33 600 train. Solidne wyniki.
- **Pełna pula**: 8000 → SOTA.

### 6.4 Czego NIE wrzucać do treningu

- **Wpisy `type="oryginal"`**: wszystkie mają `pseudo_mos=1.0` → degenerują target, model nauczy się mówić "zawsze 1" na łatwych patchach. Filtruj `if e['type']=='oryginal': continue` w `__getitem__`.
- **`is_stress_test=True`**: idą tylko do osobnej ewaluacji końcowej (szachownice, AI-gen, rysunki).
- **Sample z `crop.fallback_used=True`**: opcjonalnie odfiltruj — to są patche które nie przeszły progów entropii. Jeśli jest ich <5% datasetu, możesz zostawić.

---

## 7. Protokół ewaluacji końcowej

Po treningu uruchom ewaluację na 4 zbiorach osobno i raportuj PLCC+SROCC dla każdego:

| Zbiór | Co mierzy |
|---|---|
| `test` (zwykły) | generalizację na nowe zdjęcia naturalne |
| `test` ∩ `pair_id=1` | jak model radzi sobie z aliasingiem |
| `test` ∩ `pair_id=3` | jak radzi sobie z halucynacjami AI |
| `stress_test` | obrazy poza dystrybucją (szachownice, rysunki) |

Plus tabela per `scale_factor` (×2 / ×4 / ×8) — pokaż, że model nie zawala się na ekstremalnych SF.

---

## 8. Co dostarczasz na końcu (artefakty)

```
model/
├── checkpoint_best.pt          # state_dict + config
├── feature_scaler.json         # mean/std cech tabelarycznych
├── pipeline_config.json        # config datasetu (z KROK 3)
├── training_log.csv            # loss/PLCC/SROCC per epoch
├── eval_results.json           # PLCC/SROCC per split & per pair
└── architecture.txt            # `print(model)` dla referencji
```

---

## 9. Kolejność prac (rekomendowana)

1. Napisz `IQADataset` + `feature_scaler` (test na N=10 sampli).
2. Zaimplementuj `HybridIQA` zgodnie z tabelami w sekcji 2.
3. Pętla treningowa z PLCC/SROCC validation — puść na 5 epokach na małym podzbiorze (1000 sampli), żeby zweryfikować że loss spada i metryki rosną.
4. Pełen run 50 epok na 67k sampli treningowych — na T4 ~6-10h.
5. Ewaluacja per sekcja 7.

Człowiek-test: pokaż 20 losowych patchy trzem osobom, niech ocenią 1-10. Sprawdź czy wasze pseudo_mos i predykcje modelu korelują z ich ocenami. Jeśli model > pseudo_mos w korelacji z ludźmi → model nauczył się czegoś więcej niż jego etykiety, sukces.