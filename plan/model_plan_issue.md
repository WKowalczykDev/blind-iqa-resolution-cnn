1. Teraz jest 15 cech - został dodane w patch_stats_result
2. Brak listy LEAKÓw:

| Pole | Czy leak? | Dlaczego |
|------|-----------|----------|
| `crop.entropy_bits` | **TAK** | Obliczone na ORYGINALE (przed degradacją) |
| `crop.noise_var_laplacian` | **TAK** | Obliczone na ORYGINALE |
| `fft_orig` | **TAK** | FFT oryginału — idealna referencja |
| `fft_low_res` | NIE | To wynik downscale, model NR "widzi" podobne rzeczy |
| `fft_result` | NIE | To FFT obrazu wynikowego — model NR ma dostęp |
| `patch_stats_result` | NIE | Obliczone na `result_256` — model NR widzi to samo |
