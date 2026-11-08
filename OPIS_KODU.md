# Opis kodu — P1 „Ukryta łatka”

Kod jest w dwóch folderach, tak jak przy uruchamianiu:
- `bumas1_kod/` — generator danych i pierwszy własny U-Net (uruchamiane z folderu z `p1_student/`),
- `kolega_kod/` — rozwiązanie bazowe kolegi + moje rozszerzenia (skrypty importują się nawzajem, muszą leżeć razem z `exp/`).

Oznaczenia: **[moje]** — napisane w tym projekcie przeze mnie, **[kolega]** — rozwiązanie bazowe,
**[kolega + zmiany]** — kod kolegi z moimi poprawkami.

---

## bumas1_kod/ — dane

### `generate_v2.py` [moje] — generator obrazków z łatką
Tworzy nowe obrazki treningowe tak samo jak organizator.
- `load_backbone` — ładuje atakowany ResNet-18.
- `jpeg_u8`, `jpeg_t`, `through_jpeg` — kompresja JPEG; `through_jpeg` to BPDA: w przód prawdziwy JPEG,
  wstecz gradient tożsamościowy, dzięki czemu atak działa także po zapisie pliku.
- `pred_saved` — klasa modelu na obrazku „po zapisie” (zaokrąglenie do uint8 + JPEG).
- `d4_img`, `d4_box` — odbicie bazy (i ramek) przed atakiem.
- `rand_rect` — losowy prostokąt: 8–15% pola, proporcje 1:2–2:1, bez nakładania (sprawdzane po zaokrągleniu).
- `attack_linf` — PGD bez celu, PGD z celem, MI-FGSM; szum tylko w masce prostokąta, ε osobne dla każdego obrazka.
- `attack_cw` — Carlini–Wagner L2, 100 kroków, zapamiętuje **najmniejszy** szum, który działa po zapisie, |δ| ≤ 20/255.
- `decoy_noise` — wabik: szum jednostajny, gaussowski lub ±ε.
- `main` — dla każdego zadania: odbicie bazy, prostokąty, atak, **wyszukiwanie binarne najmniejszego ε (1–20/255)**,
  dodanie wabików, zapis JPEG; obrazek jest odrzucany, jeśli łatka nie myli modelu albo wabik go myli.
  Siła i liczba wabików losowane z rozkładu prawdziwego treningu. Bazy z walidacji są pomijane (brak wycieku).
  `--part` pozwala liczyć w częściach.

### `polacz.py` [moje] — skleja `train_labels_part*.csv` z części generatora w jeden `train_labels.csv`.

### `porownaj_dane.py` [moje] — porównanie danych z generatora z danymi organizatora
Udziały ataków i rodzajów obrazków, rozkład ε, jakość JPEG, typy i liczba wabików oraz „widoczność” łatki
(stosunek szumu wysokiej częstotliwości w ramce do reszty obrazka).

### `solve.py` [kolega + zmiany] — pierwszy własny U-Net (`moj`)
Uczy U-Net (szerokość 48) na 24 cechach. Moje zmiany: zapis kopii sieci co 10 epok (`--save_every`),
douczanie / sama ocena (`--init`, `--epochs 0`), diagnoza błędów (`diagnose`), szybsze czytanie danych
dużymi kawałkami w tle (wersja `solve_fast.py`).

---

## kolega_kod/ — model i odpowiedzi

### `ensemble4.py` [kolega] — wspólne funkcje
- `build_features` — 24 cechy na piksel: residuum (obraz − mediana 3×3), energia bloków DCT 8×8 (ślady JPEG),
  gradient ResNetu względem obrazu, korelacja szumu z gradientem i in.
- `PatchUNet` — U-Net: 3 klasy na piksel (tło / łatka / wabik) + liczba „czy jest łatka”.
- `d4_np`, `d4_t`, `d4_t_inv` — 8 obrotów/odbić (augmentacja i test).
- `label_mask`, `read_train`, `load_u8` — etykiety i wczytywanie danych.

### `exp/causal_map.py`, `exp/causal_cache.py` [kolega] — **mapa przyczynowa**
- `win_masks` — okna 14–28 px przesuwane co 4 px.
- `causal_maps` — w każdym oknie obraz zastępowany wersją po JPEG q30 / q50, mierzony spadek marginesu ResNetu;
  wynik rozkładany na piksele → 2 mapy 64×64.
- `compute_causal` — liczenie z zapisem co 100 obrazków (można przerwać i wznowić).

### `train_v3.py` [kolega] — U-Net v3 z mapą przyczynową jako wejściem
- `Stack` — skleja 24 cechy i 2 znormalizowane mapy przyczynowe w 26 kanałów.
- `train` / `infer` — trening (OneCycle, 8 symetrii) i mapy z uśrednieniem 8 symetrii.
Użyte przeze mnie z danymi z mojego generatora (15 tys. i 50 tys. obrazków, 2–3 sieci).

### `train_v4.py` [moje] — U-Net v4 / v5 (douczanie v3)
- `BoxUNet` — `PatchUNet` + druga „głowa”: każdy piksel przewiduje odległość do lewej, górnej, prawej i dolnej
  krawędzi (jak w detektorze FCOS).
- `d4_box`, `box_targets`, `iou_loss` — cele odległości po obrocie i strata IoU.
- `dist_to_coords`, `coords_inv`, `vote` — zamiana odległości na ramkę, odwrócenie obrotu, głosowanie pikseli.
- `train` — większa waga pikseli przy krawędzi (`--edge_w`), prawdziwe obrazki częściej (`--real_rep`),
  **słabe łatki ε ≤ 4/255 częściej (`--eps_boost`)**, opcja `--full`.
- `report` — IoU na walidacji: z mapy, z mapy × przyczynowej, z głosowania, średnia.

### `solve_v2.py` [kolega + zmiany] — od map do zgłoszenia
- `cached_unet` — mapy kilku zestawów sieci (`--unet unet,moj,unet3a,...`) uśrednione.
- `analyse` — mapa łatki × mapa przyczynowa → prostokąt o najwyższym oczekiwanym IoU + cechy obrazka.
- `fit_and_score` — decyzja ramka / 0,0,0,0 (regresja logistyczna, próg z walidacji 5-fold).
  Moja zmiana: `--zero_frac` (stały odsetek odpowiedzi 0,0,0,0).
- `--refine` — proste dopracowanie krawędzi (reguła z 3 cech, `exp/refine_v2.py`).

### `eksport_moj.py` [moje] — mapy sieci `moj` w formacie `solve_v2` (`cache_v2/moj_*.npz`).

### `refine_ml.py` [moje] — wybór krawędzi modelem ML
- `var_feats` — 27 cech każdego wariantu ramki: oczekiwane IoU na mapach, średnia w ramce, kontrast z pierścieniem
  2 px na zewnątrz, spadek marginesu ResNetu (i jego ranga w obrazku), pole, proporcje, przesunięcia krawędzi.
- `build_table` — ~2000 wariantów na ramkę (każda krawędź −3…+3 px), spadek marginesu z `refine_v2.rect_margins`.
- `kolega_ctx`, `v3_ctx` — mapy sieci dla obrazków z `synth_v2`, których **żadna sieć nie widziała**.
- `fit` / `pick` — `HistGradientBoostingRegressor` przewiduje IoU wariantu, wybierany jest najlepszy.
- uczciwa ocena na 400 prawdziwych obrazkach walidacji, potem test z zapisem postępu.

### `apply_box.py` [moje] — 75% ramki z ML + 25% ramki z głosowania v4 (`mix`), z oceną wag na walidacji.

### Analiza [moje]
- `diag_v2.py` — gdzie giną punkty: przeoczenia, wabiki, złe miejsce, niedokładna ramka; wynik wg ataku, ε, liczby wabików.
- `refine_test.py` — wyrocznia: najlepszy możliwy wariant ramki (sufit wyboru krawędzi).
- `refine_ml_cv.py` — czy model wyboru krawędzi lepiej uczyć na prawdziwych czy syntetycznych obrazkach.

---

## Przepływ danych

```
p1_student ──► generate_v2.py ──► synth_v2 (55 tys.)
                                     │
          ┌──────────────────────────┼─────────────────────────────┐
          ▼                          ▼                             ▼
   solve.py (moj)        train_v3.py (v3, 26 kanałów)    train_v4.py (v4, v5)
          │                          │                             │
   eksport_moj.py                    │                             │
          └────────────► cache_v2/*.npz (mapy każdej sieci) ◄───────┘
                                     │      + causal_*.npy (exp/causal_cache.py)
                                     ▼
                     solve_v2.py  → decyzja ramka / 0,0,0,0 + ramka startowa
                                     ▼
                     refine_ml.py → wybór krawędzi (ML)
                                     ▼
                     apply_box.py → +25% krawędzi v4  ──► sub_A_ml_box25.csv (0,6808)
```

## Literatura
- Athalye, Carlini, Wagner, *Obfuscated Gradients Give a False Sense of Security* (BPDA), 2018 — https://arxiv.org/abs/1802.00420
- Carlini, Wagner, *Towards Evaluating the Robustness of Neural Networks*, 2017 — https://arxiv.org/abs/1608.04644
- Madry i in., *Towards Deep Learning Models Resistant to Adversarial Attacks*, 2018 — https://arxiv.org/abs/1706.06083
- Dong i in., *Boosting Adversarial Attacks with Momentum*, 2018 — https://arxiv.org/abs/1710.06081
- Zeiler, Fergus, *Visualizing and Understanding Convolutional Networks*, 2014 — https://arxiv.org/abs/1311.2901
- Ronneberger i in., *U-Net*, 2015 — https://arxiv.org/abs/1505.04597
- Tian i in., *FCOS*, 2019 — https://arxiv.org/abs/1904.01355
- Jiang i in., *IoU-Net*, 2018 — https://arxiv.org/abs/1807.11590
- Ke i in., *LightGBM* (gradient boosting na histogramach), 2017 — https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree
