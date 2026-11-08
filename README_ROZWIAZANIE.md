# P1 „Ukryta łatka” — rozwiązanie zespołu Kryptonim S

**Najlepszy wynik na tablicy: 0,6808** (plik `sub_A_ml_box25.csv`, 3. miejsce w chwili oddania).
Punkt wyjścia: rozwiązanie bazowe `solve_v2.py` (0,5851), baseline organizatora 0,2200.

---

## 1. Idea w trzech zdaniach

1. **Gdzie jest szum?** Zespół sieci U-Net oznacza każdy piksel jako tło / łatka / wabik. Wejście to nie sam obrazek,
   tylko 24 cechy (szum wysokiej częstotliwości, ślady JPEG, gradient ResNetu) + **2 mapy przyczynowe**.
2. **Czy ten szum oszukuje model?** Mapa przyczynowa: przesuwamy okno po obrazku, w oknie „zmywamy” szum mocnym
   JPEG-iem (q30, q50) i mierzymy spadek pewności ResNetu. Łatka ma minimalne ε, więc jej zmycie przywraca
   prawdziwą klasę; zmycie wabika nic nie zmienia. Dzięki temu **wabik nie jest wskazywany ani razu** (diagnoza).
3. **Dokładna ramka.** Wokół ramki z map sprawdzamy ~2000 wariantów (każda krawędź −3…+3 px), a model
   gradient boosting przewiduje IoU każdego wariantu i wybiera najlepszy.

## 2. Co dało ile (tablica, publiczna połowa testu)

| Krok | Wynik |
|---|---|
| Rozwiązanie bazowe (6 U-Netów + mapa przyczynowa na wyjściu) | 0,5851 |
| + własny U-Net uczony na 55 tys. obrazków z własnego generatora | 0,6081 |
| + U-Net **v3**: mapa przyczynowa jako **wejście** sieci (15 tys. obrazków) | 0,6572 |
| + druga sieć v3 | 0,6594 |
| + wybór krawędzi modelem ML zamiast reguły | 0,6627 |
| + sieci v3 na 50 tys. obrazków | 0,6714 |
| + 25% krawędzi z U-Netu v4 | 0,6747 |
| + ML na zespole z 2× v3 (50 tys.) | 0,6756 |
| + trzecia sieć v3 (50 tys.) | 0,6801 |
| **zespół: 2× v3 (50 tys.) + U-Net v5 (słabe łatki ×4) zamiast 3. sieci v3 — wynik końcowy** | **0,6808** |

Walidacja: 500 prawdziwych obrazków treningowych odłożonych na stałe (permutacja `seed 0`), nigdy nie używanych
w treningu sieci ani jako bazy dla generatora. Wynik zespołu na niej (5-fold dla decyzji ramka/brak): **0,709**.

## 3. Składniki

### 3.1 Generator danych — `bumas1/generate_v2.py`
Nowe obrazki z łatką, zrobione tak jak u organizatora:
- bazy: czyste obrazki i obrazki z samymi wabikami z treningu (**bez** 500 obrazków walidacji), losowe odbicie lewo-prawo,
- 4 ataki z maską prostokąta i gradientem liczonym **przez JPEG** (BPDA): PGD bez celu, PGD z celem
  (cel = jedna z klas 2–5), MI-FGSM (20 kroków), C&W L2 (100 kroków, najmniejszy udany szum, |δ| ≤ 20/255),
- ε = **najmniejsza całkowita** wartość 1–20/255, przy której atak działa po zapisie (wyszukiwanie binarne),
- wabiki: jednostajne, gaussowskie i ±ε; siła i liczba losowane z rozkładu prawdziwego treningu,
- prostokąty 8–15% pola, proporcje 1:2–2:1, bez nakładania; JPEG q 70–95,
- każdy obrazek **sprawdzony**: łatka myli ResNet po zapisie, wabik nie.
Wynik: **55 283 obrazki** w `synth_v2/`. Zgodność z danymi organizatora sprawdzał `porownaj_dane.py`.

### 3.2 Sieci w zespole końcowym (`--unet unet,moj,unet3a,unet3b2,unet5a`)

| Nazwa | Skąd | Wejście | Dane | Uwagi |
|---|---|---|---|---|
| `unet` | 6 sieci z `modele_natrenowane/` (rozwiązanie bazowe) | 24 cechy | prawdziwy trening | szerokość 96 |
| `moj` | `bumas1/solve.py`, 100 epok | 24 cechy | 4,5 tys. prawdziwych + 55 tys. synth | szerokość 48 |
| `unet3a` | `train_v3.py`, 2 sieci, 100 epok | 24 cechy + 2 mapy przyczynowe | 4,5 tys. + 15 tys. synth | szerokość 96 |
| `unet3b2` | `train_v3.py`, 2 sieci, 60 epok | 24 + 2 | 4,5 tys. + 50 tys. synth | szerokość 96 |
| `unet5a` | `train_v4.py --eps_boost 4`, 25 epok (start z v3) | 24 + 2 | jw., łatki ε ≤ 4/255 ×4, prawdziwe ×3 | douczanie |
| `unet4` (tylko krawędzie) | `train_v4.py`, 50 epok (start z v3) | 24 + 2 | jw., prawdziwe ×3 | każdy piksel przewiduje odległość do 4 krawędzi |

Każda sieć: U-Net, 3 klasy na piksel + „czy jest łatka”, strata = entropia krzyżowa + dice + BCE,
augmentacja 8 obrotów/odbić, OneCycle. Mapy zespołu = średnia map wszystkich zestawów, test z 8 symetriami.

### 3.3 Od map do odpowiedzi — `solve_v2.py` + `refine_ml.py` + `apply_box.py`
1. mapa łatki zespołu × mapa przyczynowa → prostokąt o najwyższym oczekiwanym IoU (kształty z regulaminu),
2. **decyzja ramka / 0,0,0,0**: regresja logistyczna na cechach zespołu, próg dobrany na walidacji,
3. **wybór krawędzi (`refine_ml.py`)**: ~2000 wariantów ramki, dla każdego spadek pewności ResNetu po zmyciu,
   zgodność z mapami, kontrast przy brzegu itd.; `HistGradientBoostingRegressor` uczony na 2000 obrazkach
   z `synth_v2`, których **żadna sieć nie widziała**, + walidacji; IoU walidacji 0,654 → 0,659,
4. **25% krawędzi z v4 (`apply_box.py`)**: końcowa ramka = 0,75 × ramka z punktu 3 + 0,25 × ramka z głosowania v4.

## 4. Diagnoza (walidacja, zespół z v3)

| Gdzie giną punkty | Strata |
|---|---|
| niedokładna ramka (łatka trafiona w 92%, średnie IoU 0,68) | 0,237 |
| ramka w złym miejscu | 0,048 |
| przeoczona łatka (0,0,0,0) | 0,016 |
| fałszywa ramka na obrazku bez łatki | 0,008 |
| **wskazany wabik** | **0,000** |

Najtrudniejsze są słabe łatki (ε 2–4/255: wynik 0,49, ε 15–20: 0,79) i MI-FGSM. Wyrocznia wyboru krawędzi
(najlepszy z ~2000 wariantów) daje IoU 0,83 — tu jest największa niewykorzystana rezerwa.

**Co nie zadziałało:** ramki z samego głosowania v4 (IoU 0,638 < 0,651), trzecia identyczna sieć v3
(na walidacji bez zysku), więcej epok po wypłaszczeniu straty.

## 5. Odtworzenie wyniku 0,6808 krok po kroku

Środowisko: WSL (Ubuntu/Kali), Python 3, `pip install torch --index-url https://download.pytorch.org/whl/cu128`,
`pip install numpy pillow scikit-learn`. Struktura folderów:
```
bumas/
  bumas1/   p1_student/  synth_v2/  generate_v2.py  polacz.py  solve.py  porownaj_dane.py
  kolega/   solve_v2.py  ensemble4.py  train_v3.py  train_v4.py  refine_ml.py  apply_box.py
            eksport_moj.py  diag_v2.py  exp/  modele_natrenowane/
```
W `kolega/` ścieżka do danych: `sed -i "s#'data/p1_student'#'../bumas1/p1_student'#" solve_v2.py exp/*.py`.

Czasy na RTX 3060 Laptop (6 GB):

```bash
# --- A. dane (bumas1/), ~12 h
for p in 0 1 2 3 4 5 6 7 8; do
  python generate_v2.py --data p1_student --out synth_v2 --per_base 10 --bs 64 --part $p
done
python polacz.py synth_v2

# --- B. własny U-Net "moj" (bumas1/), ~7 h
python solve.py --data p1_student --extra synth_v2 --epochs 100 --bs 64 --lr 2e-3

# --- C. mapy przyczynowe walidacji i testu (kolega/), ~45 min
python exp/causal_cache.py val
python exp/causal_cache.py test
python eksport_moj.py --moj ../bumas1 --skrypt solve          # -> cache_v2/moj_*.npz
python solve_v2.py --out sub_kolega.csv                       # -> cache_v2/unet_*.npz (6 sieci bazowych)

# --- D. v3 na 15 tys. (kolega/), ~8 h
python train_v3.py --synth ../bumas1/synth_v2 --synth_max 15000 --synth_val 500 \
                   --epochs 100 --width 96 --bs 32 --seeds 0,1
cp cache_v2/unet3_val.npz cache_v2/unet3a_val.npz; cp cache_v2/unet3_test.npz cache_v2/unet3a_test.npz

# --- E. v3 na 50 tys. (kolega/), ~12 h; mapy pierwszych 20 tys. obrazków kopiowane z cache_v3
#     (fragment w historii: kopia causal_train.npy do cache_v3_full, N = 55000)
python train_v3.py --synth ../bumas1/synth_v2 --synth_max 50000 --synth_val 500 \
                   --epochs 60 --width 96 --bs 32 --seeds 0,1 --cache cache_v3_full
cp cache_v2/unet3_val.npz cache_v2/unet3b2_val.npz; cp cache_v2/unet3_test.npz cache_v2/unet3b2_test.npz
python train_v3.py ... --seeds 0,1,2 --cache cache_v3_full    # 3. sieć: potrzebna do cech refine_ml (unet3)

# --- F. v4 (krawędzie) i v5 (słabe łatki), ~3 h + ~4 h
python train_v4.py --synth ../bumas1/synth_v2 --synth_max 50000 --cache cache_v3_full \
                   --init cache_v3_full/unet3_1.pt --epochs 50
python train_v4.py --synth ../bumas1/synth_v2 --synth_max 50000 --cache cache_v3_full \
                   --init cache_v3_full/unet3_1.pt --epochs 25 --eps_boost 4 --box_w 0.3 --tag unet5
cp cache_v2/unet5_val.npz cache_v2/unet5a_val.npz; cp cache_v2/unet5_test.npz cache_v2/unet5a_test.npz

# --- G. zgłoszenie 0,6808, ~1 h
U=unet,moj,unet3a,unet3b2,unet5a
python solve_v2.py --unet $U --out sub_A_baza.csv
python refine_ml.py --sub sub_A_baza.csv --start_unet $U --cache cache_ref2 --out sub_A_ml.csv
python apply_box.py --sub sub_A_ml.csv --unet $U --mode w --w 0.25 --out sub_A_ml_box25.csv
```
Uwaga: w czasie tworzenia rekordu mapa pojedynczej v5 nazywała się `unet5`; po treningu drugiej v5 została
zachowana jako `unet5a`. Sieci są losowe (ziarno, kolejność na GPU), więc odtworzony wynik może różnić się o ~0,001.

## 6. Pliki

**Potrzebne do odtworzenia:** `generate_v2.py`, `polacz.py`, `solve.py`, `eksport_moj.py`, `solve_v2.py`,
`ensemble4.py`, `exp/` (`common.py`, `causal_cache.py`, `causal_map.py`, `refine_v2.py`), `train_v3.py`,
`train_v4.py`, `refine_ml.py`, `apply_box.py`, `modele_natrenowane/`.

**Narzędzia pomocnicze (analiza, nie potrzebne do wyniku):** `porownaj_dane.py`, `diag_v2.py`, `refine_test.py`,
`refine_ml_cv.py`, `solve_fast.py` (szybsze czytanie danych).

**Do pominięcia:** `generate.py` i `synth_final/` (stary generator, bazy z walidacji — wyciek), `selektor/`,
stare `sub_*.csv` poza najlepszymi, foldery `cache_*` (odtwarzalne, kilkadziesiąt GB).

## Literatura
- Madry i in., *Towards Deep Learning Models Resistant to Adversarial Attacks* (PGD), 2018 — https://arxiv.org/abs/1706.06083
- Dong i in., *Boosting Adversarial Attacks with Momentum* (MI-FGSM), 2018 — https://arxiv.org/abs/1710.06081
- Carlini, Wagner, *Towards Evaluating the Robustness of Neural Networks* (C&W), 2017 — https://arxiv.org/abs/1608.04644
- Athalye, Carlini, Wagner, *Obfuscated Gradients Give a False Sense of Security* (BPDA przez JPEG), 2018 — https://arxiv.org/abs/1802.00420
- Zeiler, Fergus, *Visualizing and Understanding Convolutional Networks* (zasłanianie fragmentów), 2014 — https://arxiv.org/abs/1311.2901
- Petsiuk, Das, Saenko, *RISE*, 2018 — https://arxiv.org/abs/1806.07421
- Fridrich, Kodovský, *Rich Models for Steganalysis of Digital Images*, 2012 — https://doi.org/10.1109/TIFS.2012.2190402
- Ronneberger i in., *U-Net*, 2015 — https://arxiv.org/abs/1505.04597
- Tian i in., *FCOS* (odległości do krawędzi), 2019 — https://arxiv.org/abs/1904.01355
- Jiang i in., *IoU-Net* (wybór ramki wg przewidywanego IoU), 2018 — https://arxiv.org/abs/1807.11590
- Lakshminarayanan i in., *Deep Ensembles*, 2017 — https://arxiv.org/abs/1612.01474
- He, Garcia, *Learning from Imbalanced Data*, 2009 — https://doi.org/10.1109/TKDE.2008.239
