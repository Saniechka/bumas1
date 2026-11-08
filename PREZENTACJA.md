# P1 „Ukryta łatka” — co pokazać na prezentacji

Ten plik to „ściąga”: **6 miejsc w kodzie**, w których jest całe rozwiązanie, z fragmentami kodu i wyjaśnieniem.
Pełny opis wszystkich plików: `OPIS_KODU.md`, strategia i odtworzenie wyniku: `README_ROZWIAZANIE.md`.

| # | Co pokazać | Plik | Funkcja / fragment | Kto |
|---|---|---|---|---|
| 1 | Generator danych | `bumas1_kod/generate_v2.py` | `through_jpeg`, wyszukiwanie najmniejszego ε, sprawdzanie łatki i wabika | moje |
| 2 | **Mapa przyczynowa** | `kolega_kod/exp/causal_cache.py` | `compute_causal` (pętla po oknach) | kolega |
| 3 | Wejście sieci: 24 cechy + 2 mapy | `kolega_kod/ensemble4.py`, `train_v3.py` | `build_features`, klasa `Stack` | kolega |
| 4 | Trening sieci v4 / v5 | `kolega_kod/train_v4.py` | `BoxUNet`, `train` (strata, `--eps_boost`) | moje |
| 5 | Od map do ramki i decyzji | `kolega_kod/solve_v2.py` | `analyse`, `fit_and_score` | kolega |
| 6 | Wybór krawędzi modelem ML | `kolega_kod/refine_ml.py` | `var_feats`, `fit`, `pick` | moje |
| + | Diagnoza | `kolega_kod/diag_v2.py`, logi | tabela strat | moje |

---

## 0. Zadanie w jednym akapicie

Obrazki 64×64, atakowany ResNet-18. Na 80% obrazków w jednym prostokącie (8–15% pola) jest słaby szum
adwersaryjny — **łatka** — który zmienia odpowiedź modelu. Na części obrazków są też **wabiki**: prostokąty
z losowym szumem tej samej siły, które odpowiedzi nie zmieniają. Trzeba podać ramkę łatki albo 0,0,0,0.
Wynik = średnie IoU (z 1 punktem za poprawne 0,0,0,0). Wskazanie wabika = 0 punktów.

**Główna trudność:** łatka i wabik **wyglądają tak samo** (szum ±ε w prostokącie), szum jest bardzo słaby
(ε od 2/255) i po kompresji JPEG. Różni je tylko to, **czy zmieniają decyzję modelu**.

---

## 1. Generator danych (`generate_v2.py`) — 55 tys. nowych obrazków

Organizator dał 5000 obrazków treningowych (500 odłożyłem na walidację). To za mało dla sieci, więc
wygenerowałem nowe, **dokładnie wg opisu z regulaminu**.

**Gradient przez JPEG (BPDA).** Atak musi działać także po zapisie pliku, a JPEG nie ma gradientu.
W przód liczymy prawdziwy JPEG, wstecz udajemy, że to tożsamość:
```python
def through_jpeg(x, qs):
    """BPDA: w przod prawdziwy JPEG, wstecz gradient tozsamosciowy."""
    return x + (jpeg_t(x, qs) - x).detach()
```

**Najmniejsze ε, przy którym atak działa** (regulamin: „ε to najmniejsza wartość, przy której atak działa”).
Wyszukiwanie binarne po liczbach całkowitych 1–20/255, osobno dla każdego obrazka:
```python
run(cand, {i: 20 for i in cand})               # najpierw eps = 20; jak nie dziala -> obrazek odpada
while True:
    active = [i for i in active if hi[i] - lo[i] > 1]
    if not active:
        break
    run(active, {i: (lo[i] + hi[i]) // 2 for i in active})   # srodek przedzialu; sukces -> hi, porazka -> lo
```

**Każdy obrazek jest sprawdzany przez ResNet po zapisie** — tak jak u organizatora:
```python
if j['new_dec'] and pred_of(jpeg_u8(u8f(only_dec), j['q'])) != clean_pred:   # wabik NIE moze zmienic klasy
    stat['fail_dec'] += 1; continue
if j['patch'] and pred_of(saved) == clean_pred:                              # latka MUSI zmienic klase
    stat['fail_atk'] += 1; continue
```

Pozostałe zasady: 4 ataki (PGD bez celu, PGD z celem, MI-FGSM, C&W L2 z najmniejszym udanym szumem),
prostokąty 8–15% pola i proporcje 1:2–2:1, wabiki jednostajne / gaussowskie / ±ε o sile z rozkładu prawdziwych
łatek, JPEG q 70–95, **bazy z walidacji pominięte** (wcześniejsza wersja miała tu wyciek).
Zgodność z danymi organizatora sprawdził `porownaj_dane.py` (rozkład ε, typy wabików, „widoczność” łatki).

---

## 2. Mapa przyczynowa — najważniejszy pomysł

**Pytanie, na które odpowiada:** „czy szum w tym miejscu naprawdę oszukuje ResNet?”

**Jak się ją liczy (dla jednego obrazka):**
1. ResNet ocenia obrazek → klasa `c0` (na obrazku z łatką to klasa *błędna*, wymuszona atakiem)
   i **margines** `m0` = logit `c0` minus najwyższy inny logit (jak bardzo model jest pewny).
2. Tworzymy kopię obrazka po mocnym JPEG (jakość 30 i osobno 50). Mocny JPEG „zmywa” słaby szum.
3. Bierzemy okno (5 rozmiarów: 16×16, 20×20, 24×24, 14×28, 28×14, przesuwane co 4 px — **ok. 700 okien**).
   **Tylko w tym oknie** podmieniamy obraz na wersję po JPEG, reszta zostaje oryginalna.
4. Liczymy margines `md` klasy `c0` na takim obrazku. Spadek marginesu = jak bardzo to okno „trzymało” decyzję.
5. Wartość piksela mapy = średni (znormalizowany) spadek po wszystkich oknach, które go zawierają.

Razem ~700 okien × 2 jakości ≈ **1400 przejść przez ResNet na obrazek** (ok. 0,5 s na RTX 3060).

```python
for s in range(0, len(M), 512):                       # M = maski ~700 okien
    m = M[s:s + 512, None]
    md.append(margin(bb(x + m * (f - x)).float(),      # w oknie: obraz po JPEG (f), poza oknem: oryginal (x)
                     c0.repeat(len(m))))
md = torch.cat(md)
drop = (m0 - md) / (abs(m0 - mf) + 0.5)                # spadek marginesu, znormalizowany przez spadek dla calego JPEG
out[n, j] = torch.einsum('w,wij->ij', drop, M) / cnt   # rozlozenie na piksele: srednia po oknach zawierajacych piksel
```
`margin(z, c)` = logit klasy `c` minus największy inny logit (`exp/common.py`).

**Dlaczego to działa:**
- **Łatka** ma *minimalne* ε — ledwo-ledwo oszukuje model. Zmycie choćby jej części psuje atak →
  margines klasy błędnej mocno spada → **jasne miejsce na mapie**.
- **Wabik** to losowy szum, który nic nie zmienia → jego zmycie też nic nie zmienia → **ciemno**.
- **Czysty obrazek** — nie ma czego zmywać → ciemno.

To idea „zasłaniania fragmentów” z literatury (Zeiler & Fergus, RISE), tylko zamiast szarego prostokąta
**zmywamy szum JPEG-iem**, więc obraz zostaje naturalny, a znika tylko perturbacja.

**Efekt:** wskazania wabika spadły do **0%** (diagnoza), sama ta sama sieć zyskała +0,05.

---

## 3. Co widzi sieć: 24 cechy + 2 mapy przyczynowe

Sieć nie dostaje gołego obrazka, tylko 26 kanałów na piksel (`ensemble4.build_features` + `train_v3.Stack`):

| Grupa | Kanały | Po co |
|---|---|---|
| obraz RGB | 3 | kontekst |
| residuum: obraz − mediana 3×3 (×20) | 3 | szum wysokiej częstotliwości — tam „siedzi” perturbacja (pomysł ze steganalizy) |
| energia residuum, energia / lokalne odchylenie obrazu, filtr górnoprzepustowy, kanał „poprzeczny” | 4 | ile szumu względem tekstury tła |
| DCT bloków 8×8 (wysokie częstotliwości, ile niezerowych) | 2 | ślady JPEG: łatka zmienia rozkład współczynników |
| gradient ResNetu: klasa top-1 i margines top-1 − top-2 | 6 | które piksele najbardziej wpływają na decyzję (mapa saliency) |
| korelacja residuum z gradientem (3 skale), energia gradientu, residuum × znak gradientu | 5 | **szum łatki jest ułożony zgodnie z gradientem**, szum wabika — losowo |
| odsetek prześwietlonych / czarnych pikseli | 1 | tam szum jest obcięty |
| **mapa przyczynowa JPEG q30, q50** (znormalizowane do 0–1) | **2** | czy szum w tym miejscu zmienia decyzję (pkt 2) |

```python
class Stack:
    """Leniwe sklejenie cech (N,24,..) i znormalizowanych map przyczynowych (N,NC,..)."""
    def __getitem__(self, b):
        c = np.clip(np.asarray(self.c[b], np.float32), 0, None)
        c = c / (c.max((-2, -1), keepdims=True) + 1e-6)
        return np.concatenate([np.asarray(self.f[b], np.float32), c], 1)    # 24 + 2 = 26 kanalow
```

---

## 4. Sieci i trening

### Architektura (wszystkie)
**U-Net**: koder zmniejsza obraz 64 → 32 → 16 → 8 px (szeroki kontekst), dekoder wraca do 64×64 z połączeniami
skrótowymi (dokładne położenie). Na wyjściu:
- **3 klasy na piksel**: tło / łatka / wabik,
- **1 liczba** „czy na obrazku jest łatka”,
- (tylko v4/v5) **4 liczby na piksel**: odległość do lewej, górnej, prawej, dolnej krawędzi łatki.

### Strata i trening
```python
ce = (F.cross_entropy(logit, m, weight=wcls, reduction='none') * wmap).mean()   # wcls = [0.25, 1.0, 0.6]
loss = ce + 0.5 * soft_dice(p_latka, maska_latki) \
          + 0.3 * F.binary_cross_entropy_with_logits(g, jest_latka) \
          + box_w * iou_loss(odleglosci, cele, piksele_latki)                    # tylko v4/v5
```
- entropia krzyżowa na pikselach (łatka liczy się 4× mocniej niż tło; w v4/v5 piksele 2 px od krawędzi łatki ×3),
- **dice** — wprost premiuje dobre pokrycie łatki,
- BCE dla „czy jest łatka”,
- augmentacja: każdy obrazek losowo obracany / odbijany (8 wariantów), cele krawędzi obracane razem z nim,
- optymalizator AdamW, krok uczenia OneCycle (rośnie, potem maleje do zera), paczki 32–64 obrazków,
- test: średnia z 8 obrotów każdego obrazka.

### Które sieci, na czym i jak długo (RTX 3060 Laptop, 6 GB)

| Sieć | Wejście | Szer. | Dane treningowe | Epoki | Czas | IoU walid.* |
|---|---|---|---|---|---|---|
| `unet` — 6 sieci bazowych (kolega) | 24 | 96 | prawdziwy trening | — | — | 0,581 (6 razem) |
| `moj` — pierwszy własny U-Net | 24 | 48 | 4,5 tys. prawdziwych + **55 tys.** z generatora | 100 | ~7 h | 0,497 |
| `unet3a` — 2× v3 | **26** | 96 | 4,5 tys. + 15 tys. | 100 | ~2 h / sieć | 0,640 (2 razem) |
| `unet3b2` — 2× v3 | **26** | 96 | 4,5 tys. + **50 tys.** | 60 | ~5 h / sieć | ~0,65 |
| `unet4` — v4 (krawędzie) | 26 | 96 | jw., prawdziwe ×3, start z v3 | 50 | ~4 h | głosowanie 0,638 |
| `unet5a` — v5 (słabe łatki) | 26 | 96 | jw., **łatki ε ≤ 4/255 ×4**, prawdziwe ×3, start z v3 | 25 | ~3,5 h | 0,637 |

\* IoU ramki z mapy × mapa przyczynowa na 400 prawdziwych obrazkach walidacji z łatką.

**v4 i v5 to douczanie** (`--init` z wagami v3): sieć już umie znajdować łatki, doucza się nowa „głowa”
i trudne przypadki. Fragment `train_v4.py`, który robi „balanced”:
```python
parts = [tr_idx] + [tr_real] * (a.real_rep - 1)                  # prawdziwe obrazki czesciej
if eps is not None and a.eps_boost > 1:
    weak = tr_idx[(eps[tr_idx] > 0) & (eps[tr_idx] <= a.eps_max)]
    parts += [weak] * (a.eps_boost - 1)                          # slabe latki (eps <= 4/255) czesciej
pool = np.concatenate(parts)
```

---

## 5. Od map do odpowiedzi (`solve_v2.py`)

1. Mapy wszystkich sieci w zespole są **uśredniane** (`--unet unet,moj,unet3a,unet3b2,unet5a`).
2. Mapa łatki **× mapa przyczynowa** → prostokąt o największym oczekiwanym IoU, tylko o kształtach dozwolonych
   regulaminem (`best_rect`):
```python
cz = np.clip(c.mean(0), 0, None); cz = cz / (cz.max() + 1e-9)    # mapa przyczynowa 0..1
pc = p[1] * cz ** a                                               # p[1] = prawdopodobienstwo "latka" z zespolu
b, v = best_rect(pc)
```
3. **Decyzja ramka / 0,0,0,0**: regresja logistyczna na 13 cechach obrazka (suma mapy łatki i wabika, średnia
   w ramce, marginesy ResNetu…). Ramka jest podawana, gdy `P(łatka) × oczekiwane IoU > 1 − P(łatka) + próg`,
   czyli gdy w oczekiwaniu daje więcej punktów niż 0,0,0,0. Próg dobierany walidacją 5-krotną.

---

## 6. Wybór krawędzi modelem ML (`refine_ml.py`)

Diagnoza pokazała, że **łatkę znajdujemy w 92% przypadków, ale ramki są niedokładne** (strata 0,237 z 0,29).
Wyrocznia (najlepszy z wariantów) daje IoU 0,83, a ramka startowa 0,65 — więc trzeba lepiej **wybierać**.

1. Wokół ramki startowej ~2000 wariantów: każda krawędź −3…+3 px (przesunięcie, powiększenie, zmniejszenie).
2. Dla każdego wariantu ResNet sprawdza spadek marginesu po zmyciu szumu w tej ramce + 27 cech
   (`var_feats`: oczekiwane IoU na mapach, kontrast z pierścieniem 2 px na zewnątrz, ranga spadku, pole, przesunięcia).
3. Model **gradient boosting** przewiduje IoU każdego wariantu; bierzemy najlepszy:
```python
def fit(T):
    X, Y = stack(T, np.random.default_rng(0))                    # cechy wariantow i ich prawdziwe IoU
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
                                      min_samples_leaf=50, l2_regularization=1.0, random_state=0)
    return m.fit(X, Y)

def pick(m, T):
    return [tuple(int(v) for v in rects[int(np.argmax(m.predict(F)))]) for rects, F, _ in T]
```
4. **Na czym się uczy:** 2000 obrazków z generatora, **których żadna sieć nie widziała** (zostawione z 55 tys.)
   + walidacja. Uczciwie sprawdzone na walidacji: IoU 0,654 → 0,659.
5. Na koniec (`apply_box.py`) ramka = 75% ramki z ML + 25% ramki z głosowania v4.

---

## 7. Wyniki i diagnoza

| Krok | Tablica |
|---|---|
| rozwiązanie bazowe | 0,5851 |
| + własny U-Net (55 tys. obrazków) | 0,6081 |
| **+ mapa przyczynowa jako wejście (v3)** | **0,6572** |
| + wybór krawędzi ML | 0,6627 |
| + v3 na 50 tys. obrazków | 0,6714 |
| + 25% krawędzi v4 | 0,6747 |
| + trzecia v3 / zespół z v5 | 0,6801 / **0,6808** |

Gdzie zostały punkty (walidacja): niedokładna ramka 0,237, ramka w złym miejscu 0,048, przeoczona łatka 0,016,
fałszywa ramka 0,008, **wabik 0,000**. Najtrudniejsze: łatki ε 2–4/255 (wynik 0,49) i MI-FGSM.

**Co nie zadziałało (warto powiedzieć):** ramki z samego głosowania v4 (gorsze od map), trzecia identyczna sieć v3,
więcej epok po wypłaszczeniu straty. **Największa niewykorzystana rezerwa:** wybór krawędzi (0,66 → wyrocznia 0,83).

## Literatura
- Zeiler, Fergus, *Visualizing and Understanding Convolutional Networks* (zasłanianie fragmentów), ECCV 2014 — https://arxiv.org/abs/1311.2901
- Petsiuk, Das, Saenko, *RISE*, BMVC 2018 — https://arxiv.org/abs/1806.07421
- Athalye, Carlini, Wagner, *Obfuscated Gradients Give a False Sense of Security* (BPDA), 2018 — https://arxiv.org/abs/1802.00420
- Dziugaite, Ghahramani, Roy, *A study of the effect of JPG compression on adversarial images*, 2016 — https://arxiv.org/abs/1608.00853
- Fridrich, Kodovský, *Rich Models for Steganalysis of Digital Images*, IEEE TIFS 2012 — https://doi.org/10.1109/TIFS.2012.2190402
- Simonyan, Vedaldi, Zisserman, *Deep Inside Convolutional Networks* (mapy gradientu), 2014 — https://arxiv.org/abs/1312.6034
- Ronneberger i in., *U-Net*, 2015 — https://arxiv.org/abs/1505.04597
- Milletari i in., *V-Net* (strata dice), 2016 — https://arxiv.org/abs/1606.04797
- Tian i in., *FCOS* (odległości do krawędzi), 2019 — https://arxiv.org/abs/1904.01355
- Jiang i in., *IoU-Net*, 2018 — https://arxiv.org/abs/1807.11590
- Lakshminarayanan i in., *Deep Ensembles*, 2017 — https://arxiv.org/abs/1612.01474
- Smith, Topin, *Super-Convergence* (OneCycle), 2018 — https://arxiv.org/abs/1708.07120
