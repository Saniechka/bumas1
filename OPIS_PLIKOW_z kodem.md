# Opis plików — P1 „Ukryta łatka” (wynik 0,6808)

Pliki w kolejności, w jakiej działa rozwiązanie.

```
generate_v2 → (obrazki) → causal_cache → (mapy) → train_v3 → train_v4
                                                       ↓          ↓
                                                  solve_v2 (zespół + decyzja)
                                                       ↓
                                                  refine_ml (krawędzie)
                                                       ↓
                                                  apply_box → sub_A_ml_box25.csv (0,6808)
```

---

## 1. `generate_v2.py` — generator danych 


**Co robi:**
- bierze czyste obrazki z treningu (bez 500 odłożonych do sprawdzania wyników),
- wybiera losowy prostokąt (8–15% obrazka, proporcje od 1:2 do 2:1),
- atakuje ResNet jednym z 4 ataków (PGD, PGD z celem, MI-FGSM, C&W) **tylko w tym prostokącie**,
- liczy atak „przez JPEG” (`through_jpeg`), żeby działał też po zapisaniu pliku,
- szuka **najmniejszej siły ataku ε**, przy której model się myli (wyszukiwanie binarne 1–20/255),
- dokłada wabiki (losowy szum tej samej siły),
- **sprawdza każdy obrazek**: łatka musi oszukać ResNet, wabik nie — inaczej obrazek wylatuje.

**Wynik:** ok. 55 tys. obrazków i plik z prawdziwymi ramkami.

---

## 2. `causal_cache.py` — mapa przyczynowa (kolega)

**Po co:** łatka i wabik wyglądają identycznie; różnią się tylko tym, czy zmieniają odpowiedź ResNetu.
Ten plik to mierzy.

**Co robi (dla każdego obrazka):**
- pyta ResNet o odpowiedź i jego pewność,
- przesuwa po obrazku okno (~700 położeń i rozmiarów),
- **tylko w oknie** zastępuje obraz wersją po mocnym JPEG (jakość 30 i 50), który „zmywa” słaby szum,
- mierzy, o ile spadła pewność ResNetu,
- składa z tego mapę 64×64: jasno tam, gdzie zmycie zmieniło decyzję — czyli tam, gdzie jest łatka.

**Wynik:** 2 mapy na obrazek (JPEG 30 i 50). Postęp zapisywany co 100 obrazków (można przerwać i wznowić).

---

## 3. `train_v3.py` — sieć v3 (kolega, na moich danych)

**Po co:** najważniejsza sieć — **największy skok wyniku (+0,049)**.

**Co robi:**
- przygotowuje dla każdego obrazka 26 kanałów: 24 cechy (szum, ślady JPEG, gradient ResNetu)
  + 2 mapy przyczynowe (klasa `Stack`),
- uczy sieć U-Net, która dla każdego piksela mówi: tło / łatka / wabik, a dla obrazka — „czy jest łatka”,
- przy treningu obraca i odbija obrazki (8 wariantów),
- na końcu liczy mapy dla walidacji i testu.

**Wynik:** wytrenowane sieci (`.pt`) i ich mapy. Użyte 2 wersje: na 15 tys. i na 50 tys. wygenerowanych obrazków.

---

## 4. `train_v4.py` — sieci v4 i v5 (moje)

**Po co:** poprawić dwie słabe strony v3 — niedokładne krawędzie i słabe łatki.

**Co robi:**
- startuje z gotowej sieci v3 i ją **douczą** (nie od zera),
- **v4:** druga „głowa” — każdy piksel łatki przewiduje odległość do 4 krawędzi (`BoxUNet`);
  ramka powstaje z głosowania pikseli,
- **v5:** łatki o najmniejszej sile (ε ≤ 4/255) pokazywane 4× częściej (`--eps_boost`), prawdziwe obrazki 3× częściej,
- błędy przy krawędziach łatki liczą się mocniej.

**Wynik:** sieci v4 i v5 z mapami. v5 weszła do najlepszego zespołu, z v4 brane jest 25% krawędzi.

---

## 5. `solve_v2.py` — zespół i decyzja (kolega + moje zmiany)

**Po co:** zamienić mapy wielu sieci w odpowiedź do zgłoszenia.

**Co robi:**
- uśrednia mapy wszystkich sieci w zespole (np. `--unet unet,moj,unet3a,unet3b2,unet5a`),
- mnoży je przez mapę przyczynową — zostaje szum, który naprawdę oszukuje ResNet,
- znajduje prostokąt o najlepszym oczekiwanym IoU, tylko o kształtach dozwolonych regulaminem (`analyse`),
- decyduje: podać ramkę czy 0,0,0,0 (regresja logistyczna na cechach obrazka, `fit_and_score`),
- wypisuje wynik walidacji, żeby porównać różne zespoły.

**Wynik:** plik zgłoszenia z ramkami startowymi.

---

## 6. `refine_ml.py` — wybór krawędzi (moje)

**Po co:** łatka jest znajdowana w 92% przypadków, ale ramki są niedokładne — to była największa strata punktów.

**Co robi:**
- wokół każdej ramki tworzy ~2000 wariantów (każda krawędź przesunięta o −3…+3 px),
- dla każdego wariantu liczy cechy: spadek pewności ResNetu po zmyciu szumu w tej ramce, zgodność z mapami,
  różnicę szumu w środku i tuż na zewnątrz (`var_feats`),
- model gradient boosting przewiduje IoU każdego wariantu (`fit`) i wybiera najlepszy (`pick`),
- model uczy się na 2000 wygenerowanych obrazkach, **których żadna sieć nie widziała**.

**Wynik:** zgłoszenie z dokładniejszymi ramkami (IoU walidacji 0,654 → 0,659).

---

## 7. `apply_box.py` — końcowa ramka (moje)

**Co robi:** łączy dwie ramki: 75% z `refine_ml.py` + 25% z głosowania sieci v4.
Przykład: ML — lewa krawędź na x = 12, v4 — na x = 16 → wynik 13.

**Wynik:** `sub_A_ml_box25.csv` — **0,6808**.

---

## Literatura
- Madry i in., *Towards Deep Learning Models Resistant to Adversarial Attacks* (PGD), 2018 — https://arxiv.org/abs/1706.06083
- Athalye, Carlini, Wagner, *Obfuscated Gradients Give a False Sense of Security* (gradient przez JPEG), 2018 — https://arxiv.org/abs/1802.00420
- Zeiler, Fergus, *Visualizing and Understanding Convolutional Networks* (zasłanianie fragmentów), 2014 — https://arxiv.org/abs/1311.2901
- Ronneberger i in., *U-Net*, 2015 — https://arxiv.org/abs/1505.04597
- Tian i in., *FCOS* (odległości do krawędzi), 2019 — https://arxiv.org/abs/1904.01355
- Jiang i in., *IoU-Net*, 2018 — https://arxiv.org/abs/1807.11590
- Dietterich, *Ensemble Methods in Machine Learning*, 2000 — https://doi.org/10.1007/3-540-45014-9_1
