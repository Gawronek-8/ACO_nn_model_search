# Jak uruchamiać eksperymenty

Krótka instrukcja dla osób, które robią eksperymenty E1–E4. Opis algorytmu jest w `README.md`.

## 1. Przed pierwszym uruchomieniem

```powershell
uv sync
uv run python -c "import torch; print(torch.cuda.is_available())"   # czy masz GPU
uv run python -m ACO.Datasets                                       # pobiera Adult, Wine, Letter do data/openml
uv run python -m pytest                                             # wszystkie testy powinny przejść
```

Zbiory pobierają się raz i są cache'owane w `data/openml/` (git ich nie śledzi).

## 2. Szybki test, że wszystko działa

```powershell
uv run python main.py configs/smoke.json
```

To mikro-przebieg (4 sieci, 3 epoki, kilka sekund). Wynik jest bez znaczenia, służy tylko do sprawdzenia, że całość działa na Twojej maszynie. Wyniki zapisują się w `runs/smoke/`, który jest ignorowany przez git.

## 3. Jeden eksperyment: `main.py`

```powershell
uv run python main.py configs/default.json --dataset adult --strategy random --seeds 0 1 2 --device gpu
```

Opcje nadpisują wartości z pliku JSON: `--dataset`, `--strategy`, `--seeds`, `--device`, `--name`, `--out`, `--data-dir`.

## 4. Siatka eksperymentów: `grid.py`

Uruchamia każdą kombinację dataset × strategia, każdą na wszystkich ziarnach.

```powershell
# najpierw sprawdź, ile to będzie pracy (nic nie uruchamia)
uv run python grid.py configs/default.json --datasets all --strategies all --seeds 0 1 2 --name e2 --dry-run

# potem uruchom
uv run python grid.py configs/default.json --datasets all --strategies all --seeds 0 1 2 --name e2 --device gpu
```

| Opcja | Znaczenie |
|---|---|
| `--datasets` | `adult`, `wine`, `letter` albo `all` (domyślnie dataset z JSON) |
| `--strategies` | `independent`, `param_dependent`, `config_dependent`, `random` albo `all` (domyślnie strategia z JSON) |
| `--seeds` | lista ziaren, jeden przebieg na ziarno |
| `--device` | `cpu`, `gpu` albo `auto` |
| `--name` | nazwa eksperymentu, czyli folder w `runs/` |
| `--dry-run` | tylko wypisuje plan i liczbę modeli |
| `--no-resume` | liczy wszystko od nowa, nawet skończone przebiegi |

`--device gpu` kończy się błędem, jeśli GPU nie ma (nie przełącza się po cichu na CPU). `auto` używa GPU, jeśli jest, a w przeciwnym razie CPU.

Na końcu dostajesz tabelę ze średnimi po ziarnach dla każdej pary dataset × strategia.

### Przerwany grid

Jeśli przerwiesz grid (albo komputer się wyłączy), uruchom tę samą komendę jeszcze raz. Skończone przebiegi są pomijane, a reszta się dolicza. Przebieg uznajemy za skończony tylko wtedy, gdy jego zapisane ustawienia są identyczne z bieżącymi, więc po zmianie np. `epochs` policzy się od nowa.

## 5. Konfiguracja (plik JSON)

Skopiuj `configs/default.json` i zmień to, co potrzebujesz. Pominięte pola mają wartości domyślne, więc plik może być krótki (zobacz `configs/smoke.json`).

| Pole | Co robi |
|---|---|
| `name` | nazwa folderu z wynikami w `runs/` |
| `dataset`, `seeds` | zbiór i lista ziaren |
| `data_seed` | ziarno podziału train/val/test, wspólne dla wszystkich metod (nie zmieniaj w trakcie badania) |
| `min_class_count` | usuwa klasy rzadsze niż ta liczba (Wine ma klasę z kilkoma próbkami), `0` zostawia wszystko |
| `colony.strategy` | `independent`, `param_dependent`, `config_dependent`, `random` |
| `colony.n_ants`, `colony.n_iterations` | liczba mrówek i iteracji, razem budżet |
| `colony.greediness` | `q0`, szansa wyboru najlepszej opcji zamiast ruletki |
| `colony.pheromone.start` / `local_decay` / `evaporation` | `τ0`, aktualizacja lokalna, aktualizacja globalna |
| `colony.search_space` | opcje dla neuronów, aktywacji, dropoutu i głębokości |
| `colony.reuse_weights` | warm-start z wag sieci o wspólnym początku ścieżki |
| `evaluator.epochs`, `batch_size`, `learning_rate`, `patience` | trening każdej sieci (`patience: null` wyłącza early stopping) |

`colony.seed` w pliku jest ignorowane, bo każde ziarno z `seeds` je nadpisuje.

**Budżet** to liczba ocenionych modeli, czyli `n_ants × n_iterations`, i jest taki sam dla Random Search. Przy porównaniu metod nie zmieniaj tych wartości między nimi.

## 6. Jak zrobić poszczególne eksperymenty

**E1: ACO kontra Random Search** (ten sam budżet):
```powershell
uv run python grid.py configs/default.json --datasets all --strategies param_dependent random --seeds 0 1 2 3 4 --name e1
```

**E2: feromony niezależne i zależne:**
```powershell
uv run python grid.py configs/default.json --datasets all --strategies independent param_dependent config_dependent --seeds 0 1 2 3 4 --name e2
```

**E3: wpływ parametrów.** `grid.py` nie ma flag do parametrów kolonii, więc robisz osobny plik JSON na każdą wartość, np. `configs/e3_evap_0.3.json`:
```json
{
  "name": "e3_evap_0.3",
  "colony": { "pheromone": { "evaporation": 0.3 } }
}
```
**Każdy wariant musi mieć inne `name`.** Gdyby dwa warianty miały to samo, wyniki jednego nadpisałyby drugi. Zmieniaj w E3: `n_ants`, `pheromone.evaporation`, `greediness` i `pheromone.local_decay` (ten ostatni mocno hamuje zbieżność). Zmiana `n_ants` zmienia też budżet, więc porównuj to ostrożnie.

**E4: efektywność wyszukiwania** nie wymaga osobnych przebiegów. Dane są już w wynikach E1 i E2: `ants.csv` ma kolumnę `best_so_far` (jak najlepszy wynik rośnie z kolejnymi modelami), a `summary.json` ma czas.

## 7. Co powstaje w `runs/`

```
runs/<name>/
  grid_summary.csv                          jeden wiersz na przebieg (cała siatka)
  summary_<dataset>_<strategy>.csv          jeden wiersz na ziarno
  <dataset>_<strategy>_s<seed>/
    ants.csv                                każda oceniona sieć, w kolejności oceny
    summary.json                            najlepsza sieć, wynik na teście, pełna konfiguracja
```

Kolumny `ants.csv`: `ant`, `iteration`, `n_layers`, `layers` (np. `64-relu-0.1 | 32-tanh-0.0`, czyli neurony-aktywacja-dropout), `val_accuracy`, `val_loss`, `train_time`, `n_params`, `best_so_far`.

W `summary.json` i `grid_summary.csv`: `best_val_accuracy`, `test_accuracy`, `train_time_s` (suma treningu wszystkich sieci), `search_time_s` (cały czas przebiegu), `device`.

Wynik na teście (`test_accuracy`) pochodzi z ponownego treningu najlepszej architektury od zera. Mrówki nigdy nie widzą zbioru testowego. Do wyboru architektury służy walidacja.

## 8. Zasady w zespole

- **Te same ziarna dla wszystkich.** Ustalcie listę (np. `0 1 2 3 4`) i używajcie jej w każdym eksperymencie, żeby wyniki dało się porównać.
- **Nazwy.** Ostateczne wyniki mają nazwę eksperymentu (`e1`, `e2`, `e3_evap_0.3`...) i je commitujemy (pliki są małe). Próbne przebiegi nazywamy `smoke` albo `test*`, a te są ignorowane przez git.
- **GPU i CPU.** Wyniki z różnych urządzeń różnią się minimalnie (inna kolejność operacji zmiennoprzecinkowych). Kolumna `device` zapisuje, na czym liczyło się dany przebieg. Dla uczciwego porównania metod licz je na tej samej maszynie.
- **Czas.** Zanim odpalisz duży grid, zmierz jeden przebieg (`--datasets letter --strategies param_dependent --seeds 0`) i zobacz `search_time_s`. Pomnóż przez liczbę przebiegów z `--dry-run`.

## 9. Gdy coś nie działa

| Objaw | Co zrobić |
|---|---|
| `error: device='gpu' was requested but no CUDA or MPS device is available` | ta maszyna nie ma GPU, użyj `--device cpu` albo `auto` |
| Pobieranie zbioru się wywala | sprawdź połączenie z internetem i uruchom jeszcze raz `uv run python -m ACO.Datasets` (OpenML bywa chwilowo niedostępny) |
| Chcę policzyć wszystko od nowa | dodaj `--no-resume` albo użyj nowej `--name` |
| `uv run python -m pytest` nie przechodzi | uruchamiaj z katalogu głównego repo, testy czytają `configs/` |
