# 🐜 ACO_nn_model_search

> Dokumentacja wierszem, bo każde zdanie ma tu swój rym,\
> więc czytaj ją na głos, a mrówki zatańczą przy nim.

Ta dokumentacja rymem pisana jest od początku do końca,\
by czytało się ją lekko, jak spacer w promieniach słońca.

Nazwy z kodu zapisuję w `takich ramkach` dla jasności,\
by od razu było widać, co pochodzi z kodu, bez żadnych wątpliwości.

Wzory, tabele i kod zostają zwykłe, bo kod się nie rymuje,\
ale każde zdanie dookoła rymem was poczęstuje.

## Spis treści

1. [O co chodzi w projekcie](#1-o-co-chodzi-w-projekcie)
2. [ACO od zera](#2-aco-od-zera)
3. [DeepSwarm, czyli skąd pomysł](#3-deepswarm-czyli-skąd-pomysł)
4. [Graf i tablice feromonów](#4-graf-i-tablice-feromonów)
5. [Strategie i czym się różnią](#5-strategie-i-czym-się-różnią)
6. [Przebieg wyszukiwania i hooki](#6-przebieg-wyszukiwania-i-hooki)
7. [Struktura repozytorium](#7-struktura-repozytorium)
8. [Plik po pliku](#8-plik-po-pliku)
9. [Najtrudniejsze momenty](#9-najtrudniejsze-momenty)
10. [Konfiguracja](#10-konfiguracja)
11. [Jak uruchomić](#11-jak-uruchomić)
12. [Co jest do zrobienia](#12-co-jest-do-zrobienia)

---

## 1. O co chodzi w projekcie

Projekt sam szuka, jak sieć MLP najlepiej złożyć,\
żeby nikt nie musiał ręcznie warstw na chybił trafił mnożyć.

Dla każdej warstwy ukrytej mrówka wybiera trzy rzeczy do ustawienia:\
liczbę neuronów, funkcję aktywacji i dropout, który wyłącza część neuronów podczas uczenia.

Wybiera też, ile takich warstw ma być w sieci w sumie,\
a z każdą iteracją kolonia coraz więcej z tego rozumie.

Algorytm to ACO, czyli optymalizacja kolonią mrówek, z natury wzięta,\
a inspiracją był DeepSwarm, choć nasza implementacja jest od zera zaczęta.

Testy zrobimy na danych z tabel, bo MLP do obrazków się nie nadaje,\
czyli na Adult, Wine Quality i Letter Recognition, które każdy z OpenML i UCI dostaje.

Każdy zbiór dzielimy stratyfikowanie na trening, walidację i test w proporcji 60/20/20, jak w planie,\
a zbioru testowego mrówki nigdy nie widzą, bo tylko zwycięska sieć dostaje go na ostatnie sprawdzanie.

Projekt powstaje na przedmiot Algorytmy Inteligentne Inspirowane Naturą,\
a pełny plan pracy jest w PDF-ie zespołu, razem z eksperymentami i całą strukturą.

---

## 2. ACO od zera

### Mrówki w przyrodzie

Prawdziwe mrówki na początku szukają jedzenia całkiem losowo,\
a gdy znajdą dobre źródło, zostawiają feromon i wracają tam na nowo.

Kolejne mrówki chętniej idą tam, gdzie feromonu jest więcej,\
więc dobre trasy rosną w siłę coraz prędzej i prędzej.

Feromon z czasem paruje, więc nieużywana trasa słabnie i blednie,\
dlatego kolonia nie trzyma się kurczowo starej drogi niepotrzebnie.

### Mrówki w kodzie

W kodzie mrówka to procedura, która buduje rozwiązanie krok po kroku,\
a feromon to liczba przy każdej możliwej decyzji, którą mamy zawsze na oku.

U nas mrówka buduje sieć, dokładając kolejne warstwy po kolei,\
a feromon mówi jej, które wybory dawały najwięcej nadziei.

### Jedna iteracja krok po kroku

1. Każda mrówka buduje swoją sieć warstwa po warstwie, z feromonami w tle,\
   aż wybierze `END` albo limit głębokości zatrzyma ją w tej grze.
2. Z jej ścieżki powstaje sieć, którą trenujemy i sprawdzamy na walidacji,\
   a dokładność to `score` mrówki, czyli miara jej racji.
3. Ścieżka mrówki dostaje aktualizację lokalną, która feromon na niej zmniejsza,\
   żeby następna mrówka nie szła tą samą drogą i eksploracja była większa.
4. Gdy wszystkie mrówki skończą, najlepsza dostaje globalną nagrodę,\
   więc jej wybory zyskują feromon i kolejne mrówki chętniej wchodzą na tę drogę.
5. Potem zaczyna się kolejna iteracja i wszystko leci od nowa,\
   aż skończą się iteracje, a najlepsza sieć będzie gotowa.

Mrówki w jednej iteracji chodzą u nas po kolei, a nie jednocześnie,\
więc każda kolejna widzi feromony zmienione przez mrówki, które szły wcześniej.

### Jak mrówka wybiera, czyli reguła ACS

Do wyboru opcji używamy reguły ACS, która ma dwa tryby działania,\
a o tym, który zadziała, decyduje parametr `greediness` i jedno losowanie.

Z prawdopodobieństwem `greediness`, w literaturze zwanym `q0`, mrówka bierze opcję z największym feromonem bez wahania,\
a w przeciwnym razie kręci ruletką, w której większy feromon daje większą szansę wylosowania.

```text
r = losowa liczba z [0, 1)
jeśli r < q0:        wybierz argmax(τ)                    # tryb zachłanny, remis rozstrzyga los
w przeciwnym razie:  wybierz i z prawdopodobieństwem τ_i / Σ τ   # ruletka
```

Przy remisie w trybie zachłannym decyduje los, żeby żadna opcja nie wygrywała tylko dlatego, że stoi pierwsza w kolejce,\
a maska blokuje opcje zakazane, na przykład `END`, gdy sieć nie ma jeszcze `min_depth` warstw, a musi mieć ich więcej.

Nie mamy heurystyki η, więc decyzję podejmujemy wyłącznie na podstawie feromonu, bez dodatkowej wagi,\
i na starcie każda opcja jest równie prawdopodobna, bo wszystkie mają `τ0` i żadna nie ma przewagi.

### Aktualizacja lokalna i globalna

Lokalna aktualizacja przesuwa użyty feromon w stronę wartości startowej z powrotem,\
więc ta komórka jest dla kolejnych mrówek trochę mniej kusząca potem.

```text
τ ← (1 − ρ) · τ + ρ · τ0        ρ = pheromone.local_decay,   τ0 = pheromone.start
```

Globalna aktualizacja przesuwa feromon w stronę wyniku najlepszej mrówki,\
więc im lepsza była jej sieć, tym silniejsze zostawia wskazówki.

```text
τ ← (1 − α) · τ + α · score     α = pheromone.evaporation,   score = dokładność walidacyjna
```

Którą mrówkę nagradzamy, ustala `global_update_ant` w konfiguracji:\
`global_best` to najlepsza z całego przebiegu, a `iteration_best` to najlepsza z bieżącej iteracji.

Obie aktualizacje zmieniają tylko komórki leżące na ścieżce mrówki,\
a reszta tablicy zostaje bez zmian, bez żadnej przeróbki.

Aktualizacja lokalna na komórce, która wciąż ma `τ0`, nic nie zmienia, bo jej wartość już jest startowa,\
więc komórki, których najlepsza mrówka nie dotknęła, zostają na `τ0` — i to nie awaria, tylko rzecz całkiem zdrowa.

### Progresywna głębokość

Gdy `progressive_depth` jest włączone, limit głębokości rośnie o jedną warstwę na każdą iterację,\
więc najpierw sprawdzamy płytkie sieci, a głębsze dopiero później, gdy feromon ma już jakąś informację.

```text
depth_limit(i) = min(max(min_depth, 1) + i, max_depth)    # progresywnie
depth_limit(i) = max_depth                                 # bez progresji
```

Mrówka może skończyć wcześniej, wybierając `END`, ale nie przejdzie limitu tej iteracji,\
a gdy go osiągnie, budowa po prostu się urywa, bez żadnej dodatkowej akcji.

---

## 3. DeepSwarm, czyli skąd pomysł

DeepSwarm to biblioteka, która mrówkami szukała sieci konwolucyjnych do obrazków,\
a my bierzemy z niej pomysły, ale do MLP i danych z tabel, bez jej kodu i bez dodatków.

Najważniejsze cechy DeepSwarm wyglądają tak, w skrócie i po kolei,\
żebyście wiedzieli, skąd się wzięło to, co nasz kod dalej powieli.

- Graf rośnie leniwie: węzły powstają dopiero wtedy, gdy mrówka dojdzie tam, gdzie ich jeszcze brak,\
  a na każdej głębokości jest jeden węzeł na typ warstwy, więc mrówki go dzielą i graf jest DAG-iem, a nie drzewem, o tak.
- Typ warstwy mrówka wybiera z feromonu pomnożonego przez stałą heurystykę zapisaną w konfiguracji,\
  a ustawienia warstwy, jak rozmiar filtra, wybiera osobno ze słownika feromonów w węźle, już bez heurystyki i bez komplikacji.
- Maksymalna głębokość rośnie z każdą iteracją, więc najpierw powstają sieci płytkie,\
  a głębsze przychodzą później, gdy feromon już odsiał pomysły brzydkie.
- Wytrenowane wagi zapisywane są pod hashem prefiksu ścieżki w pamięci,\
  więc mrówka idąca tą samą drogą startuje z gotowych wag i trening szybciej się kręci.
- Węzeł wyjściowy działa jak nasze `END`, bo kończy budowę sieci,\
  a najlepsza mrówka dostaje globalną aktualizację, gdy cała iteracja przeleci.

Czym różnimy się od DeepSwarm, mówi poniższa lista, krótka i szczera,\
bo bez niej łatwo się pogubić, gdy ktoś oba rozwiązania porównuje i rozbiera.

- Nie mamy grafu z obiektami węzłów, tylko tablice NumPy z feromonami,\
  bo w MLP jest tylko jeden typ warstwy, więc graf to głębokość i wybór, opisane liczbami.
- DeepSwarm ma feromony ustawień niezależne, a my dodaliśmy warianty zależne od tego, co było wcześniej,\
  bo w eksperymencie drugim sprawdzamy, czy z zależnościami mrówki szukają sprawniej.
- Nie używamy heurystyki η, więc wybór zależy tylko od tego, co feromon powie,\
  a ponowne użycie wag czeka w planach i na razie nie siedzi w żadnej metodzie ani w głowie.
- Decyzję o końcu podejmujemy jawnie, osobną tablicą `type` z wyborem `FC` albo `END`,\
  więc kolonia uczy się też, kiedy przestać, i widać to w danych jako wyraźny trend.

---

## 4. Graf i tablice feromonów

Każdy feromon siedzi w klasie `PheromoneTable`, w trójwymiarowej tablicy NumPy w środku,\
a jej indeksy to `[depth, context, value]`, więc omówmy je po kolei, w porządku.

- `depth` to numer warstwy, którą mrówka teraz wybiera, liczony od zera,\
  więc ta sama opcja na warstwie pierwszej i trzeciej ma osobny feromon, który osobno się zbiera.
- `context` mówi, od czego zależy wybór, czyli jaka była poprzednia warstwa w kolejce,\
  a w wariancie niezależnym jest zawsze zero, bo przeszłości nikt tam nie trzyma w ręce.
- `value` to indeks opcji do wybrania, na przykład pozycja liczby neuronów na liście,\
  a liczba w komórce to feromon, czyli jak mocno kolonia ten wybór poleca, oczywiście.

Każda strategia z feromonami ma też tablicę `type`, która ma tylko dwie kolumny do wyboru:\
`FC` dokłada kolejną warstwę, a `END` kończy budowę sieci bez sporu.

Graf nie istnieje jako osobny obiekt, bo tablice już go w pełni opisują,\
węzłami są pary głębokość i kontekst, a krawędziami komórki, które feromon przechowują.

| Strategia | `type` | `neurons` | `activation` | `dropout` | `config` |
|---|---|---|---|---|---|
| `independent` | (6, 1, 2) | (6, 1, 5) | (6, 1, 4) | (6, 1, 4) | – |
| `param_dependent` | (6, 6, 2) | (6, 6, 5) | (6, 5, 4) | (6, 5, 4) | – |
| `config_dependent` | (6, 81, 2) | – | – | – | (6, 81, 80) |
| `random` | – | – | – | – | – |

Kształty dotyczą domyślnej przestrzeni: pięć opcji neuronów, a aktywacji i dropoutów po cztery,\
przy głębokości sześć, a pełnych konfiguracji warstwy jest osiemdziesiąt, bo tyle daje pięć razy cztery razy cztery.

### Przykład przejścia mrówki w `param_dependent`

Prześledźmy jedną mrówkę w strategii `param_dependent` krok po kroku,\
z `min_depth = 1` i limitem czterech warstw, żeby nic nie umknęło naszemu oku.

| Głębokość | Kontekst (`prev`) | Maska `type` | Wybór |
|---|---|---|---|
| 0 | brak poprzedniej warstwy → kontekst 0 | `END` zablokowany | `FC` → 64, relu, 0.1 |
| 1 | 64 neurony (indeks 2) → kontekst 3 | wolna | `FC` → 32, relu, 0.0 |
| 2 | 32 neurony (indeks 1) → kontekst 2 | wolna | `END` |

Wynik to `steps = [warstwa 64, warstwa 32, END]`, a do Evaluatora trafia ścieżka bez `END`, czyli dwie warstwy, bez dodatków,\
choć limit pozwalał na cztery, bo mrówka sama uznała, że dość już tych klocków.

Na głębokości zero mrówka czyta wiersz `τ_neurons[0, 0, :]`, bo przed nią nie ma jeszcze żadnej warstwy,\
a na dalszych głębokościach kontekst zero się nie pojawia, więc tam ten wiersz jest praktycznie martwy.

Na głębokości jeden czyta `τ_neurons[1, 3, :]`, bo 64 ma na liście indeks 2, a jedynkę dodajemy za start,\
więc każdy kontekst jest przesunięty o jeden, i tego szczegółu trzeba się trzymać, bo jest go wart.

---

## 5. Strategie i czym się różnią

Wszystkie strategie dziedziczą po `PheromoneStrategy` i mają ten sam zestaw metod,\
a różnią się tym, od czego zależą feromony, i to jest ich istnienia cały powód.

### `independent`: feromony niezależne

W strategii `independent` każdy parametr ma tablicę, gdzie kontekst jest zawsze zerowy,\
więc wybór zależy tylko od głębokości, a poprzednia warstwa nie zaprząta mrówce głowy.

To najprostszy wariant, który uczy się najszybciej, bo ma najmniej komórek do wypełnienia,\
ale nie umie zapamiętać, że dwie konkretne warstwy dobrze do siebie pasują, i to są jego ograniczenia.

### `param_dependent`: zależne parametr→parametr (domyślna)

W strategii `param_dependent` każdy parametr patrzy wstecz na samego siebie,\
czyli wybór neuronów zależy od neuronów warstwy poprzedniej, i tak przy każdej potrzebie.

Dzięki temu kolonia może się nauczyć przejść, na przykład że po 128 neuronach dobrze idzie 64, a nie odwrotnie,\
a każda tablica ma o jeden kontekst więcej niż opcji, bo kontekst zero jest dla startu, osobno i samotnie.

Decyzja `END` patrzy tu na liczbę neuronów w poprzedniej warstwie, bo stała `END_CONTEXT_PARAM` tak to ustawia,\
więc kolonia może się nauczyć, że po małej warstwie warto skończyć, jeśli to się w wynikach objawia.

To kompromis, bo zależności są, a tablice wciąż są małe i uczą się w miarę szybko,\
choć zależności między różnymi parametrami, jak neurony i aktywacja, nie są tu nawet blisko.

To strategia domyślna w `ColonyConfig`, kompromis między prostotą a pełną zależnością,\
i nie ma jej w PDF-ie, więc w raporcie trzeba ją opisać z należytą starannością.

### `config_dependent`: zależne konfiguracja→konfiguracja

W strategii `config_dependent` cała warstwa to jedna opcja, czyli neurony, aktywacja i dropout naraz,\
a w domyślnej przestrzeni takich konfiguracji jest osiemdziesiąt i spośród nich mrówka wybiera teraz.

Feromon zależy tu od całej poprzedniej konfiguracji, więc model zależności jest najpełniejszy,\
ale tablica ma 81 kontekstów na 80 opcji na każdą warstwę, więc potrzebuje ocen najwięcej, a jej budżet musi być największy.

To właśnie ten wariant opisuje PDF jako feromony zależne od przejść między konfiguracjami,\
a decyzja `END` korzysta z tych samych kontekstów, więc kończenie zależy od poprzedniej warstwy razem z jej parametrami.

Numer konfiguracji liczy funkcja `config_index`, a `config_at` zamienia numer z powrotem w trójkę ustawień,\
a szczegóły są w sekcji o trudnych momentach, wśród innych wyjaśnień.

### `random`: Random Search

Random Search to punkt odniesienia, który podejmuje te same decyzje, ale wszystko losuje równo, bez żadnej pamięci,\
żeby w eksperymencie pierwszym sprawdzić, czy feromony w czymś pomagają, czy ACO tylko ładnie się kręci.

Na każdym kroku kończy z szansą pół na pół, więc jego sieci są zwykle krótkie,\
dlatego przy porównaniu trzeba pilnować budżetu, bo inaczej wyjdą z tego dziwne skutki.

Nie ma żadnych tablic, więc `snapshot()` daje pusty słownik, a aktualizacje nic nie zmieniają,\
ale pętla, przestrzeń i Evaluator są te same, więc wyniki obu metod uczciwie się zestawiają.

### Porównanie w jednej tabeli

| Strategia | Od czego zależą ustawienia warstwy | Od czego zależy `END` | Liczba komórek | Tempo nauki |
|---|---|---|---|---|
| `independent` | tylko głębokość | tylko głębokość | najmniej | najszybciej |
| `param_dependent` | głębokość + ten sam parametr poprzedniej warstwy | głębokość + neurony poprzedniej warstwy | średnio | średnio |
| `config_dependent` | głębokość + cała poprzednia konfiguracja | głębokość + cała poprzednia konfiguracja | najwięcej | najwolniej |
| `random` | nic (rozkład jednostajny) | nic (50/50) | brak tablic | wcale |

W tabeli widać zasadę: im więcej zależności, tym więcej komórek do nauczenia,\
a eksperyment drugi sprawdzi, czy taka pamięć jest warta docenienia.

---

## 6. Przebieg wyszukiwania i hooki

`Colony` steruje pętlą, a strategia i obserwatory tylko reagują na zdarzenia, które ona im zgłasza,\
więc żeby zrozumieć kod, warto znać kolejność hooków, którą poniższy schemat ogłasza.

```text
Colony.run()
├─ seed_everything(seed)
├─ strategy.on_search_start(SearchContext)             → zapamiętuje ustawienia i woła build_tables()
├─ observers.on_search_start(ctx, snapshot)
├─ dla każdej iteracji:
│   ├─ strategy.on_iteration_start(IterationContext)   → domyślnie nic
│   ├─ observers.on_iteration_start(ctx, snapshot)
│   ├─ dla każdej mrówki (_run_ant):
│   │   ├─ _walk: dla każdej głębokości < depth_limit:
│   │   │   ├─ strategy.on_ant_movement(MoveContext)  → zwraca LayerConfig albo END
│   │   │   ├─ strategy.on_ant_step_end(ctx, step)    → lokalna aktualizacja, gdy local_update = "per_move"
│   │   │   ├─ observers.on_ant_step_end(ctx, step, snapshot)
│   │   │   └─ koniec pętli, jeśli step to END
│   │   ├─ evaluator.evaluate_path(ant.path, reuse_weights)
│   │   ├─ strategy.on_local_step_end(AntContext)     → lokalna aktualizacja, gdy local_update = "per_ant"
│   │   └─ observers.on_local_step_end(ctx, snapshot)
│   ├─ wybór iteration_best, best_ant i update_ant
│   ├─ strategy.on_global_step_end(IterationContext)  → globalna aktualizacja ścieżki update_ant
│   └─ observers.on_global_step_end(ctx, snapshot)
├─ observers.on_search_end(best_ant, snapshot)
└─ return best_ant
```

Strategia zawsze dostaje zdarzenie pierwsza, a obserwatory zaraz po niej, w drugiej kolejności,\
więc obserwator widzi feromony już po aktualizacji, bez żadnych opóźnień i bez zaległości.

| Kontekst | Pola | Kiedy przychodzi |
|---|---|---|
| `SearchContext` | `space`, `config`, `rng` | raz, na start |
| `IterationContext` | `iteration`, `depth_limit`, `ants`, `best_ant`, `update_ant` | na początku iteracji (puste `ants`) i na jej końcu (pełne) |
| `MoveContext` | `depth`, `prev`, `steps`, `depth_limit`, `can_end` | przy każdym kroku mrówki |
| `AntContext` | `iteration`, `ant` | po ocenie każdej mrówki |

Każdy kontekst to zamrożona dataklasa, więc po utworzeniu nikt jej nie zmienia,\
i żaden hook nie popsuje danych innym, co oszczędza dużo debugowania i cierpienia.

---

## 7. Struktura repozytorium

```text
ACO_nn_model_search/
├─ ACO/                        # cały algorytm jako pakiet
│  ├─ __init__.py              # publiczne API pakietu
│  ├─ Colony.py                # konfiguracja, mrówka i pętla wyszukiwania
│  ├─ Evaluator.py             # trening i ocena sieci (na razie zaślepka)
│  ├─ utils.py                 # LayerConfig, END, reguła ACS, PheromoneTable, ziarna
│  ├─ Strategies/              # modele feromonów
│  │  ├─ __init__.py           # rejestr STRATEGIES i typ StrategyName
│  │  ├─ base.py               # konteksty i klasa bazowa PheromoneStrategy
│  │  ├─ independent.py
│  │  ├─ param_dependent.py
│  │  ├─ config_dependent.py
│  │  └─ random_search.py
│  └─ Observers/               # słuchacze zdarzeń (logowanie, wykresy)
│     ├─ __init__.py
│     ├─ base.py               # BaseObserver z pustymi hookami
│     └─ Observers.py          # grupa obserwatorów działająca jak jeden
├─ data/                       # zbiory danych (na razie tylko .gitkeep)
├─ runs/                       # wyniki przebiegów (na razie tylko .gitkeep)
├─ main.py                     # punkt wejścia (na razie Hello World)
├─ settings.py                 # ścieżki projektu (na razie pusty)
├─ pyproject.toml              # zależności i indeks PyTorch z CUDA
├─ uv.lock                     # zablokowane wersje pakietów
└─ .python-version             # Python 3.12 dla uv
```

Cały algorytm siedzi w pakiecie `ACO`, a reszta repozytorium to na razie konfiguracja i puste foldery na dane,\
które czekają na wczytywanie zbiorów i wyniki eksperymentów, już zaplanowane.

---

## 8. Plik po pliku

### `pyproject.toml`, `uv.lock` i `.python-version`

Plik `pyproject.toml` wymaga Pythona 3.12 i wymienia biblioteki potrzebne do działania,\
czyli PyTorch, NumPy, scikit-learn, pandas, Matplotlib, Seaborn, MLflow i pydantic do sprawdzania.

Python jest przypięty do 3.12, bo nowsze wersje gorzej dogadują się z CUDA i PyTorchem,\
a `.python-version` mówi `uv`, której wersji użyć, żeby nikt nie walczył z tym potem.

PyTorch pobieramy z osobnego indeksu `pytorch-cu130`, żeby sieci liczyły się na karcie, a nie na procesorze,\
a `uv.lock` pilnuje dokładnych wersji, więc każdy dostaje to samo środowisko o każdej porze.

Indeks ma ustawione `explicit = true`, więc tylko `torch` jest z niego brany,\
a cała reszta pakietów przychodzi ze zwykłego PyPI, bez zmiany.

### `main.py`, `settings.py`, `data/` i `runs/`

Plik `main.py` to na razie tylko Hello World, które `uv init` stworzył na początek,\
a docelowo ma uruchamiać eksperymenty, gdy reszta kodu da z siebie pożytek.

Plik `settings.py` jest pusty i ma trzymać wyłącznie ścieżki do danych, modeli i wyników,\
bo parametry algorytmu siedzą w `ColonyConfig` i nie mieszamy tych dwóch rodzajów ustawień w jednym z plików.

Foldery `data/` i `runs/` mają tylko pliki `.gitkeep`, żeby Git je w ogóle zachował,\
bo pustych folderów Git nie śledzi i bez tego pliku folder by się schował.

### `ACO/__init__.py`

Ten plik wystawia publiczne API pakietu, czyli to, czego zwykle potrzeba na zewnątrz,\
więc wystarczy `from ACO import Colony, ColonyConfig`, a szczegóły zostają wewnątrz.

Na zewnątrz wychodzą `Ant`, `Colony`, `ColonyConfig`, `PheromoneConfig`, `SearchSpace`, `EvaluationResult`, `Evaluator`, `BaseObserver`, `Observers`, `END` i `LayerConfig` w jednym pakiecie,\
a strategie i konteksty importuje się z podpakietu `ACO.Strategies`, gdy ktoś pisze własną strategię albo obserwatora na tym świecie.

### `ACO/utils.py`

Tu leżą klocki, z których korzysta cały pakiet, a same nie importują niczego z pakietu,\
dzięki czemu każdy plik może je wczytać bez pętli importów i bez żadnego kłopotu.

- `LayerConfig` to zamrożona dataklasa, która opisuje jedną ukrytą warstwę trzema polami,\
  czyli `neurons`, `activation` i `dropout`, i da się ją hashować oraz porównywać z innymi warstwami.
- `END` to jedyna wartość enuma `End`, którą mrówka zwraca, gdy kończy budowę sieci,\
  a typ `Step` to warstwa albo koniec, więc nic innego do ścieżki nie wleci.
- Enum zamiast zwykłego napisu daje pewność, że `END` nie pomyli się z nazwą warstwy ani aktywacji,\
  a porównanie `step is END` jest szybkie i jednoznaczne w każdej sytuacji.
- `PARAMS` to krotka nazw parametrów, które mrówka wybiera dla każdej warstwy w ustalonej kolejności,\
  a `SearchSpace` ma listy opcji pod dokładnie tymi samymi nazwami, co upraszcza pętle bez zbędnej złożoności.
- `acs_select` to reguła ACS z sekcji drugiej, która dostaje wartości, `q0`, generator i opcjonalną maskę,\
  a zwraca indeks wybranej opcji, i to ona daje mrówkom tę losowo-zachłanną łaskę.
- Gdy maska blokuje wszystko, `acs_select` rzuca `ValueError`, a gdy suma feromonów wynosi zero, ruletka losuje jednostajnie,\
  więc wyzerowane feromony nie wywalą programu, tylko mrówka wybierze zwyczajnie.
- `PheromoneTable` trzyma tablicę `tau` o kształcie `(depth, n_contexts, n_values)`, na starcie wypełnioną wszędzie jednakowo,\
  a jej metody `select`, `local_update` i `global_update` to wybór i dwa wzory aktualizacji z sekcji drugiej, przepisane słowo w słowo.
- `seed_everything` ustawia ziarno dla `random`, NumPy i PyTorcha jednym wywołaniem,\
  żeby przebieg z tym samym ziarnem dał te same wyniki z każdym kolejnym uruchamianiem.

### `ACO/Colony.py`

To serce projektu, gdzie jest konfiguracja, mrówka i cała pętla wyszukiwania,\
więc jeśli ktoś ma przeczytać tylko jeden plik, to właśnie ten jest do czytania.

**`SearchSpace`** to model pydantic z listami opcji `neurons`, `activation` i `dropout` oraz z `min_depth` i `max_depth` jako granicami,\
a walidator sprawdza, czy `min_depth` nie przekracza `max_depth` i czy na listach nie ma duplikatów między opcjami.

Metody `options` i `value_index` zwracają listę opcji i pozycję wartości na niej, żeby strategie nie musiały nic liczyć ręcznie,\
a `shape`, `n_configs`, `config_index` i `config_at` numerują pełne konfiguracje warstwy i robią to szybko i zręcznie.

**`PheromoneConfig`** ma trzy liczby: `start`, czyli `τ0`, `local_decay`, czyli ρ dla aktualizacji lokalnej, i `evaporation`, czyli α dla globalnej,\
a domyślnie wszystkie wynoszą 0.1, jak w wersji wyjściowej i normalnej.

**`ColonyConfig`** to główna konfiguracja przebiegu, z rozsądną wartością domyślną dla każdego pola,\
więc w JSON-ie podajesz tylko to, co zmieniasz, a resztę uzupełnia pydantic, bo to jego rola.

| Pole | Domyślnie | Znaczenie |
|---|---|---|
| `strategy` | `"param_dependent"` | strategia feromonów: `independent`, `param_dependent`, `config_dependent` albo `random` |
| `n_ants` | `10` | liczba mrówek w jednej iteracji |
| `n_iterations` | `6` | liczba iteracji |
| `greediness` | `0.5` | `q0`, czyli szansa na wybór zachłanny |
| `progressive_depth` | `True` | czy limit głębokości rośnie z iteracjami |
| `local_update` | `"per_ant"` | aktualizacja lokalna po całej mrówce (`per_ant`) albo po każdym kroku (`per_move`) |
| `global_update_ant` | `"global_best"` | która mrówka dostaje aktualizację globalną (`global_best` albo `iteration_best`) |
| `reuse_weights` | `True` | czy Evaluator ma używać wag ponownie (jeszcze niezaimplementowane) |
| `seed` | `42` | ziarno losowości |
| `search_space` | `SearchSpace()` | opcje warstw i granice głębokości |
| `pheromone` | `PheromoneConfig()` | `τ0`, ρ i α |

**`Ant`** to dataklasa, która trzyma numer iteracji, listę kroków `steps` i wynik `result` z Evaluatora po treningu,\
a dopóki wyniku nie ma, `result` to `None`, bo mrówka jest jeszcze w biegu.

Właściwość `path` zwraca same warstwy bez `END`, czyli dokładnie to, z czego Evaluator ma zbudować sieć,\
a `score` zwraca `val_accuracy` albo rzuca `RuntimeError`, gdy mrówki nie oceniono, żeby nikt nie liczył na wynik, którego nie ma, i nie dał się zwieść.

Metoda `transitions` zwraca trójki `(depth, prev, step)` dla każdego kroku, gdzie `prev` to poprzednia warstwa albo `None` na początku,\
a z tych trójek obie aktualizacje odtwarzają komórki, przez które mrówka szła, w odpowiednim porządku.

**`Colony`** przyjmuje konfigurację, Evaluator, strategię i obserwatorów, a gdy czegoś brak, sięga po to, co jest gotowe,\
więc strategia powstaje z nazwy w konfiguracji przez słownik `STRATEGIES`, i to rozwiązanie zwykłe i wzorcowe.

Metoda klasowa `from_config` czyta plik JSON i tworzy z niego `ColonyConfig`, a brakujące pola same się uzupełniają,\
więc w pliku trzymasz tylko różnice, a resztę wartości domyślne ustalają.

Metoda `run` ustawia ziarno, woła hooki w kolejności ze schematu w sekcji szóstej i na końcu zwraca najlepszą mrówkę z całego przebiegu,\
a najlepsza mrówka zmienia się tylko wtedy, gdy nowa ma wynik ostro większy, więc przy remisie wygrywa ta, która była pierwsza w biegu.

Metoda `register_observer` przyjmuje jednego obserwatora, listę, krotkę albo całą grupę `Observers` i dopisuje je do kolonii,\
a przypadek grupy sprawdzany jest przed pojedynczym obserwatorem, bo grupa też jest obserwatorem, i inaczej trafiłaby w złą gałąź, wbrew harmonii.

Metoda `depth_limit` liczy limit głębokości dla iteracji według wzoru z sekcji drugiej,\
a `_walk` prowadzi mrówkę krok po kroku, i `_run_ant` dokłada do tego ocenę i aktualizację lokalną po tej drodze długiej.

### `ACO/Evaluator.py`

To na razie zaślepka, w której każda metoda ma tylko `pass`, bo trening zaprojektujemy razem z obsługą danych,\
a interfejs jest już ustalony, więc reszta kodu wie, czego się spodziewać od metod zaplanowanych.

`EvaluationResult` to wynik oceny z dokładnością i stratą walidacyjną, czasem treningu i liczbą parametrów w sieci,\
a `Colony` używa tylko `evaluate_path` i bierze `val_accuracy` jako `score`, więc reszta to wewnętrzne sprawy Evaluatora, gdy ktoś go napisze i zadanie mu się zleci.

| Metoda | Co ma robić |
|---|---|
| `evaluate_path(path, reuse_weights)` | cały proces dla jednej mrówki: budowa, opcjonalnie wczytanie wag, trening, ocena, zapis wag |
| `build_model(path)` | `nn.Sequential` z `Linear → aktywacja → Dropout` dla każdej warstwy i warstwą wyjściową na końcu |
| `load_reusable_weights(path, model)` | kopiowanie wag z zapisanego modelu o najdłuższym wspólnym prefiksie ścieżki |
| `store_weights(path, model, result)` | zapis modelu, jeśli jest najlepszy dla któregoś ze swoich prefiksów |
| `train(model)` | trening na zbiorze treningowym |
| `evaluate(model)` | ocena na walidacji i zwrot `EvaluationResult` |

Uwaga: z tą zaślepką `Colony.run()` wywróci się już przy pierwszym wyborze najlepszej mrówki,\
bo `evaluate_path` zwraca `None`, a `ant.score` rzuca wtedy `RuntimeError`, więc do testów podstaw sztuczny Evaluator, zgodnie z sekcją jedenastą, gdzie są gotowe wskazówki.

### `ACO/Strategies/__init__.py`

Tu jest słownik `STRATEGIES`, który mapuje nazwę strategii z konfiguracji na jej klasę,\
a typ `StrategyName` to `Literal` z tymi samymi nazwami, żeby pydantic odrzucił literówkę, zanim zmarnujesz na liczenie czas i kasę.

Nową strategię dodajesz w trzech prostych krokach:\
piszesz klasę dziedziczącą po `PheromoneStrategy`, dopisujesz jej nazwę do `StrategyName` i klasę do `STRATEGIES`, a `Colony` niczego więcej nie potrzebuje, bez zmian w innych blokach.

### `ACO/Strategies/base.py`

Ten plik to fundament wszystkich strategii: konteksty dla hooków i klasa bazowa, która robi robotę wspólną,\
a konkretna strategia dopisuje tylko swoje tablice i wybór, bo resztę dziedziczy jako część ogólną.

`TYPE_FC = 0` i `TYPE_END = 1` to numery kolumn w tablicy `type`, żeby nikt nie pisał gołych zer i jedynek w kodzie,\
a `Cell` to trójka `(tabela, kontekst, wartość)`, czyli adres jednej komórki, z którym obie aktualizacje żyją w zgodzie.

Funkcja `type_mask` zwraca maskę, która blokuje `END`, dopóki sieć nie ma `min_depth` warstw, a potem nie blokuje niczego wcale,\
więc mrówka nie skończy za wcześnie i pusta sieć nie trafi do treningu ani teraz, ani dalej.

Strategia musi zaimplementować trzy metody abstrakcyjne, a wszystko inne dostaje w spadku,\
i oto one w skrócie, bez żadnego wyjątku.

- `build_tables` tworzy tablice w słowniku `self.tables`, z rozmiarami wziętymi z przestrzeni wyszukiwania,\
  i woła ją `on_search_start`, więc tablice są gotowe, zanim zacznie się pierwsze losowanie.
- `on_ant_movement` dostaje `MoveContext` i zwraca kolejny krok, czyli warstwę albo `END`, w roli wyboru,\
  i to jedyny hook, który podejmuje decyzję, a pozostałe tylko reagują i nie mają w tym żadnego głosu ani sporu.
- `cells(depth, prev, step)` zwraca listę komórek feromonu, które obejmuje dany krok, jako trójki `Cell`,\
  i to jedno źródło prawdy dla obu aktualizacji, więc każda strategia opisuje swoje komórki tylko raz, i to jest ich cel.

Hook `on_search_start` zapisuje przestrzeń, konfigurację i generator losowy, a potem buduje tablice,\
a `on_iteration_start` domyślnie nic nie robi i czeka na przyszłe strategie, jak pusta półka w piwnicy.

Hooki `on_ant_step_end` i `on_local_step_end` robią aktualizację lokalną, ale zawsze tylko jeden z nich, zależnie od `local_update` w konfiguracji,\
a `on_global_step_end` robi aktualizację globalną na ścieżce `update_ant` na koniec każdej iteracji.

Metoda `snapshot` zwraca kopie wszystkich tablic w słowniku, żeby obserwator mógł je zachować i rysować z nich mapy cieplne,\
bo bez kopii każda zapisana mapa pokazywałaby stan po ostatniej aktualizacji, a wszystkie byłyby identyczne i bezcelne.

Metody pomocnicze `select` i `_local_update` same podają `greediness`, generator i `local_decay` z konfiguracji,\
więc w konkretnej strategii wystarczy nazwa tablicy, głębokość i kontekst, bez powtarzania tej samej informacji.

Konteksty importują `Ant`, `ColonyConfig` i `SearchSpace` tylko pod `TYPE_CHECKING`, bo inaczej powstałaby pętla importów między plikami,\
a szczegóły tego triku są w sekcji o trudnych momentach, razem z innymi takimi tematami.

### `independent.py`, `param_dependent.py`, `config_dependent.py` i `random_search.py`

Te cztery pliki implementują strategie z sekcji piątej, każdą w osobnym pliku i osobnej klasie,\
i każdy ma tylko `build_tables`, `on_ant_movement`, `cells` i ewentualnie pomocnicze `_context`, więc czyta się je w krótkim czasie.

| Plik | Klasa | Tablice | Jak liczy kontekst |
|---|---|---|---|
| `independent.py` | `IndependentStrategy` | `type`, `neurons`, `activation`, `dropout` | zawsze `0` |
| `param_dependent.py` | `ParamDependentStrategy` | `type`, `neurons`, `activation`, `dropout` | `0` na starcie, potem indeks wartości poprzedniej warstwy `+ 1` |
| `config_dependent.py` | `ConfigDependentStrategy` | `type`, `config` | `0` na starcie, potem `config_index(prev) + 1` |
| `random_search.py` | `RandomSearchStrategy` | brak | nie dotyczy |

### `ACO/Observers/base.py`

Tu jest `BaseObserver`, czyli klasa bazowa dla słuchaczy, którzy patrzą na przebieg, ale go nie zmieniają,\
a wszystkie hooki są puste, więc obserwator nadpisuje tylko te, które coś mu dają.

Hooki są te same co w strategii, tylko każdy dostaje jeszcze argument `pheromones`, czyli kopię tablic zaraz po aktualizacji,\
a dodatkowy hook `on_search_end` dostaje najlepszą mrówkę i końcowe feromony na pożegnanie, po ostatniej iteracji.

Obserwator nie ma `on_ant_movement`, bo wybór kroku należy do strategii, a obserwator tylko patrzy z boku,\
i wybrany krok i tak dostaje chwilę później w `on_ant_step_end`, więc nic mu nie umknie w tym kroku.

Typ `Pheromones` to po prostu `dict[str, np.ndarray]`, gdzie kluczem jest nazwa tablicy, a wartością jej kopia, jak zdjęcie w albumie,\
a dla Random Search ten słownik jest pusty, bo tam nie ma tablic, i obserwator musi to mieć w rozumie.

### `ACO/Observers/Observers.py`

Klasa `Observers` to lista obserwatorów, która sama udaje jednego obserwatora,\
więc kolonia woła jeden obiekt, a ten rozsyła zdarzenia wszystkim, w roli koordynatora.

Ten wzorzec nazywa się kompozytem, bo grupa ma ten sam interfejs co pojedynczy element,\
więc można nawet włożyć grupę w grupę i nic się nie psuje, bo każda warstwa działa w ten sam moment.

Metody `__iter__`, `__len__` i `__getitem__` sprawiają, że grupę można przeglądać pętlą, mierzyć funkcją `len` i indeksować jak tablicę,\
a `extend` kopiuje wejście, bo bez tego `obs.extend(obs)` rośnie bez końca, przekraczając każdą granicę.

Pusta grupa `Observers` ma długość zero, więc w warunku `if` zachowuje się jak fałsz, dlatego kod sprawdza `is not None`, a nie posługuje się samą prawdziwością,\
bo inaczej pusta grupa podana przez użytkownika zostałaby po cichu podmieniona na nową, co byłoby przykrą złośliwością.

### `ACO/Observers/__init__.py`

Ten plik wystawia `BaseObserver` i `Observers` na zewnątrz przez względne importy z kropką na początku,\
a `__all__` zawiera nazwy jako napisy, bo Python tak tego wymaga, i wszystko jest w porządku.

---

## 9. Najtrudniejsze momenty

### 9.1. Dlaczego kontekstów jest `n + 1`

W strategiach zależnych kontekst mówi, co było w poprzedniej warstwie, ale pierwsza warstwa nie ma poprzedniej na początku,\
więc kontekst zero zarezerwowano na start, a opcja o indeksie `i` dostaje kontekst `i + 1`, żeby wszystko było w porządku.

| Kontekst | Znaczenie dla `neurons = [16, 32, 64, 128, 256]` |
|---|---|
| 0 | brak poprzedniej warstwy, mrówka wybiera pierwszą warstwę |
| 1 | poprzednia warstwa miała 16 neuronów |
| 2 | poprzednia warstwa miała 32 neurony |
| 3 | poprzednia warstwa miała 64 neurony |
| 4 | poprzednia warstwa miała 128 neuronów |
| 5 | poprzednia warstwa miała 256 neuronów |

Bez tej jedynki pierwsza warstwa musiałaby udawać, że przed nią stało 16 neuronów, i dwie różne sytuacje zlałyby się w jedno,\
a tak start ma własny wiersz, więc kolonia uczy się, od czego zaczynać, osobno.

Skutkiem ubocznym są nieużywane wiersze, bo na głębokości zero działa tylko kontekst zero, a dalej kontekst zero nie występuje wcale,\
więc na mapach cieplnych te wiersze zostaną na `τ0` i to nie błąd, tylko cena prostego kształtu tablicy, płacona stale.

### 9.2. `END` a wymuszony koniec

Mrówka kończy budowę na dwa sposoby: albo sama wybiera `END`, albo dochodzi do limitu głębokości i pętla się kończy,\
a tylko ten pierwszy sposób dodaje krok `END` do ścieżki, więc tylko on z feromonem się łączy.

| Jak się kończy | `ant.steps` | Feromon na komórce `END` |
|---|---|---|
| mrówka wybrała `END` na głębokości `d` | `[L0, …, L(d-1), END]` | tak, przy aktualizacji lokalnej i globalnej |
| mrówka doszła do `depth_limit` | `[L0, …, L(limit-1)]`, bez `END` | nie, bo żadnej decyzji nie było |

Dzięki temu kolonia uczy się kończenia tylko z prawdziwych decyzji mrówek,\
a wymuszony koniec nie trafia do tablicy `type`, bo to była ściana, a nie wybór, więc nie daje kolonii żadnych wskazówek.

Na początku, gdy feromony są równe, szansa na `END` na każdym dozwolonym kroku wynosi pół na pół,\
więc połowa sieci kończy się po jednej warstwie, ćwierć po dwóch, i tak dalej, coraz mniej w dół.

Jeśli to za płytko, można podnieść `min_depth`, dać `FC` więcej feromonu na start albo wyłączyć `END` i budować do końca,\
a każda z tych zmian inaczej przesunie rozkład głębokości, więc sprawdźcie ją wcześniej, zanim eksperymenty ruszą i zabraknie wam czasu i słońca.

### 9.3. `per_ant` i `per_move` dają dziś ten sam wynik

Ustawienie `local_update` mówi, czy aktualizacja lokalna dzieje się po każdym kroku, czy po całej mrówce, już po jej ocenie,\
ale w obecnym kodzie obie opcje dają identyczne feromony i ścieżki, co potwierdził test na wszystkich strategiach, bez jednej różnicy w tej zamianie.

- Każda tablica ma wymiar głębokości, a mrówka odwiedza każdą głębokość tylko raz,\
  więc nie czyta komórki, którą sama przed chwilą zmieniła, i jej wybory są takie same za każdym razem.
- Aktualizacja lokalna nie zależy od wyniku, więc nie ma znaczenia, czy dzieje się przed oceną sieci, czy po niej,\
  a różne komórki nie wpływają na siebie, więc nie liczy się, która idzie po kolei.
- Mrówki chodzą po kolei, a cała praca jednej mrówki kończy się przed startem następnej w kolejce,\
  więc następna widzi dokładnie te same tablice, niezależnie od ustawienia, i nie dostaje niczego więcej.

Różnica pojawiłaby się dopiero przy mrówkach budujących sieci równolegle, krok po kroku, jak w klasycznym ACS, albo przy tablicach bez wymiaru głębokości,\
więc opcja zostaje w konfiguracji na przyszłość, choć dziś na nic nie wpływa, i warto to wiedzieć dla własnej pewności.

### 9.4. Słaba nagroda globalna

W teście na łatwym zbiorze wine najlepsza ścieżka co iterację dostawała nagrodę z wynikiem równym jeden, czyli idealnym,\
a mimo to po trzydziestu iteracjach jej feromon wynosił tylko około 0.25 przy 0.1 w reszcie tablicy, co jest wynikiem raczej banalnym.

Powód jest taki, że każda mrówka, która przejdzie tą samą komórką, ciągnie ją aktualizacją lokalną z powrotem w stronę `τ0`, a mrówek w iteracji jest dziesięć, a nagroda globalna tylko jedna,\
więc lokalne cofnięcia zjadają większość nagrody i najlepsza ścieżka zostaje dość biedna.

Dla eksploracji to dobrze, ale jeśli kolonia ma się szybciej skupić, trzeba zmniejszyć `local_decay` albo zwiększyć `evaporation` w konfiguracji,\
i warto to zbadać w eksperymencie trzecim, który dotyczy wpływu parametrów na przebieg optymalizacji.

### 9.5. Remisy przy najlepszym wyniku

`best_ant` zmienia się tylko wtedy, gdy nowa mrówka ma wynik ostro lepszy, więc przy remisie zostaje ta, która była pierwsza na mecie,\
a na łatwym zbiorze wiele sieci ma sto procent, więc pierwsza taka mrówka zgarnia nagrody na zawsze, a lepsze sieci z niższą stratą nie dostają nic, bo nikt ich nie zauważy w tym świecie.

Proste rozwiązanie to rozstrzygać remisy stratą walidacyjną albo użyć innej miary jako `score`, na przykład zbalansowanej dokładności przy niezbalansowanych klasach,\
i to jest na liście zadań, żeby wyniki na trudniejszych zbiorach nie stały w miejscu jak na wakacjach.

### 9.6. `config_index` i `np.ravel_multi_index`

Strategia `config_dependent` potrzebuje jednej liczby na całą warstwę, a mamy trzy indeksy: neuronów, aktywacji i dropoutu w szeregu,\
więc `np.ravel_multi_index` składa je w jedną liczbę tak, jak cyfry na liczniku w biegu.

```text
shape   = (5, 4, 4)                      # neurony, aktywacje, dropouty
indeksy = (2, 1, 3)                      # 64, tanh, 0.5
numer   = 2·(4·4) + 1·4 + 3 = 39         # np.ravel_multi_index(indeksy, shape)
np.unravel_index(39, shape) → (2, 1, 3)  # i z powrotem
```

Ostatni indeks zmienia się najszybciej, jak jedności na liczniku, a pierwszy najwolniej, więc każda trójka dostaje numer od 0 do 79 włącznie,\
a `np.unravel_index` robi to w drugą stronę, więc z numeru odtwarza trójkę równie zręcznie.

### 9.7. `tuple[X, ...]` i zamrożone konteksty

Zapis `tuple[Ant, ...]` to podpowiedź typu, która znaczy krotkę dowolnej długości, gdzie każdy element jest mrówką, a wielokropek mówi: i tak dalej, bez ograniczeń,\
natomiast `tuple[Ant]` to krotka dokładnie jednej mrówki, i to częste źródło nieporozumień.

Konteksty używają krotek zamiast list, bo są zamrożone, a krotki nie da się zmienić, więc żaden hook nie doda ani nie usunie mrówki z cudzej listy,\
a domyślne `()` to pusta krotka, bezpieczna jako wartość domyślna, bo pusta lista byłaby wspólna dla wszystkich obiektów, a to klasyczny błąd, wcale nie czysty.

### 9.8. Importy pod `TYPE_CHECKING`

`Colony.py` importuje strategie i obserwatorów, a one potrzebują typów `Ant` i `ColonyConfig` z `Colony.py` w swoich adnotacjach,\
więc zwykły import w obie strony stworzyłby pętlę, i Python wywaliłby się już przy samych deklaracjach.

Dlatego te typy importujemy w bloku `if TYPE_CHECKING:`, który widzi tylko edytor i sprawdzanie typów, a w czasie działania ten blok jest pomijany w całości,\
a `from __future__ import annotations` zamienia adnotacje w napisy, więc brak tych nazw w czasie działania nie robi żadnej przykrości.

Z tego samego powodu `LayerConfig` i `END` siedzą w `utils.py`, a nie w `Colony.py`, bo strategie potrzebują ich naprawdę, w czasie działania, a nie tylko w podpowiedziach,\
a `utils.py` nie importuje niczego z pakietu, więc wczytuje się pierwszy, bez żadnych kłopotów w powiązaniach.

### 9.9. `cells()` jako jedno źródło prawdy

Każda strategia ma metodę `cells`, która dla jednego kroku mówi, które komórki feromonu ten krok obejmuje,\
a obie aktualizacje z niej korzystają, więc opis komórek jest w jednym miejscu i nikt go nie dubluje.

Dla kroku `END` zwraca tylko komórkę `END` w tablicy `type`, a dla warstwy zwraca komórkę `FC` i po jednej komórce na parametr albo jedną komórkę konfiguracji,\
więc jeśli dodasz nową strategię i dobrze napiszesz `cells`, obie aktualizacje zadziałają same, bez żadnej modyfikacji.

### 9.10. Kolejność `match` w `register_observer`

Instrukcja `match` sprawdza przypadki od góry i bierze pierwszy pasujący, a zapis `case Klasa():` działa jak `isinstance`, więc kolejność ma tu znaczenie,\
bo `Observers` dziedziczy po `BaseObserver`, i gdyby `BaseObserver()` był pierwszy, cała grupa zostałaby dodana jako jeden obserwator, wbrew zamierzeniu.

Nawiasy w `case Klasa():` są obowiązkowe, bo bez nich Python uzna nazwę za nową zmienną, która złapie wszystko, nawet śmieci,\
i zgłosi błąd składni, że kolejne przypadki są nieosiągalne, więc taki błąd wychodzi, zanim program w ogóle przeleci.

### 9.11. Typ `best_ant` i komentarz dla Pylance

`best_ant` startuje jako `None`, więc jego typ to `Ant | None`, a Pylance nie wie, że po pętli zawsze jest ustawiony, bo konfiguracja wymaga co najmniej jednej iteracji i jednej mrówki w kolonii,\
dlatego przy `on_search_end` stoi `# pyright: ignore[reportArgumentType]`, który wycisza tylko ten jeden błąd w tej jednej linii, bez zbędnej ceremonii.

Czystszą alternatywą jest `assert self.best_ant is not None` tuż przed wywołaniem, bo wtedy Pylance sam zawęża typ i komentarz staje się zbędny,\
a `run` mógłby znów zwracać samo `Ant`, bo `None` po pętli to przypadek błędny.

---

## 10. Konfiguracja

Konfigurację można stworzyć w kodzie albo wczytać z pliku JSON przez `Colony.from_config`, a w pliku podajesz tylko to, co chcesz zmienić,\
a resztę pydantic uzupełni sam, więc pliki są krótkie i łatwo je porównać i ocenić.

```json
{
  "strategy": "config_dependent",
  "n_ants": 8,
  "n_iterations": 20,
  "greediness": 0.3,
  "progressive_depth": false,
  "search_space": { "max_depth": 4, "neurons": [32, 64, 128] },
  "pheromone": { "evaporation": 0.2 }
}
```

```python
from ACO import Colony

colony = Colony.from_config("configs/e2_config_dependent.json", evaluator=my_evaluator)
best = colony.run()
```

Błędna wartość, na przykład `greediness` większe od jeden albo nieznana nazwa strategii, zatrzyma program od razu z czytelnym błędem walidacji,\
więc literówka w konfiguracji nie zmarnuje godzin liczenia, tylko wyjdzie na jaw przed startem pierwszej iteracji.

---

## 11. Jak uruchomić

Środowisko instalujesz jednym poleceniem `uv sync`, które stworzy `.venv` z Pythonem 3.12 i wszystkimi pakietami,\
a PyTorch przyjdzie z CUDA, więc na karcie NVIDIA sieci będą się liczyć szybko, a nie ślimakami.

```bash
uv sync
```

Czy PyTorch widzi kartę graficzną, sprawdzisz jedną linijką w terminalu,\
a jeśli wypisze `True`, to GPU działa bez żadnego skandalu.

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Dopóki nie ma prawdziwego Evaluatora, kolonię testujesz sztucznym, który wynik wymyśla ze ścieżki od ręki,\
więc mechanikę mrówek, feromonów i hooków sprawdzisz w kilka sekund, bez treningu i bez męki.

```python
from ACO import BaseObserver, Colony, ColonyConfig, EvaluationResult


class FakeEvaluator:
    """Zamiast trenować, nagradza szerokie sieci."""

    def evaluate_path(self, path, reuse_weights=True):
        acc = sum(layer.neurons for layer in path) / 1000
        return EvaluationResult(val_accuracy=acc, val_loss=1 - acc, train_time=0.0, n_params=0)


class PrintBest(BaseObserver):
    def on_global_step_end(self, ctx, pheromones):
        print(f"iteracja {ctx.iteration}: najlepszy wynik {ctx.best_ant.score:.3f}")

    def on_search_end(self, best_ant, pheromones):
        print("najlepsza sieć:", best_ant.path)


config = ColonyConfig(strategy="independent", n_ants=5, n_iterations=5)
Colony(config, FakeEvaluator(), observers=PrintBest()).run()
```

Taki skrypt zapisz w katalogu głównym repozytorium i uruchom przez `uv run python nazwa_pliku.py`, a zobaczysz, jak wynik rośnie z iteracji na iterację,\
i właśnie tak sprawdzaliśmy hooki i strategie, zanim powstał prawdziwy trening i jego konfigurację.

---

## 12. Co jest do zrobienia

Lista wynika z planu w PDF-ie i z tego, co wyszło w testach, a zadania są ułożone od najpilniejszych,\
bo Evaluator i dane blokują wszystkie eksperymenty, więc stoją na liście najważniejszych.

### Evaluator

- [ ] Napisać `build_model`, który zamienia ścieżkę na `nn.Sequential` z `Linear`, aktywacją i `Dropout` dla każdej warstwy i z warstwą wyjściową na końcu,\
  i trzymać się nazw aktywacji z `SearchSpace`, czyli `relu`, `tanh`, `elu` i `leaky_relu`, żeby nic się nie rozjechało w tym tańcu.
- [ ] Napisać `train` i `evaluate` z treningiem na GPU, rozsądną liczbą epok i najlepiej z wczesnym zatrzymaniem,\
  a `evaluate` ma zwracać `EvaluationResult` z dokładnością, stratą, czasem treningu i liczbą parametrów, zgodnie z zadaniem.
- [ ] Dodać końcową ocenę najlepszej architektury na zbiorze testowym, którego mrówki nie dotykają,\
  bo tylko ten wynik idzie do raportu, a walidacja służy mrówkom do wyboru i z testem się nie spotykają.
- [ ] Opcjonalnie dopisać ponowne użycie wag, czyli `load_reusable_weights` i `store_weights`, z kluczem w postaci prefiksu ścieżki, jak w DeepSwarm to bywa,\
  bo wtedy mrówki z tym samym początkiem ścieżki zaczynają trening z gotowych wag, i oszczędność czasu jest prawdziwa.

### Dane

- [ ] Napisać wczytywanie zbiorów Adult, Wine Quality i Letter Recognition, na przykład przez `fetch_openml`, z zapisem w `data/` w pamięci podręcznej,\
  a ścieżki trzymać w `settings.py`, żeby nie były wpisane na sztywno w każdej funkcji osobno i nie robiły bałaganu w strukturze wewnętrznej.
- [ ] Zrobić stratyfikowany podział 60/20/20 na trening, walidację i test, a standaryzację dopasowywać tylko na treningu,\
  bo inaczej informacja z walidacji i testu przeciekłaby do modelu, i wyniki byłyby zawyżone od pierwszego biegu.
- [ ] W Adult zakodować kolumny kategoryczne i obsłużyć braki danych, a w Wine Quality połączyć wino czerwone i białe i zdecydować, co zrobić z bardzo rzadkimi klasami,\
  bo klasy 3 i 9 mają tylko garstkę próbek i mogą zepsuć podział stratyfikowany albo trening, więc trzeba to sprawdzić przed eksperymentami.
- [ ] Ustalić, jak dane trafiają do Evaluatora, na przykład jako `TensorDataset` albo własna klasa zbioru,\
  bo `Colony` o danych nic nie wie i tak ma zostać, więc dane żyją tylko w Evaluatorze, bez wyjątku i bez sporu.

### Testy na trudniejszych zbiorach (nowe zadanie)

- [ ] Zrobić przebiegi pilotażowe na wszystkich trzech zbiorach i sprawdzić, czy wynik najlepszej mrówki rośnie, czy stoi w miejscu jak wryty,\
  bo na łatwym zbiorze wine wszystko wychodziło około 98 procent, więc pilotaż ma pokazać różnice, których tam nie było, a które tu muszą być odkryte.
- [ ] Zmierzyć czas treningu jednej sieci i dobrać liczbę epok, mrówek i iteracji tak, żeby przebiegi zmieściły się na lokalnym GPU albo w Colabie w rozsądnym czasie,\
  bo eksperymenty z wieloma ziarnami szybko mnożą koszt, a budżet trzeba znać przed startem, a nie po fakcie, w smutnym grymasie.
- [ ] Wybrać miarę `score`, bo zwykła dokładność jest słaba dla niezbalansowanych zbiorów Adult i Wine Quality, i rozważyć F1 albo zbalansowaną dokładność w tej roli,\
  a przy okazji rozstrzygać remisy stratą walidacyjną, żeby najlepsza mrówka nie była pierwszą lepszą, którą los wyróżnił w tej niedoli.
- [ ] Dostroić `local_decay` i `evaporation`, bo w teście na wine aktualizacja lokalna zjadała prawie całą nagrodę globalną, jak opisano w sekcji o trudnościach,\
  i sprawdzić, przy jakich wartościach kolonia się skupia, a przy jakich dalej szeroko eksploruje, we wszystkich kierunkach i możliwościach.

### Obserwatory i logowanie

- [ ] Napisać obserwatora MLflow, który zapisuje parametry konfiguracji, wyniki każdej iteracji i najlepszą sieć w jednym przebiegu,\
  i zamyka przebieg w `on_search_end`, bo to ostatni hook, który się wykonuje, zanim `run` zwróci wynik po całym tym biegu.
- [ ] Napisać obserwatora, który zapisuje każdą ocenioną architekturę do tabeli pandas, z numerem iteracji, warstwami, dokładnością i czasem treningu,\
  i zapisuje ją do pliku w `runs/`, żeby analiza i wykresy korzystały z gotowych danych, bez ponownego przebiegu.
- [ ] Napisać obserwatora, który zbiera kopie feromonów co iterację, żeby narysować z nich mapy cieplne i pokazać, jak kolonia się uczy,\
  a dla Random Search pominąć te wykresy, bo jego słownik jest pusty, więc mapa niczego nie wyłuszczy.

### Uruchamianie eksperymentów

- [ ] Przygotować pliki JSON z konfiguracją dla każdego eksperymentu i skrypt, który uruchamia je z kilkoma ziarnami po kolei,\
  żeby każdy wynik był średnią z kilku przebiegów, a nie szczęśliwym trafem jednej mrówki, która miała więcej nadziei.
- [ ] Zapewnić ten sam budżet dla ACO i Random Search, a wcześniej ustalić, czy budżet to liczba ocenionych modeli, czy czas treningu,\
  bo Random Search buduje płytsze, szybsze sieci, więc przy równej liczbie modeli zużyje mniej czasu, i porównanie byłoby krzywe od pierwszego biegu.
- [ ] Podpiąć to wszystko pod `main.py`, żeby jedno polecenie uruchamiało wybrany eksperyment, a wyniki zapisywały się w MLflow i w `runs/` bez pytania,\
  więc nikt nie traci czasu na kopiowanie plików ani na ręczne przepisywanie.
- [ ] Dodać do repozytorium plik `.python-version`, bo na razie nie jest zacommitowany, a bez niego `uv` może wybrać inną wersję Pythona, niż trzeba,\
  i dopisać kilka testów jednostkowych dla strategii i tablic feromonów, żeby zmiany nie psuły tego, co już działa, bo to podstawowa potrzeba.

### Eksperymenty z PDF-u

- [ ] **E1**: porównać ACO z Random Search przy tym samym budżecie, żeby sprawdzić, czy feromony w ogóle pomagają w szukaniu,\
  i pokazać rozkłady dokładności obu metod, a nie tylko najlepszy wynik, bo pojedynczy rekord nie mówi wiele o całym działaniu.
- [ ] **E2**: porównać feromony niezależne z zależnymi, czyli `independent`, `param_dependent` i `config_dependent`, na tych samych zbiorach,\
  i sprawdzić, czy zależności między warstwami są warte większej liczby komórek, czy tylko spowalniają naukę w tak dużych rozmiarach.
- [ ] **E3**: zbadać wpływ liczby mrówek, parowania feromonów, czyli `evaporation`, i zachłanności, czyli `greediness`, na jakość architektur i przebieg optymalizacji,\
  a przy okazji też `local_decay`, bo testy pokazały, że mocno hamuje zbieżność, więc warto go zbadać w tej samej serii i w tej samej konfiguracji.
- [ ] **E4**: porównać efektywność wyszukiwania, czyli jak najlepszy wynik rośnie z iteracjami, ile modeli trzeba było ocenić i ile czasu to kosztowało,\
  i zestawić to dla wszystkich metod, żeby było widać nie tylko, kto wygrał, ale też jakim kosztem i jak szybko się to opłacało.

### Wizualizacja i raport

- [ ] Narysować krzywe zbieżności, rozkłady dokładności ACO i Random Search oraz mapy cieplne feromonów w Matplotlib i Seaborn, w jednym spójnym stylu,\
  żeby wykresy w raporcie wyglądały jak z jednej ręki, a nie jak zbiór przypadkowych obrazków bez ładu i profilu.
- [ ] Opisać w raporcie strategię `param_dependent`, bo nie ma jej w PDF-ie, a jest domyślna w kodzie, więc recenzent musi wiedzieć, skąd się wzięła i czym się od reszty różni,\
  a każdą część raportu niech pisze osoba, która ją robiła, a jedna osoba złoży całość w spójny dokument, żeby nikt w ostatniej chwili nie błądził w próżni.

---

Gdy kod się zmieni, poprawcie też to README, żeby opis i kod szły ze sobą w parze,\
a mrówki niech szukają dobrych sieci, zanim termin oddania projektu nas wszystkich ukarze.
