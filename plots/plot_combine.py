import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Wczytanie danych (zmień 'wyniki.csv' na nazwę swojego pliku)
df = pd.read_csv("summary_local_decay.csv")


# 2. Uniwersalna funkcja parsująca dla różnych nazw hiperparametrów
def parse_series(series_str):
  parts = series_str.split("/")
  filename = parts[0]
  if filename.endswith(".json"):
    filename = filename[:-5]
  strategy = parts[1] if len(parts) > 1 else "unknown"

  # Usuwa opcjonalny prefiks eksperymentu (np. 'e3_', 'e1_' itp., jeśli występuje)
  clean_name = re.sub(r"^e\d+_", "", filename)

  # Wyciąga nazwę parametru i wartość (obsługuje np. n_ants_30 oraz greed_0_3)
  # Wzorzec szuka tekstu na początku i liczby (w tym z podkreśleniem np. 0_3) na końcu
  match = re.search(r"(.+?)_([0-9]+(?:_[0-9]+)?)$", clean_name)
  if match:
    param_name = match.group(1)
    val_str = match.group(2).replace("_", ".")
    param_val = float(val_str)
  else:
    # Awaryjnie, gdyby format był zupełnie inny
    param_name = clean_name
    param_val = 0.0

  return pd.Series([param_val, param_name, strategy])


# Aplikowanie parsowania (tworzy kolumny z wartością, nazwą parametru i strategią)
df[["param_value", "param_name", "strategy"]] = df["series"].apply(
    parse_series
)

# Pobranie nazwy badanego parametru dynamicznie (np. 'n_ants' lub 'greed')
current_param = df["param_name"].iloc[0]

# Sortowanie danych po wartości parametru
df = df.sort_values(by="param_value")

# 3. Konfiguracja stylu wykresów
sns.set_theme(style="whitegrid", palette="muted")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

# --- WYKRES 1: Test Accuracy ---
sns.lineplot(
    data=df,
    x="param_value",
    y="test_accuracy",
    hue="strategy",
    marker="o",
    linewidth=2.5,
    ax=axes[0],
)
axes[0].set_title(
    f"Dokładność testowa (Test Accuracy)", fontsize=12, fontweight="bold"
)
axes[0].set_xlabel(f"Wartość parametru: {current_param}", fontsize=11)
axes[0].set_ylabel("Test Accuracy", fontsize=11)
axes[0].legend(title="Strategia")

# --- WYKRES 2: Czas trenowania ---
sns.lineplot(
    data=df,
    x="param_value",
    y="train_time_s",
    hue="strategy",
    marker="s",
    linewidth=2.5,
    ax=axes[1],
)
axes[1].set_title(
    f"Czas trenowania (Train Time)", fontsize=12, fontweight="bold"
)
axes[1].set_xlabel(f"Wartość parametru: {current_param}", fontsize=11)
axes[1].set_ylabel("Czas trenowania [s]", fontsize=11)
axes[1].legend(title="Strategia")

# Ogólny nagłówek wykresu
plt.suptitle(
    f"Analiza hiperparametru [{current_param}] – Dataset: {df['dataset'].iloc[0]}",
    fontsize=14,
    fontweight="bold",
    y=1.03,
)
plt.tight_layout()

# Automatyczny zapis pliku z nazwą badanego parametru w nazwie pliku graficznego
plt.savefig(
    f"combined_plot_{current_param}.png", dpi=300, bbox_inches="tight"
)
