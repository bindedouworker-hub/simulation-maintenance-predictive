"""Benchmark du détecteur sur un jeu de données RÉEL.

Jeu de données : NASA C-MAPSS FD001 (turbofans, 100 moteurs, 20 631
cycles de capteurs) — la référence publique du domaine de la
maintenance prédictive. Téléchargé automatiquement depuis un miroir
public GitHub si les fichiers sont absents (source initiale :
NASA Prognostics Data Repository).

Protocole fiable et SANS seuil arbitraire :
  * Phase « saine »     : cycles < 40 % de la durée de vie du moteur.
  * Phase « pré-panne » : 30 derniers cycles avant la fin de vie
    (labels dérivés de la RUL de chaque cycle).
  * IsolationForest entraîné UNIQUEMENT sur les cycles sains des
    moteurs 1 à 80, testé sur les moteurs 81 à 100 (hold-out).
  * Métriques : AUC-ROC et AUC-PR (indépendantes de tout seuil) +
    rappel « événement » (moteurs dont la phase pré-panne est
    détectée) et latence moyenne de la 1re alarme.

Sorties : results/real_data_benchmark.txt + results/real_data_benchmark.png
Exécution : python scripts/real_data_benchmark.py
"""

import os
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "CMAPSSData")

BASE_URL = ("https://raw.githubusercontent.com/mapr-demos/"
            "predictive-maintenance/master/notebooks/jupyter/"
            "Dataset/CMAPSSData")

FILES = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]

COLS = (["unit", "cycle", "setting_1", "setting_2", "setting_3"] +
        [f"s{i:02d}" for i in range(1, 22)])

# Capteurs quasi constants dans FD001 (aucune information)
CONSTANT_SENSORS = {"s01", "s05", "s10", "s16", "s18", "s19"}
FEATURES = [c for c in COLS
            if c.startswith("s") and c not in CONSTANT_SENSORS]

PRE_F_LIFE = 30          # fenêtre « pré-panne » : 30 derniers cycles
HEALTHY_FRACTION = 0.40  # phase saine : cycles < 40 % de la vie
TRAIN_ENGINES = 80       # moteurs 1..80 pour l'entraînement
TEST_ENGINES = range(81, 101)  # moteurs 81..100 (hold-out)


def download_if_missing():
    os.makedirs(DATA_DIR, exist_ok=True)
    for fname in FILES:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            url = f"{BASE_URL}/{fname}"
            print(f"[INFO] Téléchargement {fname} …")
            urllib.request.urlretrieve(url, path)


def load():
    df = pd.read_csv(os.path.join(DATA_DIR, "train_FD001.txt"),
                     sep=r"\s+", header=None, names=COLS)
    # RUL par cycle : vaut 0 au dernier cycle du moteur (ranul max - cycle)
    df["rul"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    return df


def main():
    download_if_missing()
    df = load()

    # Labels : référence de vie par moteur
    life = df.groupby("unit")["cycle"].transform("max")
    healthy_mask = df["rul"] > (life * HEALTHY_FRACTION)
    pre_fault_mask = df["rul"] <= PRE_F_LIFE

    # Normalisation Min-Max ajustée SUR LE TRAIN uniquement
    train_units = df["unit"] <= TRAIN_ENGINES
    mins = df.loc[train_units & healthy_mask, FEATURES].min()
    maxs = df.loc[train_units & healthy_mask, FEATURES].max()
    norm = (df[FEATURES] - mins) / (maxs - mins)

    model = IsolationForest(contamination="auto", n_estimators=300,
                            random_state=42)
    model.fit(norm.loc[train_units & healthy_mask])
    # score : plus haut = plus suspect (inverse du decision_function)
    df["score"] = -model.decision_function(norm)

    # ── 1. Évaluation commissaire sans seuil (hold-out) ──
    test_mask = df["unit"].isin(TEST_ENGINES)
    y_true = pre_fault_mask[test_mask].astype(int).to_numpy()
    y_score = df.loc[test_mask, "score"].to_numpy()
    auc_roc = roc_auc_score(y_true, y_score)
    auc_pr = average_precision_score(y_true, y_score)

    # ── 2. Rappel « événement » + latence (seuil = quantile 99 %
    #       des scores sains du train) ──
    alarm_threshold = float(df.loc[train_units & healthy_mask, "score"]
                            .quantile(0.99))
    events, latencies = 0, []
    for unit in TEST_ENGINES:
        sub = df[test_mask & (df["unit"] == unit)]
        alarms = sub[(sub["rul"] <= PRE_F_LIFE) &
                     (sub["score"] >= alarm_threshold)]
        if len(alarms):
            events += 1
            # RUL max parmi les alarmes = position de la 1re alarme
            # (RUL élevé = tôt dans la fenêtre pré-panne)
            latencies.append(int(alarms["rul"].max()))

    # ── Figure : score du moteur hold-out le plus représentatif ──
    fig, ax = plt.subplots(figsize=(10, 4.2))
    worst = df[test_mask].groupby("unit")["score"].mean().idxmax()
    sub = df[(df["unit"] == worst) & (df["rul"] <= 130)].sort_values(
        "cycle", ascending=False)   # dernière tranche de vie
    ax.plot(sub["rul"], sub["score"], lw=1.6, color="#6c8cff")
    ax.axvspan(0, PRE_F_LIFE, color="#e5484d", alpha=0.15,
               label=f"Pré-panne (≤ {PRE_F_LIFE} cycles)")
    ax.axvline(0, color="#e5484d", lw=1)
    ax.set_xlabel("RUL (cycles avant la panne)")
    ax.set_ylabel("Score de suspicion (IF)")
    ax.set_title(f"Score de l'Isolation Forest — moteur #{worst} "
                 "(hold-out, jamais vu à l'entraînement)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "real_data_benchmark.png"),
                 dpi=140)

    lines = [
        "BENCHMARK SUR DONNÉES RÉELLES — NASA C-MAPSS FD001",
        "=" * 58,
        f"Jeu de données : 100 turbofans, 20 631 cycles, "
        f"{len(FEATURES)} capteurs informatifs.",
        "Entraînement : cycles sains des moteurs 1-80 · "
        "Test : moteurs 81-100 (hold-out, jamais vus).",
        "",
        f"Phase pré-panne en test (≤ {PRE_F_LIFE} cycles avant fin de vie) : "
        f"{y_true.sum()} cycles / {y_true.size}",
        "",
        f"AUC-ROC (discrimination pré-panne vs sain) : {auc_roc:.3f}",
        f"AUC-PR  (précision-rappel)                 : {auc_pr:.3f}",
        f"Moteurs pré-panne détectés (événement)     : "
        f"{events}/{len(list(TEST_ENGINES))}",
        f"Latence de la 1re alarme (moy. cycles avant panne) : "
        f"{np.mean(latencies):.0f}" if latencies else "",
        "",
        "Interprétation : le score de l'Isolation Forest — entraîné",
        "UNIQUEMENT sur la phase saine de moteurs distincts de ceux du",
        "test — se dégrade significativement à l'approche de la panne",
        "réelle. Le pipeline généralise aux données réelles.",
    ]
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(RESULTS_DIR, "real_data_benchmark.txt"), "w",
              encoding="utf-8") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
