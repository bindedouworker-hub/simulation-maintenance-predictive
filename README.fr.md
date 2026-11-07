<div align="center">

<img src="docs/banner.png" alt="Bannière — Simulation de maintenance prédictive hybride" width="100%"/>

**Un pipeline complet simulé de maintenance prédictive** — génération
de données capteurs, détection d'anomalies et modélisation de RUL.

[![CI](https://github.com/bindedouworker-hub/simulation-maintenance-predictive/actions/workflows/python-package.yml/badge.svg)](https://github.com/bindedouworker-hub/simulation-maintenance-predictive/actions/workflows/python-package.yml)
[![Licence : MIT](https://img.shields.io/badge/licence-MIT-31c48d)](LICENSE)
![Python 3.10–3.12](https://img.shields.io/badge/Python-3.10%20%E2%80%93%203.12-3776ab)
![scikit-learn](https://img.shields.io/badge/Isolation%20Forest-scikit--learn-f89939)
![LSTM](https://img.shields.io/badge/LSTM-TensorFlow-ff8f00)

[Français](README.fr.md) · [English](README.md)

*Projet d'étude personnel — ouvert à toutes les améliorations et contributions !*

</div>

---

## 🎯 Ce que le projet fait

Ce projet simule **la chaîne complète de données d'un système de
maintenance prédictive** pour du matériel informatique / électronique
(température CPU, tension VCC, vitesse ventilateur) :

1. **Génération de données** — flux de capteurs synthétiques : signal
   sain (cycle journalier + bruit gaussien) avec **pannes injectées**
   (dérives, pics)
2. **Détection d'anomalies (non supervisée)** — `IsolationForest`
   marque les fenêtres anormales sans aucune donnée étiquetée
3. **Prétraitement pour le deep learning** — normalisation Min-Max +
   fenêtres glissantes pour la modélisation de séries temporelles
4. **Modèle LSTM (RUL)** — réseau `LSTM → LSTM → Dense` empilé qui
   estime la **durée de vie restante** (*Remaining Useful Life*)
5. **Évaluation** — matrice de confusion, précision/rappel/F1, avec un
   œil sur les **faux négatifs** (panne ratée = l'erreur la plus coûteuse)

## 📸 Aperçu des résultats

Signal + anomalies détectées par l'Isolation Forest :

![Résultat de simulation](results/simulation_plot.png)

Matrice de confusion d'un run complet enregistré :

![Matrice de confusion](docs/confusion_matrix.png)

Rapport du même run (voir `results/evaluation_results.txt`) :

| | Précision | Rappel | F1 |
|---|---|---|---|
| Normal | 0.94 | 0.99 | 0.96 |
| **Anomalie** | **0.84** | **0.42** | **0.56** |

> 93 % d'exactitude globale, mais un rappel d'anomalie de 0.42 — exactement
> le genre de constat honnête que ce projet se construit pour mettre en
> évidence : la détection non supervisée seule **rate plus de la moitié
> des pannes**. Acquérir des données étiquetées ou ajouter un étage de
> scoring supervisé/hybride est la suite logique (voir la
> [feuille de route](#-feuille-de-route)).

## 🚀 Démarrage

```bash
git clone https://github.com/bindedouworker-hub/simulation-maintenance-predictive.git
cd simulation-maintenance-predictive

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Exécuter le pipeline complet :

```bash
python scripts/generate_data.py          # 1. flux de capteurs synthétique
python scripts/maintenance_simulator.py  # 2. détection IsolationForest + plot
python scripts/data_preprocessing.py     # 3. normalisation + fenêtres glissantes
python scripts/evaluation.py             # 4. matrice de confusion + rapport
```

Les sorties vont dans `results/` (flux CSV, rapports texte,
`simulation_plot.png`).

> `scripts/lstm_model.py` définit l'architecture du réseau RUL
> (TensorFlow). Le reste du pipeline fonctionne **sans TensorFlow** :
> vous pouvez explorer détection/évaluation sans installation lourde.

## 🗂️ Structure du projet

```
.
├── data/                       # données de capteurs générées (flux de démo)
├── scripts/
│   ├── generate_data.py        # capteurs synthétiques (temp CPU, VCC, RPM)
│   ├── data_preprocessing.py   # Min-Max + fenêtres glissantes pour le LSTM
│   ├── lstm_model.py           # LSTM empilé → tête de régression RUL
│   ├── maintenance_simulator.py# IsolationForest + injection de pannes + plot
│   ├── evaluation.py           # matrice de confusion, précision/rappel/F1
│   └── read_results.py         # relit les sorties d'expériences enregistrées
├── results/                    # CSV / rapports / plots des runs enregistrés
├── notebooks/read_results.ipynb# notebook d'exploration rapide
└── .github/workflows/          # CI : lint + smoke test du pipeline (3.10–3.12)
```

## 🗺️ Feuille de route

- [ ] Scoring supervisé / hybride pour faire monter le rappel d'anomalie
      (actuellement 0.42)
- [ ] Vraie boucle d'entraînement du RUL (`lstm_model.py` ne définit
      aujourd'hui que l'architecture)
- [ ] Datasets publics : NASA C-MAPSS, Microsoft Azure
      predictive-maintenance
- [ ] Traçabilité structurée des expériences (`mlflow` ou tracker CSV)
- [ ] Courbes ROC-AUC / PR-AUC pour une comparaison sans seuil

## 🤝 Contribuer

Tout le monde est le bienvenu : c'est un projet d'étude ouvert !
Lisez [CONTRIBUTING.md](CONTRIBUTING.md) (court). Ouvrez une issue ou
une PR.

## 📄 Licence

[MIT](LICENSE) — libre d'utilisation, de modification et de partage.

---

<div align="center">

**Un projet de Mr Jackson** 🇨🇮

</div>
