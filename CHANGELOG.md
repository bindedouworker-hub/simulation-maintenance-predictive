# Changelog

Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/).

## [1.2.0] — 2026-11

### Added
- **Audit de fiabilité scientifique** (`scripts/reliability_audit.py`)
  : Monte-Carlo multi-graines, sensibilité à la magnitude de dérive,
  comparaison IF seul vs **détecteur hybride IF + résidu saisonnier**,
  métriques industrielles (épisodes détectés, latence d'alarme,
  fausses alertes partie saine). Rapport + figure dans `results/`.
- **Benchmark sur données réelles** (`scripts/real_data_benchmark.py`)
  : NASA C-MAPSS FD001 (100 turbofans, 20 631 cycles) téléchargé
  automatiquement ; protocole hold-out (moteurs 1-80 entraînement /
  81-100 test), métriques sans seuil.
- Graines aléatoires fixées et paramétrables (`seed=` par défaut 42)
  dans `generate_data.py` et `maintenance_simulator.py` → runs
  strictement reproductibles.
- CI : étapes audit + benchmark réel, dataset d'entraînement C-MAPSS
  versionné (`data/CMAPSSData/train_FD001.txt`).

### Changed
- Détection : l'audit révèle que l'IF seul plafonne sur les dérives
  persistantes (plateau ~0.44 de rappel à 16σ) ; le détecteur hybride
  proposé atteint un rappel 1.0 dès ~4σ du bruit, à latence quasi nulle.

### Verified (données réelles)
- Discrimination pré-panne vs sain sur moteurs jamais vus :
  **AUC-ROC 0.984 · AUC-PR 0.923 · 20/20 moteurs détectés · alarme en
  moyenne 29 cycles avant la fin de vie**.

## [1.1.0] — 2026-11

### Added
- README bilingue (EN + FR) avec bannière et illustrations réelles
  (plot de simulation, matrice de confusion).
- `CONTRIBUTING.md` + templates GitHub (issue bug/feature, PR).
- Licence **MIT** (projet ouvert aux contributions).
- Figure de matrice de confusion (`docs/confusion_matrix.png`)
  générée depuis un run réel du pipeline.

### Changed
- `requirements.txt` : versions minimales modernes (`pandas>=2.0`,
  `numpy>=1.24`, `scikit-learn>=1.3`, `matplotlib>=3.7`) ; TensorFlow
  documenté comme optionnel (inutile pour le pipeline de détection).
- CI remplacée : Python 3.10–3.12, `setup-python@v5`, lint flake8
  bloquant sur E9/F63/F7/F82, **smoke test du pipeline complet**
  (ancienne CI : `pytest` sans aucun test → échec systématique).
- `.gitignore` rendu cohérent (les démos `data/` et `results/` sont
  volontairement versionnées ; caches et gros fichiers ignorés).

### Removed
- `# Code Citations.md` (artefact Copilot).
- `package.json` (wrapper npm sans objet pour un projet Python).
- `scripts/__pycache__/*.pyc` versionné.

## [1.0.0] — 2026-11

- Simulation initiale : génération de données capteurs, prétraitement,
  architecture LSTM (RUL), simulateur Isolation Forest, évaluation.
