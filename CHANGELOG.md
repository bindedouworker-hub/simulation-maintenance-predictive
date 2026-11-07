# Changelog

Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/).

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
