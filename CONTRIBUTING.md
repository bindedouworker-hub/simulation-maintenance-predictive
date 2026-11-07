# Contribuer à Simulation de Maintenance Prédictive

Merci de votre intérêt ! Ce projet est **sous licence MIT** et ouvert à
toutes les améliorations.

## Point de départ rapide

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_data.py && python scripts/maintenance_simulator.py
```

## Bonnes pratiques

1. **Branche** : partez de `main` avec un nom clair
   (`fix/…`, `feat/…`, `docs/…`).
2. **[Conventional Commits](https://www.conventionalcommits.org/)**
   : `feat:`, `fix:`, `docs:`, `refactor:`, `ci:` …
3. **PR petite et ciblée** : un sujet par PR.
4. **Avant de pousser** : la CI exécute le pipeline complet
   (génération → simulateur → prétraitement → évaluation) sur Python
   3.10 à 3.12 — vérifiez qu'il passe en local.
5. **Pas de fichiers lourds** : datasets > 5 Mo, modèles entraînés
   (`.h5`, `.pkl`) ou sorties de labo → utilisez un lien (utilisez
   Git LFS si vraiment nécessaire).
6. **Reproductibilité** : si votre code tire au hasard, fixez la graine
   (`np.random.default_rng(seed)` / `random_state=42`) — c'est actuellement
   le point faible d'`evaluation.py` ✔ bienvenue.

## Où aider en priorité ?

Voir la feuille de route dans le README (rappel d'anomalie faible,
entraînement LSTM réel, datasets publics, courbes ROC/PR).
