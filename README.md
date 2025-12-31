# Simulation de Maintenance Prédictive Hybride

Ce projet contient les simulations et fichiers pour le rapport technique sur la maintenance prédictive utilisant des approches hybrides (IT et Électronique).

## Structure du Projet

- `data/`: Données générées et datasets
- `scripts/`: Scripts Python pour les simulations
- `notebooks/`: Notebooks Jupyter pour l'analyse et visualisation

## Simulations Implémentées

1. **Génération de Données**: Script pour créer des données de capteurs simulées.
2. **Prétraitement des Données**: Normalisation et création de séquences temporelles.
3. **Modèle LSTM**: Architecture pour la prédiction de RUL.
4. **Simulateur de Maintenance**: Détection d'anomalies avec Isolation Forest.
5. **Évaluation**: Analyse des performances avec matrice de confusion.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

1. Générer les données:
   ```bash
   python scripts/generate_data.py
   ```

2. Exécuter le simulateur:
   ```bash
   python scripts/maintenance_simulator.py
   ```

3. Évaluer les résultats:
   ```bash
   python scripts/evaluation.py
   ```

## Résultats

Les simulations démontrent la capacité des algorithmes à détecter des anomalies dans des signaux temporels, avec des performances détaillées dans le rapport.