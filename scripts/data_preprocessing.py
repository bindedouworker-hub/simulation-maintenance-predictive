import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

# Chargement des données brutes (Simulation ou Export Prometheus)
data_path = os.path.join(os.path.dirname(__file__), '../results/sensor_data_stream.csv')
data = pd.read_csv(data_path)

# Sélection des fonctionnalités (Features) critiques
features = ['cpu_temperature', 'voltage_vcc', 'fan_speed_rpm']

# Normalisation Min-Max pour ramener les valeurs entre 0 et 1
# Indispensable pour la convergence rapide des réseaux de neurones
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Création de fenêtres temporelles (Sliding Windows) pour l'analyse séquentielle
def create_sequences(data, window_size=50):
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i+window_size])
    return np.array(sequences)

# Exemple d'utilisation
sequences = create_sequences(data[features].values, window_size=50)
print(f"Sequences shape: {sequences.shape}")

# Sauvegarder les résultats
results_dir = os.path.join(os.path.dirname(__file__), '../results')
os.makedirs(results_dir, exist_ok=True)

# Sauvegarder les données normalisées
data.to_csv(os.path.join(results_dir, 'normalized_sensor_data.csv'), index=False)

# Sauvegarder les séquences (premières 10 pour exemple)
with open(os.path.join(results_dir, 'preprocessing_results.txt'), 'w') as f:
    f.write(f"Données chargées: {len(data)} échantillons\n")
    f.write(f"Features sélectionnées: {features}\n")
    f.write(f"Forme des séquences: {sequences.shape}\n")
    f.write(f"Paramètres de normalisation:\n")
    f.write(f"Min: {scaler.data_min_}\n")
    f.write(f"Max: {scaler.data_max_}\n")
    f.write("Exemple de séquence (première):\n")
    f.write(f"{sequences[0]}\n")