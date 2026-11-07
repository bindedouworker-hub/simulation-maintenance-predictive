import numpy as np
import pandas as pd
import os

def generate_sensor_data(n_samples=1000, seed=42):
    """
    Génère des données de capteurs simulées pour la maintenance prédictive.
    Inclut cpu_temperature, voltage_vcc, fan_speed_rpm.

    Graine fixe par défaut (seed=42) : deux exécutions produisent
    exactement les mêmes données — conditions requises pour toute
    comparaison de résultats. Choisir seed=None pour un tirage aléatoire.
    """
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 100, n_samples)

    # Température CPU: base 40°C + sinusoïde + bruit
    cpu_temp = 40 + 10 * np.sin(time / 10) + rng.normal(0, 2, n_samples)

    # Tension VCC: stable autour de 3.3V + petit bruit
    voltage = 3.3 + rng.normal(0, 0.1, n_samples)

    # Vitesse ventilateur: proportionnelle à la température + bruit
    fan_speed = 1000 + (cpu_temp - 40) * 50 + rng.normal(0, 50, n_samples)
    
    data = pd.DataFrame({
        'cpu_temperature': cpu_temp,
        'voltage_vcc': voltage,
        'fan_speed_rpm': fan_speed
    })
    
    return data

if __name__ == "__main__":
    data = generate_sensor_data(1000)
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(results_dir, exist_ok=True)
    data.to_csv(os.path.join(results_dir, 'sensor_data_stream.csv'), index=False)
    print("Données générées et sauvegardées dans results/sensor_data_stream.csv")
    
    # Sauvegarder un résumé
    with open(os.path.join(results_dir, 'generate_data_results.txt'), 'w') as f:
        f.write(f"Données générées: {len(data)} échantillons\n")
        f.write(f"Colonnes: {list(data.columns)}\n")
        f.write(f"Statistiques:\n{data.describe()}\n")