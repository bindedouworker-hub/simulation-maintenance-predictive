import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import os

class MaintenanceSimulator:
    def __init__(self, n_samples=1000, seed=42):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        self.data = None
        self.model = IsolationForest(contamination=0.05, random_state=42)

    def generate_healthy_data(self):
        """
        Simule le fonctionnement normal d'un composant (ex: Température CPU).
        Combine une onde sinusoïdale (cycle journalier) et du bruit gaussien.
        Graine fixe par défaut : runs reproductibles (voir scripts/reliability_audit.py).
        """
        time = np.linspace(0, 50, self.n_samples)
        # Signal de base + Bruit aléatoire
        signal = 20 + 5 * np.sin(time) + self.rng.normal(0, 0.5, self.n_samples)
        self.data = pd.DataFrame({'value': signal})
        print("[INFO] Données saines générées.")

    def inject_fault(self, start_index, end_index, magnitude=10):
        """
        Simule une panne (ex: surchauffe due à un ventilateur défaillant).
        Ajoute une dérive artificielle aux données.
        """
        if self.data is not None:
            self.data.iloc[start_index:end_index] += magnitude
            print(f"[INFO] Panne injectée entre les index {start_index} et {end_index}.")

    def train_and_detect(self):
        """
        Entraîne le modèle sur les données et détecte les anomalies.
        Retourne les prédictions (-1 = anomalie, 1 = normal).
        """
        # Entraînement du modèle (apprentissage non supervisé)
        self.model.fit(self.data[['value']])
        self.data['anomaly'] = self.model.predict(self.data[['value']])
        
        # Compter les anomalies détectées
        n_anomalies = (self.data['anomaly'] == -1).sum()
        print(f"[RESULTAT] Analyse terminée. {n_anomalies} points anormaux détectés.")
        return self.data

    def visualize(self):
        """Génère un graphique montrant le signal et les alertes."""
        plt.figure(figsize=(12, 6))
        
        # Tracer le signal complet
        plt.plot(self.data.index, self.data['value'], label='Données Capteur (Température)', color='blue', alpha=0.6)
        
        # Mettre en évidence les anomalies détectées (points rouges)
        anomalies = self.data[self.data['anomaly'] == -1]
        plt.scatter(anomalies.index, anomalies['value'], color='red', label='Anomalie Détectée (IA)', s=50, zorder=5)
        
        plt.title("Simulation de Maintenance Prédictive : Détection d'Anomalie")
        plt.xlabel("Temps (Cycles)")
        plt.ylabel("Valeur Capteur")
        plt.legend()
        plt.grid(True)
        plt.show()

# --- Exécution du Scénario de Test ---
if __name__ == "__main__":
    # 1. Initialisation
    sim = MaintenanceSimulator(n_samples=500)
    
    # 2. Génération d'un comportement sain
    sim.generate_healthy_data()
    
    # 3. Injection d'une panne critique (Surchauffe) vers la fin de la série
    sim.inject_fault(start_index=400, end_index=450, magnitude=8)
    
    # 4. L'IA analyse les données
    results = sim.train_and_detect()
    
    # 5. Sauvegarder les résultats
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(results_dir, exist_ok=True)
    results.to_csv(os.path.join(results_dir, 'simulation_results.csv'), index=False)
    
    with open(os.path.join(results_dir, 'simulator_results.txt'), 'w') as f:
        f.write("Résultats de la Simulation de Maintenance\n")
        f.write("=====================================\n")
        f.write(f"Échantillons totaux: {len(results)}\n")
        f.write(f"Anomalies détectées: {(results['anomaly'] == -1).sum()}\n")
        f.write(f"Paramètres du modèle: contamination=0.05\n")
        f.write(f"Intervalle de panne injectée: [400, 450]\n")
        f.write(f"Magnitude de la panne: 8\n")
    
    # 6. Visualisation des résultats (sauvegardée en image)
    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results['value'], label='Données Capteur (Température)', color='blue', alpha=0.6)
    anomalies = results[results['anomaly'] == -1]
    plt.scatter(anomalies.index, anomalies['value'], color='red', label='Anomalie Détectée (IA)', s=50, zorder=5)
    plt.title("Simulation de Maintenance Prédictive : Détection d'Anomalie")
    plt.xlabel("Temps (Cycles)")
    plt.ylabel("Valeur Capteur")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'simulation_plot.png'))
    print("Graphique sauvegardé dans results/simulation_plot.png")
    
    # (Décommentez la ligne suivante pour afficher le graphique dans un environnement local)
    # sim.visualize()