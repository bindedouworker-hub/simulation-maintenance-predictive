import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os

# Supposons que nous avons des vraies étiquettes (pour simulation)
# Dans un vrai scénario, ces étiquettes seraient manuelles ou d'un dataset étiqueté

def simulate_labels(data, fault_start=400, fault_end=450):
    """
    Simule les vraies étiquettes: 0 = normal, 1 = anomalie
    """
    labels = np.zeros(len(data))
    labels[fault_start:fault_end] = 1
    return labels

def evaluate_model(predictions, true_labels):
    """
    Évalue le modèle avec matrice de confusion et rapport de classification
    """
    # Convertir prédictions IsolationForest (-1,1) en (1,0)
    pred_labels = (predictions == -1).astype(int)
    
    cm = confusion_matrix(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, target_names=['Normal', 'Anomalie'])
    
    print("Matrice de Confusion:")
    print(cm)
    print("\nRapport de Classification:")
    print(report)
    
    return cm, report

if __name__ == "__main__":
    # Charger les données du simulateur
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    results = pd.read_csv(os.path.join(results_dir, 'simulation_results.csv'))
    
    true_labels = simulate_labels(results)
    predictions = results['anomaly'].values
    
    cm, report = evaluate_model(predictions, true_labels)
    
    # Sauvegarder les résultats
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'evaluation_results.txt'), 'w') as f:
        f.write("Évaluation des Performances du Modèle\n")
        f.write("===================================\n")
        f.write("Matrice de Confusion:\n")
        f.write(f"{cm}\n\n")
        f.write("Rapport de Classification:\n")
        f.write(f"{report}\n")
        f.write("\nInterprétation:\n")
        f.write("- Vrais Positifs (TP): Pannes correctement prédites\n")
        f.write("- Faux Négatifs (FN): Pannes non détectées (risque élevé)\n")
        f.write("- Faux Positifs (FP): Fausses alertes\n")
        f.write("- Vrais Négatifs (TN): Fonctionnement normal correctement identifié\n")