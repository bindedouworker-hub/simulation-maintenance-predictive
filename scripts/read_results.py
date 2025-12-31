import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def read_results():
    """
    Lit et affiche tous les fichiers de résultats générés par les simulations
    """
    results_dir = os.path.join(os.path.dirname(__file__), '../results')

    print("=" * 60)
    print("LECTURE DES RÉSULTATS DES SIMULATIONS")
    print("=" * 60)

    # 1. Résultats de génération de données
    print("\n1. GÉNÉRATION DE DONNÉES")
    print("-" * 30)
    try:
        with open(os.path.join(results_dir, 'generate_data_results.txt'), 'r') as f:
            print(f.read())
    except FileNotFoundError:
        print("Fichier generate_data_results.txt non trouvé")

    # 2. Résultats du prétraitement
    print("\n2. PRÉTRAITEMENT DES DONNÉES")
    print("-" * 30)
    try:
        with open(os.path.join(results_dir, 'preprocessing_results.txt'), 'r') as f:
            print(f.read())
    except FileNotFoundError:
        print("Fichier preprocessing_results.txt non trouvé")

    # 3. Résultats du simulateur
    print("\n3. SIMULATION DE MAINTENANCE")
    print("-" * 30)
    try:
        with open(os.path.join(results_dir, 'simulator_results.txt'), 'r') as f:
            print(f.read())

        # Charger et afficher un aperçu des données de simulation
        sim_data = pd.read_csv(os.path.join(results_dir, 'simulation_results.csv'))
        print(f"\nAperçu des données de simulation ({len(sim_data)} échantillons):")
        print(sim_data.head(10))
        print(f"\nAnomalies détectées: {(sim_data['anomaly'] == -1).sum()}")

    except FileNotFoundError:
        print("Fichiers de simulation non trouvés")

    # 4. Résultats de l'évaluation
    print("\n4. ÉVALUATION DES PERFORMANCES")
    print("-" * 30)
    try:
        with open(os.path.join(results_dir, 'evaluation_results.txt'), 'r') as f:
            print(f.read())
    except FileNotFoundError:
        print("Fichier evaluation_results.txt non trouvé")

    # 5. Description du modèle LSTM
    print("\n5. MODÈLE LSTM")
    print("-" * 30)
    try:
        with open(os.path.join(results_dir, 'lstm_model_description.txt'), 'r') as f:
            print(f.read())
    except FileNotFoundError:
        print("Fichier lstm_model_description.txt non trouvé")

    # 6. Afficher le graphique si disponible
    print("\n6. GRAPHIQUE DE SIMULATION")
    print("-" * 30)
    plot_path = os.path.join(results_dir, 'simulation_plot.png')
    if os.path.exists(plot_path):
        print("Graphique disponible: simulation_plot.png")
        print("Pour visualiser: ouvrir le fichier dans un visionneur d'images")
    else:
        print("Graphique non trouvé")

    print("\n" + "=" * 60)
    print("FIN DE LA LECTURE DES RÉSULTATS")
    print("=" * 60)

if __name__ == "__main__":
    read_results()