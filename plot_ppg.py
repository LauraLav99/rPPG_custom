import sys
import pandas as pd
import matplotlib.pyplot as plt

def plot_ppg_raw(df):
            # Création du graphique
        plt.figure(figsize=(12, 6))
        
        # Tracé des deux signaux sur le même graphique
        plt.plot(df['time_s'], df['ppg_raw'], label='PPG Raw', alpha=0.6, color='blue')
        plt.plot(df['time_s'], df['ppg_filtered'], label='PPG Filtered', alpha=0.9, color='red', linewidth=1.5)
        
        # Configuration des axes et de la légende
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Visualisation du signal PPG (Brut vs Filtré)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Affichage du graphique
        plt.tight_layout()
        plt.show()

def plot_ppg_pos(df):
    # Création du graphique
    fig,axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Visualisation du signal PPG (POS) vs green raw', fontsize=16)
    

    # Tracé du signal PPG POS
    axs[0].plot(df['time_s'], df['ppg_pos'], label='PPG POS', alpha=0.9, color='green', linewidth=1.5)

    # Configuration des axes et de la légende
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title('Visualisation du signal PPG (POS)')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # Tracé du signal green raw
    axs[1].plot(df['time_s'], df['green_raw'], label='Green Raw', alpha=0.9, color='blue', linewidth=1.5)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_title('Visualisation du signal Green (Raw)')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.5)

    # Affichage du graphique
    plt.tight_layout()
    plt.show()

def main():
    # Vérifie si le nom du fichier est passé en argument
    if len(sys.argv) < 2:
        print("Usage: python plot_ppg.py <chemin_du_fichier.csv>")
        sys.exit(1)
        
    csv_file = sys.argv[1]

    try:
        # Lecture du fichier CSV
        df = pd.read_csv(csv_file)
        
        # Vérification de la présence des colonnes requises
        required_cols = ['time_s', 'ppg_raw', 'ppg_filtered', 'green_raw', 'ppg_pos']
        #if not all(col in df.columns for col in required_cols):
        if not any(col in df.columns for col in required_cols):
            print(f"Erreur: Le CSV doit contenir les colonnes: {required_cols}")
            sys.exit(1)
        
        if 'ppg_pos' in df.columns:
            plot_ppg_pos(df)
        elif 'ppg_raw' in df.columns and 'ppg_filtered' in df.columns:
            plot_ppg_raw(df)


    except Exception as e:
        print(f"Une erreur est survenue lors de la lecture du fichier : {e}")

if __name__ == "__main__":
    main()
