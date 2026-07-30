import sys
import pandas as pd
import matplotlib
#matplotlib.set_loglevel("debug")
matplotlib.use('TkAgg')
#matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import argparse
import os

def plot_ppg_pos(dfrppg,dfmediapipe):
    # Création du graphique
    #fig,axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig,axs = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Visualisation du signal PPG (POS) vs rppg toolbox', fontsize=16)
    

    # Tracé du signal PPG POS
    axs[0].plot(dfmediapipe['time_s'], dfmediapipe['ppg_pos'], label='PPG mediapipe', alpha=0.9, color='blue', linewidth=1.5)

    # Configuration des axes et de la légende
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title('Visualisation du signal PPG (POS)')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # Tracé du signal PPG RPPG
    axs[1].plot(dfrppg['time_s'], dfrppg['ppg_rppg'], label='PPG RPPG', alpha=0.9, color='blue', linewidth=1.5)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_title('Visualisation du signal PPG (RPPG)')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.5)

    # Affichage du graphique
    plt.tight_layout()
    plt.show()
    
    
def main():
    parser = argparse.ArgumentParser(description="plotta ppg estratat da rppgtoolbox vs ppg estratto da mediapipe POS")
    parser.add_argument("--rppgfolder", type=str, required=True, help="Cartella contenente i file CSV estratti da rppgtoolbox")
    parser.add_argument("--mediapipefolder", type=str, required=True, help="Cartella contenente i file CSV estratti da mediapipe POS")
    parser.add_argument("--subject", type=str, required=True, help="Nome del soggetto (es. S01_01)")
    args = parser.parse_args()  
    
    rppgpath=args.rppgfolder
    rppgfiles=os.listdir(rppgpath)
    rppgfile=[f for f in rppgfiles if args.subject in f and f.endswith('.csv')][0]
    df_rppg=pd.read_csv(os.path.join(rppgpath,rppgfile),names=['ppg_rppg'], header=None)
   
    df_rppg['time_s'] = df_rppg.index / 30.
    
    mediapipepath=args.mediapipefolder
    mediapipefiles=os.listdir(mediapipepath)
    mediapipefile=[f for f in mediapipefiles if args.subject in f and f.endswith('.csv')][0]
    df_mediapipe=pd.read_csv(os.path.join(mediapipepath,mediapipefile))
    print(df_mediapipe.head())
    plot_ppg_pos(df_rppg,df_mediapipe)
                           
    


if __name__ == "__main__":
    main()