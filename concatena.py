import os
import re
import argparse
import numpy as np

def main():
    # Configura i parametri da linea di comando
    parser = argparse.ArgumentParser(description="Concatena file .npy dello stesso gruppo in formato .npy e .csv")
    parser.add_argument("-o", "--origine", required=True, help="Percorso della cartella contenente i file .npy")
    parser.add_argument("-d", "--destinazione", required=True, help="Percorso della cartella dove salvare i risultati")
    args = parser.parse_args()

    # Crea la cartella di destinazione se non esiste
    os.makedirs(args.destinazione, exist_ok=True)

    # Espressione regolare per estrarre il gruppo (tutto ciò che precede '_label')
    pattern = re.compile(r"^(S\d{2}_\d{2})_label\d+\.npy$")

    gruppi = {}

    # 1. Raggruppamento dei file
    for file in os.listdir(args.origine):
        match = pattern.match(file)
        if match:
            gruppo = match.group(1)  # Estrae es. 'S05_06'
            if gruppo not in gruppi:
                gruppi[gruppo] = []
            gruppi[gruppo].append(os.path.join(args.origine, file))

    if not gruppi:
        print(f"Nessun file corrispondente al pattern trovato in: {args.origine}")
        return

    # 2. Concatenazione e salvataggio
    for gruppo, lista_file in gruppi.items():
        # Ordina i file (es. label0 prima di label1) per mantenere la sequenza
        lista_file.sort()
        print(f"Elaborazione gruppo {gruppo} ({len(lista_file)} file)...")
        
        # Carica e concatena gli array del gruppo
        array_da_concatenare = [np.load(f) for f in lista_file]
        array_finale = np.concatenate(array_da_concatenare, axis=0)
        
        # Definisci i percorsi di salvataggio
        percorso_npy = os.path.join(args.destinazione, f"{gruppo}.npy")
        percorso_csv = os.path.join(args.destinazione, f"{gruppo}.csv")
        
        # Salva in formato .npy
        np.save(percorso_npy, array_finale)
        
        # Salva in formato .csv (gestisce array 1D o 2D)
        if array_finale.ndim <= 2:
            np.savetxt(percorso_csv, array_finale, delimiter=",", fmt="%g")
        else:
            # Se l'array ha più di 2 dimensioni, lo appiattisce per il CSV
            array_2d = array_finale.reshape(array_finale.shape[0], -1)
            np.savetxt(percorso_csv, array_2d, delimiter=",", fmt="%g")

    print("\nProcesso completato con successo!")

if __name__ == "__main__":
    main()
