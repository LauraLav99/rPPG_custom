# rPPG Extractor — prototipo

Estrae il segnale PPG (battito cardiaco) dalla webcam, in due modalità:

- **face**: rileva il volto, monitora il canale verde su una ROI sulla fronte.
- **finger**: il dito coperto sull'obiettivo, monitora il canale rosso sull'intero frame.

Il **video non viene mai salvato**: ogni frame viene letto, trasformato in un
singolo numero (intensità media) e scartato. Solo questa serie numerica
viene conservata e poi scritta su disco.

## Installazione

```bash
pip install -r requirements.txt
```

## Uso

Modalità viso, 30 secondi, con grafico finale:

```bash
python rppg_extractor.py --mode face --duration 30 --plot
```

Modalità dito, 20 secondi:

```bash
python rppg_extractor.py --mode finger --duration 20
```

Senza finestra di anteprima (solo terminale):

```bash
python rppg_extractor.py --mode face --duration 30 --no-preview
```

Premi `q` nella finestra di anteprima per interrompere prima della durata
impostata.

## Output

Tre file con lo stesso nome base (timestamp automatico se non specificato
con `--output`):

- `ppg_<mode>_<timestamp>.csv` — colonne `time_s, ppg_raw, ppg_filtered`
- `ppg_<mode>_<timestamp>.json` — metadati: frequenza di campionamento,
  durata, frequenza cardiaca stimata
- `ppg_<mode>_<timestamp>.png` — grafico del segnale (solo con `--plot`)

## Consigli pratici

- **Face**: buona luce frontale, viso fermo, evita movimenti/luci pulsanti.
- **Finger**: copri bene l'obiettivo con il polpastrello, senza schiacciare
  troppo (toglie il flusso sanguigno) né troppo poco (entra luce ambiente).
  Su laptop senza flash, tieni il dito vicino a una lampada o fai aderire
  bene il dito alla lente: la luce che lo attraversa basta per modulare
  il canale rosso.

## Limiti del prototipo / possibili miglioramenti

- Il rilevamento volto usa Haar Cascade (semplice, dipendenza minima);
  per maggiore robustezza al movimento si può passare a Mediapipe
  Face Mesh per un tracking della ROI più stabile.
- L'algoritmo usa solo il canale verde (face) o rosso (finger). Algoritmi
  come **CHROM** o **POS** combinano più canali e sono più robusti a
  variazioni di luce e piccoli movimenti.
- La frequenza di campionamento effettiva dipende dalla velocità della
  webcam: lo script la stima automaticamente e ricampiona il segnale.
