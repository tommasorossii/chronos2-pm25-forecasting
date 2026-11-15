# Chronos-2 per la previsione del PM2.5 da sensori low-cost
 
Progetto del corso di **Graph Analytics** (Università di Modena e Reggio Emilia, prof.ssa Laura Po), a cura di **Tommaso Rossi**.
 
Il progetto aggiunge **Chronos-2** (Amazon, `amazon/chronos-2`) al confronto del paper:
 
> F. Rollo et al. (2026), *Beyond traditional models: Foundation models for accurate particulate matter prediction*.
 
Si usano gli stessi dati, gli stessi 10 sensori OSC e le stesse regole del paper, così che i risultati di Chronos-2 siano confrontabili con quelli di TimesFM, Granite TTM-R2, dei modelli di deep learning (CNN, LSTM, Transformer) e di ARIMAX.
 
---
 
## I due scenari
 
| | Scenario X | Scenario Y |
|---|---|---|
| **Cosa si prevede** | PM2.5 del sensore low-cost (LCS, `pm2p5_x`) | PM2.5 della centralina di riferimento (RS, `pm2p5_y`) |
| **Input** | passato dell'LCS + meteo | passato dell'LCS + meteo (mai i valori della RS) |
| **Modello** | zero-shot (nessun addestramento) | fine-tuning su ogni sensore |
| **Script** | `03b_run_scenario_x.py` | `03_run_scenario_y.py` |
 
Variabili meteo usate come covariate: `temperature`, `relative_humidity`, `pressure`, `wind_speed`.
 
Nello **Scenario Y** Chronos-2 ha bisogno del passato della serie da prevedere, ma la RS non può essere data in input. Al suo posto si usa l'**LCS calibrato con una regressione lineare** stimata solo sul training set (`--target-mode cal`, la configurazione adottata). Le altre due modalità (`nan`: passato vuoto, `lcs`: LCS grezzo) sono state provate e scartate.
 
---
 
## Struttura della repository
 
```
.
├── config.py                  percorsi, sensori, griglia (I, H), iperparametri
├── common.py                  finestre, calibrazione, fine-tuning, previsione, metriche
├── 01_prepare_data.py         10 sensori OSC, griglia oraria, split 75/25
├── 01b_prepare_nan_data.py    stessa griglia ma con i valori NaN non gestiti
├── 02_smoke_test.py           prova veloce su 1 sensore e 1 configurazione
├── 03_run_scenario_y.py       Scenario Y: fine-tuning + previsione su tutta la griglia
├── 03b_run_scenario_x.py      Scenario X: previsione zero-shot su tutta la griglia
├── 04_evaluate.py             metriche per sensore, media OSC, heatmap
├── 05_compare_paper.py        confronto con il paper, Scenario Y
├── 05b_compare_paper_x.py     confronto con il paper, Scenario X
├── file_*.sh                  job SLURM usati sul cluster
├── data/                      dataset di partenza e dataset preparati
├── results/                   previsioni, metriche e confronti
└── archivio_prove/            prove preliminari per la scelta degli iperparametri
```
 
---
 
## Requisiti
 
- Python 3.10
- GPU NVIDIA con CUDA (gli esperimenti sono stati eseguiti su una **H100** del cluster di Ateneo)
- Pacchetti: vedi `requirements.txt`
```bash
conda create -n chronos2 python=3.10 -y
conda activate chronos2
pip install -r requirements.txt
```
 
La versione di `chronos-forecasting` usata è la **2.3.3.dev0**, installata dal repository GitHub di Amazon; serve una versione che includa `Chronos2Pipeline` e il metodo `fit` per il fine-tuning.
 
Gli script SLURM impostano `HF_HUB_OFFLINE=1`, quindi il modello deve essere già in cache. Scaricalo una volta da un nodo con accesso a Internet:
 
```bash
huggingface-cli download amazon/chronos-2
```
 
---
 
## Dati
 
I file in `data/` provengono dal dataset del paper:
 
| File | Contenuto |
|---|---|
| `preprocessed_data.csv` | dati pre-processati del paper: valori NaN gestiti per interpolazione, filtro degli outlier (IQR) |
| `data_without_preprocessing.csv` | dati grezzi, con i valori NaN |
| `osc_prepared.csv` | creato da `01_prepare_data.py` |
| `osc_prepared_nan.csv` | creato da `01b_prepare_nan_data.py` |
 
Colonne principali: `valid_at` (ora), `sensor_id`, `city`, `pm2p5_x` (LCS), `pm2p5_y` (RS), variabili meteo, `is_missing` (valore originariamente mancante o interpolato).
 
**Sensori OSC (10 città):** `ari-2433` Aosta, `fec01` Badajoz, `sensor bangalore` Bangalore, `sensor calgary` Calgary, `sensor delhi` Delhi, `sensor hamirpur` Hamirpur, `5e8` Lima, `ari-2172` Reggio Emilia, `ari-2263` Trento, `pms5003-2018062702491` Southampton.
 
---
 
## Impostazioni sperimentali
 
| Parametro | Valore |
|---|---|
| Split | cronologico per sensore: primo 75% training set, ultimo 25% test set |
| Finestra di input I (ore) | 12, 24, 48, 72, 512 |
| Orizzonte H (ore) | 1, 12, 24, 48, 72 |
| Configurazioni | tutte le coppie con H ≤ 2·I → **22 configurazioni × 10 sensori = 220 esperimenti per scenario** |
| Previsione | rolling sul test set con passo 1 ora; le previsioni sovrapposte vengono mediate; si usa la mediana dei quantili |
| Fine-tuning (solo Y) | `full`, learning rate 1e-5, 1000 step, batch 32 |
| Seed | 42 |
| Metriche | MAE, RMSE, SMAPE, R², calcolate **solo sui valori osservati** (`is_missing = False`) e mediate sui 10 sensori |
 
Gli iperparametri del fine-tuning sono stati scelti con una prova su due sensori molto diversi (Aosta e Delhi, configurazione I = 24, H = 12) e poi fissati per tutti: i risultati sono in `archivio_prove/` (prodotti da `file_sweep.sh`).
 
---
 
## Come riprodurre i risultati
 
Tutti i comandi si lanciano dalla cartella principale della repository. Sul cluster, negli script `file_*.sh` sostituisci il percorso dopo `cd` e i parametri `#SBATCH` (partizione, account, qos) con i tuoi.
 
### 1. Preparazione dei dati
 
```bash
python 01_prepare_data.py          # → data/osc_prepared.csv
python 01b_prepare_nan_data.py     # → data/osc_prepared_nan.csv (richiede il file precedente)
```
 
### 2. Prova veloce (facoltativa)
 
```bash
python 02_smoke_test.py            # 1 sensore (Aosta), I=24, H=12, 200 step
```
 
### 3. Scenario X: zero-shot
 
```bash
python 03b_run_scenario_x.py                                                # valori NaN gestiti
PREPARED_FILE=osc_prepared_nan.csv python 03b_run_scenario_x.py --tag X_zeroshot_nan   # valori NaN non gestiti
 
python 04_evaluate.py --tag X_zeroshot
python 04_evaluate.py --tag X_zeroshot_nan
python 05b_compare_paper_x.py --tags X_zeroshot X_zeroshot_nan --pair X_zeroshot:X_zeroshot_nan
```
 
Su SLURM: `sbatch file_X_zeroshot.sh`
 
### 4. Scenario Y: fine-tuning
 
```bash
python 03_run_scenario_y.py --target-mode cal --steps 1000 --lr 1e-5 --tag Y_cal_full
PREPARED_FILE=osc_prepared_nan.csv python 03_run_scenario_y.py --target-mode cal --steps 1000 --lr 1e-5 --tag Y_cal_full_nan
 
python 04_evaluate.py --tag Y_cal_full
python 04_evaluate.py --tag Y_cal_full_nan
python 05_compare_paper.py --tags Y_lcs_full Y_cal_full Y_cal_full_nan
```
 
Su SLURM: `sbatch file_run_finale.sh`, poi `sbatch file_run_finale_nan.sh`.
 
La griglia Y richiede diverse ore su una H100; gli script sono **riprendibili**: rilanciandoli, le configurazioni già calcolate vengono saltate (usa `--overwrite` per ricalcolarle). Per lanciare solo una parte della griglia: `--sensors "sensor delhi" --configs 24:12 72:1`.
 
La griglia `Y_lcs_full` (LCS grezzo come passato, scartata) si ottiene con `--target-mode lcs --tag Y_lcs_full`.
 
### Output
 
Per ogni esperimento `results/<tag>/` contiene:
 
| File | Contenuto |
|---|---|
| `preds/<sensore>_I<I>_H<H>.csv` | previsioni e valori reali sul test set |
| `run_args.json`, `run_log.csv` | parametri e tempi di esecuzione |
| `calibration/` | coefficienti della regressione lineare (solo Y, modalità `cal`) |
| `metrics_per_sensor.csv` | metriche per sensore e configurazione |
| `metrics_osc.csv` | metriche medie sui 10 sensori |
| `best_per_horizon.csv` | miglior finestra I per ogni orizzonte |
| `heatmap_mae_osc.png` | MAE medio per ogni coppia (I, H) |
 
I confronti con il paper sono in `results/comparison/` (Scenario Y) e `results/comparison_X/` (Scenario X).
 
---
 
## Risultati principali
 
MAE medio sui 10 sensori (µg/m³), miglior finestra di input per ogni orizzonte.
 
| Chronos-2 | H = 1 | H = 12 |
|---|---|---|
| Scenario X, valori NaN gestiti | 4,63 | 9,46 |
| Scenario X, valori NaN non gestiti | 6,41 | 11,28 |
| Scenario Y, valori NaN gestiti | 12,39 | 14,00 |
| Scenario Y, valori NaN non gestiti | 12,83 | 15,78 |
 
- **Scenario X:** a 1 ora Chronos-2 zero-shot è il modello più preciso; a 12 ore è vicino a TimesFM ZS.
- **Scenario Y:** Granite tuned resta il migliore; Chronos-2 supera il Transformer a 1 ora e riduce l'errore rispetto all'LCS grezzo in 9 città su 10.
- **Valori NaN non gestiti:** Chronos-2 li accetta direttamente, ma con l'interpolazione i risultati sono più stabili (con NaN non gestiti, 15 configurazioni su 220 nello Scenario X e 20 nello Scenario Y risultano instabili).
---
 
## Nota sulla riproducibilità
 
Il seed è fissato, ma su GPU alcune operazioni non sono perfettamente deterministiche: rilanciando gli esperimenti di fine-tuning i valori possono differire leggermente nelle ultime cifre decimali.
 
---
 
## Riferimenti
 
- F. Rollo et al. (2026), *Beyond traditional models: Foundation models for accurate particulate matter prediction*.
- A. F. Ansari et al., *Chronos-2* — modello `amazon/chronos-2` su Hugging Face; libreria [chronos-forecasting](https://github.com/amazon-science/chronos-forecasting).
 
