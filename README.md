# Codice_sorgente_TOPTW_con_incertezza
# Robust Optimization for the Team Orienteering Problem with Time Windows (TOPTW)

Repository ufficiale del codice sorgente, dei dataset e degli strumenti analitici sviluppati nell'ambito della **Tesi Magistrale in Ingegneria Gestionale** (A.A. 2025/2026).

Il progetto affronta la pianificazione logistica nel trasporto merci su scala europea con orizzonte settimanale continuo. L'architettura proposta combina una meta-euristica memetica **GA-LNS** (Algoritmi Genetici integrati con *Large Neighborhood Search*) con il paradigma di **ottimizzazione robusta con budget d'incertezza $\Gamma$ (Bertsimas & Sim)**, garantendo la protezione del margine operativo netto contro la volatilità tariffaria e le contrazioni dei ricavi.

---

## 📁 Struttura del Repository

├── data/
│   └── toptw_bks_benchmark_reference.csv   # Soluzioni note di riferimento (BKS) per la validazione
├── instances/                              # Dataset di benchmark standard di letteratura (C, R, RC, PR)
├── instances_real/                         # Dataset operativi del caso industriale ("Bridge Logistica")
├── results/                                # Output, file di log e metriche generati sui benchmark accademici
├── results_real/                           # Piani di rotta, report e visualizzazioni cartografiche del caso reale
│
├── main_solver_3.py                        # Algoritmo euristico modulare GA-LNS di riferimento
├── main_validation_benchmark_3.py          # Validazione algoritmica e confronto su BKS di letteratura
├── main_tuning_parallel_3.py               # Procedura di tuning parallelo degli iperparametri del modello
├── confronto_gurobi_vs_ma_3.py             # Risoluzione esatta (Gurobi MILP) vs approccio euristico (GA-LNS)
├── stress_test_montecarlo_batch_3.py       # Motore di simulazione stocastica batch Monte Carlo
├── simulazione_montecarlo_monmin_3.py      # Analisi di resilienza e quantificazione del rischio di violazione
└── final_solver_4.py                       # Pipeline integrata di ottimizzazione robusta per il caso reale

---

## ⚙️ Descrizione dei Moduli di Calcolo

* **`main_solver_3.py` (Algoritmo Euristico Modulare di Riferimento)**  
  Implementa il motore memetico GA-LNS: generazione guidata della popolazione iniziale, operatore di crossover genetico BCRC (*Best Cost Route Crossover*), euristiche di rimozione/distruzione e operatori LNS di riparazione e reinserimento basati sull'efficienza marginale.

* **`main_validation_benchmark_3.py` (Validazione Istanze Benchmark)**  
  Gestisce la lettura e l'esecuzione serializzata delle istanze canoniche di letteratura (famiglie Solomon, Montemanni e Ropke), valutando lo scostamento percentuale (gap) rispetto ai valori BKS archiviati nel file `data/toptw_bks_benchmark_reference.csv`.

* **`main_tuning_parallel_3.py` (Tuning Parallelo degli Iperparametri)**  
  Esegue la calibrazione parallela multi-core per identificare la combinazione ottimale dei parametri euristici (dimensione della popolazione, tasso di crossover $p_{\text{cross}}$, probabilità di ricerca locale $p_{\text{lns}}$, ampiezza di distruzione e cardinalità della *Restricted Candidate List*).

* **`confronto_gurobi_vs_ma_3.py` (Confronto Esatto vs Euristico)**  
  Modella e risolve il problema tramite programmazione lineare intera mista (MILP) su solutore commerciale Gurobi, fornendo il riferimento esatto per quantificare l'optimality gap e le differenze nei tempi di calcolo rispetto a GA-LNS.

* **`stress_test_montecarlo_batch_3.py` (Simulazione Stocastica Batch Monte Carlo)**  
  Sottopone i piani operativi a perturbazioni tariffarie simulate su campioni estesi ($N = 1000$ repliche per scenario), calcolando il Margine Operativo Netto atteso ($E[\text{MON}]$), la variabilità empirica ($\text{Std } \text{MON}$) e il Win Rate tra configurazioni nominali e robuste.

* **`simulazione_montecarlo_monmin_3.py` (Valutazione della Resilienza Operativa)**  
  Analizza la probabilità empirica di violazione delle soglie di break-even operative ($P(\text{MON} < \alpha \cdot Z_{\text{nom}})$) e certifica il valore minimo di profitto residuo ($\text{Min } \text{MON}$) negli scenari avversi di mercato.

* **`final_solver_4.py` (Pipeline Integrata per il Caso Reale)**  
  Pipeline parallela multi-processo ottimizzata con compilazione JIT (Numba) per l'istanza industriale da 500 nodi e 14 veicoli pesanti. Include il preprocessing di ammissibilità temporale e di orizzonte, il soddisfacimento dei vincoli di quota minima sui clienti prioritari e la generazione automatizzata dei grafici delle rotte territoriali e dei carichi giornalieri.

---

## 💻 Requisiti di Sistema e Installazione

Il codice richiede un ambiente **Python 3.9 o superiore**.

1. Clonare il repository:
   git clone https://github.com/<TUO-USERNAME>/<NOME-REPOSITORY>.git
   cd <NOME-REPOSITORY>

2. Creare e attivare un ambiente virtuale (consigliato):
   # Creazione ambiente virtuale
   python -m venv venv

   # Attivazione su Windows:
   venv\Scripts\activate

   # Attivazione su macOS/Linux:
   source venv/bin/activate

3. Installare le dipendenze:
   pip install -r requirements.txt

### Dipendenze software principali
* `numpy`, `pandas`: Elaborazione dati matriciali, calcolo statistico e gestione dataset.
* `numba`: Accelerazione JIT (*Just-In-Time*) per la valutazione ad alta velocità delle sequenze temporali.
* `matplotlib`: Rendering dei grafici distributivi, mappe di carico e convergenza euristica.
* `contextily`: Integrazione cartografica con mappe stradali di sfondo (Esri WorldStreetMap).
* `gurobipy`: Risoluzione esatta dei modelli MILP (richiede licenza accademica o commerciale attiva; necessario unicamente per `confronto_gurobi_vs_ma_3.py`).

---

## 🚀 Istruzioni per l'Esecuzione

### 1. Risoluzione del Caso Industriale Reale
Per avviare l'ottimizzazione parallela sulle configurazioni $\Gamma \in \{0\%, 50\%, 100\%\}$ ed eseguire lo stress-test Monte Carlo:
python final_solver_4.py

Tutti i report tabellari (.csv) e le mappe delle rotte (.png) vengono salvati automaticamente nella cartella results_real/.

### 2. Validazione e Confronto su Benchmark
Per eseguire la campagna di test sulle istanze standard di letteratura:
python main_validation_benchmark_3.py

Le metriche di convergenza e i gap calcolati rispetto ai valori BKS vengono archiviati nella cartella results/.

### 3. Simulazioni Monte Carlo di Dettaglio
Per eseguire le simulazioni stocastiche e l'analisi di resilienza sui piani preesistenti:
python stress_test_montecarlo_batch_3.py
python simulazione_montecarlo_monmin_3.py

---

## 📜 Riferimento Accademico

Il codice e i modelli contenuti in questo repository costituiscono parte integrante della Tesi di Laurea Magistrale in Ingegneria Gestionale:
* Titolo: Modelli e algoritmi per il Team Orienteering Problem con ricavi incerti: formulazione robusta e applicazione al caso del trasporto a lunga distanza
* Anno Accademico: 2025/2026
