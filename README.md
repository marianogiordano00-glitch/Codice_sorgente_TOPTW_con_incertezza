# Codice_sorgente_TOPTW_con_incertezza
# Robust Optimization for the Team Orienteering Problem with Time Windows (TOPTW)

Repository ufficiale del codice e degli strumenti analitici sviluppati nell'ambito della **Tesi Magistrale in Ingegneria Gestionale**.

Il progetto implementa un'architettura ibrida memetica basata sull'integrazione di **Algoritmi Genetici** e ricerca locale **Large Neighborhood Search (GA-LNS)**, combinata con il paradigma di **ottimizzazione robusta di Bertsimas & Sim**, applicata alla pianificazione logistica su scala europea con orizzonte settimanale continuo.

---

## 📁 Struttura del Repository

```text
├── data/
│   └── toptw_bks_benchmark_reference.csv   # Best Known Solutions (BKS) per la validazione
├── instances/                              # Dataset di benchmark accademici (C, R, RC, PR)
├── instances_real/                         # Dataset operativi del caso reale ("Bridge Logistica")
├── results/                                # Output, metriche e log delle istanze di benchmark
├── results_real/                           # Piani di rotta, report e grafici del caso reale
│
├── main_solver_3.py                        # Algoritmo euristico modulare GA-LNS di riferimento
├── main_validation_benchmark_3.py          # Validazione algoritmica e benchmark su BKS di letteratura
├── main_tuning_parallel_3.py               # Procedura di tuning parallelo degli iperparametri
├── confronto_gurobi_vs_ma_3.py             # Validazione e confronto esatto (Gurobi) vs euristico (GA-LNS)
├── stress_test_montecarlo_batch_3.py       # Motore di simulazione stocastica Monte Carlo batch
├── simulazione_montecarlo_monmin_3.py      # Analisi di resilienza e stima empirica del rischio di violazione
└── final_solver_4.py                       # Pipeline unificata parallela per il caso industriale
