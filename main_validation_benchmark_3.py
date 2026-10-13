import concurrent.futures
from dataclasses import dataclass
import os
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd

# Importa le strutture dati e il solver ottimizzato v3
from main_solver_3 import (
    hybrid_genetic_lns_top_solver,
    parse_toptw_instance,
)

# ==============================================================================
# WORKER PER MULTIPROCESSING
# ==============================================================================

def esegui_singola_validazione(task_info: dict) -> dict:
    file_path = task_info["file_path"]
    nome_istanza = task_info["nome_istanza"]
    bks_val = task_info["bks_val"]
    seed = task_info["seed"]
    params = task_info["params"]

    inst_obj = parse_toptw_instance(str(file_path), Gamma=0.0)

    random.seed(seed)
    np.random.seed(seed)

    t0 = time.time()
    sol, z, _ = hybrid_genetic_lns_top_solver(inst_obj, verbose=False, **params)
    tempo_cpu = time.time() - t0

    gap_pct = (
        ((bks_val - z) / bks_val * 100.0)
        if (not np.isnan(bks_val) and bks_val > 0)
        else np.nan
    )

    return {
        "Istanza": nome_istanza,
        "Replica": seed,
        "BKS": bks_val,
        "Z_Trovato": z,
        "Gap_Pct": gap_pct,
        "Tempo_CPU_s": tempo_cpu,
    }


# ==============================================================================
# MAIN ESECUZIONE SU ISTANZE BENCHMARK
# ==============================================================================

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    INSTANCES_DIR = BASE_DIR / "instances"
    DATA_DIR = BASE_DIR / "data"
    RESULTS_DIR = BASE_DIR / "results"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Configurazione ottimale identificata nella calibrazione OAT
    OPTIMAL_PARAMS = {
        "pop_size": 40,
        "max_gen": 150,
        "p_cross": 0.90,
        "p_lns": 0.30,
        "distruzione_pct": 0.20,
        "n_elite": 2,
        "tournament_k": 2,
        "rcl_size": 3,
        "risk_aware": False,
    }

    SEEDS = list(range(100, 110))  # 10 repliche: 100, 101, ..., 109
    NUM_WORKERS = max(1, os.cpu_count() - 2)

    file_dettaglio = RESULTS_DIR / "dettaglio_validazione_34_istanze_3.csv"
    file_riassunto = RESULTS_DIR / "riassunto_validazione_34_istanze_3.csv"

    # Caricamento automatico file BKS di riferimento da /data
    bks_file = DATA_DIR / "toptw_bks_benchmark_reference.csv"
    bks_dict = {}
    if bks_file.exists():
        df_bks = pd.read_csv(bks_file)
        df_bks.columns = df_bks.columns.str.strip()
        col_inst = [c for c in df_bks.columns if "instance" in c.lower()][0]
        col_score = [c for c in df_bks.columns if "bks" in c.lower() or "optimal" in c.lower()][0]

        for _, row in df_bks.iterrows():
            nome = str(row[col_inst]).strip()
            val = float(row[col_score])
            bks_dict[nome] = val
            if not nome.endswith(".txt"):
                bks_dict[nome + ".txt"] = val
        print(f"✓ Caricati {len(bks_dict)} BKS da {bks_file.name}")
    else:
        print(f"ATTENZIONE: File BKS non trovato in {bks_file}")

    # Rilevazione di tutte le istanze .txt presenti nella cartella
    file_istanze = sorted(list(INSTANCES_DIR.glob("*.txt")))
    if not file_istanze:
        print(f"Nessun file .txt trovato in {INSTANCES_DIR}")
        exit()

    print(f"Trovate {len(file_istanze)} istanze in {INSTANCES_DIR}. Processi paralleli: {NUM_WORKERS}")

    df_storico = (
        pd.read_csv(file_dettaglio) if file_dettaglio.exists() else pd.DataFrame()
    )

    tasks = []
    for f_inst in file_istanze:
        nome = f_inst.name
        bks = bks_dict.get(nome, bks_dict.get(f_inst.stem, np.nan))

        for seed in SEEDS:
            if not df_storico.empty:
                gia_fatto = not df_storico[
                    (df_storico["Istanza"] == nome) & (df_storico["Replica"] == seed)
                ].empty
                if gia_fatto:
                    continue

            tasks.append({
                "file_path": f_inst,
                "nome_istanza": nome,
                "bks_val": bks,
                "seed": seed,
                "params": OPTIMAL_PARAMS,
            })

    print(f"Task di validazione totali da eseguire: {len(tasks)}\n")

    t_start = time.time()
    completati = 0

    if tasks:
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futuri = {executor.submit(esegui_singola_validazione, t): t for t in tasks}

            for future in concurrent.futures.as_completed(futuri):
                res = future.result()
                completati += 1
                gap_str = (
                    f"{res['Gap_Pct']:.2f}%" if not np.isnan(res["Gap_Pct"]) else "N/D"
                )
                bks_str = f"{res['BKS']:.1f}" if not np.isnan(res["BKS"]) else "N/D"
                print(
                    f"[{completati}/{len(tasks)}] {res['Istanza']:<12} | Seed: {res['Replica']} | "
                    f"Z: {res['Z_Trovato']:<8.2f} (BKS: {bks_str}) | Gap: {gap_str:<7} | "
                    f"Tempo: {res['Tempo_CPU_s']:.2f}s"
                )

                nuova_riga = pd.DataFrame([res])
                if file_dettaglio.exists():
                    nuova_riga.to_csv(file_dettaglio, mode="a", header=False, index=False)
                else:
                    nuova_riga.to_csv(file_dettaglio, mode="w", header=True, index=False)

    # Aggregazione finale delle istanze
    df_completo = pd.read_csv(file_dettaglio).drop_duplicates(subset=["Istanza", "Replica"])
    
    # Aggiorna i BKS per eventuali righe pregresse prive di valore
    df_completo["BKS"] = df_completo["Istanza"].map(bks_dict)
    df_completo["Gap_Pct"] = ((df_completo["BKS"] - df_completo["Z_Trovato"]) / df_completo["BKS"]) * 100.0
    df_completo.to_csv(file_dettaglio, index=False)

    riepilogo = (
        df_completo.groupby("Istanza")
        .agg(
            BKS=("BKS", "first"),
            Z_Migliore=("Z_Trovato", "max"),
            Z_Medio=("Z_Trovato", "mean"),
            Z_Std=("Z_Trovato", "std"),
            Gap_Migliore_Pct=("Gap_Pct", "min"),
            Gap_Medio_Pct=("Gap_Pct", "mean"),
            Gap_Std_Pct=("Gap_Pct", "std"),
            Tempo_Medio_s=("Tempo_CPU_s", "mean"),
            N_Run=("Replica", "count"),
        )
        .reset_index()
    )

    riepilogo = riepilogo.round(2)
    riepilogo.to_csv(file_riassunto, index=False)

    print(f"\n=======================================================")
    print(f"✓ Validazione completata in {(time.time() - t_start)/60:.2f} minuti.")
    print(f"✓ Tabella di riepilogo salvata in: {file_riassunto}")
    print(f"=======================================================")