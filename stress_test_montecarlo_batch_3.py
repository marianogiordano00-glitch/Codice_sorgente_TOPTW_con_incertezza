import os
import json
import random
import time
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict

import numpy as np
import numba
from numba import njit
import pandas as pd

# ==============================================================================
# 1. STRUTTURA DATI E FUNZIONI COMPILATE JIT (NUMBA)
# ==============================================================================

@dataclass
class IstanzaTOPTW:
    nome: str
    n_tot: int
    n_clienti: int
    K: int
    T_max: float
    coords: np.ndarray
    p: np.ndarray
    delta: np.ndarray
    e: np.ndarray
    l: np.ndarray
    s: np.ndarray
    t: np.ndarray
    c: np.ndarray
    somma_profitti: float
    penale: float

@njit(fastmath=True)
def simula_rotta_fast(route: np.ndarray, t_mat: np.ndarray, e_arr: np.ndarray, l_arr: np.ndarray, s_arr: np.ndarray, T_max_val: float, c_mat: np.ndarray):
    if len(route) == 0:
        return True, 0.0, 0.0
    curr_time = 0.0
    curr_node = 0
    costo = 0.0
    for i in range(len(route)):
        nxt = route[i]
        curr_time += t_mat[curr_node, nxt]
        if curr_time < e_arr[nxt]:
            curr_time = e_arr[nxt]
        if curr_time > l_arr[nxt] + 1e-9:
            return False, -1.0, -1.0
        curr_time += s_arr[nxt]
        costo += c_mat[curr_node, nxt]
        curr_node = nxt
    curr_time += t_mat[curr_node, 0]
    if curr_time > T_max_val + 1e-9:
        return False, -1.0, -1.0
    costo += c_mat[curr_node, 0]
    return True, curr_time, costo

@njit(fastmath=True)
def valuta_inserimento_rotta_pos_fast(route: np.ndarray, nodo: int, pos: int, t_mat: np.ndarray, e_arr: np.ndarray, l_arr: np.ndarray, s_arr: np.ndarray, T_max_val: float, c_mat: np.ndarray):
    n_len = len(route) + 1
    curr_time = 0.0
    prev = 0
    costo = 0.0
    for idx in range(n_len):
        if idx < pos:
            curr = route[idx]
        elif idx == pos:
            curr = nodo
        else:
            curr = route[idx - 1]
            
        curr_time += t_mat[prev, curr]
        if curr_time < e_arr[curr]:
            curr_time = e_arr[curr]
        if curr_time > l_arr[curr] + 1e-9:
            return False, -1.0, -1.0
        curr_time += s_arr[curr]
        costo += c_mat[prev, curr]
        prev = curr
        
    curr_time += t_mat[prev, 0]
    if curr_time > T_max_val + 1e-9:
        return False, -1.0, -1.0
    costo += c_mat[prev, 0]
    return True, curr_time, costo

@njit(fastmath=True)
def protez_bertsimas_fast(deviazioni: np.ndarray, Gamma: float) -> float:
    if len(deviazioni) == 0 or Gamma <= 0.0:
        return 0.0
    dev_ord = np.sort(deviazioni)[::-1]
    k_int = int(math.floor(Gamma))
    frazione = Gamma - k_int
    n_dev = len(dev_ord)
    limite = min(k_int, n_dev)
    prot = 0.0
    for i in range(limite):
        prot += dev_ord[i]
    if frazione > 0.0 and k_int < n_dev:
        prot += frazione * dev_ord[k_int]
    return prot

# ==============================================================================
# 2. CARICATORE POLIVALENTE ISTANZE (JSON e TESTO SOLOMON/CORDEAU)
# ==============================================================================

def carica_qualsiasi_istanza(percorso_file: Path) -> IstanzaTOPTW:
    nome = percorso_file.name
    if percorso_file.suffix.lower() == ".json":
        with open(percorso_file, "r") as f:
            dati = json.load(f)
        nodes_data = dati["nodes"]
        K_val = dati["K"]
        T_max_val = float(dati["T_max"])
        coords = np.array([[n["x"], n["y"]] for n in nodes_data], dtype=float)
        p = np.array([n["p"] for n in nodes_data], dtype=float)
        e = np.array([n["e"] for n in nodes_data], dtype=float)
        l = np.array([n["l"] for n in nodes_data], dtype=float)
        s = np.array([n["s"] for n in nodes_data], dtype=float)
        delta = np.array([n.get("delta", 0.50 * n["p"]) for n in nodes_data], dtype=float)
    else:
        # Formato testo standard TOPTW / Solomon
        with open(percorso_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        testata = lines[0].split()
        K_val = int(testata[1]) if len(testata) >= 2 else 3
        coords_list, p_list, s_list, e_list, l_list = [], [], [], [], []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 8:
                try:
                    coords_list.append([float(parts[1]), float(parts[2])])
                    s_list.append(float(parts[3]))
                    p_list.append(float(parts[4]))
                    e_list.append(float(parts[-2]))
                    l_list.append(float(parts[-1]))
                except ValueError:
                    continue
        coords = np.array(coords_list, dtype=float)
        p = np.array(p_list, dtype=float)
        s = np.array(s_list, dtype=float)
        e = np.array(e_list, dtype=float)
        l = np.array(l_list, dtype=float)
        T_max_val = float(l[0])
        delta = 0.50 * p

    n_tot = len(coords)
    n_clienti = n_tot - 1
    p[0], delta[0] = 0.0, 0.0

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))
    t = dist.copy()
    c = 0.5 * dist.copy()

    somma_profitti = float(np.sum(p))
    penale = -(somma_profitti + 1.0)

    return IstanzaTOPTW(
        nome=nome,
        n_tot=n_tot,
        n_clienti=n_clienti,
        K=K_val,
        T_max=T_max_val,
        coords=coords,
        p=p,
        delta=delta,
        e=e,
        l=l,
        s=s,
        t=t,
        c=c,
        somma_profitti=somma_profitti,
        penale=penale,
    )

# ==============================================================================
# 3. MOTORE GA-LNS DI OTTIMIZZAZIONE
# ==============================================================================

def valuta_soluzione_ga(rotte: List[List[int]], inst: IstanzaTOPTW, gamma_val: float) -> float:
    nodi_serviti = []
    costo_tot = 0.0
    for r in rotte:
        if not r:
            continue
        r_arr = np.array(r, dtype=np.int64)
        ok, _, cost = simula_rotta_fast(r_arr, inst.t, inst.e, inst.l, inst.s, inst.T_max, inst.c)
        if not ok:
            return inst.penale
        costo_tot += cost
        nodi_serviti.extend(r)

    if len(nodi_serviti) != len(set(nodi_serviti)):
        return inst.penale

    if not nodi_serviti:
        return 0.0

    profitto_nom = float(np.sum(inst.p[nodi_serviti]))
    prot = protez_bertsimas_fast(inst.delta[nodi_serviti], gamma_val)
    return profitto_nom - prot - costo_tot

def costruisci_soluzione_iniziale(inst: IstanzaTOPTW, gamma_val: float, risk_aware=False, rcl_size=3):
    rotte = [[] for _ in range(inst.K)]
    nodi_disp = list(range(1, inst.n_tot))
    random.shuffle(nodi_disp)

    while True:
        candidati = []
        nodi_attuali = [node for r in rotte for node in r]
        prot_attuale = protez_bertsimas_fast(inst.delta[nodi_attuali], gamma_val) if risk_aware else 0.0
        r_arrays = [np.array(r, dtype=np.int64) for r in rotte]

        for u in nodi_disp:
            for k_idx in range(inst.K):
                for pos in range(len(rotte[k_idx]) + 1):
                    ok, t_arr, _ = valuta_inserimento_rotta_pos_fast(r_arrays[k_idx], u, pos, inst.t, inst.e, inst.l, inst.s, inst.T_max, inst.c)
                    if ok:
                        if risk_aware and gamma_val > 0:
                            nodi_nuovi = nodi_attuali + [u]
                            delta_prot = protez_bertsimas_fast(inst.delta[nodi_nuovi], gamma_val) - prot_attuale
                            p_net = inst.p[u] - delta_prot
                            gain = np.sign(p_net) * (p_net**2) / (t_arr + 1e-5)
                        else:
                            gain = (inst.p[u] ** 2) / (t_arr + 1e-5)
                        candidati.append((gain, u, k_idx, pos))

        if not candidati:
            break

        candidati.sort(key=lambda x: x[0], reverse=True)
        if risk_aware and candidati[0][0] <= 0:
            break
        top_cands = candidati[: min(rcl_size, len(candidati))]
        scelto = random.choice(top_cands)

        _, u_sel, k_sel, pos_sel = scelto
        rotte[k_sel].insert(pos_sel, u_sel)
        nodi_disp.remove(u_sel)

    return rotte

def bcrc_crossover(p1, p2, inst: IstanzaTOPTW, gamma_val: float):
    rotte_non_vuote = [k for k in range(inst.K) if p1[k]]
    if not rotte_non_vuote:
        return [r[:] for r in p2]
    k_sel = random.choice(rotte_non_vuote)

    figlio = [r[:] for r in p2]
    nodi_da_inserire = list(p1[k_sel])
    set_ins = set(nodi_da_inserire)

    for k_idx in range(inst.K):
        figlio[k_idx] = [n for n in figlio[k_idx] if n not in set_ins]

    random.shuffle(nodi_da_inserire)
    for u in nodi_da_inserire:
        miglior_val = -1e9
        miglior_pos = None
        miglior_k = None
        for k_idx in range(inst.K):
            for pos in range(len(figlio[k_idx]) + 1):
                figlio[k_idx].insert(pos, u)
                val = valuta_soluzione_ga(figlio, inst, gamma_val)
                figlio[k_idx].pop(pos)
                if val > miglior_val:
                    miglior_val = val
                    miglior_pos = pos
                    miglior_k = k_idx
        if miglior_k is not None and miglior_val > inst.penale:
            figlio[miglior_k].insert(miglior_pos, u)

    return figlio

def lns_repair(sol, inst: IstanzaTOPTW, gamma_val: float, risk_aware=False, distruzione_pct=0.20, rcl_size=3):
    rotte = [r[:] for r in sol]
    nodi_serviti = [n for r in rotte for n in r]
    if not nodi_serviti:
        return rotte

    n_da_rimuovere = max(1, int(len(nodi_serviti) * distruzione_pct))
    nodi_rimossi = set(random.sample(nodi_serviti, n_da_rimuovere))
    for k_idx in range(inst.K):
        rotte[k_idx] = [n for n in rotte[k_idx] if n not in nodi_rimossi]

    nodi_attuali_set = set(x for r in rotte for x in r)
    nodi_disp = [n for n in range(1, inst.n_tot) if n not in nodi_attuali_set]

    while True:
        candidati = []
        nodi_attuali = [node for r in rotte for node in r]
        prot_attuale = protez_bertsimas_fast(inst.delta[nodi_attuali], gamma_val) if risk_aware else 0.0
        r_arrays = [np.array(r, dtype=np.int64) for r in rotte]

        for u in nodi_disp:
            for k_idx in range(inst.K):
                for pos in range(len(rotte[k_idx]) + 1):
                    ok, t_arr, _ = valuta_inserimento_rotta_pos_fast(r_arrays[k_idx], u, pos, inst.t, inst.e, inst.l, inst.s, inst.T_max, inst.c)
                    if ok:
                        if risk_aware and gamma_val > 0:
                            nodi_nuovi = nodi_attuali + [u]
                            delta_prot = protez_bertsimas_fast(inst.delta[nodi_nuovi], gamma_val) - prot_attuale
                            p_net = inst.p[u] - delta_prot
                            gain = np.sign(p_net) * (p_net**2) / (t_arr + 1e-5)
                        else:
                            gain = (inst.p[u] ** 2) / (t_arr + 1e-5)
                        candidati.append((gain, u, k_idx, pos))

        if not candidati:
            break

        candidati.sort(key=lambda x: x[0], reverse=True)
        if risk_aware and candidati[0][0] <= 0:
            break
        top_cands = candidati[: min(rcl_size, len(candidati))]
        scelto = random.choice(top_cands)

        _, u_sel, k_sel, pos_sel = scelto
        rotte[k_sel].insert(pos_sel, u_sel)
        nodi_disp.remove(u_sel)

    return rotte

def risolvi_con_ga_lns(inst: IstanzaTOPTW, gamma_val: float, risk_aware=False, pop_size=40, max_gen=150, p_cross=0.90, p_lns=0.30, distruzione_pct=0.20, n_elite=2, rcl_size=3, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    t0 = time.perf_counter()

    pop = [costruisci_soluzione_iniziale(inst, gamma_val, risk_aware=risk_aware, rcl_size=rcl_size) for _ in range(pop_size)]
    fit = [valuta_soluzione_ga(ind, inst, gamma_val) for ind in pop]

    for _ in range(max_gen):
        nuova_pop = []
        elite_idx = np.argsort(fit)[-n_elite:]
        for idx in elite_idx:
            nuova_pop.append([r[:] for r in pop[idx]])

        while len(nuova_pop) < pop_size:
            i1, i2 = random.sample(range(pop_size), 2)
            p_a = pop[i1] if fit[i1] >= fit[i2] else pop[i2]

            i3, i4 = random.sample(range(pop_size), 2)
            p_b = pop[i3] if fit[i3] >= fit[i4] else pop[i4]

            if random.random() < p_cross:
                figlio = bcrc_crossover(p_a, p_b, inst, gamma_val)
            else:
                figlio = [r[:] for r in p_a]

            if random.random() < p_lns:
                figlio = lns_repair(figlio, inst, gamma_val, risk_aware=risk_aware, distruzione_pct=distruzione_pct, rcl_size=rcl_size)

            nuova_pop.append(figlio)

        pop = nuova_pop
        fit = [valuta_soluzione_ga(ind, inst, gamma_val) for ind in pop]

    miglior_idx = int(np.argmax(fit))
    runtime = time.perf_counter() - t0
    return {
        "obj_val": round(fit[miglior_idx], 2),
        "rotte": [r for r in pop[miglior_idx] if r],
        "runtime": round(runtime, 3),
    }

# ==============================================================================
# 4. MOTORE MONTE CARLO STRESS-TEST
# ==============================================================================

def calcola_mon_scenario(rotte: List[List[int]], inst: IstanzaTOPTW, colpiti_set: Set[int]) -> float:
    costo_tot = 0.0
    nodi_serviti = []
    for r in rotte:
        if not r:
            continue
        r_arr = np.array(r, dtype=np.int64)
        ok, _, cost = simula_rotta_fast(r_arr, inst.t, inst.e, inst.l, inst.s, inst.T_max, inst.c)
        if not ok:
            return inst.penale
        costo_tot += cost
        nodi_serviti.extend(r)

    if not nodi_serviti:
        return 0.0

    ricavo_reale = sum((inst.p[i] - inst.delta[i]) if i in colpiti_set else inst.p[i] for i in nodi_serviti)
    return ricavo_reale - costo_tot

def esegui_montecarlo_istanza(inst: IstanzaTOPTW, piani: Dict[str, List[List[int]]], z_nom: float, n_sim=1000):
    omega_levels = [0.25, 0.50, 0.75, 1.00]
    alpha_levels = [0.40, 0.50, 0.60, 0.70]
    clienti = list(range(1, inst.n_tot))

    risultati_dettaglio = []
    risultati_riassunto = []

    for omega in omega_levels:
        k_colpiti = int(round(omega * inst.n_clienti))
        num_repliche = 1 if omega == 1.00 else n_sim

        mon_scenari = {nome: [] for nome in piani}

        for _ in range(num_repliche):
            if omega == 1.00:
                colpiti_set = set(clienti)
            else:
                colpiti_set = set(random.sample(clienti, k_colpiti))

            for nome, rotte in piani.items():
                mon = calcola_mon_scenario(rotte, inst, colpiti_set)
                mon_scenari[nome].append(mon)

        arr_mon_nom = np.array(mon_scenari["Nominale (Γ=0%)"])

        for nome, lista_mon in mon_scenari.items():
            arr_mon = np.array(lista_mon)
            media_mon = float(np.mean(arr_mon))
            min_mon = float(np.min(arr_mon))
            std_mon = float(np.std(arr_mon)) if len(arr_mon) > 1 else 0.0

            # Calcolo Win Rate rispetto alla soluzione nominale
            if nome == "Nominale (Γ=0%)":
                win_rate_str = "—"
            else:
                win_count = np.sum(arr_mon >= arr_mon_nom)
                win_rate_pct = (win_count / len(arr_mon)) * 100.0
                win_rate_str = f"{win_rate_pct:.1f}%"

            riga = {
                "Istanza": inst.nome,
                "N_Clienti": inst.n_clienti,
                "Omega (%)": f"{int(omega*100)}%",
                "Configurazione": nome,
                "E[MON] (€)": round(media_mon, 2),
                "Min MON (€)": round(min_mon, 2),
                "Std MON (€)": round(std_mon, 2),
                "Win Rate vs Nom (%)": win_rate_str,
            }

            for alpha in alpha_levels:
                soglia = alpha * z_nom
                pct_violazione = float(np.mean(arr_mon < soglia)) * 100.0
                riga[f"P(MON < {int(alpha*100)}%)"] = f"{pct_violazione:.1f}%"

            risultati_dettaglio.append(riga)

            # Raccogli nel riassunto solo lo scenario cardine omega = 50%
            if omega == 0.50:
                riga_sintesi = {
                    "Istanza": inst.nome,
                    "N": inst.n_clienti,
                    "Configurazione": nome,
                    "E[MON] (€)": round(media_mon, 2),
                    "Min MON (€)": round(min_mon, 2),
                    "Win Rate vs Nom": win_rate_str,
                    "P(MON < 50%)": riga["P(MON < 50%)"],
                    "P(MON < 60%)": riga["P(MON < 60%)"],
                    "P(MON < 70%)": riga["P(MON < 70%)"],
                }
                risultati_riassunto.append(riga_sintesi)

    return risultati_dettaglio, risultati_riassunto

# ==============================================================================
# 5. PIPELINE AUTOMATICA BATCH SU TUTTE LE ISTANZE E SALVATAGGIO CSV
# ==============================================================================

# ==============================================================================
# 5. PIPELINE BATCH SU ISTANZE SELEZIONATE E SALVATAGGIO CSV
# ==============================================================================

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    INSTANCES_DIR = BASE_DIR / "instances"
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Elenco mirato delle istanze da testare (3 JSON + 2 per tipologia C, R, RC, PR)
    ISTANZE_TARGET = [
        # Istanze del modello
        "istanza_10.json",
        "istanza_25.json",
        "istanza_50.json",
        # 2 Clustered (C)
        "c102.txt", "c201.txt",
        # 2 Random (R)
        "r101.txt", "r201.txt",
        # 2 Random-Clustered (RC)
        "rc102.txt", "rc201.txt",
        # 2 Realistic/Cordeau (PR)
        "pr01.txt", "pr10.txt"
    ]

    # Filtra solo i file effettivamente presenti nella cartella
    file_istanze = [INSTANCES_DIR / nome for nome in ISTANZE_TARGET if (INSTANCES_DIR / nome).exists()]

    if not file_istanze:
        print(f"Nessuna istanza target trovata in {INSTANCES_DIR}!")
        exit()

    print("\n" + "=" * 115)
    print(f"BATCH STRESS-TEST MONTE CARLO SELEZIONATO SU {len(file_istanze)} ISTANZE")
    print("=" * 115)

    tutti_dettagli = []
    tutti_riassunti = []

    for idx, path_inst in enumerate(file_istanze, start=1):
        print(f"\n[{idx}/{len(file_istanze)}] Caricamento ed elaborazione: {path_inst.name}...")
        inst = carica_qualsiasi_istanza(path_inst)

        # 1. Ottimizzazione delle 3 configurazioni con GA-LNS
        res_nom = risolvi_con_ga_lns(inst, gamma_val=0.0, risk_aware=False)
        res_rob50 = risolvi_con_ga_lns(inst, gamma_val=float(inst.n_clienti * 0.5), risk_aware=True)
        res_rob100 = risolvi_con_ga_lns(inst, gamma_val=float(inst.n_clienti * 1.0), risk_aware=True)

        z_nom = res_nom["obj_val"]
        piani = {
            "Nominale (Γ=0%)": res_nom["rotte"],
            "Robusto (Γ=50%)": res_rob50["rotte"],
            "Conservativo (Γ=100%)": res_rob100["rotte"],
        }

        # 2. Stress-test Monte Carlo
        dett, riass = esegui_montecarlo_istanza(inst, piani, z_nom=z_nom, n_sim=1000)
        tutti_dettagli.extend(dett)
        tutti_riassunti.extend(riass)

    # 3. Creazione e salvataggio dei file CSV
    df_dettaglio = pd.DataFrame(tutti_dettagli)
    df_riassunto = pd.DataFrame(tutti_riassunti)

    csv_dettaglio_path = RESULTS_DIR / "montecarlo_dettaglio.csv"
    csv_riassunto_path = RESULTS_DIR / "montecarlo_riassunto.csv"

    df_dettaglio.to_csv(csv_dettaglio_path, index=False, sep=";")
    df_riassunto.to_csv(csv_riassunto_path, index=False, sep=";")

    print("\n" + "=" * 115)
    print("ESECUZIONE COMPLETATA CON SUCCESSO!")
    print(f"✓ Dettaglio completo (tutti gli Omega e soglie) salvato in: {csv_dettaglio_path}")
    print(f"✓ Riassunto sintetico (Omega=50%) salvato in: {csv_riassunto_path}")
    print("=" * 115)

    print("\nANTEPRIMA DELLA TABELLA SINTETICA RIASSUNTIVA (Omega = 50%):")
    print(df_riassunto.to_string(index=False))