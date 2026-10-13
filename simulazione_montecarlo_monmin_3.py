import os
import json
import random
import time
import math
import numpy as np
import numba
from numba import njit
import pandas as pd

# ==============================================================================
# 1. CARICAMENTO ISTANZA E CONFIGURAZIONE PENALE
# ==============================================================================
def carica_istanza(percorso_file):
    with open(percorso_file, "r") as f:
        dati = json.load(f)
    return dati["nodes"], dati["K"], dati["T_max"]

NOME_FILE = "istanza_10.json"  # Modificabile con istanza_10.json o istanza_25.json

cartella_script = os.path.dirname(os.path.abspath(__file__))
percorso_completo = os.path.join(cartella_script, "instances", NOME_FILE)

nodes_data, K, T_max = carica_istanza(percorso_completo)

coords = np.array([[n["x"], n["y"]] for n in nodes_data], dtype=float)
p = np.array([n["p"] for n in nodes_data], dtype=float)
delta = 0.50 * p
e = np.array([n["e"] for n in nodes_data], dtype=float)
l = np.array([n["l"] for n in nodes_data], dtype=float)
s = np.array([n["s"] for n in nodes_data], dtype=float)
n_tot = len(nodes_data)
n_clienti = n_tot - 1

diff = coords[:, None, :] - coords[None, :, :]
dist = np.sqrt((diff**2).sum(axis=-1))
t = dist.copy()
c = 0.5 * dist.copy()

# Penalità esatta standardizzata: -(somma profitti + 1.0)
somma_profitti = float(np.sum(p))
PENALE = -(somma_profitti + 1.0)


# ==============================================================================
# 2. FUNZIONI COMPILATE JIT (NUMBA)
# ==============================================================================
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
def calcola_protezione_robusta_fast(deviazioni: np.ndarray, gamma_val: float) -> float:
    if len(deviazioni) == 0 or gamma_val <= 0.0:
        return 0.0
    dev_ord = np.sort(deviazioni)[::-1]
    k_int = int(math.floor(gamma_val))
    frazione = gamma_val - k_int
    n_dev = len(dev_ord)
    limite = min(k_int, n_dev)
    prot = 0.0
    for i in range(limite):
        prot += dev_ord[i]
    if frazione > 0.0 and k_int < n_dev:
        prot += frazione * dev_ord[k_int]
    return prot


# ==============================================================================
# 3. MOTORE GA-LNS OTTIMIZZATO
# ==============================================================================
def calcola_protezione_robusta(nodi_serviti, gamma_val):
    if len(nodi_serviti) == 0 or gamma_val <= 0:
        return 0.0
    dev_arr = delta[nodi_serviti]
    return calcola_protezione_robusta_fast(dev_arr, gamma_val)


def valuta_soluzione(rotte, gamma_val):
    nodi_serviti = []
    costo_tot = 0.0
    for r in rotte:
        if not r:
            continue
        r_arr = np.array(r, dtype=np.int64)
        ok, _, cost = simula_rotta_fast(r_arr, t, e, l, s, T_max, c)
        if not ok:
            return PENALE
        costo_tot += cost
        nodi_serviti.extend(r)

    if len(nodi_serviti) != len(set(nodi_serviti)):
        return PENALE

    if not nodi_serviti:
        return 0.0

    profitto_nom = float(np.sum(p[nodi_serviti]))
    prot = calcola_protezione_robusta_fast(delta[nodi_serviti], gamma_val)
    return profitto_nom - prot - costo_tot


def costruisci_soluzione_iniziale(gamma_val, risk_aware=False, rcl_size=3):
    rotte = [[] for _ in range(K)]
    nodi_disp = list(range(1, n_tot))
    random.shuffle(nodi_disp)

    while True:
        candidati = []
        nodi_attuali = [node for r in rotte for node in r]
        prot_attuale = calcola_protezione_robusta(nodi_attuali, gamma_val) if risk_aware else 0.0
        r_arrays = [np.array(r, dtype=np.int64) for r in rotte]

        for u in nodi_disp:
            for k_idx in range(K):
                for pos in range(len(rotte[k_idx]) + 1):
                    ok, t_arr, _ = valuta_inserimento_rotta_pos_fast(r_arrays[k_idx], u, pos, t, e, l, s, T_max, c)
                    if ok:
                        if risk_aware and gamma_val > 0:
                            nodi_nuovi = nodi_attuali + [u]
                            delta_prot = calcola_protezione_robusta(nodi_nuovi, gamma_val) - prot_attuale
                            p_net = p[u] - delta_prot
                            gain = np.sign(p_net) * (p_net**2) / (t_arr + 1e-5)
                        else:
                            gain = (p[u] ** 2) / (t_arr + 1e-5)
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


def bcrc_crossover(p1, p2, gamma_val):
    rotte_non_vuote = [k for k in range(K) if p1[k]]
    if not rotte_non_vuote:
        return [r[:] for r in p2]
    k_sel = random.choice(rotte_non_vuote)

    figlio = [r[:] for r in p2]
    nodi_da_inserire = list(p1[k_sel])
    set_ins = set(nodi_da_inserire)

    for k_idx in range(K):
        figlio[k_idx] = [n for n in figlio[k_idx] if n not in set_ins]

    random.shuffle(nodi_da_inserire)
    for u in nodi_da_inserire:
        miglior_val = -1e9
        miglior_pos = None
        miglior_k = None
        for k_idx in range(K):
            for pos in range(len(figlio[k_idx]) + 1):
                figlio[k_idx].insert(pos, u)
                val = valuta_soluzione(figlio, gamma_val)
                figlio[k_idx].pop(pos)
                if val > miglior_val:
                    miglior_val = val
                    miglior_pos = pos
                    miglior_k = k_idx
        if miglior_k is not None and miglior_val > -10000.0:
            figlio[miglior_k].insert(miglior_pos, u)

    return figlio


def lns_repair(sol, gamma_val, risk_aware=False, distruzione_pct=0.20, rcl_size=3):
    rotte = [r[:] for r in sol]
    nodi_serviti = [n for r in rotte for n in r]
    if not nodi_serviti:
        return rotte

    n_da_rimuovere = max(1, int(len(nodi_serviti) * distruzione_pct))
    nodi_rimossi = set(random.sample(nodi_serviti, n_da_rimuovere))
    for k_idx in range(K):
        rotte[k_idx] = [n for n in rotte[k_idx] if n not in nodi_rimossi]

    nodi_attuali_set = set(x for r in rotte for x in r)
    nodi_disp = [n for n in range(1, n_tot) if n not in nodi_attuali_set]

    while True:
        candidati = []
        nodi_attuali = [node for r in rotte for node in r]
        prot_attuale = calcola_protezione_robusta(nodi_attuali, gamma_val) if risk_aware else 0.0
        r_arrays = [np.array(r, dtype=np.int64) for r in rotte]

        for u in nodi_disp:
            for k_idx in range(K):
                for pos in range(len(rotte[k_idx]) + 1):
                    ok, t_arr, _ = valuta_inserimento_rotta_pos_fast(r_arrays[k_idx], u, pos, t, e, l, s, T_max, c)
                    if ok:
                        if risk_aware and gamma_val > 0:
                            nodi_nuovi = nodi_attuali + [u]
                            delta_prot = calcola_protezione_robusta(nodi_nuovi, gamma_val) - prot_attuale
                            p_net = p[u] - delta_prot
                            gain = np.sign(p_net) * (p_net**2) / (t_arr + 1e-5)
                        else:
                            gain = (p[u] ** 2) / (t_arr + 1e-5)
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


def memetic_algorithm_solver(
    gamma_val,
    risk_aware,
    pop_size=40,
    max_gen=150,
    p_cross=0.90,
    p_lns=0.30,
    distruzione_pct=0.20,
    n_elite=2,
    rcl_size=3,
    seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    t0 = time.perf_counter()

    pop = [
        costruisci_soluzione_iniziale(
            gamma_val, risk_aware=risk_aware, rcl_size=rcl_size
        )
        for _ in range(pop_size)
    ]
    fit = [valuta_soluzione(ind, gamma_val) for ind in pop]

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
                figlio = bcrc_crossover(p_a, p_b, gamma_val)
            else:
                figlio = [r[:] for r in p_a]

            if random.random() < p_lns:
                figlio = lns_repair(
                    figlio,
                    gamma_val,
                    risk_aware=risk_aware,
                    distruzione_pct=distruzione_pct,
                    rcl_size=rcl_size,
                )

            nuova_pop.append(figlio)

        pop = nuova_pop
        fit = [valuta_soluzione(ind, gamma_val) for ind in pop]

    miglior_idx = int(np.argmax(fit))
    runtime = time.perf_counter() - t0
    return {
        "obj_val": round(fit[miglior_idx], 2),
        "rotte": [r for r in pop[miglior_idx] if r],
        "runtime": round(runtime, 3),
    }


# ==============================================================================
# 4. VALUTAZIONE EX-POST E SIMULAZIONE MONTE CARLO (MON_min)
# ==============================================================================
def calcola_mon_reale(rotte, nodi_colpiti_set):
    """Calcola il Margine Operativo Netto effettivo consuntivato in uno scenario perturbato."""
    nodi_serviti = []
    costo_tot = 0.0
    for r in rotte:
        if not r:
            continue
        r_arr = np.array(r, dtype=np.int64)
        ok, _, cost = simula_rotta_fast(r_arr, t, e, l, s, T_max, c)
        if not ok:
            return PENALE
        costo_tot += cost
        nodi_serviti.extend(r)

    if not nodi_serviti:
        return 0.0

    # Ricavo effettivo decurtato solo sui nodi colpiti dallo shock
    ricavo_reale = sum(
        (p[i] - delta[i]) if i in nodi_colpiti_set else p[i]
        for i in nodi_serviti
    )
    return ricavo_reale - costo_tot


def esegui_stress_test_monte_carlo(piani, z_nom, n_sim=1000):
    omega_levels = [0.25, 0.50, 0.75, 1.00]
    alpha_levels = [0.40, 0.50, 0.60, 0.70]
    tutti_i_clienti = list(range(1, n_tot))

    risultati_aggregati = []

    print("\n" + "=" * 120)
    print(f"AVVIO SIMULAZIONE MONTE CARLO STRESS-TEST (N_sim = {n_sim}, N_clienti = {n_clienti})")
    print(f"Baseline Margine Nominale Z_nom: {z_nom:.2f} €")
    print("=" * 120)

    for omega in omega_levels:
        k_colpiti = int(round(omega * n_clienti))
        num_repliche = 1 if omega == 1.00 else n_sim

        print(f"-> Simulazione per Omega = {int(omega*100)}% ({k_colpiti}/{n_clienti} nodi perturbati) | Repliche: {num_repliche}...")

        mon_scenari = {nome: [] for nome in piani}

        for _ in range(num_repliche):
            if omega == 1.00:
                colpiti_set = set(tutti_i_clienti)
            else:
                colpiti_set = set(random.sample(tutti_i_clienti, k_colpiti))

            for nome, piano_rotte in piani.items():
                mon = calcola_mon_reale(piano_rotte, colpiti_set)
                mon_scenari[nome].append(mon)

        for nome, lista_mon in mon_scenari.items():
            arr_mon = np.array(lista_mon)
            media_mon = float(np.mean(arr_mon))
            min_mon = float(np.min(arr_mon))
            std_mon = float(np.std(arr_mon)) if len(arr_mon) > 1 else 0.0

            riga = {
                "Omega (%)": f"{int(omega*100)}%",
                "Configurazione": nome,
                "E[MON] (€)": round(media_mon, 2),
                "Min MON (€)": round(min_mon, 2),
                "Std MON (€)": round(std_mon, 2),
            }

            for alpha in alpha_levels:
                soglia = alpha * z_nom
                pct_violazione = float(np.mean(arr_mon < soglia)) * 100.0
                riga[f"P(MON < {int(alpha*100)}%)"] = f"{pct_violazione:.1f}%"

            risultati_aggregati.append(riga)

    return pd.DataFrame(risultati_aggregati)


# ==============================================================================
# 5. ESECUZIONE COMPLETA
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 120)
    print(f"PIPELINE COMPLETA MONTE CARLO E ANALISI MON_min: {NOME_FILE}")
    print(f"Somma Profitti Istanza: {somma_profitti:.2f} € | Penale: {PENALE:.2f}")
    print("=" * 120)

    # 1. Generazione delle soluzioni di riferimento tramite GA-LNS
    print("\n>>> FASE 1: Ottimizzazione dei piani logistici tramite GA-LNS...")
    
    # Piano Nominale (Gamma = 0%)
    print("1/3 Ottimizzazione Piano Nominale (Gamma = 0%)...")
    res_nom = memetic_algorithm_solver(gamma_val=0.0, risk_aware=False)
    z_nom = res_nom["obj_val"]
    piano_nom = res_nom["rotte"]
    print(f"    -> Z_nom: {z_nom:.2f} € | Tempo: {res_nom['runtime']} s")

    # Piano Robusto Medio (Gamma = 50%)
    print(f"2/3 Ottimizzazione Piano Robusto Medio (Gamma = 50% = {n_clienti*0.5})...")
    res_rob50 = memetic_algorithm_solver(gamma_val=float(n_clienti * 0.5), risk_aware=True)
    piano_rob50 = res_rob50["rotte"]
    print(f"    -> Z_rob(50%): {res_rob50['obj_val']:.2f} € | Tempo: {res_rob50['runtime']} s")

    # Piano Robusto Conservativo (Gamma = 100%)
    print(f"3/3 Ottimizzazione Piano Robusto Conservativo (Gamma = 100% = {n_clienti*1.0})...")
    res_rob100 = memetic_algorithm_solver(gamma_val=float(n_clienti * 1.0), risk_aware=True)
    piano_rob100 = res_rob100["rotte"]
    print(f"    -> Z_rob(100%): {res_rob100['obj_val']:.2f} € | Tempo: {res_rob100['runtime']} s")

    piani_da_testare = {
        "Nominale (Γ=0%)": piano_nom,
        "Robusto (Γ=50%)": piano_rob50,
        "Conservativo (Γ=100%)": piano_rob100,
    }

    # 2. Esecuzione Simulazione Monte Carlo
    print("\n>>> FASE 2: Esecuzione Monte Carlo e scansione soglie MON_min...")
    df_risultati = esegui_stress_test_monte_carlo(piani_da_testare, z_nom=z_nom, n_sim=1000)

    # 3. Stampa Tabella Riassuntiva
    print("\n" + "=" * 120)
    print("TABELLA RIASSUNTIVA MONTE CARLO: RISCHIO DI FALLIMENTO E MARGINE ATTESO (MON_min)")
    print("=" * 120)
    print(df_risultati.to_string(index=False))
    print("=" * 120)

    # 4. Formato per esportazione su Word
    print("\nFORMATO TABELLA PER WORD (Copia e incolla in Word -> Inserisci -> Tabella -> Converti il testo in tabella):")
    cols = list(df_risultati.columns)
    print("\t".join(cols))
    for _, row in df_risultati.iterrows():
        print("\t".join(str(val) for val in row.values))