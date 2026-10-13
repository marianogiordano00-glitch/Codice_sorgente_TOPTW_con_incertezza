import os
import json
import copy
import random
import time
import math
import numpy as np
import numba
from numba import njit
import gurobipy as gp
from gurobipy import GRB

# ==========================================
# 1. CARICAMENTO DELL'ISTANZA DA FILE JSON
# ==========================================
def carica_istanza(percorso_file):
    with open(percorso_file, "r") as f:
        dati = json.load(f)
    return dati["nodes"], dati["K"], dati["T_max"]

# Nome del file nella sottocartella 'instances'
NOME_FILE = "istanza_50.json"

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

somma_profitti = float(np.sum(p))
PENALE = -(somma_profitti + 1.0)

# ==========================================
# 2. FUNZIONI COMPILATE JIT (NUMBA) CON C-LEVEL INSERT
# ==========================================
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


# ==========================================
# 3. MOTORE GA-LNS OTTIMIZZATO IN-PLACE
# ==========================================
def simula_rotta(rotta):
    if not rotta:
        return True, 0.0, 0.0
    r_arr = np.array(rotta, dtype=np.int64)
    return simula_rotta_fast(r_arr, t, e, l, s, T_max, c)


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
    risk_aware=False,
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


# ==========================================
# 4. RISOLUTORE ESATTO MILP CON GUROBI
# ==========================================
def risolvi_toptw_gurobi(gamma_val, time_limit=3600):
    V_c = list(range(1, n_tot))
    V = list(range(n_tot))
    Veicoli = list(range(K))

    m = gp.Model("Robust_TOPTW_Gurobi")
    m.Params.MIPFocus = 1
    m.Params.OutputFlag = 1
    m.Params.TimeLimit = time_limit

    x = m.addVars(V, V, Veicoli, vtype=GRB.BINARY, name="x")
    z = m.addVars(V_c, vtype=GRB.BINARY, name="z")
    t_arr = m.addVars(V, Veicoli, vtype=GRB.CONTINUOUS, lb=0.0, ub=T_max, name="t_arr")

    theta = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="theta")
    mu = m.addVars(V_c, vtype=GRB.CONTINUOUS, lb=0.0, name="mu")

    for i in V:
        for k_idx in Veicoli:
            m.addConstr(x[i, i, k_idx] == 0)

    for i in V_c:
        m.addConstr(gp.quicksum(x[i, j, k_idx] for j in V if j != i for k_idx in Veicoli) == z[i])

    for k_idx in Veicoli:
        m.addConstr(gp.quicksum(x[0, j, k_idx] for j in V_c) <= 1)
        m.addConstr(gp.quicksum(x[j, 0, k_idx] for j in V_c) <= 1)
        m.addConstr(gp.quicksum(x[0, j, k_idx] for j in V_c) == gp.quicksum(x[j, 0, k_idx] for j in V_c))

    for k_idx in Veicoli:
        for h in V_c:
            m.addConstr(
                gp.quicksum(x[i, h, k_idx] for i in V if i != h)
                == gp.quicksum(x[h, j, k_idx] for j in V if j != h)
            )

    M = T_max + float(np.max(s)) + float(np.max(t))
    for k_idx in Veicoli:
        m.addConstr(t_arr[0, k_idx] == 0.0)
        for i in V:
            for j in V_c:
                if i != j:
                    m.addConstr(
                        t_arr[j, k_idx] >= t_arr[i, k_idx] + s[i] + t[i, j] - M * (1 - x[i, j, k_idx])
                    )
        for i in V_c:
            m.addConstr(
                t_arr[i, k_idx] + s[i] + t[i, 0] <= T_max + M * (1 - x[i, 0, k_idx])
            )

    for i in V_c:
        for k_idx in Veicoli:
            m.addConstr(t_arr[i, k_idx] >= e[i] * gp.quicksum(x[j, i, k_idx] for j in V if j != i))
            m.addConstr(
                t_arr[i, k_idx] <= l[i] * gp.quicksum(x[j, i, k_idx] for j in V if j != i) + T_max * (1 - gp.quicksum(x[j, i, k_idx] for j in V if j != i))
            )

    for i in V_c:
        m.addConstr(theta + mu[i] >= delta[i] * z[i])

    profitto_nom = gp.quicksum(p[i] * z[i] for i in V_c)
    protezione_rob = gamma_val * theta + gp.quicksum(mu[i] for i in V_c)
    costo_trasp = gp.quicksum(c[i, j] * x[i, j, k_idx] for i in V for j in V for k_idx in Veicoli)

    m.setObjective(profitto_nom - protezione_rob - costo_trasp, GRB.MAXIMIZE)

    t_start = time.perf_counter()
    m.optimize()
    runtime = time.perf_counter() - t_start

    if m.Status == GRB.OPTIMAL:
        rotte = []
        for k_idx in Veicoli:
            rotta_k = []
            curr = 0
            while True:
                next_node = None
                for j in V:
                    if j != curr and x[curr, j, k_idx].X > 0.5:
                        next_node = j
                        break
                if next_node is None or next_node == 0:
                    break
                rotta_k.append(next_node)
                curr = next_node
            if rotta_k:
                rotte.append(rotta_k)

        return {
            "status": "OPTIMAL",
            "obj_val": round(m.ObjVal, 2),
            "rotte": rotte,
            "runtime": round(runtime, 3),
        }
    else:
        return {
            "status": "INFEASIBLE_OR_LIMITED",
            "obj_val": None,
            "rotte": [],
            "runtime": round(runtime, 3),
        }


# ==========================================
# 5. ESECUZIONE COMPARATIVA COMPLETA
# ==========================================
if __name__ == "__main__":
    scenari = [
        ("0%", 0.0),
        ("50%", float(n_clienti * 0.5)),
        ("100%", float(n_clienti * 1.0)),
    ]

    print("\n" + "=" * 115)
    print(f"ESECUZIONE SULL'ISTANZA: {NOME_FILE} (N={n_clienti} clienti, K={K} veicoli, T_max={T_max} min)")
    print("=" * 115)

    print(f"-> Somma Profitti Istanza: {somma_profitti:.2f} | Valore PENALE: {PENALE:.2f}\n")

    risultati = []

    for label_pct, gamma_val in scenari:
        print(f"\n>>> Risoluzione per Gamma = {gamma_val} ({label_pct} dei nodi)...")

        # 1. Gurobi MILP
        res_gurobi = risolvi_toptw_gurobi(gamma_val, time_limit=3600)

        # 2. GA-LNS Standard (RA = False)
        res_ma_std = memetic_algorithm_solver(gamma_val, risk_aware=False)

        # 3. GA-LNS Risk-Aware (RA = True)
        res_ma_ra = memetic_algorithm_solver(gamma_val, risk_aware=True)

        gap_std = ((res_gurobi["obj_val"] - res_ma_std["obj_val"]) / res_gurobi["obj_val"]) * 100 if res_gurobi["obj_val"] else None
        gap_ra = ((res_gurobi["obj_val"] - res_ma_ra["obj_val"]) / res_gurobi["obj_val"]) * 100 if res_gurobi["obj_val"] else None

        risultati.append({
            "gamma_pct": label_pct,
            "gamma_val": gamma_val,
            "z_milp": res_gurobi["obj_val"],
            "rotte_milp": str(res_gurobi["rotte"]),
            "t_milp": res_gurobi["runtime"],
            "z_ma_std": res_ma_std["obj_val"],
            "gap_std": round(gap_std, 2) if gap_std is not None else "-",
            "rotte_ma_std": str(res_ma_std["rotte"]),
            "t_ma_std": res_ma_std["runtime"],
            "z_ma_ra": res_ma_ra["obj_val"],
            "gap_ra": round(gap_ra, 2) if gap_ra is not None else "-",
            "rotte_ma_ra": str(res_ma_ra["rotte"]),
            "t_ma_ra": res_ma_ra["runtime"],
        })

    print("\n" + "=" * 115)
    print("TABELLA RIASSUNTIVA COMPARATIVA")
    print("=" * 115)
    header = f"{'Gamma (%)':<10} | {'Z*_MILP':<9} | {'T_MILP(s)':<10} | {'Z_MA (Std)':<10} | {'Gap Std %':<10} | {'Z_MA (RA)':<10} | {'Gap RA %':<10} | {'T_MA (s)':<10} | {'Rotte Ottime MILP'}"
    print(header)
    print("-" * 115)
    for r in risultati:
        z_milp_str = f"{r['z_milp']:.2f}" if r['z_milp'] is not None else "N/D"
        gap_s_str = f"{r['gap_std']}%" if r['gap_std'] != "-" else "-"
        gap_r_str = f"{r['gap_ra']}%" if r['gap_ra'] != "-" else "-"
        print(
            f"{r['gamma_pct']:<10} | {z_milp_str:<9} | {r['t_milp']:<10.3f} | {r['z_ma_std']:<10.2f} | {gap_s_str:<10} | {r['z_ma_ra']:<10.2f} | {gap_r_str:<10} | {r['t_ma_std']:<10.3f} | {r['rotte_milp']}"
        )
    print("=" * 115)

    print("\nFORMATO TABELLA PER WORD (COPIA E INCOLLA IN WORD -> Converti in Tabella):")
    print("Gamma (%)\tZ*_MILP (Gurobi)\tTempo MILP (s)\tZ_MA\tGap %\tTempo MA (s)\tRotte Ottimo Globale\tRotte MA")
    for r in risultati:
        print(f"{r['gamma_pct']}\t{r['z_milp']}\t{r['t_milp']}\t{r['z_ma_ra']}\t{r['gap_ra']}%\t{r['t_ma_ra']}\t{r['rotte_milp']}\t{r['rotte_ma_ra']}")