import concurrent.futures
import copy
from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Set, Tuple

import numba
from numba import njit
import numpy as np
import pandas as pd

# ==============================================================================
# 1. STRUTTURE DATI E DEFINIZIONI
# ==============================================================================

@dataclass
class Instance:
    n: int
    coords: np.ndarray
    t: np.ndarray
    c: np.ndarray
    p: np.ndarray
    delta: np.ndarray
    e: np.ndarray
    l: np.ndarray
    s: np.ndarray
    N_prio: Set[int]
    N_opt: Set[int]
    K: int
    T_max: float
    Gamma: float
    Usa_Variante_Margine: bool = False
    M_min: float = 0.0

    def copy(self):
        return Instance(
            n=self.n,
            coords=self.coords,
            t=self.t,
            c=self.c,
            p=self.p,
            delta=self.delta,
            e=self.e,
            l=self.l,
            s=self.s,
            N_prio=set(self.N_prio),
            N_opt=set(self.N_opt),
            K=self.K,
            T_max=self.T_max,
            Gamma=self.Gamma,
            Usa_Variante_Margine=self.Usa_Variante_Margine,
            M_min=self.M_min,
        )


@dataclass
class Individuo:
    routes: List[List[int]]
    fitness: Optional[float] = None

    def copy(self):
        return Individuo(routes=[r[:] for r in self.routes], fitness=self.fitness)


# ==============================================================================
# 2. FUNZIONI COMPILATE JIT (NUMBA) CON INSERIMENTO NATIVO C
# ==============================================================================

@njit(fastmath=True)
def simula_rotta_fast(route: np.ndarray, t_mat: np.ndarray, e_arr: np.ndarray, l_arr: np.ndarray, s_arr: np.ndarray, T_max: float):
    if len(route) == 0:
        return True, 0.0
    orario = 0.0
    prev = 0
    for i in range(len(route)):
        nodo = route[i]
        orario += t_mat[prev, nodo]
        if orario > l_arr[nodo] + 1e-9:
            return False, -1.0
        if orario < e_arr[nodo]:
            orario = e_arr[nodo]
        orario += s_arr[nodo]
        prev = nodo
    orario += t_mat[prev, 0]
    if orario > T_max + 1e-9:
        return False, -1.0
    return True, orario


@njit(fastmath=True)
def valuta_inserimento_rotta_pos_fast(
    route: np.ndarray,
    nodo: int,
    pos: int,
    t_mat: np.ndarray,
    e_arr: np.ndarray,
    l_arr: np.ndarray,
    s_arr: np.ndarray,
    T_max: float,
):
    """Valuta la rotta risultante dall'inserimento del nodo in posizione pos senza allocare array."""
    n_len = len(route) + 1
    orario = 0.0
    prev = 0
    for idx in range(n_len):
        if idx < pos:
            curr = route[idx]
        elif idx == pos:
            curr = nodo
        else:
            curr = route[idx - 1]

        orario += t_mat[prev, curr]
        if orario > l_arr[curr] + 1e-9:
            return False, -1.0
        if orario < e_arr[curr]:
            orario = e_arr[curr]
        orario += s_arr[curr]
        prev = curr

    orario += t_mat[prev, 0]
    if orario > T_max + 1e-9:
        return False, -1.0
    return True, orario


@njit(fastmath=True)
def trova_miglior_inserimento_rotta_fast(
    route: np.ndarray,
    nodo: int,
    t_mat: np.ndarray,
    e_arr: np.ndarray,
    l_arr: np.ndarray,
    s_arr: np.ndarray,
    T_max: float,
):
    """Esegue l'intero ciclo di inserimento su tutte le posizioni in C nativo, ritornando la migliore."""
    ok_prima, tempo_prima = simula_rotta_fast(route, t_mat, e_arr, l_arr, s_arr, T_max)
    if not ok_prima:
        return -1, 1e18, -1.0

    best_pos = -1
    best_delta = 1e18
    best_tempo_dopo = -1.0
    n_pos = len(route) + 1

    for j in range(n_pos):
        ok_dopo, tempo_dopo = valuta_inserimento_rotta_pos_fast(
            route, nodo, j, t_mat, e_arr, l_arr, s_arr, T_max
        )
        if ok_dopo:
            delta = tempo_dopo - tempo_prima
            if delta < best_delta:
                best_delta = delta
                best_pos = j
                best_tempo_dopo = tempo_dopo

    return best_pos, best_delta, best_tempo_dopo


@njit(fastmath=True)
def calcola_costo_rotta_fast(route: np.ndarray, c_mat: np.ndarray) -> float:
    if len(route) == 0:
        return 0.0
    costo = c_mat[0, route[0]]
    for i in range(len(route) - 1):
        costo += c_mat[route[i], route[i + 1]]
    costo += c_mat[route[-1], 0]
    return costo


@njit(fastmath=True)
def protez_bertsimas_fast(deviazioni: np.ndarray, Gamma: float) -> float:
    if len(deviazioni) == 0:
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
# 3. PARSER DELLE ISTANZE BENCHMARK (.TXT)
# ==============================================================================

def parse_toptw_instance(file_path: str, Gamma: float = 0.0) -> Instance:
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    riga_testata = lines[0].split()
    K_val = int(riga_testata[1]) if len(riga_testata) >= 2 else 3

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
    n = len(coords)

    T_max_val = float(l[0])
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))
    t = dist.copy()
    c = dist.copy()

    somma_profitti = float(np.sum(p))
    PENALE = -(somma_profitti + 1.0)

    delta = 0.3 * p
    delta[0] = 0.0
    N_prio = set(range(1, n))
    N_opt = set()

    return Instance(
        n=n,
        coords=coords,
        t=t,
        c=c,
        p=p,
        delta=delta,
        e=e,
        l=l,
        s=s,
        N_prio=N_prio,
        N_opt=N_opt,
        K=K_val,
        T_max=T_max_val,
        Gamma=Gamma,
    )

# ==============================================================================
# 4. FUNZIONI DI VALUTAZIONE E OPERATORI METAEURISTICI OTTIMIZZATI
# ==============================================================================

def simula_rotta(route: List[int], inst: Instance) -> Tuple[bool, Optional[float], List[float], List[float]]:
    if not route:
        return True, 0.0, [], []
    r_arr = np.array(route, dtype=np.int64)
    fattibile, tempo = simula_rotta_fast(r_arr, inst.t, inst.e, inst.l, inst.s, inst.T_max)
    if not fattibile:
        return False, None, [], []
    return True, tempo, [], []


def preprocessing_graph(inst: Instance) -> Instance:
    inst2 = inst.copy()
    da_rimuovere = set()
    for i in list(inst2.N_opt):
        r_arr = np.array([i], dtype=np.int64)
        fattibile, _ = simula_rotta_fast(r_arr, inst2.t, inst2.e, inst2.l, inst2.s, inst2.T_max)
        round_trip_cost = inst2.c[0, i] + inst2.c[i, 0]
        ricavo_netto = inst2.p[i] - round_trip_cost
        if not fattibile or ricavo_netto < 0:
            da_rimuovere.add(i)
    inst2.N_opt = inst2.N_opt - da_rimuovere
    return inst2


def calcola_profitto_robusto(inst: Instance, ind: Individuo) -> float:
    nodi_serviti = []
    for r in ind.routes:
        nodi_serviti.extend(r)
    nodi_set = set(nodi_serviti)

    if len(nodi_set & inst.N_prio) < inst.K:
        return inst.PENALE

    costo_trasporto_totale = 0.0
    for route in ind.routes:
        if not route:
            continue
        r_arr = np.array(route, dtype=np.int64)
        fattibile, _ = simula_rotta_fast(r_arr, inst.t, inst.e, inst.l, inst.s, inst.T_max)
        if not fattibile:
            return inst.PENALE
        costo_trasporto_totale += calcola_costo_rotta_fast(r_arr, inst.c)

    if not nodi_serviti:
        return 0.0

    ricavo_nominale = float(np.sum(inst.p[nodi_serviti]))
    deviazioni = inst.delta[nodi_serviti]
    protezione = protez_bertsimas_fast(deviazioni, inst.Gamma)
    profitto_robusto_lordo = ricavo_nominale - protezione

    if inst.Usa_Variante_Margine:
        margine_netto = profitto_robusto_lordo - costo_trasporto_totale
        if margine_netto < inst.M_min:
            return inst.PENALE
        return margine_netto
    return profitto_robusto_lordo


def calcola_gain_inserimento(
    nodo: int,
    delta_t: float,
    deviazioni_attuali: List[float],
    inst: Instance,
    risk_aware: bool,
) -> float:
    if not risk_aware or inst.Gamma == 0.0:
        return (inst.p[nodo] ** 2) / delta_t
    dev_arr = np.array(deviazioni_attuali, dtype=np.float64)
    prot_prima = protez_bertsimas_fast(dev_arr, inst.Gamma)
    prot_dopo = protez_bertsimas_fast(np.append(dev_arr, inst.delta[nodo]), inst.Gamma)
    p_netto = inst.p[nodo] - (prot_dopo - prot_prima)
    segno = 1.0 if p_netto >= 0 else -1.0
    return segno * (p_netto**2) / delta_t


def inserisci_nodo_migliore_posizione(
    routes: List[List[int]], nodo: int, inst: Instance
) -> bool:
    best_delta = float("inf")
    best_route_idx, best_pos = None, None

    for k, route in enumerate(routes):
        r_arr = np.array(route, dtype=np.int64)
        pos, delta, _ = trova_miglior_inserimento_rotta_fast(
            r_arr, nodo, inst.t, inst.e, inst.l, inst.s, inst.T_max
        )
        if pos != -1 and delta < best_delta:
            best_delta = delta
            best_route_idx = k
            best_pos = pos

    if best_route_idx is None:
        return False
    routes[best_route_idx].insert(best_pos, nodo)
    return True


def inizializza_individuo_randomizzato(
    inst: Instance, rcl_size: int = 3, risk_aware: bool = False
) -> Individuo:
    routes = [[] for _ in range(inst.K)]
    prio = list(inst.N_prio)
    random.shuffle(prio)
    for nodo in prio:
        inserisci_nodo_migliore_posizione(routes, nodo, inst)

    candidati = list(inst.N_opt)
    random.shuffle(candidati)

    while candidati:
        deviazioni_attuali = [inst.delta[i] for r in routes for i in r]
        valutazioni = []
        r_arrays = [np.array(r, dtype=np.int64) for r in routes]

        for nodo in candidati:
            for k in range(len(routes)):
                pos, delta_t, _ = trova_miglior_inserimento_rotta_fast(
                    r_arrays[k], nodo, inst.t, inst.e, inst.l, inst.s, inst.T_max
                )
                if pos != -1:
                    dt = max(delta_t, 1e-6)
                    gain = calcola_gain_inserimento(nodo, dt, deviazioni_attuali, inst, risk_aware)
                    valutazioni.append((gain, nodo, k, pos))

        if not valutazioni:
            break
        valutazioni.sort(key=lambda x: x[0], reverse=True)
        if risk_aware and valutazioni[0][0] <= 0:
            break
        rcl = valutazioni[:rcl_size]
        gain, nodo, k, j = random.choice(rcl)
        routes[k].insert(j, nodo)
        candidati.remove(nodo)

    return Individuo(routes=routes)


def selezione_torneo(popolazione: List[Individuo], k: int = 3) -> Individuo:
    candidati = random.sample(popolazione, min(k, len(popolazione)))
    return max(candidati, key=lambda ind: ind.fitness).copy()


def _bcrc_single(
    parent_donor: Individuo, parent_base: Individuo, inst: Instance
) -> Individuo:
    rotte_non_vuote = [r for r in parent_donor.routes if r]
    if not rotte_non_vuote:
        return parent_base.copy()
    rotta_donatrice = random.choice(rotte_non_vuote)
    nodi_da_reinserire = rotta_donatrice[:]
    set_reinserire = set(nodi_da_reinserire)
    nuove_routes = [[n for n in r if n not in set_reinserire] for r in parent_base.routes]
    figlio = Individuo(routes=nuove_routes)
    random.shuffle(nodi_da_reinserire)
    for nodo in nodi_da_reinserire:
        inserisci_nodo_migliore_posizione(figlio.routes, nodo, inst)
    return figlio


def crossover_bcrc(
    genitore_1: Individuo, genitore_2: Individuo, inst: Instance
) -> Tuple[Individuo, Individuo]:
    return _bcrc_single(genitore_1, genitore_2, inst), _bcrc_single(
        genitore_2, genitore_1, inst
    )


def distruggi_rotte_casuale(
    ind: Individuo, num_nodi_da_rimuovere: int
) -> Tuple[List[int], Individuo]:
    tutti_i_nodi = [(k, nodo) for k, route in enumerate(ind.routes) for nodo in route]
    if not tutti_i_nodi:
        return [], ind.copy()
    n_rimuovere = min(num_nodi_da_rimuovere, len(tutti_i_nodi))
    da_rimuovere = random.sample(tutti_i_nodi, n_rimuovere)
    ind_parziale = ind.copy()
    nodi_rimossi = []
    for k, nodo in da_rimuovere:
        if nodo in ind_parziale.routes[k]:
            ind_parziale.routes[k].remove(nodo)
            nodi_rimossi.append(nodo)
    return nodi_rimossi, ind_parziale


def conteggio_prio(ind: Individuo, inst: Instance) -> int:
    return len(set(nodo for r in ind.routes for nodo in r) & inst.N_prio)


def applica_lns(
    inst: Instance,
    individuo: Individuo,
    distruzione_pct: float = 0.20,
    max_iter_repair: int = 2000,
    risk_aware: bool = False,
) -> Individuo:
    tutti_i_nodi = sum(len(r) for r in individuo.routes)
    num_nodi = max(1, int(round(distruzione_pct * tutti_i_nodi))) if tutti_i_nodi > 0 else 0
    nodi_rimossi, individuo_parziale = distruggi_rotte_casuale(individuo, num_nodi)

    serviti = set()
    for r in individuo_parziale.routes:
        serviti.update(r)
    candidati = list(set(nodi_rimossi) | ((inst.N_prio | inst.N_opt) - serviti))
    prio_candidati = [n for n in candidati if n in inst.N_prio]
    random.shuffle(prio_candidati)

    it = 0
    while len(serviti & inst.N_prio) < inst.K and prio_candidati and it < max_iter_repair:
        it += 1
        m = prio_candidati.pop(0)
        if inserisci_nodo_migliore_posizione(individuo_parziale.routes, m, inst):
            if m in candidati:
                candidati.remove(m)
            serviti.add(m)

    ancora_inseribile = True
    it = 0
    while ancora_inseribile and candidati and it < max_iter_repair:
        it += 1
        deviazioni_attuali = [inst.delta[i] for r in individuo_parziale.routes for i in r]
        miglior_gain = -1.0 if not risk_aware else float("-inf")
        miglior_nodo, miglior_route_idx, miglior_pos = None, None, None

        r_arrays = [np.array(r, dtype=np.int64) for r in individuo_parziale.routes]

        for nodo in candidati:
            for k in range(len(individuo_parziale.routes)):
                pos, delta_t, _ = trova_miglior_inserimento_rotta_fast(
                    r_arrays[k], nodo, inst.t, inst.e, inst.l, inst.s, inst.T_max
                )
                if pos != -1:
                    dt = max(delta_t, 1e-6)
                    gain = calcola_gain_inserimento(nodo, dt, deviazioni_attuali, inst, risk_aware)
                    if gain > miglior_gain:
                        miglior_gain = gain
                        miglior_nodo = nodo
                        miglior_route_idx = k
                        miglior_pos = pos

        if miglior_nodo is not None and not (risk_aware and miglior_gain <= 0):
            individuo_parziale.routes[miglior_route_idx].insert(miglior_pos, miglior_nodo)
            candidati.remove(miglior_nodo)
        else:
            ancora_inseribile = False

    return individuo_parziale


def hybrid_genetic_lns_top_solver(
    inst: Instance,
    pop_size: int = 30,
    max_gen: int = 100,
    p_cross: float = 0.8,
    p_lns: float = 0.3,
    tournament_k: int = 3,
    n_elite: int = 2,
    rcl_size: int = 3,
    distruzione_pct: float = 0.20,
    risk_aware: bool = False,
    verbose: bool = False,
) -> Tuple[Individuo, float, List[float]]:
    inst_f = preprocessing_graph(inst)
    popolazione = [
        inizializza_individuo_randomizzato(inst_f, rcl_size=rcl_size, risk_aware=risk_aware)
        for _ in range(pop_size)
    ]
    for ind in popolazione:
        ind.fitness = calcola_profitto_robusto(inst_f, ind)

    sol_best = max(popolazione, key=lambda ind: ind.fitness).copy()
    z_best = sol_best.fitness
    storia_best = [z_best]

    for generazione in range(1, max_gen + 1):
        popolazione_nuova = []
        while len(popolazione_nuova) < pop_size:
            genitore_1 = selezione_torneo(popolazione, tournament_k)
            genitore_2 = selezione_torneo(popolazione, tournament_k)

            if random.random() < p_cross:
                figlio_1, figlio_2 = crossover_bcrc(genitore_1, genitore_2, inst_f)
            else:
                figlio_1, figlio_2 = genitore_1.copy(), genitore_2.copy()

            if random.random() < p_lns:
                figlio_1 = applica_lns(inst_f, figlio_1, distruzione_pct=distruzione_pct, risk_aware=risk_aware)
            if random.random() < p_lns:
                figlio_2 = applica_lns(inst_f, figlio_2, distruzione_pct=distruzione_pct, risk_aware=risk_aware)

            figlio_1.fitness = calcola_profitto_robusto(inst_f, figlio_1)
            figlio_2.fitness = calcola_profitto_robusto(inst_f, figlio_2)

            popolazione_nuova.append(figlio_1)
            if len(popolazione_nuova) < pop_size:
                popolazione_nuova.append(figlio_2)

        pop_ordinata = sorted(popolazione, key=lambda ind: ind.fitness, reverse=True)
        elite = [ind.copy() for ind in pop_ordinata[:n_elite]]
        nuova_ordinata = sorted(popolazione_nuova, key=lambda ind: ind.fitness, reverse=True)
        popolazione = elite + nuova_ordinata[: pop_size - n_elite]

        candidato_best = max(popolazione, key=lambda ind: ind.fitness)
        if candidato_best.fitness > z_best:
            sol_best = candidato_best.copy()
            z_best = candidato_best.fitness
        storia_best.append(z_best)

    return sol_best, z_best, storia_best

# ==============================================================================
# 5. WORKER MULTIPROCESSING
# ==============================================================================

def esegui_singola_run(task_info: dict) -> dict:
    nome_parametro = task_info["nome_parametro"]
    valore = task_info["valore"]
    inst_nome = task_info["inst_nome"]
    inst_obj = task_info["inst_obj"]
    bks_val = task_info["bks_val"]
    seed = task_info["seed"]
    params = task_info["params"]

    random.seed(seed)
    np.random.seed(seed)

    t0 = time.time()
    sol, z, _ = hybrid_genetic_lns_top_solver(inst_obj, verbose=False, **params)
    tempo_run = time.time() - t0

    gap_pct = (
        ((bks_val - z) / bks_val * 100.0)
        if (not np.isnan(bks_val) and bks_val > 0)
        else np.nan
    )

    return {
        "Parametro": nome_parametro,
        "Valore": valore,
        "Istanza": inst_nome,
        "Replica": seed,
        "Profitto_Z": z,
        "BKS": bks_val,
        "Gap_Pct": gap_pct,
        "Tempo_CPU_s": tempo_run,
    }

# ==============================================================================
# 6. AMBIENTE DI ESECUZIONE PARALLELA
# ==============================================================================

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    OUTPUT_DIR = BASE_DIR / "results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INSTANCES_DIR = BASE_DIR / "instances"
    DATA_DIR = BASE_DIR / "data"

    TARGET_SEEDS = list(range(100, 110))
    NUM_WORKERS = max(1, os.cpu_count() - 2)

    bks_csv_path = DATA_DIR / "toptw_bks_benchmark_reference.csv"
    BKS_MAP = {
        "c102.txt": 1810.0,
        "r201.txt": 1458.0,
        "rc102.txt": 1724.0,
        "pr01.txt": 657.0,
    }
    if bks_csv_path.exists():
        df_bks = pd.read_csv(bks_csv_path)
        df_bks.columns = df_bks.columns.str.strip()
        col_inst = [c for c in df_bks.columns if "instance" in c.lower()][0]
        col_score = [c for c in df_bks.columns if "bks" in c.lower() or "optimal" in c.lower()][0]
        for _, row in df_bks.iterrows():
            nome = str(row[col_inst]).strip()
            if nome in BKS_MAP:
                BKS_MAP[nome] = float(row[col_score])

    istanze_tuning_dict = {}
    for nome_inst, bks_val in BKS_MAP.items():
        file_p = INSTANCES_DIR / nome_inst
        if file_p.exists():
            inst_obj = parse_toptw_instance(str(file_p), Gamma=0.0)
            istanze_tuning_dict[nome_inst] = {"obj": inst_obj, "bks": bks_val}
        else:
            print(f"File non trovato: {file_p}")

    BASELINE = {
        "pop_size": 30,
        "max_gen": 100,
        "p_cross": 0.8,
        "p_lns": 0.3,
        "distruzione_pct": 0.20,
        "n_elite": 2,
        "tournament_k": 3,
        "rcl_size": 3,
    }

    RANGES = {
        "pop_size": [20, 40, 60],
        "max_gen": [60, 100, 150],
        "p_cross": [0.0, 0.5, 0.7, 0.9],
        "p_lns": [0.0, 0.2, 0.3, 0.5],
        "distruzione_pct": [0.10, 0.20, 0.30],
        "n_elite": [1, 2, 4],
        "tournament_k": [2, 3, 5],
        "rcl_size": [1, 3, 5],
    }

    file_dettaglio = OUTPUT_DIR / "dettaglio_calibrazione_4_istanze_3.csv"
    file_riassunto = OUTPUT_DIR / "riassunto_calibrazione_ottimale_3.csv"

    df_storico = pd.read_csv(file_dettaglio) if file_dettaglio.exists() else pd.DataFrame()
    print(f"Storico trovato: {len(df_storico)} run salvate. Processi paralleli: {NUM_WORKERS}")

    tasks = []
    for nome_parametro, valori in RANGES.items():
        for valore in valori:
            params_test = dict(BASELINE)
            params_test[nome_parametro] = valore
            for inst_nome, inst_info in istanze_tuning_dict.items():
                for seed in TARGET_SEEDS:
                    if not df_storico.empty:
                        gia_fatto = not df_storico[
                            (df_storico["Parametro"] == nome_parametro)
                            & (df_storico["Valore"] == valore)
                            & (df_storico["Istanza"] == inst_nome)
                            & (df_storico["Replica"] == seed)
                        ].empty
                        if gia_fatto:
                            continue

                    tasks.append({
                        "nome_parametro": nome_parametro,
                        "valore": valore,
                        "inst_nome": inst_nome,
                        "inst_obj": inst_info["obj"],
                        "bks_val": inst_info["bks"],
                        "seed": seed,
                        "params": params_test,
                    })

    print(f"Task totali rimanenti da eseguire: {len(tasks)}\n")

    t_start = time.time()
    completati = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futuri = {executor.submit(esegui_singola_run, t): t for t in tasks}

        for future in concurrent.futures.as_completed(futuri):
            res = future.result()
            completati += 1
            print(
                f"[{completati}/{len(tasks)}] Completato: {res['Parametro']} = {res['Valore']} | "
                f"{res['Istanza']} | Seed: {res['Replica']} | Gap: {res['Gap_Pct']:.2f}% | Tempo: {res['Tempo_CPU_s']:.1f}s"
            )

            nuova_riga = pd.DataFrame([res])
            if file_dettaglio.exists():
                nuova_riga.to_csv(file_dettaglio, mode="a", header=False, index=False)
            else:
                nuova_riga.to_csv(file_dettaglio, mode="w", header=True, index=False)

    df_completo = pd.read_csv(file_dettaglio).drop_duplicates(
        subset=["Parametro", "Valore", "Istanza", "Replica"]
    )
    df_completo.to_csv(file_dettaglio, index=False)

    riassunto_list = []
    for nome_parametro in RANGES.keys():
        df_p = df_completo[df_completo["Parametro"] == nome_parametro]
        if df_p.empty:
            continue

        grouped = (
            df_p.groupby("Valore")
            .agg(
                N_Repliche=("Replica", "count"),
                Gap_Medio=("Gap_Pct", "mean"),
                Gap_Std=("Gap_Pct", "std"),
                Tempo_Medio=("Tempo_CPU_s", "mean"),
            )
            .reset_index()
        )

        pivot_istanze = df_p.pivot_table(
            index="Valore", columns="Istanza", values="Gap_Pct", aggfunc="mean"
        ).reset_index()
        merged = pd.merge(grouped, pivot_istanze, on="Valore")

        for _, row in merged.iterrows():
            riassunto_list.append({
                "Parametro": nome_parametro,
                "Valore_Testato": row["Valore"],
                "N_Run_Totali": int(row["N_Repliche"]),
                "Gap_Medio_%": round(row["Gap_Medio"], 2),
                "Gap_Std_%": round(row["Gap_Std"], 2) if not np.isnan(row["Gap_Std"]) else 0.0,
                "Tempo_Medio_s": round(row["Tempo_Medio"], 2),
                "Gap_c102_%": round(row.get("c102.txt", np.nan), 2),
                "Gap_r201_%": round(row.get("r201.txt", np.nan), 2),
                "Gap_rc102_%": round(row.get("rc102.txt", np.nan), 2),
                "Gap_pr01_%": round(row.get("pr01.txt", np.nan), 2),
            })

    df_riassunto = pd.DataFrame(riassunto_list)
    df_riassunto.to_csv(file_riassunto, index=False)

    print(f"\n✓ Tabella di riepilogo salvata in: {file_riassunto}")
    print(f"Tutti i task completati in {(time.time() - t_start)/60:.2f} minuti.")