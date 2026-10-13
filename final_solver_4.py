"""
================================================================================
TESI MAGISTRALE IN INGEGNERIA GESTIONALE
PIPELINE UNIFICATA PARALLELA: GA-LNS OTTIMIZZATO (v4 - DEPOSITO AL CENTROIDE)
CASO DI STUDIO INDUSTRIALE: "BRIDGE LOGISTICA"
================================================================================
"""

import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Supporto JIT Numba con fallback
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# ==============================================================================
# 1. STRUTTURE DATI
# ==============================================================================

@dataclass
class IstanzaReale:
    nome: str
    n_tot: int
    n_clienti: int
    K: int
    T_max: float
    coords: np.ndarray        # Matrice (n_tot, 2) [lat, lon]
    p: np.ndarray             # Vettore ricavi nominali
    delta: np.ndarray         # Vettore incertezza (50% di p)
    e: np.ndarray             # Finestra temporale apertura (minuti da t=0)
    l: np.ndarray             # Finestra temporale chiusura (minuti da t=0)
    s: np.ndarray             # Tempi di servizio (minuti)
    N_prio: Set[int]          # Clienti prioritari / mandatory
    N_opt: Set[int]           # Clienti facoltativi
    t: np.ndarray             # Matrice tempi di percorrenza (minuti)
    c: np.ndarray             # Matrice costi di trasporto (€)
    somma_profitti: float
    penale: float
    depot_original_id: int    # ID originario nel file JSON del nodo scelto come deposito
    depot_city: str           # Città del deposito
    depot_country: str        # Nazione del deposito

    def copy(self):
        return IstanzaReale(
            nome=self.nome,
            n_tot=self.n_tot,
            n_clienti=self.n_clienti,
            K=self.K,
            T_max=self.T_max,
            coords=self.coords,
            p=self.p,
            delta=self.delta,
            e=self.e,
            l=self.l,
            s=self.s,
            N_prio=set(self.N_prio),
            N_opt=set(self.N_opt),
            t=self.t,
            c=self.c,
            somma_profitti=self.somma_profitti,
            penale=self.penale,
            depot_original_id=self.depot_original_id,
            depot_city=self.depot_city,
            depot_country=self.depot_country,
        )


@dataclass
class Individuo:
    routes: List[List[int]]
    fitness: Optional[float] = None

    def copy(self):
        return Individuo(routes=[r[:] for r in self.routes], fitness=self.fitness)


# ==============================================================================
# 2. FUNZIONI NATIVE C COMPILATE (JIT NUMBA FAST)
# ==============================================================================

@njit(fastmath=True)
def simula_rotta_fast(route: np.ndarray, t_mat: np.ndarray, e_arr: np.ndarray, l_arr: np.ndarray, s_arr: np.ndarray, T_max_val: float):
    if len(route) == 0:
        return True, 0.0
    curr_time = 0.0
    prev = 0
    for i in range(len(route)):
        curr = route[i]
        curr_time += t_mat[prev, curr]
        if curr_time > l_arr[curr] + 1e-9:
            return False, -1.0
        if curr_time < e_arr[curr]:
            curr_time = e_arr[curr]
        curr_time += s_arr[curr]
        prev = curr
    curr_time += t_mat[prev, 0]
    if curr_time > T_max_val + 1e-9:
        return False, -1.0
    return True, curr_time


@njit(fastmath=True)
def valuta_inserimento_rotta_pos_fast(
    route: np.ndarray,
    nodo: int,
    pos: int,
    t_mat: np.ndarray,
    e_arr: np.ndarray,
    l_arr: np.ndarray,
    s_arr: np.ndarray,
    T_max_val: float,
):
    n_len = len(route) + 1
    curr_time = 0.0
    prev = 0
    for idx in range(n_len):
        if idx < pos:
            curr = route[idx]
        elif idx == pos:
            curr = nodo
        else:
            curr = route[idx - 1]

        curr_time += t_mat[prev, curr]
        if curr_time > l_arr[curr] + 1e-9:
            return False, -1.0
        if curr_time < e_arr[curr]:
            curr_time = e_arr[curr]
        curr_time += s_arr[curr]
        prev = curr

    curr_time += t_mat[prev, 0]
    if curr_time > T_max_val + 1e-9:
        return False, -1.0
    return True, curr_time


@njit(fastmath=True)
def trova_miglior_inserimento_rotta_fast(
    route: np.ndarray,
    nodo: int,
    t_mat: np.ndarray,
    e_arr: np.ndarray,
    l_arr: np.ndarray,
    s_arr: np.ndarray,
    T_max_val: float,
):
    ok_prima, tempo_prima = simula_rotta_fast(route, t_mat, e_arr, l_arr, s_arr, T_max_val)
    if not ok_prima:
        return -1, 1e18, -1.0

    best_pos = -1
    best_delta = 1e18
    best_tempo_dopo = -1.0
    n_pos = len(route) + 1

    for j in range(n_pos):
        ok_dopo, tempo_dopo = valuta_inserimento_rotta_pos_fast(
            route, nodo, j, t_mat, e_arr, l_arr, s_arr, T_max_val
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
# 3. CARICAMENTO CON SELEZIONE DEL DEPOSITO SUL CENTROIDE
# ==============================================================================

def carica_istanza_reale_json(
    percorso_file: Path,
    velocita_kmh: float = 70.0,
    detour_factor: float = 1.25,
    max_richieste: Optional[int] = None
) -> IstanzaReale:
    with open(percorso_file, "r", encoding="utf-8") as f:
        dati = json.load(f)

    t_start = datetime.fromisoformat(dati["planning_start"])
    t_end = datetime.fromisoformat(dati["planning_end"])
    T_max = round((t_end - t_start).total_seconds() / 60.0, 2)

    K_val = len(dati.get("trucks", []))
    if K_val == 0:
        K_val = 9

    tutte_richieste = dati.get("requests", [])
    if max_richieste is not None:
        tutte_richieste = tutte_richieste[:max_richieste]

    # 1. Calcolo del centroide geografico di tutte le posizioni di pickup
    raw_coords = np.array([
        [r["pickup"]["address"]["lat"], r["pickup"]["address"]["lon"]]
        for r in tutte_richieste
    ], dtype=float)

    centroid_lat = float(np.mean(raw_coords[:, 0]))
    centroid_lon = float(np.mean(raw_coords[:, 1]))

    # Distanza Haversine di ciascun nodo dal centroide
    R_earth = 6371.0
    c_lat_rad, c_lon_rad = np.radians(centroid_lat), np.radians(centroid_lon)
    n_lats, n_lons = np.radians(raw_coords[:, 0]), np.radians(raw_coords[:, 1])
    dlat = n_lats - c_lat_rad
    dlon = n_lons - c_lon_rad
    a_cent = np.sin(dlat / 2.0)**2 + np.cos(c_lat_rad) * np.cos(n_lats) * np.sin(dlon / 2.0)**2
    dist_from_centroid = 2.0 * R_earth * np.arctan2(np.sqrt(a_cent), np.sqrt(1.0 - a_cent))

    # Identifica il nodo reale più vicino al centroide da promuovere a Deposito (indice 0)
    depot_idx = int(np.argmin(dist_from_centroid))
    depot_req = tutte_richieste[depot_idx]
    customer_reqs = [r for idx, r in enumerate(tutte_richieste) if idx != depot_idx]

    depot_addr = depot_req["pickup"]["address"]
    dep_city = depot_addr.get("city", "N/D")
    dep_country = depot_addr.get("country", "N/D")
    dep_id = depot_req["id"]

    print(f"-> Centroide geografico calcolato: Lat {centroid_lat:.3f}°N, Lon {centroid_lon:.3f}°E")
    print(f"-> Nodo reale più vicino eletto a DEPOSITO (0): ID {dep_id} ({dep_city}, {dep_country})")
    print(f"   Distanza dal centroide teorico: {dist_from_centroid[depot_idx]:.1f} km")

    n_clienti = len(customer_reqs)
    n_tot = n_clienti + 1

    coords = np.zeros((n_tot, 2), dtype=float)
    p = np.zeros(n_tot, dtype=float)
    delta = np.zeros(n_tot, dtype=float)
    e = np.zeros(n_tot, dtype=float)
    l = np.zeros(n_tot, dtype=float)
    s = np.zeros(n_tot, dtype=float)
    N_prio = set()
    N_opt = set()

    coords[0] = [depot_addr["lat"], depot_addr["lon"]]
    p[0], delta[0], e[0], l[0], s[0] = 0.0, 0.0, 0.0, T_max, 0.0

    for idx, r in enumerate(customer_reqs, start=1):
        pk = r["pickup"]
        addr = pk["address"]
        coords[idx] = [addr["lat"], addr["lon"]]
        profit = float(r.get("revenue", 0.0))
        p[idx] = profit
        delta[idx] = 0.50 * profit

        dt_start = datetime.fromisoformat(pk["time_window_start"])
        dt_end = datetime.fromisoformat(pk["time_window_end"])
        e[idx] = max(0.0, (dt_start - t_start).total_seconds() / 60.0)
        l[idx] = min(T_max, (dt_end - t_start).total_seconds() / 60.0)
        s[idx] = float(pk.get("service_time_h", 0.0)) * 60.0

        if r.get("mandatory", False):
            N_prio.add(idx)
        else:
            N_opt.add(idx)

    # Matrice distanze reali con detour factor
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2.0)**2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    dist_km = R_earth * c * detour_factor

    t_mat = (dist_km / velocita_kmh) * 60.0

    trucks_data = dati.get("trucks", [])
    all_rates = [rate[2] for t in trucks_data for rate in t.get("cost_rates", []) if len(rate) >= 3]
    costo_km_eff = float(np.mean(all_rates)) if all_rates else 1.291
    c_mat = costo_km_eff * dist_km

    somma_profitti = float(np.sum(p))
    penale = -(somma_profitti + 1.0)

    print(f"-> Costo chilometrico medio flotta: {costo_km_eff:.3f} €/km")

    return IstanzaReale(
        nome=percorso_file.name,
        n_tot=n_tot,
        n_clienti=n_clienti,
        K=K_val,
        T_max=T_max,
        coords=coords,
        p=p,
        delta=delta,
        e=e,
        l=l,
        s=s,
        N_prio=N_prio,
        N_opt=N_opt,
        t=t_mat,
        c=c_mat,
        somma_profitti=somma_profitti,
        penale=penale,
        depot_original_id=dep_id,
        depot_city=dep_city,
        depot_country=dep_country,
    )


def preprocessing_graph(inst: IstanzaReale) -> IstanzaReale:
    """Rimuove solo i nodi opzionali fisicamente irraggiungibili entro la time window o T_max."""
    inst2 = inst.copy()
    da_rimuovere = set()
    for i in list(inst2.N_opt):
        r_arr = np.array([i], dtype=np.int64)
        fattibile, _ = simula_rotta_fast(r_arr, inst2.t, inst2.e, inst2.l, inst2.s, inst2.T_max)
        if not fattibile:
            da_rimuovere.add(i)
    inst2.N_opt = inst2.N_opt - da_rimuovere
    return inst2


# ==============================================================================
# 4. MOTORE GA-LNS OTTIMIZZATO (v3)
# ==============================================================================

def calcola_profitto_robusto(inst: IstanzaReale, ind: Individuo, gamma_val: float) -> float:
    nodi_serviti = []
    costo_tot = 0.0
    for r in ind.routes:
        if not r:
            continue
        r_arr = np.array(r, dtype=np.int64)
        fattibile, _ = simula_rotta_fast(r_arr, inst.t, inst.e, inst.l, inst.s, inst.T_max)
        if not fattibile:
            return inst.penale
        costo_tot += calcola_costo_rotta_fast(r_arr, inst.c)
        nodi_serviti.extend(r)

    if len(nodi_serviti) != len(set(nodi_serviti)):
        return inst.penale

    if not nodi_serviti:
        return 0.0

    ricavo_nom = float(np.sum(inst.p[nodi_serviti]))
    deviazioni = inst.delta[nodi_serviti]
    protezione = protez_bertsimas_fast(deviazioni, gamma_val)
    return ricavo_nom - protezione - costo_tot


def calcola_gain_differenziale(nodo: int, delta_t: float, prot_prima: float, dev_attuali: np.ndarray, inst: IstanzaReale, gamma_val: float, risk_aware: bool) -> float:
    if not risk_aware or gamma_val <= 0.0:
        return (inst.p[nodo] ** 2) / delta_t
    prot_dopo = protez_bertsimas_fast(np.append(dev_attuali, inst.delta[nodo]), gamma_val)
    p_netto = inst.p[nodo] - (prot_dopo - prot_prima)
    segno = 1.0 if p_netto >= 0 else -1.0
    return segno * (p_netto ** 2) / delta_t


def inserisci_nodo_migliore_posizione(routes: List[List[int]], nodo: int, inst: IstanzaReale) -> bool:
    best_delta = 1e18
    best_route_idx, best_pos = None, None

    for k, route in enumerate(routes):
        r_arr = np.array(route, dtype=np.int64)
        pos, delta_t, _ = trova_miglior_inserimento_rotta_fast(
            r_arr, nodo, inst.t, inst.e, inst.l, inst.s, inst.T_max
        )
        if pos != -1 and delta_t < best_delta:
            best_delta = delta_t
            best_route_idx = k
            best_pos = pos

    if best_route_idx is None:
        return False
    routes[best_route_idx].insert(best_pos, nodo)
    return True


def inizializza_individuo(inst: IstanzaReale, gamma_val: float, risk_aware: bool, rcl_size: int = 3) -> Individuo:
    routes = [[] for _ in range(inst.K)]
    
    prio = list(inst.N_prio)
    random.shuffle(prio)
    for nodo in prio:
        inserisci_nodo_migliore_posizione(routes, nodo, inst)

    candidati = list(inst.N_opt)
    random.shuffle(candidati)

    while candidati:
        nodi_attuali = [i for r in routes for i in r]
        dev_attuali = inst.delta[nodi_attuali]
        prot_prima = protez_bertsimas_fast(dev_attuali, gamma_val) if (risk_aware and gamma_val > 0) else 0.0

        valutazioni = []
        r_arrays = [np.array(r, dtype=np.int64) for r in routes]

        for nodo in candidati:
            for k in range(len(routes)):
                pos, delta_t, _ = trova_miglior_inserimento_rotta_fast(
                    r_arrays[k], nodo, inst.t, inst.e, inst.l, inst.s, inst.T_max
                )
                if pos != -1:
                    dt = max(delta_t, 1e-6)
                    gain = calcola_gain_differenziale(nodo, dt, prot_prima, dev_attuali, inst, gamma_val, risk_aware)
                    valutazioni.append((gain, nodo, k, pos))

        if not valutazioni:
            break
        valutazioni.sort(key=lambda x: x[0], reverse=True)
        if risk_aware and valutazioni[0][0] <= 0:
            break
        rcl = valutazioni[: min(rcl_size, len(valutazioni))]
        _, nodo_scelto, k_scelto, pos_scelta = random.choice(rcl)
        routes[k_scelto].insert(pos_scelta, nodo_scelto)
        candidati.remove(nodo_scelto)

    return Individuo(routes=routes)


def selezione_torneo(popolazione: List[Individuo], k: int = 2) -> Individuo:
    candidati = random.sample(popolazione, min(k, len(popolazione)))
    return max(candidati, key=lambda ind: ind.fitness).copy()


def _bcrc_single(parent_donor: Individuo, parent_base: Individuo, inst: IstanzaReale) -> Individuo:
    rotte_attive = [r for r in parent_donor.routes if r]
    if not rotte_attive:
        return parent_base.copy()
    rotta_donatrice = random.choice(rotte_attive)
    da_reinserire = rotta_donatrice[:]
    set_ins = set(da_reinserire)
    nuove_routes = [[n for n in r if n not in set_ins] for r in parent_base.routes]
    figlio = Individuo(routes=nuove_routes)
    random.shuffle(da_reinserire)
    for nodo in da_reinserire:
        inserisci_nodo_migliore_posizione(figlio.routes, nodo, inst)
    return figlio


def crossover_bcrc(p1: Individuo, p2: Individuo, inst: IstanzaReale) -> Tuple[Individuo, Individuo]:
    return _bcrc_single(p1, p2, inst), _bcrc_single(p2, p1, inst)


def distruggi_rotte_casuale(ind: Individuo, n_rimuovere: int) -> Tuple[List[int], Individuo]:
    tutti = [(k, n) for k, r in enumerate(ind.routes) for n in r]
    if not tutti:
        return [], ind.copy()
    n_rim = min(n_rimuovere, len(tutti))
    scelti = random.sample(tutti, n_rim)
    ind_parziale = ind.copy()
    rimossi = []
    for k, n in scelti:
        if n in ind_parziale.routes[k]:
            ind_parziale.routes[k].remove(n)
            rimossi.append(n)
    return rimossi, ind_parziale


def applica_lns(inst: IstanzaReale, ind: Individuo, gamma_val: float, risk_aware: bool, distruzione_pct: float = 0.20) -> Individuo:
    tutti_nodi = sum(len(r) for r in ind.routes)
    n_rim = max(1, int(round(distruzione_pct * tutti_nodi))) if tutti_nodi > 0 else 0
    nodi_rimossi, ind_p = distruggi_rotte_casuale(ind, n_rim)

    serviti = set(n for r in ind_p.routes for n in r)
    candidati = list(set(nodi_rimossi) | ((inst.N_prio | inst.N_opt) - serviti))
    prio_cands = [n for n in candidati if n in inst.N_prio]
    random.shuffle(prio_cands)

    for m in prio_cands:
        if inserisci_nodo_migliore_posizione(ind_p.routes, m, inst):
            candidati.remove(m)

    ancora = True
    while ancora and candidati:
        nodi_att = [i for r in ind_p.routes for i in r]
        dev_att = inst.delta[nodi_att]
        prot_prima = protez_bertsimas_fast(dev_att, gamma_val) if (risk_aware and gamma_val > 0) else 0.0

        miglior_gain = -1e18
        miglior_nodo, miglior_k, miglior_pos = None, None, None
        r_arrays = [np.array(r, dtype=np.int64) for r in ind_p.routes]

        for nodo in candidati:
            for k in range(len(ind_p.routes)):
                pos, delta_t, _ = trova_miglior_inserimento_rotta_fast(
                    r_arrays[k], nodo, inst.t, inst.e, inst.l, inst.s, inst.T_max
                )
                if pos != -1:
                    dt = max(delta_t, 1e-6)
                    gain = calcola_gain_differenziale(nodo, dt, prot_prima, dev_att, inst, gamma_val, risk_aware)
                    if gain > miglior_gain:
                        miglior_gain = gain
                        miglior_nodo = nodo
                        miglior_k = k
                        miglior_pos = pos

        if miglior_nodo is not None and not (risk_aware and miglior_gain <= 0):
            ind_p.routes[miglior_k].insert(miglior_pos, miglior_nodo)
            candidati.remove(miglior_nodo)
        else:
            ancora = False

    return ind_p


def risolvi_istanza_ga_lns(
    inst: IstanzaReale,
    gamma_val: float,
    risk_aware: bool,
    pop_size: int = 40,
    max_gen: int = 150,
    p_cross: float = 0.90,
    p_lns: float = 0.30,
    distruzione_pct: float = 0.20,
    n_elite: int = 2,
    rcl_size: int = 3,
    seed: int = 42,
) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    t0 = time.perf_counter()

    inst_f = preprocessing_graph(inst)

    popolazione = [
        inizializza_individuo(inst_f, gamma_val, risk_aware=risk_aware, rcl_size=rcl_size)
        for _ in range(pop_size)
    ]
    for ind in popolazione:
        ind.fitness = calcola_profitto_robusto(inst_f, ind, gamma_val)

    sol_best = max(popolazione, key=lambda ind: ind.fitness).copy()
    z_best = sol_best.fitness

    for gen in range(1, max_gen + 1):
        pop_nuova = []
        while len(pop_nuova) < pop_size:
            p1 = selezione_torneo(popolazione)
            p2 = selezione_torneo(popolazione)

            if random.random() < p_cross:
                f1, f2 = crossover_bcrc(p1, p2, inst_f)
            else:
                f1, f2 = p1.copy(), p2.copy()

            if random.random() < p_lns:
                f1 = applica_lns(inst_f, f1, gamma_val, risk_aware, distruzione_pct)
            if random.random() < p_lns:
                f2 = applica_lns(inst_f, f2, gamma_val, risk_aware, distruzione_pct)

            f1.fitness = calcola_profitto_robusto(inst_f, f1, gamma_val)
            f2.fitness = calcola_profitto_robusto(inst_f, f2, gamma_val)

            pop_nuova.append(f1)
            if len(pop_nuova) < pop_size:
                pop_nuova.append(f2)

        pop_ord = sorted(popolazione, key=lambda ind: ind.fitness, reverse=True)
        elite = [ind.copy() for ind in pop_ord[:n_elite]]
        nuova_ord = sorted(pop_nuova, key=lambda ind: ind.fitness, reverse=True)
        popolazione = elite + nuova_ord[: pop_size - n_elite]

        cand_best = max(popolazione, key=lambda ind: ind.fitness)
        if cand_best.fitness > z_best:
            sol_best = cand_best.copy()
            z_best = cand_best.fitness

    runtime = time.perf_counter() - t0

    rotte_attive = [r for r in sol_best.routes if r]
    serviti = sorted([n for r in rotte_attive for n in r])
    non_serviti = sorted(list(set(range(1, inst.n_tot)) - set(serviti)))

    ricavo_lordo = float(np.sum(inst.p[serviti])) if serviti else 0.0
    costo_tot = sum(calcola_costo_rotta_fast(np.array(r, dtype=np.int64), inst.c) for r in rotte_attive)

    return {
        "obj_val": round(z_best, 2),
        "ricavo_lordo": round(ricavo_lordo, 2),
        "costo_trasporto": round(costo_tot, 2),
        "rotte": rotte_attive,
        "nodi_serviti": serviti,
        "nodi_non_serviti": non_serviti,
        "runtime": round(runtime, 2),
    }


def worker_ottimizzazione_parallela(args):
    """Funzione worker per multiprocessore concurrent.futures."""
    inst, nome_cfg, gamma_val, risk_flag, seed = args
    res = risolvi_istanza_ga_lns(inst, gamma_val=gamma_val, risk_aware=risk_flag, seed=seed)
    return nome_cfg, res


# ==============================================================================
# 5. GRAFICI E VISUALIZZAZIONE DELLE ROTTE
# ==============================================================================

def genera_grafico_rotte(
    inst: IstanzaReale,
    rotte: List[List[int]],
    out_path: Path,
    titolo: str,
    sottotitolo: str = ""
):
    """
    Genera un'immagine ad alta risoluzione con la mappa dei percorsi:
    - Deposito centrale (nodo 0): quadrato nero ('s')
    - Clienti prioritari: stella ('*')
    - Clienti non prioritari: cerchio ('o')
    - Nodi non serviti: grigio chiaro sullo sfondo
    - Rotte e nodi serviti: colorati per ciascun veicolo attivo
    """
    fig, ax = plt.subplots(figsize=(11, 9), dpi=300)
    cmap = plt.get_cmap("tab10")

    # X: Longitudine, Y: Latitudine
    x_coords = inst.coords[:, 1]
    y_coords = inst.coords[:, 0]

    serviti = set(n for r in rotte for n in r)
    non_serviti = set(range(1, inst.n_tot)) - serviti

    # 1. Nodi non serviti in grigio
    if non_serviti:
        non_serviti_prio = [n for n in non_serviti if n in inst.N_prio]
        non_serviti_opt = [n for n in non_serviti if n in inst.N_opt]

        if non_serviti_opt:
            ax.scatter(
                x_coords[non_serviti_opt],
                y_coords[non_serviti_opt],
                c="#d3d3d3",
                marker="o",
                s=40,
                alpha=0.7,
                edgecolors="none",
                zorder=2,
                label=f"Non serviti (Opzionali: {len(non_serviti_opt)})"
            )
        if non_serviti_prio:
            ax.scatter(
                x_coords[non_serviti_prio],
                y_coords[non_serviti_prio],
                c="#b0b0b0",
                marker="*",
                s=80,
                alpha=0.8,
                edgecolors="none",
                zorder=3,
                label=f"Non serviti (Prioritari: {len(non_serviti_prio)})"
            )

    # 2. Rotte e nodi serviti
    for k, r in enumerate(rotte):
        if not r:
            continue
        colore = cmap(k % 10)
        path = [0] + r + [0]
        xs = x_coords[path]
        ys = y_coords[path]

        ax.plot(
            xs,
            ys,
            color=colore,
            linewidth=2.0,
            alpha=0.85,
            linestyle="-",
            zorder=4,
            label=f"Veicolo {k+1} ({len(r)} stop)"
        )

        prio_r = [n for n in r if n in inst.N_prio]
        opt_r = [n for n in r if n in inst.N_opt]

        if opt_r:
            ax.scatter(
                x_coords[opt_r],
                y_coords[opt_r],
                color=colore,
                marker="o",
                s=65,
                edgecolors="black",
                linewidths=0.6,
                zorder=5
            )
        if prio_r:
            ax.scatter(
                x_coords[prio_r],
                y_coords[prio_r],
                color=colore,
                marker="*",
                s=160,
                edgecolors="black",
                linewidths=0.8,
                zorder=6
            )

    # 3. Deposito Centrale (nodo 0)
    label_dep = f"Deposito (0: {inst.depot_city}, {inst.depot_country})"
    ax.scatter(
        x_coords[0],
        y_coords[0],
        c="black",
        marker="s",
        s=180,
        edgecolors="white",
        linewidths=1.2,
        zorder=7,
        label=label_dep
    )

    ax.annotate(
        f"Dep. 0 ({inst.depot_city})",
        (x_coords[0], y_coords[0]),
        textcoords="offset points",
        xytext=(6, 6),
        fontsize=8.5,
        fontweight="bold",
        color="black"
    )

    titolo_completo = f"{titolo}\n{sottotitolo}" if sottotitolo else titolo
    ax.set_title(titolo_completo, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Longitudine (°E)", fontsize=10)
    ax.set_ylabel("Latitudine (°N)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35, zorder=1)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8.5,
        frameon=True,
        shadow=True
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Immagine della soluzione salvata in: {out_path}")


# ==============================================================================
# 6. STRESS-TEST MONTE CARLO (v3)
# ==============================================================================

def calcola_mon_consuntivo(rotte: List[List[int]], inst: IstanzaReale, colpiti_set: Set[int]) -> float:
    costo_tot = 0.0
    serviti = []
    for r in rotte:
        if not r:
            continue
        r_arr = np.array(r, dtype=np.int64)
        fattibile, _ = simula_rotta_fast(r_arr, inst.t, inst.e, inst.l, inst.s, inst.T_max)
        if not fattibile:
            return inst.penale
        costo_tot += calcola_costo_rotta_fast(r_arr, inst.c)
        serviti.extend(r)

    if not serviti:
        return 0.0

    ricavo_reale = sum((inst.p[i] - inst.delta[i]) if i in colpiti_set else inst.p[i] for i in serviti)
    return ricavo_reale - costo_tot


def esegui_stress_test_monte_carlo(inst: IstanzaReale, piani: Dict[str, List[List[int]]], z_nom: float, n_sim: int = 1000):
    omega_levels = [0.25, 0.50, 0.75, 1.00]
    alpha_levels = [0.40, 0.50, 0.60, 0.70]
    clienti = list(range(1, inst.n_tot))

    dettaglio_rows = []
    riassunto_rows = []

    print("\n" + "=" * 115)
    print(f"AVVIO STRESS-TEST MONTE CARLO (N_sim = {n_sim}, N_clienti = {inst.n_clienti})")
    print(f"Margine Nominale di Riferimento Z_nom: {z_nom:.2f} €")
    print("=" * 115)

    for omega in omega_levels:
        k_colpiti = int(round(omega * inst.n_clienti))
        repliche = 1 if omega == 1.00 else n_sim
        print(f"-> Simulazione Omega = {int(omega*100)}% ({k_colpiti}/{inst.n_clienti} nodi perturbati) | Repliche: {repliche}...")

        mon_scenari = {nome: [] for nome in piani}
        for _ in range(repliche):
            colpiti = set(clienti) if omega == 1.00 else set(random.sample(clienti, k_colpiti))
            for nome, r in piani.items():
                mon_scenari[nome].append(calcola_mon_consuntivo(r, inst, colpiti))

        arr_nom = np.array(mon_scenari["Nominale (Γ=0%)"])

        for nome, lista_mon in mon_scenari.items():
            arr_mon = np.array(lista_mon)
            media_mon = float(np.mean(arr_mon))
            min_mon = float(np.min(arr_mon))
            std_mon = float(np.std(arr_mon)) if len(arr_mon) > 1 else 0.0

            if nome == "Nominale (Γ=0%)":
                wr_str = "—"
            else:
                wr_pct = (np.sum(arr_mon >= arr_nom) / len(arr_mon)) * 100.0
                wr_str = f"{wr_pct:.1f}%"

            riga = {
                "Istanza": inst.nome,
                "N_Clienti": inst.n_clienti,
                "Omega (%)": f"{int(omega*100)}%",
                "Configurazione": nome,
                "E[MON] (€)": round(media_mon, 2),
                "Min MON (€)": round(min_mon, 2),
                "Std MON (€)": round(std_mon, 2),
                "Win Rate vs Nom (%)": wr_str,
            }

            for alpha in alpha_levels:
                soglia = alpha * z_nom
                pct_viol = float(np.mean(arr_mon < soglia)) * 100.0
                riga[f"P(MON < {int(alpha*100)}%)"] = f"{pct_viol:.1f}%"

            dettaglio_rows.append(riga)

            if omega == 0.50:
                riassunto_rows.append({
                    "Istanza": inst.nome,
                    "N": inst.n_clienti,
                    "Configurazione": nome,
                    "E[MON] (€)": round(media_mon, 2),
                    "Min MON (€)": round(min_mon, 2),
                    "Win Rate vs Nom": wr_str,
                    "P(MON < 50%)": riga["P(MON < 50%)"],
                    "P(MON < 60%)": riga["P(MON < 60%)"],
                    "P(MON < 70%)": riga["P(MON < 70%)"],
                })

    return pd.DataFrame(dettaglio_rows), pd.DataFrame(riassunto_rows)


# ==============================================================================
# 7. ESECUZIONE PRINCIPALE MULTI-THREAD CON EXPORT TABELLE E GRAFICI
# ==============================================================================

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    INSTANCES_DIR = BASE_DIR / "instances_real"
    RESULTS_DIR = BASE_DIR / "results_real"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    NOME_FILE_CASO_REALE = "istanza_reale.json"
    PERCORSO_CASO_REALE = INSTANCES_DIR / NOME_FILE_CASO_REALE

    if not PERCORSO_CASO_REALE.exists():
        print(f"[ERRORE] File non trovato: {PERCORSO_CASO_REALE}")
        print("Posiziona 'istanza_reale.json' nella cartella 'instances_real'.")
        exit()

    print("\n" + "=" * 115)
    print(f"CARICAMENTO E SETUP ISTANZA: {NOME_FILE_CASO_REALE}")
    print("=" * 115)

    istanza = carica_istanza_reale_json(PERCORSO_CASO_REALE, velocita_kmh=70.0, detour_factor=1.25, max_richieste=None)

    print(f"-> Nodi Totali (con Deposito): {istanza.n_tot} (Clienti: {istanza.n_clienti})")
    print(f"-> Veicoli K: {istanza.K} | Orizzonte T_max: {istanza.T_max:.1f} min ({istanza.T_max/60.0:.1f} h)")
    print(f"-> Mandatari (|N_prio|): {len(istanza.N_prio)} | Facoltativi (|N_opt|): {len(istanza.N_opt)}")
    print(f"-> Somma Profitti Potenziali: {istanza.somma_profitti:.2f} € | Penale Infattibilità: {istanza.penale:.2f}")

    configurazioni = [
        ("Nominale (Γ=0%)", 0.0, False),
        ("Robusto (Γ=50%)", float(istanza.n_clienti * 0.50), True),
        ("Conservativo (Γ=100%)", float(istanza.n_clienti * 1.00), True),
    ]

    tasks = [
        (istanza, nome_cfg, gamma_val, risk_flag, 42)
        for nome_cfg, gamma_val, risk_flag in configurazioni
    ]

    risultati_ottimizzazione = {}
    piani_rotte = {}

    print("\n" + "-" * 100)
    print(f">>> AVVIO OTTIMIZZAZIONE PARALLELA SU 3 WORKERS (Γ = 0%, 50%, 100%)...")
    print("-" * 100)
    t_start_par = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        futures = executor.map(worker_ottimizzazione_parallela, tasks)
        for nome_cfg, res in futures:
            risultati_ottimizzazione[nome_cfg] = res
            piani_rotte[nome_cfg] = res["rotte"]
            print(f"✓ Risoluzione completata: {nome_cfg:22s} | Z: {res['obj_val']:9.2f} € | Tempo CPU: {res['runtime']:6.2f} s")

    tempo_tot_par = time.perf_counter() - t_start_par
    print(f"\n-> Ottimizzazione parallela globale completata in: {tempo_tot_par:.2f} s")

    # ==========================================================================
    # SALVATAGGIO DEI RISULTATI DELLA PIANIFICAZIONE IN FILE CSV E GRAFICI PNG
    # ==========================================================================
    piani_sintesi_rows = []
    rotte_dettaglio_rows = []

    file_tag_map = {
        "Nominale (Γ=0%)": "gamma_0",
        "Robusto (Γ=50%)": "gamma_50",
        "Conservativo (Γ=100%)": "gamma_100"
    }

    for nome_cfg, gamma_val, risk_flag in configurazioni:
        res = risultati_ottimizzazione[nome_cfg]
        
        piani_sintesi_rows.append({
            "Configurazione": nome_cfg,
            "Gamma_Val": gamma_val,
            "Valore_Z_euro": res["obj_val"],
            "Ricavo_Lordo_euro": res["ricavo_lordo"],
            "Costo_Trasporto_euro": res["costo_trasporto"],
            "N_Serviti": len(res["nodi_serviti"]),
            "N_Esclusi": len(res["nodi_non_serviti"]),
            "Lista_Serviti": str(res["nodi_serviti"]),
            "Lista_Esclusi": str(res["nodi_non_serviti"]),
            "Tempo_CPU_s": res["runtime"]
        })

        print("\n" + "-" * 100)
        print(f"Dettaglio Piani Logistici: {nome_cfg}")
        print(f"    Valore Z: {res['obj_val']:.2f} € | Ricavo Lordo: {res['ricavo_lordo']:.2f} € | Costo Trasporto: {res['costo_trasporto']:.2f} €")
        print(f"    Nodi Serviti ({len(res['nodi_serviti'])}/{istanza.n_clienti}): {res['nodi_serviti']}")
        print(f"    Nodi Esclusi ({len(res['nodi_non_serviti'])}): {res['nodi_non_serviti']}")
        print("    Instradamento Flotta Veicoli:")

        for idx_r, r in enumerate(res["rotte"], start=1):
            r_arr = np.array(r, dtype=np.int64)
            _, tempo_tot = simula_rotta_fast(r_arr, istanza.t, istanza.e, istanza.l, istanza.s, istanza.T_max)
            costo_r = calcola_costo_rotta_fast(r_arr, istanza.c)
            percorso_str = f"[0] -> {' -> '.join(map(str, r))} -> [0]"

            print(f"       * Veicolo {idx_r:2d} (Stop: {len(r):2d}, Tempo: {tempo_tot:6.1f}/{istanza.T_max:.0f} min, Costo: {costo_r:7.2f} €): {percorso_str}")

            rotte_dettaglio_rows.append({
                "Configurazione": nome_cfg,
                "Veicolo": idx_r,
                "N_Stop": len(r),
                "Tempo_min": round(tempo_tot, 2),
                "T_max_min": round(istanza.T_max, 2),
                "Costo_euro": round(costo_r, 2),
                "Percorso": percorso_str
            })

        nome_img = f"rotte_caso_reale_{file_tag_map[nome_cfg]}.png"
        path_img = RESULTS_DIR / nome_img
        sottotitolo_img = f"Z = {res['obj_val']:.2f} € | Ricavo = {res['ricavo_lordo']:.2f} € | Costo = {res['costo_trasporto']:.2f} € | Serviti = {len(res['nodi_serviti'])}/{istanza.n_clienti}"
        genera_grafico_rotte(
            inst=istanza,
            rotte=res["rotte"],
            out_path=path_img,
            titolo=f"Piani di Routing Bridge Logistica - {nome_cfg}",
            sottotitolo=sottotitolo_img
        )

    df_piani_sintesi = pd.DataFrame(piani_sintesi_rows)
    df_rotte_dettaglio = pd.DataFrame(rotte_dettaglio_rows)

    path_sintesi = RESULTS_DIR / "caso_reale_piani_sintesi.csv"
    path_rotte = RESULTS_DIR / "caso_reale_rotte_dettaglio.csv"

    df_piani_sintesi.to_csv(path_sintesi, index=False, sep=";")
    df_rotte_dettaglio.to_csv(path_rotte, index=False, sep=";")

    print("\n" + "=" * 115)
    print("TABELLE DEI PIANI LOGISTICI SALVATE CON SUCCESSO:")
    print(f"✓ Tabella Sintetica Piani salvata in: {path_sintesi}")
    print(f"✓ Tabella Dettaglio Rotte Veicoli salvata in: {path_rotte}")
    print("=" * 115)

    # ==========================================================================
    # STRESS-TEST MONTE CARLO
    # ==========================================================================
    z_nom_baseline = risultati_ottimizzazione["Nominale (Γ=0%)"]["obj_val"]
    df_dettaglio, df_riassunto = esegui_stress_test_monte_carlo(istanza, piani_rotte, z_nom=z_nom_baseline, n_sim=1000)

    path_dettaglio = RESULTS_DIR / "caso_reale_montecarlo_dettaglio.csv"
    path_riassunto = RESULTS_DIR / "caso_reale_montecarlo_riassunto.csv"

    df_dettaglio.to_csv(path_dettaglio, index=False, sep=";")
    df_riassunto.to_csv(path_riassunto, index=False, sep=";")

    print("\n" + "=" * 115)
    print("ESECUZIONE COMPLETATA CON SUCCESSO!")
    print(f"✓ Output Dettagliato Monte Carlo salvato in: {path_dettaglio}")
    print(f"✓ Output Riassuntivo Monte Carlo salvato in: {path_riassunto}")
    print("=" * 115)

    print("\nTABELLA DI DETTAGLIO MONTE CARLO:")
    print(df_dettaglio.drop(columns=['Istanza', 'N_Clienti']).to_string(index=False))