from dataclasses import dataclass
from typing import Callable, Optional
import sys


# Rappresenta un intervallo [x_lower, x_higher] con il suo punto caratteristico x̂ e il valore della minorata.
@dataclass
class Interval:
    x_lower: float
    x_higher: float
    x_cap: float       # punto caratteristico: argmin della minorata sull'intervallo
    f_cap: float       # valore della minorata in x_cap: F_s(x_cap)


# Snapshot completo di una singola iterazione: punti valutati, intervalli costruiti, intervallo scelto e gap corrente.
@dataclass
class Iteration:
    step: int
    eval_points: list[float]          # punti dove f è stata valutata (ordinati)
    eval_values: list[float]          # valori f(x) nei punti valutati
    intervals: list[Interval]         # tutti gli intervalli costruiti in questa iterazione
    chosen_interval: Interval         # intervallo selezionato (minorata minima)
    chosen_index: int                 # indice dell'intervallo selezionato
    new_x_cap: float                  # nuovo punto di valutazione
    new_fval: Optional[float]         # f(new_x_cap)
    zs_plus: float                    # Z⁺_s: miglior valore di f trovato (upper bound)
    rs: float                         # R_s: minima caratteristica globale (lower bound)
    gap: float                        # Z⁺_s − R_s: gap di ottimalità, criterio di arresto


# Risultati finali dell'algoritmo: ottimo trovato, garanzia di qualità e storia completa delle iterazioni.
@dataclass
class Result:
    x_star: float                      # punto ottimale trovato
    f_star: float                      # valore ottimale trovato (Z⁺_s finale)
    R_final: float                     # R_s finale: lower bound garantito
    gap: float                         # gap di ottimalità finale: f* − R_s
    n_iterations: int                  # numero di iterazioni eseguite
    n_evaluations: int                 # numero di valutazioni di f
    iterations: list[Iteration]        # storia completa delle iterazioni
    convergenza: bool                  # True se gap <= epsilon
    epsilon: float                     # soglia di convergenza usata


# Punto caratteristico dell'intervallo [xl, xr]: x̂ = (xl + xr)/2 + (f(xl) − f(xr)) / (2L)
def x_cap_formula(xl: float, xr: float, fxl: float, fxr: float, L: float) -> float:
    return (xl + xr) / 2.0 + (fxl - fxr) / (2.0 * L)


# Valore della minorata nel punto caratteristico: F_s(x̂) = (f(xl) + f(xr))/2 − L*(xr − xl)/2
def R_formula(xl: float, xr: float, fxl: float, fxr: float, L: float) -> float:
    return (fxl + fxr) / 2.0 - L * (xr - xl) / 2.0


# Costruisce la lista di Interval da una sequenza di punti già ordinati.
def costruzione_intervallo(sorted_pts: list[float], f_vals: dict[float, float], L: float) -> list[Interval]:
    intervalli = []
    for i in range(len(sorted_pts) - 1):
        xl, xr = sorted_pts[i], sorted_pts[i + 1]
        fxl, fxr = f_vals[xl], f_vals[xr]
        xh = x_cap_formula(xl, xr, fxl, fxr, L)
        r_local = R_formula(xl, xr, fxl, fxr, L)
        intervalli.append(Interval(x_lower=xl, x_higher=xr, x_cap=xh, f_cap=r_local))
    return intervalli



def run(func: Callable[[float], float], a: float, b: float, L: float, epsilon: float = 0.0001, limite_iterazioni: int = 1000) -> Result:
    """
    Esegue l'algoritmo di Piyavskii-Shubert sull'intervallo [a, b].

    Parametri
    ---------
    func                : funzione obiettivo f(x)
    a, b                : estremi dell'intervallo di ricerca
    L                   : costante di Lipschitz (upper bound su |f'(x)|)
    epsilon             : tolleranza sul gap di ottimalità Z⁺_s − R_s (default 0.0001)
    limite_iterazioni   : numero massimo di iterazioni (default 1000)

    Restituisce
    -----------
    Result con x*, f*, gap, contatori e storia completa delle iterazioni.
    """

    # Inizializzazione: valutazione di f agli estremi dell'intervallo.
    f_vals: dict[float, float] = {}
    f_vals[a] = func(a)
    f_vals[b] = func(b)
    iterations: list[Iteration] = []
    step = 0


    # Ciclo principale — complessità O(k²) dove k è il numero di iterazioni.
    for step in range(1, limite_iterazioni + 1):

        # Punti valutati in ordine crescente: necessario per costruire gli intervalli consecutivi.
        sorted_pts = sorted(f_vals.keys())

        # Ricostruzione completa degli intervalli con i relativi x̂ e valori di minorata.
        intervals = costruzione_intervallo(sorted_pts, f_vals, L)

        # Selezione dell'intervallo con la minorata minima (R_s).
        selected_index = 0
        min_f_cap = intervals[0].f_cap

        eps_macchina = sys.float_info.epsilon

        for i in range(1, len(intervals)):
            if intervals[i].f_cap - min_f_cap < -eps_macchina:
                min_f_cap = intervals[i].f_cap
                selected_index = i

        selected = intervals[selected_index]

        # Z⁺_s: miglior valore di f trovato finora, usato come upper bound nel criterio di arresto.
        zs = next(iter(f_vals.values()))
        for x, fx in f_vals.items():
            if fx - zs < -eps_macchina:
                zs = fx

        rs = selected.f_cap     # lower bound globale: min F_s(x̂_j) su tutti gli intervalli
        gap = zs - rs           # gap di ottimalità: condizione di arresto se <= epsilon

        # Valutazione di f nel punto caratteristico dell'intervallo selezionato.
        new_x = selected.x_cap
        new_f = func(new_x)
        f_vals[new_x] = new_f

        it = Iteration(
            step=step,
            eval_points=sorted_pts,
            eval_values=[f_vals[p] for p in sorted_pts],
            intervals=intervals,
            chosen_interval=selected,
            chosen_index=selected_index,
            new_x_cap=new_x,
            new_fval=new_f,
            zs_plus=zs,
            rs=rs,
            gap=gap,
        )
        iterations.append(it)

        if gap <= epsilon:
            break

    sorted_pts = sorted(f_vals.keys())


    zs_final = next(iter(f_vals.values()))
    x_opt = next(iter(f_vals.keys()))
    for x, fx in f_vals.items():
        if fx - zs_final < -eps_macchina:
            zs_final = fx
            x_opt = x

    intervals_final = costruzione_intervallo(sorted_pts, f_vals, L)


    rs_final = intervals_final[0].f_cap
    for iv in intervals_final[1:]:
        if iv.f_cap - rs_final < -eps_macchina:
            rs_final = iv.f_cap

    return Result(
        x_star=x_opt,
        f_star=zs_final,
        R_final=rs_final,
        gap=zs_final - rs_final,
        n_iterations=step,
        n_evaluations=len(f_vals),
        iterations=iterations,
        convergenza=(zs_final - rs_final) <= epsilon,
        epsilon=epsilon,
    )
