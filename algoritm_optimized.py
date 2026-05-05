from dataclasses import dataclass
from typing import Callable, Optional
import sys
import heapq


# Rappresenta un intervallo [x_lower, x_higher] con il suo punto caratteristico x̂ e il valore della minorata.
@dataclass
class Interval:
    x_lower: float
    x_higher: float
    x_cap: float       # punto caratteristico: argmin della minorata sull'intervallo
    f_cap: float       # valore della minorata in x_cap: F_s(x_cap)

    # Ordinamento per f_cap: consente l'inserimento diretto in una min-heap.
    def __lt__(self, other):
        return self.f_cap < other.f_cap


# Snapshot di una singola iterazione: punti valutati, intervallo scelto e gap corrente.
@dataclass
class Iteration:
    step: int
    eval_points: list[float]          # punti dove f è stata valutata (ordinati)
    eval_values: list[float]          # valori f(x) nei punti valutati
    chosen_interval: Interval         # intervallo estratto dall'heap (minorata minima)
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


# Restituisce il minimo di un iterabile usando eps_macchina come soglia per il confronto tra float.
def min_with_eps(values, eps_macchina: float) -> float:
    it = iter(values)
    current_min = next(it)
    for v in it:
        if current_min - v > eps_macchina:
            current_min = v
    return current_min


# Crea un Interval per [xl, xr] calcolando x̂ e il valore della minorata.
def crea_intervallo(xl: float, xr: float, f_vals: dict, L: float) -> Interval:
    fxl, fxr = f_vals[xl], f_vals[xr]
    xh = x_cap_formula(xl, xr, fxl, fxr, L)
    r  = R_formula(xl, xr, fxl, fxr, L)
    return Interval(x_lower=xl, x_higher=xr, x_cap=xh, f_cap=r)


def run(func: Callable[[float], float], a: float, b: float, L: float,
        epsilon: float = 0.003, limite_iterazioni: int = 1000) -> Result:
    """
    Esegue l'algoritmo di Piyavskii-Shubert sull'intervallo [a, b] usando una min-heap.

    Rispetto alla versione classica, la selezione dell'intervallo ottimale avviene in
    O(log k) invece di O(k), riducendo la complessità media a O(k log k).

    Parametri
    ---------
    func                : funzione obiettivo f(x)
    a, b                : estremi dell'intervallo di ricerca
    L                   : costante di Lipschitz (upper bound su |f'(x)|)
    epsilon             : tolleranza sul gap di ottimalità Z⁺_s − R_s (default 0.003)
    limite_iterazioni   : numero massimo di iterazioni (default 1000)

    Restituisce
    -----------
    Result con x*, f*, gap, contatori e storia completa delle iterazioni.
    """

    eps_macchina = sys.float_info.epsilon

    f_vals: dict[float, float] = {}
    f_vals[a] = func(a)
    f_vals[b] = func(b)

    # Inizializzazione della min-heap con il solo intervallo [a, b].
    heap: list[Interval] = []
    iv0 = crea_intervallo(a, b, f_vals, L)
    heapq.heappush(heap, iv0)

    iterations: list[Iteration] = []
    step = 0

    # Ciclo principale — O(k log k) in media grazie alla min-heap.
    for step in range(1, limite_iterazioni + 1):

        # Estrazione dell'intervallo con la minorata minima dalla coda di priorità.
        selected = heapq.heappop(heap)

        # Z⁺_s: miglior valore di f trovato finora (upper bound).
        zs = min_with_eps(f_vals.values(), eps_macchina)
        rs = selected.f_cap

        gap = zs - rs


        # Valutazione di f nel punto caratteristico dell'intervallo estratto.
        new_x = selected.x_cap
        new_f = func(new_x)
        f_vals[new_x] = new_f

        # L'intervallo estratto viene suddiviso in due sotto-intervalli, entrambi inseriti nell'heap.
        iv_left  = crea_intervallo(selected.x_lower, new_x, f_vals, L)
        iv_right = crea_intervallo(new_x, selected.x_higher, f_vals, L)
        heapq.heappush(heap, iv_left)
        heapq.heappush(heap, iv_right)

        sorted_pts = sorted(f_vals.keys())

        it = Iteration(
            step=step,
            eval_points=sorted_pts,
            eval_values=[f_vals[p] for p in sorted_pts],
            chosen_interval=selected,
            new_x_cap=new_x,
            new_fval=new_f,
            zs_plus=zs,
            rs=rs,
            gap=gap,
        )
        iterations.append(it)

        if gap <= epsilon:
            break

    # Calcolo dei risultati finali sull'insieme completo dei punti valutati.
    zs_final = min_with_eps(f_vals.values(), eps_macchina)
    x_opt = next(x for x, fx in f_vals.items() if abs(fx - zs_final) <= eps_macchina)
    rs_final = min_with_eps((iv.f_cap for iv in heap), eps_macchina)

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
