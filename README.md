# Ottimizzazione Globale con l'Algoritmo di Piyavskii-Shubert

Implementazione dell'**algoritmo di Piyavskii-Shubert** per la ricerca del minimo globale di funzioni univariate Lipschitz-continue. Il progetto include due versioni dell'algoritmo (classica e ottimizzata con heap), una suite di 20 funzioni di test e un modulo di visualizzazione iterazione per iterazione.

## Descrizione

L'algoritmo di Piyavskii-Shubert è un metodo deterministico di ottimizzazione globale che trova il minimo di una funzione `f: [a, b] → ℝ` nota la costante di Lipschitz `L`. Ad ogni iterazione:

1. Costruisce una **minorata** (lower bound) lineare a tratti di `f` usando i coni di Lipschitz centrati nei punti già valutati.
2. Seleziona l'intervallo il cui punto caratteristico `x̂` raggiunge il valore di minorata minore.
3. Valuta `f(x̂)` e suddivide l'intervallo.
4. Si ferma quando il **gap di ottimalità** `Z⁺_s − R_s ≤ ε`, dove `Z⁺_s` è il miglior valore trovato e `R_s` è il lower bound globale.

## Struttura del Progetto

```
workspaceCalcolo/
├── algorithm.py            # Implementazione classica
├── algoritm_optimized.py   # Implementazione ottimizzata con heap
├── test_functions.py       # 20 funzioni di benchmark + runner principale
└── plotting.py             # Visualizzazione iterazione per iterazione
```

### `algorithm.py` — Implementazione Classica

Implementa l'algoritmo ricostruendo per intero la lista degli intervalli ad ogni iterazione.

| Classe / Funzione | Descrizione |
|---|---|
| `Interval` | Estremi dell'intervallo, punto caratteristico `x̂` e lower bound `f_cap` |
| `Iteration` | Snapshot completo di un'iterazione: punti valutati, intervalli, gap |
| `Result` | Output finale: `x*`, `f*`, gap, contatori, flag di convergenza |
| `x_cap_formula` | Calcola `x̂ = (xl+xr)/2 + (f(xl)−f(xr))/(2L)` |
| `R_formula` | Calcola il valore della minorata `(f(xl)+f(xr))/2 − L(xr−xl)/2` |
| `costruzione_intervallo` | Divide i punti ordinati in intervalli consecutivi |
| `run(func, a, b, L, ε, max_iter)` | Funzione principale |

**Complessità:** O(k²) — gli intervalli vengono riordinati e ricostruiti da zero a ogni iterazione.

---

### `algoritm_optimized.py` — Implementazione Ottimizzata con Heap

Sostituisce la ricostruzione completa degli intervalli con una **min-heap** (coda di priorità) indicizzata sul valore di minorata `f_cap`.

| Classe / Funzione | Descrizione |
|---|---|
| `Interval` | Come sopra, con `__lt__` per l'ordinamento nell'heap |
| `crea_intervallo` | Crea un singolo intervallo (usata dopo ogni suddivisione) |
| `run(func, a, b, L, ε, max_iter)` | Funzione principale con heap |

**Complessità:** O(k log k) in media — ogni iterazione esegue push/pop sull'heap in O(log k) invece di una scansione lineare O(k).

---

### `test_functions.py` — Suite di Benchmark

Definisce 20 funzioni di test standard usate per valutare algoritmi di ottimizzazione globale.

| Tag | Funzione | Intervallo | Note |
|---|---|---|---|
| F1 | Polinomio di grado 6 | [−1.5, 11] | Altamente non-lineare |
| F2 | sin(x) + sin(10x/3) | [2.7, 7.5] | Multimodale |
| F3 | −Σ k·sin((k+1)x+k) | [−10, 10] | Somma di sinusoidi pesate |
| F4 | −(16x²−24x+5)e^(−x) | [1.9, 3.9] | Polinomio × esponenziale |
| F5 | (3x−1.4)·sin(18x) | [0, 1.2] | Modulazione sinusoidale |
| F6 | −(x+sin(x))·e^(−x²) | [−10, 10] | Sinusoide smorzata con gaussiana |
| F7 | sin(x)+sin(10x/3)+ln(x) | [2.7, 7.5] | Multimodale con logaritmo |
| F8 | −Σ k·cos((k+1)x+k) | [−10, 10] | Somma di cosinusoidi pesate |
| F9 | sin(x) + sin(2x/3) | [3.1, 20.4] | Sinusoidi a due frequenze |
| F10 | −x·sin(x) | [0, 10] | Prodotto lineare-sinusoidale |
| F11 | 2cos(x) + cos(2x) | [−π/2, 2π] | Somma di cosinusoidi |
| F12 | sin³(x) + cos³(x) | [0, 2π] | Polinomi trigonometrici cubici |
| F13 | −∛(x²) + ∛(x²−1) | [0.001, 0.99] | Funzione con radici cubiche |
| F14 | −e^(−x)·sin(2πx) | [0, 4] | Sinusoide smorzata esponenzialmente |
| F15 | (x²−5x+6)/(x²+1) | [−5, 5] | Funzione razionale |
| F16 | 2(x−3)² + e^(0.5x²) | [−3, 3] | Parabola + esponenziale |
| F17 | x⁶−15x⁴+27x²+250 | [−4, 4] | Polinomio di grado 6 |
| F18 | (x−2)² se x≤3 / 2ln(x−2)+1 se x>3 | [0, 6] | Funzione a tratti, discontinuità in x=3 |
| F19 | −x + sin(3x) − 1 | [0, 6.5] | Lineare + sinusoide |
| F20 | (sin(x)−x)·e^(−x²) | [−10, 10] | Differenza smorzata |

Ogni voce in `FUNCTIONS` specifica: `func`, `a`, `b`, `L`, `epsilon`, `name`, `tag`.

**Esecuzione del benchmark:**

```bash
python test_functions.py
```

Esempio di output per una funzione:

```
[F2] sin(x) + sin(10x/3)
  Iterazioni : 12
  x*         : 5.1457...
  f(x*)      : -1.8995...
  Gap        : 0.3821
  Convergenza: True
  Tempo      : 0.0003 s
```

---

### `plotting.py` — Visualizzazione

Genera una griglia di grafici che mostrano lo stato dell'algoritmo ad ogni iterazione.

```python
from algoritm_optimized import run
from plotting import plot_all

result = run(f2, a=2.7, b=7.5, L=4.29, epsilon=0.5)
plot_all(f2, a=2.7, b=7.5, L=4.29, result=result, cols=3, save_path="output/f2.png")
```

Ogni pannello mostra:
- **Linea blu continua** — funzione `f(x)`
- **Linea rossa tratteggiata** — minorata di Lipschitz `F_s(x)`
- **Punti verdi** — punti di `f` già valutati
- **Titolo** — numero dell'iterazione e gap corrente

## Requisiti

```
numpy
matplotlib
```

Installazione:

```bash
pip install numpy matplotlib
```

## Utilizzo

```python
from algoritm_optimized import run
import numpy as np

def mia_funzione(x):
    return np.sin(x) + np.sin(10 * x / 3)

result = run(
    func=mia_funzione,
    a=2.7,
    b=7.5,
    L=4.29,       # costante di Lipschitz (upper bound)
    epsilon=0.5,  # tolleranza sul gap di ottimalità
    limite_iterazioni=1000
)

print(f"x*         = {result.x_star:.6f}")
print(f"f(x*)      = {result.f_star:.6f}")
print(f"Gap        = {result.gap:.6f}")
print(f"Iterazioni : {result.n_iterations}")
print(f"Valutazioni: {result.n_evaluations}")
```

### Scelta di `L`

`L` deve essere un upper bound su `|f'(x)|` nell'intervallo `[a, b]`. Sottostimare `L` può far mancare il minimo globale; sovrastimare rallenta la convergenza. Per funzioni differenziabili, una scelta sicura è `L = max |f'(x)|` sull'intervallo.

## Dettagli dell'Algoritmo

La minorata nel punto `x` è definita come:

```
F_s(x) = max_j [ f(x_j) − L · |x − x_j| ]
```

Il punto caratteristico dell'intervallo `[x_l, x_r]` è:

```
x̂ = (x_l + x_r) / 2 + (f(x_l) − f(x_r)) / (2L)
```

e il lower bound sull'intervallo vale:

```
R = (f(x_l) + f(x_r)) / 2 − L · (x_r − x_l) / 2
```

L'algoritmo termina quando `Z⁺_s − R_s ≤ ε`, garantendo che il punto restituito sia a distanza al massimo `ε` dal valore del minimo globale reale.

## Licenza

MIT
