import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#from algorithm import run
from algoritm_optimized import run
from plotting import plot_last
import time


def testL(x):
    """sin(5x) + 0.2x — usata per testare il comportamento dell'algoritmo con L sottostimata.
    Intervallo consigliato: [0, 4],  L_reale ≈ 5.2"""
    return np.sin(5*x) + 0.2 * x



def f1(x):
    """(1/6)x^6 - (52/25)x^5 + (39/80)x^4 + (71/10)x^3 - (79/20)x^2 - x + 1/10
    Intervallo: [-1.5, 11],  L = 13870"""
    return (1/6)*x**6 - (52/25)*x**5 + (39/80)*x**4 + (71/10)*x**3 - (79/20)*x**2 - x + 1/10
 
 
def f2(x):
    """sin(x) + sin(10x/3)
    Intervallo: [2.7, 7.5],  L = 4.29"""
    return np.sin(x) + np.sin(10*x / 3)
 
 
def f3(x):
    """-sum_{k=1}^{5} k * sin((k+1)*x + k)
    Intervallo: [-10, 10],  L = 67"""
    return -sum(k * np.sin((k + 1)*x + k) for k in range(1, 6))
 
 
def f4(x):
    """-(16x^2 - 24x + 5) * e^{-x}
    Intervallo: [1.9, 3.9],  L = 3"""
    return -(16*x**2 - 24*x + 5) * np.exp(-x)
 
 
def f5(x):
    """(3x - 1.4) * sin(18x)
    Intervallo: [0, 1.2],  L = 36"""
    return (3*x - 1.4) * np.sin(18*x)
 
 
def f6(x):
    """-(x + sin(x)) * e^{-x^2}
    Intervallo: [-10, 10],  L = 2.5"""
    return -(x + np.sin(x)) * np.exp(-x**2)
 
 
def f7(x):
    """sin(x) + sin(10x/3) + ln(x) - 0.84x + 3
    Intervallo: [2.7, 7.5],  L = 6"""
    return np.sin(x) + np.sin(10*x / 3) + np.log(x) - 0.84*x + 3
 
 
def f8(x):
    """-sum_{k=1}^{5} k * cos((k+1)*x + k)
    Intervallo: [-10, 10],  L = 67"""
    return -sum(k * np.cos((k + 1)*x + k) for k in range(1, 6))
 
 
def f9(x):
    """sin(x) + sin(2x/3)
    Intervallo: [3.1, 20.4],  L = 1.7"""
    return np.sin(x) + np.sin(2*x / 3)
 
 
def f10(x):
    """-x * sin(x)
    Intervallo: [0, 10],  L = 11"""
    return -x * np.sin(x)
 
 
def f11(x):
    """2*cos(x) + cos(2x)
    Intervallo: [-1.57, 6.28],  L = 3"""
    return 2*np.cos(x) + np.cos(2*x)
 
 
def f12(x):
    """sin^3(x) + cos^3(x)
    Intervallo: [0, 6.28],  L = 2.2"""
    return np.sin(x)**3 + np.cos(x)**3
 
 
def f13(x):
    """-x^(2/3) + (x^2 - 1)^(1/3)
    Intervallo: [0.001, 0.99],  L = 8.5
    Nota: si usa np.cbrt per ottenere la radice cubica reale."""
    return -np.cbrt(x**2) + np.cbrt(x**2 - 1)
 
 
def f14(x):
    """-e^{-x} * sin(2*pi*x)
    Intervallo: [0, 4],  L = 6.5"""
    return -np.exp(-x) * np.sin(2*np.pi*x)
 
 
def f15(x):
    """(x^2 - 5x + 6) / (x^2 + 1)
    Intervallo: [-5, 5],  L = 6.5"""
    return (x**2 - 5*x + 6) / (x**2 + 1)
 
 
def f16(x):
    """2*(x - 3)^2 + e^{0.5*x^2}
    Intervallo: [-3, 3],  L = 85"""
    return 2*(x - 3)**2 + np.exp(0.5*x**2)
 
 
def f17(x):
    """x^6 - 15x^4 + 27x^2 + 250
    Intervallo: [-4, 4],  L = 2520"""
    return x**6 - 15*x**4 + 27*x**2 + 250
 
 
def f18(x):
    """(x-2)^2        se x <= 3
    2*ln(x-2) + 1   se x > 3
    Intervallo: [0, 6],  L = 4"""
    x = np.asarray(x, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(x <= 3, (x - 2)**2, 2*np.log(x - 2) + 1)
 
 
def f19(x):
    """-x + sin(3x) - 1
    Intervallo: [0, 6.5],  L = 4"""
    return -x + np.sin(3*x) - 1
 
 
def f20(x):
    """(sin(x) - x) * e^{-x^2}
    Intervallo: [-10, 10],  L = 1.3"""
    return (np.sin(x) - x) * np.exp(-x**2)



# ── configurazioni ───────────────────────────────────────────────────────────

FUNCTIONS = [
    {
        "name":    "(1/6)x^6 - (52/25)x^5 + (39/80)x^4 + (71/10)x^3 - (79/20)x^2 - x + 1/10",
        "func":    f1,
        "a":       -1.5,
        "b":       11.0,
        "L":       13870,
        "epsilon": 0.5,
        "tag":     "F1",
    },
    {
        "name":    "sin(x) + sin(10x/3)",
        "func":    f2,
        "a":       2.7,
        "b":       7.5,
        "L":       4.29,
        "epsilon": 0.5,
        "tag":     "F2",
    },
    {
        "name":    "-sum_{k=1}^{5} k * sin((k+1)*x + k)",
        "func":    f3,
        "a":       -10.0,
        "b":       10.0,
        "L":       67.0,
        "epsilon": 0.5,
        "tag":     "F3",
    },
    {
        "name":    "-(16x^2 - 24x + 5) * e^{-x}",
        "func":    f4,
        "a":       1.9,
        "b":       3.9,
        "L":       3.0,
        "epsilon": 0.5,
        "tag":     "F4",
    },
    {
        "name":    "(3x - 1.4) * sin(18x)",
        "func":    f5,
        "a":       0.0,
        "b":       1.2,
        "L":       36,
        "epsilon": 0.5,
        "tag":     "F5",
    },
    {
        "name":    "-(x + sin(x)) * e^{-x^2}",
        "func":    f6,
        "a":       -10.0,
        "b":       10.0,
        "L":       2.5,
        "epsilon": 0.5,
        "tag":     "F6",
    },
    {
        "name":    "sin(x) + sin(10x/3) + ln(x) - 0.84x + 3",
        "func":    f7,
        "a":       2.7,
        "b":       7.5,
        "L":       6,
        "epsilon": 0.5,
        "tag":     "F7",
    },
    {
        "name":    "-sum_{k=1}^{5} k * cos((k+1)*x + k)",
        "func":    f8,
        "a":       -10.0,
        "b":       10.0,
        "L":       67,
        "epsilon": 0.5,
        "tag":     "F8",
    },
    {
        "name":    "sin(x) + sin(2x/3)",
        "func":    f9,
        "a":       3.1,
        "b":       20.4,
        "L":       1.7,
        "epsilon": 0.5,
        "tag":     "F9",
    },
    {
        "name":    "-x * sin(x)",
        "func":    f10,
        "a":       0.0,
        "b":       10.0,
        "L":       11,
        "epsilon": 0.5,
        "tag":     "F10",
    },
    {
        "name":    "2*cos(x) + cos(2x)",
        "func":    f11,
        "a":       -1.57,
        "b":       6.28,
        "L":       3,
        "epsilon": 0.5,
        "tag":     "F11",
    },
    {
        "name":    "sin^3(x) + cos^3(x)",
        "func":    f12,
        "a":       0.0,
        "b":       6.28,
        "L":       2.2,
        "epsilon": 0.5,
        "tag":     "F12",
    },
    {
        "name":    "-x^(2/3) + (x^2 - 1)^(1/3)",
        "func":    f13,
        "a":       0.001,
        "b":       0.99,
        "L":       8.5,
        "epsilon": 0.5,
        "tag":     "F13",
    },
    {
        "name":    "-e^{-x} * sin(2*pi*x)",
        "func":    f14,
        "a":       0.0,
        "b":       4.0,
        "L":       6.5,
        "epsilon": 0.5,
        "tag":     "F14",
    },
    {
        "name":    "(x^2 - 5x + 6) / (x^2 + 1)",
        "func":    f15,
        "a":       -5.0,
        "b":       5.0,
        "L":       6.5,
        "epsilon": 0.5,
        "tag":     "F15",
    },
    {
        "name":    "2*(x - 3)^2 + e^{0.5*x^2}",
        "func":    f16,
        "a":       -3.0,
        "b":       3.0,
        "L":       85,
        "epsilon": 0.5,
        "tag":     "F16",
    },
    {
        "name":    "x^6 - 15x^4 + 27x^2 + 250",
        "func":    f17,
        "a":       -4.0,
        "b":       4.0,
        "L":       2520,
        "epsilon": 0.5,
        "tag":     "F17",
    },
    {
        "name":    "(x-2)^2 se x <= 3; 2*ln(x-2) + 1 se x > 3",
        "func":    f18,
        "a":       0.0,
        "b":       6.0,
        "L":       4,
        "epsilon": 0.5,
        "tag":     "F18",
    },
    {
        "name":    "-x + sin(3x) - 1",
        "func":    f19,
        "a":       0.0,
        "b":       6.5,
        "L":       4,
        "epsilon": 0.5,
        "tag":     "F19",
    },
    {
        "name":    "(sin(x) - x) * e^{-x^2}",
        "func":    f20,
        "a":       -10.0,
        "b":       10.0,
        "L":       1.3,
        "epsilon": 0.5,
        "tag":     "F20",
    },
]



# Configurazione separata per testare il comportamento con L intenzionalmente sottostimata.
FUNCTION = [
    {
        "name":    "TestL",
        "func":    testL,
        "a":       0.0,
        "b":       4.0,
        "L":       5.2/1.1,
        "epsilon": 0.05,
        "tag":     "testLsottostimato",
    },
]


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    """Esegue il benchmark su tutte le funzioni in FUNCTIONS e stampa i risultati."""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    for f in FUNCTIONS:
        
        start_time = time.perf_counter()

        result = run(f["func"], f["a"], f["b"], f["L"])
        
        end_time=time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000

        print(f"[{f['name']}]  iterazioni={result.n_iterations}  "
              f"x*={result.x_star:.4f}  f*={result.f_star:.4f}  "
              f"gap={result.gap:.4f}  convergenza={result.convergenza}, tempo={elapsed_ms} ms")
        
        fig = plot_last(
            f["func"], f["a"], f["b"], f["L"], result,
            save_path=os.path.join(output_dir, f["tag"] + "_iterazioni.png"),
        )
        plt.close(fig)
        

    print("\nGrafici salvati in ./" + output_dir + "/")


if __name__ == "__main__":
    main()
