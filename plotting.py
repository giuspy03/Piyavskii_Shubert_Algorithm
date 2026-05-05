"""
Modulo di visualizzazione per l'algoritmo di Piyavskii-Shubert.

Genera una griglia di pannelli — uno per iterazione — dove ogni pannello mostra:
f(x) in blu, la minorata F_s(x) in rosso e i punti valutati in verde.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional
from algorithm import Result, Iteration

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

BLUE  = "#185FA5"
RED   = "#D32020"
GREEN = "#1D9E75"


def _minorata(x: np.ndarray, pts: list[float], vals: list[float], L: float) -> np.ndarray:
    """F_s(x): massimo puntuale dei coni di Lipschitz."""
    result = np.full_like(x, -np.inf)
    for p, fp in zip(pts, vals):
        result = np.maximum(result, fp - L * np.abs(x - p))
    return result


def _draw_panel(ax: plt.Axes, func: Callable, a: float, b: float, L: float, it: Iteration):
    """Disegna su ax: f(x), minorata F_s(x), punti valutati."""
    xs = np.linspace(a, b, 600)

    # f(x) reale
    ax.plot(xs, func(xs), color=BLUE, lw=1.8, label="f(x)")

    # minorata F_s(x)
    minorata = _minorata(xs, it.eval_points, it.eval_values, L)
    ax.plot(xs, minorata, color=RED, lw=1.4, ls="--", label="F_s(x)")

    # punti valutati
    ax.plot(it.eval_points, it.eval_values, "o", color=GREEN, ms=5, zorder=5, label="punti")

    ax.set_xlim(a, b)
    ax.set_title(f"iter {it.step}  gap={it.gap:.3f}", fontsize=8, fontweight="bold")
    ax.tick_params(labelsize=7)


def plot_all(
    func: Callable,
    a: float,
    b: float,
    L: float,
    result: Result,
    cols: int = 3,
    save_path: Optional[str] = None,
) -> plt.Figure:
    
    
    """Griglia di tutti i pannelli, uno per iterazione."""
    n = len(result.iterations)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.0 * rows))
    axes = np.array(axes).flatten()

    for i, it in enumerate(result.iterations):
        _draw_panel(axes[i], func, a, b, L, it)
        if i == 0:
            axes[i].legend(fontsize=7, loc="upper right")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Piyavskii-Shubert — gap finale: {result.gap:.4f}, ε={result.epsilon}",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
