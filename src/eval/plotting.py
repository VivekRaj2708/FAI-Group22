"""
Figures for the evaluation harness (matplotlib, headless).

* ``plot_curves``      - across-seed mean +/- 95% CI band vs round t, one line
                         per configuration (fixed categorical order).
* ``plot_heatmaps``    - small multiples (one panel per m) of a summary metric
                         over users n x budget tightness kappa, sequential map.
"""
import math
from typing import Callable, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Validated categorical order (light surface); never cycled - >8 series must be faceted.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

METRIC_LABELS = {
    "cum_regret": "Cumulative regret (focal advertiser)",
    "regret": "Per-round regret (focal advertiser)",
    "eq_gap": r"Equilibrium gap $\varepsilon_t$",
    "eq_distance": r"Distance to $b^*$  $\|b^t-b^*\|_1$",
    "efficiency": r"Allocation efficiency $\mathrm{Eff}_t$",
    "revenue": r"Platform revenue $\mathrm{Rev}(b^t)$",
    "welfare": "Expected welfare",
    "total_profit": r"Total advertiser profit $\sum_i p_i(b^t)$",
    "focal_profit": "Focal advertiser profit",
    "final_regret": r"Final regret $\mathrm{Reg}_T$",
    "regret_slope": "Regret log-log slope",
    "eq_gap_last": r"Final equilibrium gap",
    "eq_distance_last": r"Final $\|b^t-b^*\|_1$",
    "efficiency_last": "Final efficiency",
    "pol_total_profit": "Price of learning (total profit)",
    "pol_efficiency": "Price of learning (efficiency)",
    "pol_revenue": "Price of learning (revenue)",
    "t1_efficiency": r"Tier-1 efficiency $\mathrm{Eff}(b^*)$",
    "t1_revenue": r"Tier-1 revenue $\mathrm{Rev}(b^*)$",
    "t1_welfare_ratio": "Tier-1 welfare / first-best welfare",
    "t1_eq_gap": r"Tier-1 gap $\varepsilon(b^*)$",
    "t1_total_profit": r"Tier-1 total profit $\sum_i p_i^*$",
}


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def plot_curves(
    aggs: Sequence[Dict],
    metric: str,
    path: str,
    label_fn: Callable[[Dict], str] = lambda a: a["label"],
    title: Optional[str] = None,
    logx: bool = False,
) -> Optional[str]:
    series = [a for a in aggs if metric in a.get("curves", {})]
    if not series:
        return None
    if len(series) > len(SERIES):
        raise ValueError(f"{len(series)} series exceed the {len(SERIES)}-colour palette; facet instead")
    fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=150)
    for color, a in zip(SERIES, series):
        t = np.asarray(a["curves"]["t"])
        mu = np.asarray(a["curves"][metric]["mean"])
        ci = np.asarray(a["curves"][metric]["ci"])
        ax.fill_between(t, mu - ci, mu + ci, color=color, alpha=0.18, linewidth=0)
        ax.plot(t, mu, color=color, linewidth=1.5, label=f"{label_fn(a)} (k={a['n_used']})")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel("Round $t$")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    _style(ax)
    if len(series) >= 2:
        ax.legend(fontsize=7, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_heatmaps(
    aggs: Sequence[Dict],
    metric: str,
    path: str,
    title: Optional[str] = None,
    cmap: str = "Blues",
    fmt: str = "{:.2f}",
) -> Optional[str]:
    """One panel per m; x = n, y = kappa; colour = across-seed mean of ``metric``."""
    rows = [a for a in aggs if metric in a.get("summary", {})]
    if not rows:
        return None
    ms = sorted({a["config"]["m"] for a in rows})
    ns = sorted({a["config"]["n"] for a in rows})
    ks = sorted({a["config"]["kappa"] for a in rows})
    grids = {}
    for m in ms:
        g = np.full((len(ks), len(ns)), np.nan)
        for a in rows:
            if a["config"]["m"] == m:
                g[ks.index(a["config"]["kappa"]), ns.index(a["config"]["n"])] = a["summary"][metric]["mean"]
        grids[m] = g
    allv = np.concatenate([g[~np.isnan(g)] for g in grids.values()]) if grids else np.array([0.0])
    vmin, vmax = (float(allv.min()), float(allv.max())) if allv.size else (0.0, 1.0)
    if math.isclose(vmin, vmax, rel_tol=1e-6, abs_tol=1e-9):
        vmin, vmax = vmin - 0.5, vmax + 0.5

    fig, axes = plt.subplots(1, len(ms), figsize=(2.6 * len(ms) + 0.8, 2.8), dpi=150, squeeze=False)
    im = None
    for ax, m in zip(axes[0], ms):
        g = grids[m]
        im = ax.imshow(g, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
        for (r, c), val in np.ndenumerate(g):
            if not np.isnan(val):
                frac = (val - vmin) / (vmax - vmin)
                ax.text(c, r, fmt.format(val), ha="center", va="center", fontsize=6.5,
                        color="white" if frac > 0.6 else INK)
        ax.set_xticks(range(len(ns)), [str(n) for n in ns])
        ax.set_yticks(range(len(ks)), [f"{k:g}" for k in ks])
        ax.set_xlabel("users $n$", fontsize=8)
        ax.set_title(f"m = {m}", fontsize=8.5, color=INK)
        ax.tick_params(labelsize=7, colors=INK_2, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0][0].set_ylabel(r"budget tightness $\kappa$", fontsize=8)
    cb = fig.colorbar(im, ax=list(axes[0]), shrink=0.85, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=INK_2)
    cb.ax.yaxis.get_major_formatter().set_useOffset(False)
    cb.outline.set_visible(False)
    fig.suptitle(title or METRIC_LABELS.get(metric, metric), fontsize=9, color=INK, x=0.02, y=1.04, ha="left")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def pick(aggs: Sequence[Dict], **conds) -> List[Dict]:
    """Filter aggregates by config fields, e.g. pick(aggs, scenario='tier2', m=2)."""
    return [a for a in aggs if all(a["config"].get(k) == v for k, v in conds.items())]
