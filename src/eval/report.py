"""
Turns a sweep directory into report assets:

    <report_dir>/figures/*.pdf          - figures
    <report_dir>/generated/*.tex        - booktabs tables
    <report_dir>/generated/manifest.tex - \\input by Doc/report/report.tex

Everything is regenerated from ``aggregates.json``, so re-running the sweep
(e.g. after Tier-2/3 learners are registered) refreshes the report.
"""
import os
from typing import Dict, List, Optional, Sequence, Tuple

from src.eval.plotting import METRIC_LABELS, pick, plot_curves, plot_heatmaps
from src.eval.sweep import load_aggregates


def _tex(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def _pm(a: Dict, key: str, fmt: str = "{:.2f}") -> str:
    s = a["summary"].get(key)
    if s is None or s["mean"] != s["mean"]:  # NaN
        return "--"
    return f"${fmt.format(s['mean'])} \\pm {fmt.format(s['ci'])}$"


def _closest(values: Sequence[float], target: float) -> float:
    return min(values, key=lambda v: (abs(v - target), v))


def representative(aggs: Sequence[Dict]) -> Tuple[int, int, float]:
    ms = sorted({a["config"]["m"] for a in aggs})
    ns = sorted({a["config"]["n"] for a in aggs})
    ks = sorted({a["config"]["kappa"] for a in aggs})
    return ms[0], int(_closest(ns, 10)), _closest(ks, 0.5)


def _table(caption: str, label: str, cols: List[Tuple[str, str, str]], rows: List[Dict],
           row_key: Tuple[str, ...] = ("m", "n", "kappa")) -> str:
    head = " & ".join([rk if rk != "kappa" else r"$\kappa$" for rk in row_key] + [c[1] for c in cols])
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\begin{tabular}{@{}" + "r" * len(row_key) + "c" * len(cols) + "@{}}", r"\toprule",
        head + r" \\", r"\midrule",
    ]
    prev = None
    for a in rows:
        key = tuple(a["config"][k] for k in row_key)
        if prev is not None and key[0] != prev:
            lines.append(r"\midrule")
        prev = key[0]
        cells = [_tex(f"{v:g}" if isinstance(v, float) else str(v)) for v in key]
        cells += [_pm(a, c[0], c[2]) for c in cols]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _figure(path_rel: str, caption: str, label: str, width: str = r"\linewidth") -> str:
    return "\n".join([
        r"\begin{figure}[htbp]", r"\centering",
        rf"\includegraphics[width={width}]{{{path_rel}}}",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{figure}", ""])


def _sort(rows: List[Dict]) -> List[Dict]:
    return sorted(rows, key=lambda a: (a["config"]["m"], a["config"]["n"], a["config"]["kappa"]))


def build_report(sweep_dir: str, report_dir: str, fig_ext: str = "pdf") -> str:
    aggs = [a for a in load_aggregates(sweep_dir) if a["n_used"] > 0]
    gen_dir = os.path.join(report_dir, "generated")
    fig_dir = os.path.join(report_dir, "figures")
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    parts: List[str] = [f"% Auto-generated from {_tex(os.path.basename(os.path.normpath(sweep_dir)))} "
                        "by run_evaluation.py report. Do not edit by hand.", ""]
    T = aggs[0]["config"]["horizon"] if aggs else 0
    seeds = aggs[0]["n_seeds"] if aggs else 0

    def fig(name: str) -> Tuple[str, str]:
        return os.path.join(fig_dir, f"{name}.{fig_ext}"), f"figures/{name}.{fig_ext}"

    # ------------------------------------------------------------------ Tier 1
    t1 = pick(aggs, scenario="tier1", round_structure="canonical")
    if t1:
        parts.append(r"\subsection{Tier 1 across the experimental grid}")
        parts.append(
            rf"Each cell averages {seeds} random instances (Beta click probabilities, $v="
            rf"{aggs[0]['config']['click_value']:g}$); $\pm$ is the 95\% Student-$t$ half-width. "
            r"Welfare ratio is $\mathrm{Wel}(b^*)$ over the first-best welfare of the omniscient planner.")
        for metric, title in [("t1_efficiency", None), ("t1_welfare_ratio", None),
                              ("t1_revenue", None), ("t1_eq_gap", None)]:
            p, rel = fig(f"tier1_{metric}")
            if plot_heatmaps(t1, metric, p, title=title):
                parts.append(_figure(rel, rf"{METRIC_LABELS[metric]} across $(m,n,\kappa)$; mean over {seeds} seeds.",
                                     f"fig:tier1-{metric}"))
        cols = [("budget", "$B_i$", "{:.0f}"), ("realized_kappa", r"$\kappa_i(b^*)$", "{:.2f}"),
                ("frac_pure", "pure", "{:.2f}"), ("t1_eq_gap", r"$\varepsilon(b^*)$", "{:.2f}"),
                ("t1_revenue", "Rev", "{:.2f}"), ("t1_efficiency", "Eff", "{:.2f}"),
                ("t1_welfare_ratio", "Wel/FB", "{:.2f}")]
        for m in sorted({a["config"]["m"] for a in t1}):
            name = f"tier1_m{m}"
            with open(os.path.join(gen_dir, f"{name}.tex"), "w", encoding="utf-8") as fh:
                fh.write(_table(f"Tier-1 reference profile, $m={m}$ advertisers.", f"tab:{name}", cols,
                                _sort(pick(t1, m=m))))
            parts.append(rf"\input{{generated/{name}}}")
        parts.append(r"\clearpage")

    # ------------------------------------------------------------------ learners / dynamics
    m0, n0, k0 = representative(aggs) if aggs else (0, 0, 0.0)
    for scenario, title in (("tier2", "Tier 2: single learner vs.\\ fixed Tier-1 opponents"),
                            ("tier3", "Tier 3: all advertisers adapt concurrently")):
        sc = pick(aggs, scenario=scenario)
        if not sc:
            continue
        parts.append(rf"\subsection{{{title}}}")
        learners = sorted({a["config"]["learner"] for a in sc})
        rep_can = [a for a in pick(sc, m=m0, n=n0, kappa=k0, round_structure="canonical")]
        rep_can.sort(key=lambda a: learners.index(a["config"]["learner"]))
        curve_metrics = ["cum_regret", "total_profit"] if scenario == "tier2" else ["eq_gap", "eq_distance", "efficiency", "total_profit"]
        for metric in curve_metrics:
            p, rel = fig(f"{scenario}_curves_{metric}")
            if plot_curves(rep_can, metric, p, label_fn=lambda a: a["config"]["learner"],
                           title=f"m={m0}, n={n0}, kappa={k0:g}, canonical, T={T}"):
                parts.append(_figure(rel, rf"{METRIC_LABELS[metric]} vs.\ round, mean and 95\% CI band over seeds "
                                     rf"($m={m0}$, $n={n0}$, $\kappa={k0:g}$, canonical).",
                                     f"fig:{scenario}-{metric}", width=r"0.8\linewidth"))
        for learner in learners:
            ls = pick(sc, learner=learner)
            rep_struct = pick(ls, m=m0, n=n0, kappa=k0)
            metric = "cum_regret" if scenario == "tier2" else "eq_gap"
            if metric != "cum_regret":
                # Discounting only re-weights cumulative sums: per-round curves equal canonical.
                rep_struct = [a for a in rep_struct if a["config"]["round_structure"] != "discounted"]
            p, rel = fig(f"{scenario}_{learner}_structures_{metric}")
            if len(rep_struct) > 1 and plot_curves(rep_struct, metric, p,
                                                   label_fn=lambda a: a["config"]["round_structure"],
                                                   title=f"{learner}: round-structure robustness"):
                note = (r"discounted regret uses weights $\gamma^{t-1}$" if metric == "cum_regret" else
                        r"the discounted variant changes only cumulative sums, so its per-round curve equals canonical")
                parts.append(_figure(rel, rf"\texttt{{{_tex(learner)}}}: {METRIC_LABELS[metric]} under different "
                                     rf"round structures (cumulative is measured against the fluid LP benchmark; {note}).",
                                     f"fig:{scenario}-{learner}-structures", width=r"0.8\linewidth"))
            can = pick(ls, round_structure="canonical")
            heat = (["final_regret", "pol_total_profit"] if scenario == "tier2"
                    else ["eq_gap_last", "eq_distance_last", "pol_total_profit", "efficiency_last"])
            for metric in heat:
                p, rel = fig(f"{scenario}_{learner}_heat_{metric}")
                if plot_heatmaps(can, metric, p, title=f"{learner}: {METRIC_LABELS[metric]}"):
                    parts.append(_figure(rel, rf"\texttt{{{_tex(learner)}}} ({scenario}, canonical): "
                                         rf"{METRIC_LABELS[metric]} across $(m,n,\kappa)$.",
                                         f"fig:{scenario}-{learner}-{metric}"))
            cols = ([("final_regret", r"$\mathrm{Reg}_T$", "{:.1f}"), ("regret_slope", "slope", "{:.2f}")]
                    if scenario == "tier2" else
                    [("eq_gap_last", r"$\varepsilon$", "{:.2f}"), ("eq_distance_last", r"$\|b-b^*\|_1$", "{:.1f}")])
            cols += [("efficiency_last", "Eff", "{:.2f}"), ("revenue_last", "Rev", "{:.2f}"),
                     ("pol_total_profit", r"PoL$_\pi$", "{:.2f}"), ("pol_efficiency", r"PoL$_\mathrm{Eff}$", "{:.2f}")]
            for rs in sorted({a["config"]["round_structure"] for a in ls}):
                name = f"{scenario}_{learner}_{rs}"
                rows = _sort(pick(ls, round_structure=rs))
                with open(os.path.join(gen_dir, f"{name}.tex"), "w", encoding="utf-8") as fh:
                    fh.write(_table(rf"\texttt{{{_tex(learner)}}}, {scenario}, {rs} round structure, $T={T}$ "
                                    r"(final metrics averaged over the last 10\% of rounds; PoL = Tier-1 minus learned).",
                                    f"tab:{name}", cols, rows))
                parts.append(rf"\input{{generated/{name}}}")
        parts.append(r"\clearpage")

    manifest = os.path.join(gen_dir, "manifest.tex")
    with open(manifest, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts) + "\n")
    return manifest
