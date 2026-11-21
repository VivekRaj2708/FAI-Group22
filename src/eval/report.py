"""
Turns a sweep directory into report assets:

    <report_dir>/figures/*.pdf          - figures
    <report_dir>/generated/*.tex        - booktabs tables
    <report_dir>/generated/manifest.tex - \\input by Doc/report/report.tex

Everything is regenerated from ``aggregates.json``, so re-running the sweep
(e.g. after Tier-2/3 learners are registered) refreshes the report.

Configurations at the main horizon (the most common one) feed the per-tier
sections. Two extra studies are reported compactly if present:

* UCB exploration sensitivity - learners named ``ucb_a*`` (alpha variants),
  compared with ``ucb`` (alpha = 2) in one table per tier.
* Long-horizon study - configurations at any other horizon.
"""
from collections import Counter
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

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


UCB_ALPHAS = {"ucb_a05": 0.5, "ucb_a1": 1.0, "ucb": 2.0}


def _is_alpha_variant(learner: str) -> bool:
    return learner.startswith("ucb_a")


def _compare_table(caption: str, label: str, aggs: List[Dict], learners: List[Tuple[str, str]],
                   metric: str, fmt: str) -> str:
    """Rows (m, n, kappa); one column per learner showing ``metric``."""
    cells = sorted({(a["config"]["m"], a["config"]["n"], a["config"]["kappa"]) for a in aggs})
    by = {(a["config"]["learner"], a["config"]["m"], a["config"]["n"], a["config"]["kappa"]): a for a in aggs}
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\begin{tabular}{@{}rrr" + "c" * len(learners) + "@{}}", r"\toprule",
        r"$m$ & $n$ & $\kappa$ & " + " & ".join(h for _, h in learners) + r" \\", r"\midrule",
    ]
    prev = None
    for m, n, k in cells:
        if prev is not None and m != prev:
            lines.append(r"\midrule")
        prev = m
        row = [str(m), str(n), f"{k:g}"]
        for name, _ in learners:
            a = by.get((name, m, n, k))
            row.append(_pm(a, metric, fmt) if a else "--")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _sort(rows: List[Dict]) -> List[Dict]:
    return sorted(rows, key=lambda a: (a["config"]["m"], a["config"]["n"], a["config"]["kappa"]))


def build_report(sweep_dir: Union[str, Sequence[str]], report_dir: str, fig_ext: str = "pdf") -> str:
    """Build tables/figures from one or more sweep directories.

    With several directories, configurations are merged by label; the first
    directory listing a label wins (so put the most authoritative sweep first).
    """
    dirs = [sweep_dir] if isinstance(sweep_dir, str) else list(sweep_dir)
    seen, aggs = set(), []
    for d in dirs:
        for a in load_aggregates(d):
            if a["n_used"] > 0 and a["label"] not in seen:
                seen.add(a["label"])
                aggs.append(a)
    gen_dir = os.path.join(report_dir, "generated")
    fig_dir = os.path.join(report_dir, "figures")
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    names = ", ".join(_tex(os.path.basename(os.path.normpath(d))) for d in dirs)
    parts: List[str] = [f"% Auto-generated from {names} "
                        "by run_evaluation.py report. Do not edit by hand.", ""]
    T = Counter(a["config"]["horizon"] for a in aggs).most_common(1)[0][0] if aggs else 0
    seeds = aggs[0]["n_seeds"] if aggs else 0
    long_aggs = [a for a in aggs if a["config"]["horizon"] != T]
    aggs = [a for a in aggs if a["config"]["horizon"] == T]
    alpha_aggs = [a for a in aggs if _is_alpha_variant(a["config"]["learner"])
                  or (a["config"]["learner"] == "ucb" and a["config"]["round_structure"] == "canonical")]
    has_alpha = any(_is_alpha_variant(a["config"]["learner"]) for a in alpha_aggs)
    aggs = [a for a in aggs if not _is_alpha_variant(a["config"]["learner"])]

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
                           title=f"m={m0}, n={n0}, kappa={k0:g}, canonical, T={T}") or os.path.exists(p):
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
            if (len(rep_struct) > 1 and plot_curves(rep_struct, metric, p,
                                                   label_fn=lambda a: a["config"]["round_structure"],
                                                   title=f"{learner}: round-structure robustness")) or os.path.exists(p):
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
        parts.append(r"\clearpage")

    # ------------------------------------------------------------------ UCB exploration sensitivity
    if has_alpha:
        parts.append(r"\subsection{Sensitivity to the UCB exploration parameter $\alpha$}")
        parts.append(r"UCB index $\min\{1,\hat f_j+\sqrt{\alpha\ln t/N_j}\}$ with $\alpha\in\{0.5,1,2\}$ "
                     rf"(canonical, $T={T}$, {seeds} seeds; $\alpha=2$ is the default of Algorithm~1).")
        present = {a["config"]["learner"] for a in alpha_aggs}
        cols = [(l, rf"$\alpha={UCB_ALPHAS[l]:g}$") for l in ("ucb_a05", "ucb_a1", "ucb") if l in present]
        for scenario, metric, what, fmt in (("tier2", "final_regret", r"final regret $\mathrm{Reg}_T$", "{:.0f}"),
                                            ("tier2", "regret_slope", "regret log-log slope", "{:.2f}"),
                                            ("tier3", "eq_gap_last", r"final equilibrium gap $\varepsilon$", "{:.2f}"),
                                            ("tier3", "efficiency_last", "final efficiency", "{:.2f}")):
            rows = pick(alpha_aggs, scenario=scenario, round_structure="canonical")
            if not rows:
                continue
            name = f"ucb_alpha_{scenario}_{metric}"
            with open(os.path.join(gen_dir, f"{name}.tex"), "w", encoding="utf-8") as fh:
                fh.write(_compare_table(rf"UCB exploration sensitivity, {scenario}: {what}.",
                                        f"tab:{name}", rows, cols, metric, fmt))
            parts.append(rf"\input{{generated/{name}}}")
        parts.append(r"\clearpage")

    # ------------------------------------------------------------------ long horizon
    if long_aggs:
        TL = max(a["config"]["horizon"] for a in long_aggs)
        long_aggs = [a for a in long_aggs if a["config"]["horizon"] == TL]
        parts.append(rf"\subsection{{Long-horizon study ($T={TL}$)}}")
        parts.append(r"A subset of the grid re-run at the longest horizon of the proposal, canonical round "
                     r"structure, to check that the regret growth rates measured at the main horizon persist.")
        cells = sorted({(a["config"]["m"], a["config"]["n"], a["config"]["kappa"]) for a in long_aggs})
        for scenario, metric in (("tier2", "cum_regret"), ("tier3", "eq_gap")):
            sc = [a for a in pick(long_aggs, scenario=scenario) if a["config"]["learner"] != "fixed_bstar"]
            if not sc:
                continue
            learners = sorted({a["config"]["learner"] for a in sc})
            for m, n, k in cells:
                rows = sorted(pick(sc, m=m, n=n, kappa=k), key=lambda a: learners.index(a["config"]["learner"]))
                tag = f"{scenario}_long_{metric}_m{m}_n{n}"
                p, rel = fig(tag)
                if (rows and plot_curves(rows, metric, p, label_fn=lambda a: a["config"]["learner"],
                                        title=f"{scenario}: m={m}, n={n}, kappa={k:g}, T={TL}")) or os.path.exists(p):
                    parts.append(_figure(rel, rf"{METRIC_LABELS[metric]} over $T={TL}$ rounds, {scenario} "
                                         rf"($m={m}$, $n={n}$, $\kappa={k:g}$, canonical).",
                                         f"fig:{tag}", width=r"0.8\linewidth"))
            cols = ([("final_regret", r"$\mathrm{Reg}_T$", "{:.0f}"), ("regret_slope", "slope", "{:.2f}")]
                    if scenario == "tier2" else
                    [("eq_gap_last", r"$\varepsilon$", "{:.2f}"), ("eq_distance_last", r"$\|b-b^*\|_1$", "{:.1f}")])
            cols += [("efficiency_last", "Eff", "{:.2f}"), ("pol_total_profit", r"PoL$_\pi$", "{:.2f}")]
            for learner in learners:
                name = f"{scenario}_long_{learner}"
                with open(os.path.join(gen_dir, f"{name}.tex"), "w", encoding="utf-8") as fh:
                    fh.write(_table(rf"\texttt{{{_tex(learner)}}}, {scenario}, canonical, $T={TL}$.",
                                    f"tab:{name}", cols, _sort(pick(sc, learner=learner))))
                parts.append(rf"\input{{generated/{name}}}")
        parts.append(r"\clearpage")

    manifest = os.path.join(gen_dir, "manifest.tex")
    with open(manifest, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts) + "\n")
    return manifest
