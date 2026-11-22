"""
Sensitivity sweep over (m, n, kappa) x round structure x scenario/learner
(proposal Table 2), parallelised over instances.

One job = one synthetic instance (m, n, kappa, seed); the Tier-1 reference
is solved once per job and re-used by every scenario/learner/structure on that
instance. Results are written to ``out_dir``:

* ``spec.json``       - the sweep specification
* ``per_seed.csv``    - one row per (configuration, seed)
* ``summary.csv``     - one row per configuration, mean and 95% CI half-width
* ``aggregates.json`` - summary + across-seed curves (for plotting)
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass, field
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.eval.experiment import ExperimentConfig, SeedResult, aggregate, run_seed


@dataclass
class SweepSpec:
    ms: List[int] = field(default_factory=lambda: [2, 3, 5])
    ns: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    kappas: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 1.0])
    structures: List[str] = field(default_factory=lambda: ["canonical", "discounted", "cumulative"])
    # (scenario, learner) pairs; learner is ignored for tier1.
    runs: List[Tuple[str, str]] = field(default_factory=lambda: [("tier1", "-"), ("tier3", "oracle_br")])
    learner_kwargs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    horizon: int = 1000
    num_seeds: int = 30
    seed_offset: int = 0
    click_value: float = 5.0
    beta_params: Tuple[float, float] = (2.0, 5.0)
    gamma: float = 0.999
    cumulative_rho: float = 0.5
    prefer_exact: bool = True
    # tier1 (fixed b*) is a static profile; evaluating it under every structure is redundant.
    tier1_canonical_only: bool = True

    @property
    def seeds(self) -> List[int]:
        return list(range(self.seed_offset, self.seed_offset + self.num_seeds))

    def configs_for(self, m: int, n: int, kappa: float) -> List[ExperimentConfig]:
        out = []
        for scenario, learner in self.runs:
            for rs in self.structures:
                if scenario == "tier1" and self.tier1_canonical_only and rs != "canonical":
                    continue
                out.append(ExperimentConfig(
                    m=m, n=n, kappa=kappa, click_value=self.click_value, beta_params=self.beta_params,
                    scenario=scenario, learner=learner if scenario != "tier1" else "fixed_bstar",
                    learner_kwargs=self.learner_kwargs.get(learner, {}), horizon=self.horizon,
                    round_structure=rs, gamma=self.gamma, cumulative_rho=self.cumulative_rho,
                    seeds=self.seeds, prefer_exact=self.prefer_exact,
                ))
        return out

    def all_configs(self) -> List[ExperimentConfig]:
        return [c for m in self.ms for n in self.ns for k in self.kappas for c in self.configs_for(m, n, k)]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["runs"] = [list(r) for r in self.runs]
        d["beta_params"] = list(self.beta_params)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SweepSpec":
        d = dict(d)
        d["runs"] = [tuple(r) for r in d.get("runs", [])]
        d["beta_params"] = tuple(d.get("beta_params", (2.0, 5.0)))
        return cls(**d)


PRESETS: Dict[str, SweepSpec] = {
    # Seconds: plumbing check.
    "smoke": SweepSpec(ms=[2], ns=[5], kappas=[0.5], structures=["canonical", "cumulative"],
                       runs=[("tier1", "-"), ("tier2", "random"), ("tier3", "oracle_br")],
                       horizon=200, num_seeds=3),
    # Full (m, n, kappa) grid for the full-information tiers (available now).
    "tier1": SweepSpec(runs=[("tier1", "-"), ("tier3", "oracle_br")], horizon=300, num_seeds=30),
    # Full study (proposal Table 2) once Tier-2/3 learners are registered.
    "paper": SweepSpec(runs=[("tier1", "-")]
                       + [("tier2", l) for l in ("ucb", "thompson", "greedy", "eps_greedy")]
                       + [("tier3", l) for l in ("ucb", "thompson")],
                       horizon=10_000, num_seeds=30),
}


def _instance_job(spec_dict: Dict[str, Any], m: int, n: int, kappa: float, seed: int):
    spec = SweepSpec.from_dict(spec_dict)
    rows = []
    for cfg in spec.configs_for(m, n, kappa):
        rows.append((cfg.label, run_seed(cfg, seed)))
    return rows


def run_sweep(spec: SweepSpec, out_dir: str, workers: Optional[int] = None, verbose: bool = True) -> List[Dict[str, Any]]:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "spec.json"), "w", encoding="utf-8") as fh:
        json.dump(spec.to_dict(), fh, indent=2)

    jobs = [(m, n, k, s) for m in spec.ms for n in spec.ns for k in spec.kappas for s in spec.seeds]
    configs = {c.label: c for c in spec.all_configs()}
    per_label: Dict[str, List[SeedResult]] = {lbl: [] for lbl in configs}
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    spec_dict = spec.to_dict()

    def _collect(rows):
        for lbl, res in rows:
            per_label[lbl].append(res)

    if workers == 1:
        for idx, job in enumerate(jobs, 1):
            _collect(_instance_job(spec_dict, *job))
            if verbose and idx % max(1, len(jobs) // 20) == 0:
                print(f"  [{idx}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_instance_job, spec_dict, *job) for job in jobs]
            for idx, fut in enumerate(as_completed(futs), 1):
                _collect(fut.result())
                if verbose and idx % max(1, len(jobs) // 20) == 0:
                    print(f"  [{idx}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)

    aggs, seed_rows = [], []
    for lbl, cfg in configs.items():
        results = sorted(per_label[lbl], key=lambda r: r.seed)
        aggs.append(aggregate(cfg, results))
        seed_rows += [{**_cfg_row(cfg), "seed": r.seed, **r.summary} for r in results]
    write_per_seed(out_dir, seed_rows)
    write_outputs(aggs, out_dir)
    if verbose:
        print(f"Sweep finished: {len(configs)} configurations x {len(spec.seeds)} seeds in {time.time() - t0:.0f}s")
    return aggs


# --------------------------------------------------------------------------- output

CFG_COLS = ("scenario", "learner", "round_structure", "m", "n", "kappa", "horizon")


def _cfg_row(cfg: ExperimentConfig) -> Dict[str, Any]:
    d = cfg.to_dict()
    return {k: d[k] for k in CFG_COLS}


def write_per_seed(out_dir: str, rows: List[Dict[str, Any]]) -> None:
    header: List[str] = list(CFG_COLS) + ["seed"]
    for r in rows:
        for k in r:
            if k not in header:
                header.append(k)
    with open(os.path.join(out_dir, "per_seed.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def write_outputs(aggs: List[Dict[str, Any]], out_dir: str) -> None:
    with open(os.path.join(out_dir, "aggregates.json"), "w", encoding="utf-8") as fh:
        json.dump(aggs, fh)
    metric_keys: List[str] = []
    for a in aggs:
        for k in a["summary"]:
            if k not in metric_keys:
                metric_keys.append(k)
    header = list(CFG_COLS) + ["n_used", "n_excluded"]
    for k in metric_keys:
        header += [k, f"{k}_ci"]
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for a in aggs:
            row = {k: a["config"][k] for k in CFG_COLS}
            row.update(n_used=a["n_used"], n_excluded=a["n_excluded"])
            for k in metric_keys:
                if k in a["summary"]:
                    row[k] = a["summary"][k]["mean"]
                    row[f"{k}_ci"] = a["summary"][k]["ci"]
            w.writerow(row)


def load_aggregates(out_dir: str) -> List[Dict[str, Any]]:
    agg_path = os.path.join(out_dir, "aggregates.json")
    if os.path.exists(agg_path):
        with open(agg_path, encoding="utf-8") as fh:
            return json.load(fh)
    csv_path = os.path.join(out_dir, "summary.csv")
    if os.path.exists(csv_path):
        aggs = []
        with open(csv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                cfg = {
                    "scenario": row["scenario"],
                    "learner": row["learner"],
                    "round_structure": row["round_structure"],
                    "m": int(row["m"]),
                    "n": int(row["n"]),
                    "kappa": float(row["kappa"]),
                    "horizon": int(row["horizon"]),
                    "click_value": 5.0,
                }
                who = "bstar" if row["scenario"] == "tier1" else row["learner"]
                label = (f"{row['scenario']}|{who}|{row['round_structure']}|m{row['m']}|n{row['n']}"
                         f"|k{float(row['kappa']):g}|T{row['horizon']}")
                summary = {}
                for k, v in row.items():
                    if k.endswith("_ci"):
                        base = k[:-3]
                        if base in row and row[base] != "":
                            try:
                                mean_val = float(row[base])
                                ci_val = float(v) if v != "" else 0.0
                                summary[base] = {"mean": mean_val, "ci": ci_val}
                            except ValueError:
                                pass
                aggs.append({
                    "label": label,
                    "config": cfg,
                    "n_seeds": int(row["n_used"]) + int(row["n_excluded"]),
                    "n_used": int(row["n_used"]),
                    "n_excluded": int(row["n_excluded"]),
                    "summary": summary,
                    "curves": {},
                })
        return aggs
    raise FileNotFoundError(f"Neither aggregates.json nor summary.csv found in {out_dir}")
