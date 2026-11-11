"""
Phase 4/5 evaluation harness: repeated-auction simulator, metrics, across-seed
statistics, sensitivity sweeps, plots and report generation.
"""
from src.eval.simulator import ROUND_STRUCTURES, SimulationConfig, SimulationTrace, run_simulation
from src.eval.experiment import ExperimentConfig, SeedResult, aggregate, run_experiment, run_seed
from src.eval.sweep import PRESETS, SweepSpec, run_sweep

__all__ = [
    "ROUND_STRUCTURES",
    "SimulationConfig",
    "SimulationTrace",
    "run_simulation",
    "ExperimentConfig",
    "SeedResult",
    "aggregate",
    "run_experiment",
    "run_seed",
    "PRESETS",
    "SweepSpec",
    "run_sweep",
]
