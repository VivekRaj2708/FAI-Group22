import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


def compute_equilibrium_distance(
    joint_bids: List[List[int]],
    reference_bids: List[List[int]],
) -> float:
    """
    Computes L1 distance between the played joint bid profile b and reference profile b*:
    ||b - b*||_1 = sum_{i=1}^m sum_{j=1}^n |b_{ij} - b_{ij}^*|
    """
    dist = 0.0
    m = min(len(joint_bids), len(reference_bids))
    for i in range(m):
        n = min(len(joint_bids[i]), len(reference_bids[i]))
        for j in range(n):
            dist += abs(joint_bids[i][j] - reference_bids[i][j])
    return float(dist)


def compute_normalized_distance(
    joint_bids: List[List[int]],
    reference_bids: List[List[int]],
) -> float:
    """
    Computes per-user, per-advertiser average L1 distance:
    ||b - b*||_1 / (m * n)
    """
    m = min(len(joint_bids), len(reference_bids))
    if m == 0:
        return 0.0
    n = min(len(joint_bids[0]), len(reference_bids[0])) if m > 0 else 0
    if n == 0:
        return 0.0
    return compute_equilibrium_distance(joint_bids, reference_bids) / (m * n)


def compute_convergence_rate(
    gaps: List[float],
    burn_in_fraction: float = 0.1,
) -> Dict[str, float]:
    """
    Fits an empirical power law decay epsilon_t approx c * t^(-beta) via log-log regression.
    A positive beta indicates that the equilibrium gap decreases with round horizon t.

    Returns:
        dict with keys:
            'beta': decay exponent (slope in log-log space: -beta)
            'constant': scale factor c
            'r_squared': coefficient of determination R^2
            'is_converging': 1.0 if beta > 0 and r_squared > 0.1, else 0.0
    """
    T = len(gaps)
    start_t = max(2, int(T * burn_in_fraction))
    if T - start_t < 5:
        start_t = 2 if T > 2 else 1

    x_log: List[float] = []
    y_log: List[float] = []

    for t_idx in range(start_t, T):
        round_num = t_idx + 1
        gap_val = gaps[t_idx]
        # Treat tiny gaps as 1e-6 to avoid log(0)
        gap_val_pos = max(1e-6, gap_val)
        x_log.append(math.log(round_num))
        y_log.append(math.log(gap_val_pos))

    if len(x_log) < 3:
        return {"beta": 0.0, "constant": 0.0, "r_squared": 0.0, "is_converging": 0.0}

    x_arr = np.array(x_log)
    y_arr = np.array(y_log)
    n_pts = len(x_arr)

    # Simple linear regression: y = m*x + b where m = -beta
    x_mean = np.mean(x_arr)
    y_mean = np.mean(y_arr)

    ss_xx = np.sum((x_arr - x_mean) ** 2)
    ss_xy = np.sum((x_arr - x_mean) * (y_arr - y_mean))
    ss_yy = np.sum((y_arr - y_mean) ** 2)

    if ss_xx < 1e-12:
        return {"beta": 0.0, "constant": math.exp(y_mean), "r_squared": 0.0, "is_converging": 0.0}

    slope = float(ss_xy / ss_xx)
    intercept = float(y_mean - slope * x_mean)
    beta = -slope

    if ss_yy > 1e-12:
        r_squared = float((ss_xy ** 2) / (ss_xx * ss_yy))
    else:
        r_squared = 1.0 if abs(beta) < 1e-6 else 0.0

    is_converging = 1.0 if (beta > 0.05 and r_squared > 0.1) or (np.mean(gaps[-10:]) < 1e-4) else 0.0

    return {
        "beta": float(beta),
        "constant": float(math.exp(min(20.0, intercept))),
        "r_squared": float(max(0.0, min(1.0, r_squared))),
        "is_converging": is_converging,
    }


def detect_cycling_or_stability(
    bid_history: List[List[List[int]]],
    window_size: int = 50,
) -> Dict[str, Any]:
    """
    Analyzes the terminal window of joint bid profiles to detect:
    1. Convergence to a fixed profile ('converged_fixed_point')
    2. Periodic cycling among K profiles ('limit_cycle')
    3. Non-periodic or stochastic variation ('wandering_or_stochastic')
    """
    total_rounds = len(bid_history)
    if total_rounds == 0:
        return {
            "status": "empty",
            "unique_profiles_count": 0,
            "cycle_period": None,
            "most_frequent_profile": None,
            "profile_stability_fraction": 0.0,
        }

    w = min(total_rounds, window_size)
    window = bid_history[-w:]

    # Convert profiles to hashable tuples
    tuples = [tuple(tuple(b_i) for b_i in prof) for prof in window]
    freqs: Dict[Tuple[Tuple[int, ...], ...], int] = {}
    for t_prof in tuples:
        freqs[t_prof] = freqs.get(t_prof, 0) + 1

    unique_count = len(freqs)
    most_frequent_prof, max_freq = max(freqs.items(), key=lambda x: x[1])
    stability_fraction = max_freq / w

    # Check 1: Fixed point
    if unique_count == 1:
        return {
            "status": "converged_fixed_point",
            "unique_profiles_count": 1,
            "cycle_period": 1,
            "most_frequent_profile": [list(b) for b in most_frequent_prof],
            "profile_stability_fraction": 1.0,
        }

    # Check 2: Limit cycle with period k in [2, w // 2]
    detected_period = None
    for k in range(2, min(w // 2 + 1, 15)):
        # Check if window[t] == window[t - k] for all t in [k, w)
        is_periodic = True
        for idx in range(k, w):
            if tuples[idx] != tuples[idx - k]:
                is_periodic = False
                break
        if is_periodic:
            detected_period = k
            break

    if detected_period is not None:
        return {
            "status": "limit_cycle",
            "unique_profiles_count": unique_count,
            "cycle_period": detected_period,
            "most_frequent_profile": [list(b) for b in most_frequent_prof],
            "profile_stability_fraction": stability_fraction,
        }

    # High frequency of one profile
    if stability_fraction >= 0.8:
        status = "quasi_stable_fixed_point"
    else:
        status = "wandering_or_stochastic"

    return {
        "status": status,
        "unique_profiles_count": unique_count,
        "cycle_period": None,
        "most_frequent_profile": [list(b) for b in most_frequent_prof],
        "profile_stability_fraction": float(stability_fraction),
    }


def compute_time_to_convergence(
    gaps: List[float],
    threshold: float = 0.05,
    consecutive_rounds: int = 10,
) -> Optional[int]:
    """
    Finds the earliest round t where epsilon_tau <= threshold for all
    tau in [t, t + consecutive_rounds - 1].
    Returns 1-based round index, or None if never converged.
    """
    T = len(gaps)
    if T < consecutive_rounds:
        return None

    run_count = 0
    for t_idx in range(T):
        if gaps[t_idx] <= threshold:
            run_count += 1
            if run_count >= consecutive_rounds:
                # Return the start of this successful run (1-based)
                return t_idx - consecutive_rounds + 2
        else:
            run_count = 0

    return None


def compute_price_of_learning(
    tier1_advertiser_profits: List[float],
    tier1_platform_revenue: float,
    tier1_allocation_efficiency: float,
    tier1_expected_welfare: float,
    tier3_mean_profits: List[float],
    tier3_mean_revenue: float,
    tier3_mean_efficiency: float,
    tier3_mean_welfare: float,
) -> Dict[str, Any]:
    """
    Computes the Price of Learning (Delta_metric = Metric(Tier 1) - Metric(Tier 3))
    quantifying economic and allocative losses incurred due to incomplete information
    and multi-agent non-stationary learning dynamics.
    """
    m = min(len(tier1_advertiser_profits), len(tier3_mean_profits))
    per_agent_profit_loss = [
        float(tier1_advertiser_profits[i] - tier3_mean_profits[i])
        for i in range(m)
    ]
    total_profit_loss = float(sum(tier1_advertiser_profits) - sum(tier3_mean_profits))
    revenue_loss = float(tier1_platform_revenue - tier3_mean_revenue)
    efficiency_loss = float(tier1_allocation_efficiency - tier3_mean_efficiency)
    welfare_loss = float(tier1_expected_welfare - tier3_mean_welfare)

    return {
        "delta_total_profit": total_profit_loss,
        "delta_per_agent_profit": per_agent_profit_loss,
        "delta_platform_revenue": revenue_loss,
        "delta_allocation_efficiency": efficiency_loss,
        "delta_social_welfare": welfare_loss,
    }

