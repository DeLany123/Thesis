"""
HPC Benchmark Script — Train & Evaluate a Single Agent Across K Folds
======================================================================

Designed to be submitted as a job on a High Performance Computing cluster.
Each fold is trained and evaluated in its own thread for parallelism.

Usage examples::

    python benchmark_agent.py --agent PPO --steps 500000
    python benchmark_agent.py --agent DQN --steps 1000000 --iterations 5
    python benchmark_agent.py --agent SAC --steps 250000 --folds-path /data/folds

Arguments:
    --agent         : One of {PPO, DQN, A2C, SAC, DDPG}
    --steps         : Total training timesteps per run
    --iterations    : Number of independent train-from-scratch runs per fold (default: 3)
    --folds-path    : Directory containing cached fold pickles (fold_0_train.pkl, …)
    --output-dir    : Directory to write result CSVs into (default: ./results)
    --k-folds       : Number of folds to load (default: 5)
    --days-per-ep   : Days per episode for the environment (default: 3)
    --battery-cap   : Battery capacity in MWh (default: 10.0)
    --charge-rate   : Charge/discharge rate in MW (default: 5.0)
    --cycle-cost    : Cycle degradation cost in EUR (default: 6.25)
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN, A2C, SAC, DDPG
from stable_baselines3.common.base_class import BaseAlgorithm


# ═══════════════════════════════════════════════════════════════════════
#  AGENT REGISTRY
# ═══════════════════════════════════════════════════════════════════════
# Maps CLI name → (SB3 class, policy name, needs continuous wrapper)
AGENT_REGISTRY: Dict[str, Tuple[type, str, bool]] = {
    "PPO":  (PPO,  "MlpPolicy", False),
    "DQN":  (DQN,  "MlpPolicy", False),
    "A2C":  (A2C,  "MlpPolicy", False),
    "SAC":  (SAC,  "MlpPolicy", True),
    "DDPG": (DDPG, "MlpPolicy", True),
}


# ═══════════════════════════════════════════════════════════════════════
#  ENVIRONMENT & WRAPPERS
# ═══════════════════════════════════════════════════════════════════════
class ExtendedBatteryEnv(gym.Env):
    """
    Placeholder — replace with your actual ExtendedBatteryEnv.

    Expected constructor signature::

        ExtendedBatteryEnv(
            battery_capacity_mwh: float,
            charge_discharge_rate_mw: float,
            all_data: pd.DataFrame,
            days_per_episode: int,
            cycle_cost_eur: float,
        )
    """
    raise NotImplementedError(
        "Replace this class with your real ExtendedBatteryEnv import or paste."
    )


class ContinuousActionWrapper(gym.ActionWrapper):
    """
    Wraps a Discrete(3) environment so that SAC / DDPG can interact with it.
    Maps a continuous scalar in [-1, 1] → {0, 1, 2}.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def action(self, continuous_action):
        if continuous_action < -0.33:
            return 2   # Discharge
        elif continuous_action > 0.33:
            return 1   # Charge
        else:
            return 0   # Idle


# ═══════════════════════════════════════════════════════════════════════
#  EVALUATION RESULT
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class EvaluationResult:
    """Minimal evaluation output returned by run_evaluation."""
    real_rewards: List[float]
    episodic_rewards: List[float]


# ═══════════════════════════════════════════════════════════════════════
#  EVALUATION LOOP  (fill in your own implementation)
# ═══════════════════════════════════════════════════════════════════════
def run_evaluation(
    env: gym.Env,
    model: BaseAlgorithm,
    number_of_episodes: int = 1,
    is_masked: bool = False,
) -> EvaluationResult:
    """
    Placeholder — replace with your actual run_evaluation.

    Should run *number_of_episodes* episodes inside *env* using *model*,
    collecting per-step rewards and per-episode totals.
    """
    raise NotImplementedError(
        "Replace this function with your real run_evaluation import or paste."
    )


# ═══════════════════════════════════════════════════════════════════════
#  FOLD LOADING
# ═══════════════════════════════════════════════════════════════════════
def load_folds(
    folds_path: str,
    k: int,
) -> List[Tuple[pd.DataFrame, List[pd.DataFrame], List[pd.DataFrame]]]:
    """
    Load K pre-computed folds from *folds_path*.

    Expected files per fold::

        fold_0_train.pkl   fold_0_val.pkl   fold_0_test.pkl
        fold_1_train.pkl   …
    """
    folds = []
    for i in range(k):
        train_p = os.path.join(folds_path, f"fold_{i}_train.pkl")
        val_p   = os.path.join(folds_path, f"fold_{i}_val.pkl")
        test_p  = os.path.join(folds_path, f"fold_{i}_test.pkl")

        for p in (train_p, val_p, test_p):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing fold file: {p}")

        train_df  = pd.read_pickle(train_p)
        val_eps   = pd.read_pickle(val_p)
        test_eps  = pd.read_pickle(test_p)

        folds.append((train_df, val_eps, test_eps))
        print(f"  Fold {i}: train={len(train_df):,} rows, "
              f"val={len(val_eps)} eps, test={len(test_eps)} eps")

    return folds


# ═══════════════════════════════════════════════════════════════════════
#  SINGLE-FOLD WORKER
# ═══════════════════════════════════════════════════════════════════════
def _run_fold(
    fold_idx: int,
    train_df: pd.DataFrame,
    val_episodes: List[pd.DataFrame],
    agent_class: type,
    policy_name: str,
    is_continuous: bool,
    total_steps: int,
    n_iterations: int,
    days_per_episode: int,
    battery_capacity: float,
    charge_rate: float,
    cycle_cost: float,
) -> Dict[str, Any]:
    """
    Train an agent *n_iterations* times from scratch on one fold and
    evaluate on its validation episodes each time.

    Returns a dict with aggregated statistics for this fold.
    """
    val_df_combined = pd.concat(val_episodes, ignore_index=True)

    raw_revenues: List[float] = []
    train_times: List[float] = []

    for run in range(1, n_iterations + 1):
        # ── Build training environment ────────────────────────────────
        train_env = ExtendedBatteryEnv(
            battery_capacity_mwh=battery_capacity,
            charge_discharge_rate_mw=charge_rate,
            all_data=train_df,
            days_per_episode=days_per_episode,
            cycle_cost_eur=cycle_cost,
        )
        if is_continuous:
            train_env = ContinuousActionWrapper(train_env)

        # ── Train ─────────────────────────────────────────────────────
        model = agent_class(policy_name, train_env, verbose=0)
        t0 = time.time()
        model.learn(total_timesteps=total_steps, reset_num_timesteps=True)
        t_train = time.time() - t0

        # ── Evaluate on validation set ────────────────────────────────
        val_env = ExtendedBatteryEnv(
            battery_capacity_mwh=battery_capacity,
            charge_discharge_rate_mw=charge_rate,
            all_data=val_df_combined,
            days_per_episode=days_per_episode,
            cycle_cost_eur=cycle_cost,
        )
        if is_continuous:
            val_env = ContinuousActionWrapper(val_env)

        result: EvaluationResult = run_evaluation(
            val_env, model,
            number_of_episodes=len(val_episodes),
            is_masked=False,
        )

        revenue = sum(result.real_rewards)
        raw_revenues.append(revenue)
        train_times.append(t_train)

        print(f"  [Fold {fold_idx}] Run {run}/{n_iterations} — "
              f"Revenue: €{revenue:,.2f}  (train {t_train:.1f}s)")

    return {
        "fold": fold_idx,
        "n_iterations": n_iterations,
        "mean_revenue": float(np.mean(raw_revenues)),
        "std_revenue": float(np.std(raw_revenues)),
        "min_revenue": float(np.min(raw_revenues)),
        "max_revenue": float(np.max(raw_revenues)),
        "all_revenues": raw_revenues,
        "mean_train_time_sec": float(np.mean(train_times)),
    }


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="HPC benchmark: train & evaluate one RL agent across K folds."
    )
    parser.add_argument(
        "--agent", type=str, required=True,
        choices=list(AGENT_REGISTRY.keys()),
        help="Agent algorithm to benchmark.",
    )
    parser.add_argument(
        "--steps", type=int, required=True,
        help="Total training timesteps per run.",
    )
    parser.add_argument(
        "--iterations", type=int, default=3,
        help="Independent train-from-scratch runs per fold (default: 3).",
    )
    parser.add_argument(
        "--folds-path", type=str,
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Directory containing fold_*_{train,val,test}.pkl files.",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="Directory to write result CSVs.",
    )
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--days-per-ep", type=int, default=3)
    parser.add_argument("--battery-cap", type=float, default=10.0)
    parser.add_argument("--charge-rate", type=float, default=5.0)
    parser.add_argument("--cycle-cost", type=float, default=6.25)

    args = parser.parse_args()

    agent_class, policy_name, is_continuous = AGENT_REGISTRY[args.agent]
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load folds ────────────────────────────────────────────────────
    print(f"Loading {args.k_folds} folds from '{args.folds_path}' …")
    folds = load_folds(args.folds_path, args.k_folds)

    # ── Launch one thread per fold ────────────────────────────────────
    print(f"\nBenchmarking {args.agent} | {args.steps:,} steps | "
          f"{args.iterations} iterations/fold | {args.k_folds} folds (parallel)\n")

    fold_results: List[Dict[str, Any]] = [None] * args.k_folds

    with ThreadPoolExecutor(max_workers=args.k_folds) as pool:
        future_to_fold = {}
        for fold_idx, (train_df, val_eps, test_eps) in enumerate(folds):
            future = pool.submit(
                _run_fold,
                fold_idx=fold_idx,
                train_df=train_df,
                val_episodes=val_eps,
                agent_class=agent_class,
                policy_name=policy_name,
                is_continuous=is_continuous,
                total_steps=args.steps,
                n_iterations=args.iterations,
                days_per_episode=args.days_per_ep,
                battery_capacity=args.battery_cap,
                charge_rate=args.charge_rate,
                cycle_cost=args.cycle_cost,
            )
            future_to_fold[future] = fold_idx

        for future in as_completed(future_to_fold):
            idx = future_to_fold[future]
            try:
                fold_results[idx] = future.result()
            except Exception as exc:
                print(f"  *** Fold {idx} raised an exception: {exc}")
                raise

    # ── Aggregate & save ──────────────────────────────────────────────
    rows = []
    for r in fold_results:
        rows.append({
            "agent": args.agent,
            "total_steps": args.steps,
            "fold": r["fold"],
            "iterations": r["n_iterations"],
            "mean_revenue": round(r["mean_revenue"], 4),
            "std_revenue": round(r["std_revenue"], 4),
            "min_revenue": round(r["min_revenue"], 4),
            "max_revenue": round(r["max_revenue"], 4),
            "mean_train_time_sec": round(r["mean_train_time_sec"], 2),
        })

    df_results = pd.DataFrame(rows)

    csv_name = f"{args.agent}_{args.steps}_results.csv"
    csv_path = os.path.join(args.output_dir, csv_name)
    df_results.to_csv(csv_path, index=False)

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print(f"  RESULTS — {args.agent} @ {args.steps:,} steps")
    print(f"{'=' * 64}")
    print(df_results.to_string(index=False))

    overall_mean = df_results["mean_revenue"].mean()
    overall_std  = df_results["mean_revenue"].std()
    print(f"\nCross-fold mean revenue : €{overall_mean:,.2f}  ± €{overall_std:,.2f}")
    print(f"Results saved to        : {csv_path}")


if __name__ == "__main__":
    main()

