"""
Oracle Agent vs PPO — hv-Block Cross-Validation
================================================

Evaluates how close a PPO agent comes to the theoretical optimum across
K cross-validation folds.

Oracle (upper bound)
    A dynamic-programming agent with perfect foresight.  It knows every
    future settlement price and solves for the battery schedule that
    maximises total profit.  No learned model can beat this.

Key insight — why the quarter-level DP is exact:
    Within a 15-min quarter the reward is linear in the number of active
    minutes, so mixing charge/discharge inside one quarter is always
    dominated by a single all-or-nothing action.  The SoC discretises
    perfectly into units of ``charge_rate × 15/60`` MWh (1.25 MWh with the
    default 5 MW / 10 MWh setup → 9 SoC levels), meaning zero
    approximation error.

Comparison metrics
    Optimality Ratio  = PPO_reward / Oracle_reward
        Fraction of the available value captured.  Closer to 1 = better.
    Regret            = Oracle_reward − PPO_reward
        Absolute € left on the table.  More robust when oracle profit is small.

If the ratio is consistent across folds the model generalises well and the
folds are representative.  If it varies widely, certain price regimes are
harder for the agent.
"""

import sys, os

from stable_baselines3 import PPO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from Simulation.suite_simple_trading.pre_processing import clean_data
from Simulation.suite_simple_trading.data_splitting import generate_hv_block_k_folds
from Simulation.suite_simple_trading.model import ExtendedBatteryEnv
from Simulation.suite_simple_trading.observation_wrappers import RobustScalingWrapper
from Simulation.suite_simple_trading.agent_trainer import EpisodeRewardCallback

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
DAYS_PER_EPISODE = 4
K_FOLDS          = 5
BATTERY_CAPACITY = 10.0   # MWh
CHARGE_RATE      = 5.0    # MW
CYCLE_COST       = 6.25   # EUR per full cycle

RAW_DATA_PATH  = os.path.join(os.path.dirname(__file__), "../data/2025_minute.csv")
FOLD_SAVE_PATH = os.path.join(os.path.dirname(__file__), "../data/")
RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
#  ORACLE — BACKWARD DYNAMIC PROGRAMMING
# ═══════════════════════════════════════════════════════════════════════
def compute_oracle_actions(
    prices: np.ndarray,
    battery_capacity: float,
    charge_rate: float,
    cycle_cost: float,
):
    """
    Solve for the optimal action per 15-min quarter using backward DP.

    The DP state is (quarter_index, SoC_in_units) where one unit equals
    charge_rate × 15/60 MWh.  With the default 5 MW / 10 MWh setup
    this gives exactly 9 SoC levels — no approximation.

    Parameters
    ----------
    prices : np.ndarray
        Minute-level imbalance prices for the full episode.
    battery_capacity, charge_rate, cycle_cost : float
        Battery parameters (same meaning as in ``BaseBatteryEnv``).

    Returns
    -------
    actions : list[int]
        One action per quarter (0 = idle, 1 = charge, 2 = discharge).
    max_reward : float
        Theoretical maximum reward from the DP value table.
    """
    energy_per_q   = charge_rate * (15 / 60)                          # MWh
    n_soc          = int(round(battery_capacity / energy_per_q)) + 1  # levels
    marginal_cost  = cycle_cost / (2 * battery_capacity)              # EUR/MWh

    n_quarters = len(prices) // 15

    # Settlement price = price at the last minute of each quarter
    settle = np.array([prices[q * 15 + 14] for q in range(n_quarters)])

    # Value-to-go and policy tables
    dp     = np.zeros((n_quarters + 1, n_soc))
    policy = np.zeros((n_quarters, n_soc), dtype=int)

    # ── Backward pass ──────────────────────────────────────────────────
    for q in range(n_quarters - 1, -1, -1):
        p = settle[q]
        for s in range(n_soc):
            best_val = dp[q + 1, s]          # idle
            best_a   = 0

            if s + 1 < n_soc:                # charge
                r = energy_per_q * (-p - marginal_cost)
                v = r + dp[q + 1, s + 1]
                if v > best_val:
                    best_val, best_a = v, 1

            if s - 1 >= 0:                   # discharge
                r = energy_per_q * (p - marginal_cost)
                v = r + dp[q + 1, s - 1]
                if v > best_val:
                    best_val, best_a = v, 2

            dp[q, s]     = best_val
            policy[q, s] = best_a

    # ── Forward pass — extract optimal trajectory from SoC = 0 ─────────
    actions = []
    soc = 0
    for q in range(n_quarters):
        a = int(policy[q, soc])
        actions.append(a)
        if a == 1:
            soc += 1
        elif a == 2:
            soc -= 1

    return actions, float(dp[0, 0])


# ═══════════════════════════════════════════════════════════════════════
#  EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════════════════
def evaluate_oracle_on_episodes(episodes, days_per_episode, capacity, rate, cost):
    """
    Run the DP oracle on every episode *through the real environment* so
    rewards are computed identically to the agent evaluation.

    Returns (env_rewards, dp_rewards) — the latter is a sanity-check;
    both lists should contain nearly identical values.
    """
    env_rewards, dp_rewards = [], []

    for ep_df in episodes:
        prices = ep_df["Imbalance Price"].to_numpy()
        quarter_actions, dp_reward = compute_oracle_actions(
            prices, capacity, rate, cost,
        )
        dp_rewards.append(dp_reward)

        env = ExtendedBatteryEnv(
            battery_capacity_mwh=capacity,
            charge_discharge_rate_mw=rate,
            all_data=ep_df,
            days_per_episode=days_per_episode,
            cycle_cost_eur=cost,
        )
        obs, _ = env.reset()
        total_reward, done, step = 0.0, False, 0

        while not done:
            q = step // 15
            action = quarter_actions[q] if q < len(quarter_actions) else 0
            # Respect physical constraints via action masks
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1

        env_rewards.append(total_reward)

    return env_rewards, dp_rewards


# ═══════════════════════════════════════════════════════════════════════
#  MAIN — CALCULATE ORACLE BASELINE
# ═══════════════════════════════════════════════════════════════════════
def main():
    import glob

    # ── 1. Load folds ──────────────────────────────────────────────────
    print("Loading folds ...")
    # For now we'll just read from the existing fold_xx_test.pkl files
    # if they exist, or you can supply the folds generated by data_splitting.
    fold_files = sorted(glob.glob(os.path.join(FOLD_SAVE_PATH, "fold_*_val.pkl")))

    if not fold_files:
        print(f"No fold files found in {FOLD_SAVE_PATH}. Please generate them first.")
        return

    print(f"Loaded {len(fold_files)} test folds.\n")

    # ── 2. Per-fold: run oracle ─────────────────────────────────────────
    all_results = []

    for fold_idx, filepath in enumerate(fold_files):
        print(f"\n{'=' * 60}")
        print(f"  FOLD {fold_idx} ")
        print(f"{'=' * 60}")

        test_episodes = pd.read_pickle(filepath)

        # Oracle on test episodes ──────────────────────────────────────
        print(f"  Evaluating Oracle on {len(test_episodes)} validation episodes...")
        oracle_env_rewards, oracle_dp_rewards = evaluate_oracle_on_episodes(
            test_episodes, DAYS_PER_EPISODE,
            BATTERY_CAPACITY, CHARGE_RATE, CYCLE_COST,
        )

        for ep_idx, (env_r, dp_r) in enumerate(zip(oracle_env_rewards, oracle_dp_rewards)):
            all_results.append({
                "fold": fold_idx,
                "episode": ep_idx,
                "oracle_env_reward": env_r,
                "oracle_dp_reward": dp_r
            })

        print(f"  Oracle (env) Total : €{sum(oracle_env_rewards):>10.2f}")

    # ── 3. Save results ───────────────────────────────────────────────
    df_results = pd.DataFrame(all_results)

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(df_results.groupby("fold")[["oracle_env_reward"]].sum())

    csv_path = os.path.join(RESULTS_DIR, "oracle_baseline_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nOracle baseline results saved to {csv_path}")


def debug_first_episode():
    """
    Debugs the first episode of the first fold to trace exactly where the
    environment reward diverges from the Dynamic Programming reward.
    """
    import glob

    fold_files = sorted(glob.glob(os.path.join(FOLD_SAVE_PATH, "fold_*_val.pkl")))
    if not fold_files:
        print("No folds found.")
        return

    # Load only the first fold, first episode
    test_episodes = pd.read_pickle(fold_files[0])
    ep_df = test_episodes[5]

    prices = ep_df["Imbalance Price"].to_numpy()
    quarter_actions, dp_reward = compute_oracle_actions(
        prices, BATTERY_CAPACITY, CHARGE_RATE, CYCLE_COST
    )

    env = ExtendedBatteryEnv(
        battery_capacity_mwh=BATTERY_CAPACITY,
        charge_discharge_rate_mw=CHARGE_RATE,
        all_data=ep_df,
        days_per_episode=DAYS_PER_EPISODE,
        cycle_cost_eur=CYCLE_COST,
    )
    obs, _ = env.reset()

    print("\n" + "=" * 60)
    print(" DEBUGGING FOLD 0 | EPISODE 0 ")
    print("=" * 60)
    print(f"Theoretical DP Reward (Quarterly): €{dp_reward:.2f}")

    total_env_reward = 0.0
    done = False
    step = 0

    print(
        f"\n{'Minute':>6} | {'Q_idx':>5} | {'Action':>6} | {'Price':>8} | {'Env Reward':>10} | {'Cumulative Env':>15}")
    print("-" * 70)

    # Let's keep track of DP quarter rewards and Env quarter rewards
    energy_per_q = CHARGE_RATE * (15 / 60)
    marginal_cost = CYCLE_COST / (2 * BATTERY_CAPACITY)

    while not done:
        q = step // 15
        action = quarter_actions[q] if q < len(quarter_actions) else 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_env_reward += reward

        dt_minute = env.all_data['Datetime'].iloc[env.current_step - 1].minute

        # Calculate expected DP reward for the quarter when the quarter ends
        if (step % 15) == 14:
            p = prices[step]
            if action == 1:
                dp_q_reward = energy_per_q * (-p - marginal_cost)
            elif action == 2:
                dp_q_reward = energy_per_q * (p - marginal_cost)
            else:
                dp_q_reward = 0.0

            print(f"{step:>6} | {q:>5} | {action:>6} | {prices[step]:>8.2f} | Env: {reward:>8.4f} | DP: {dp_q_reward:>8.4f} | Diff: {(reward - dp_q_reward):>8.4f} | dt.min: {dt_minute}")

        # Always print if unexpected rewards happen exactly outside the 14th minute
        elif reward != 0.0:
            print(f"{step:>6} | {q:>5} | {action:>6} | {prices[step]:>8.2f} | {reward:>10.4f} | unexpected reward at dt.min: {dt_minute}")

        done = terminated or truncated
        step += 1

    print("-" * 70)
    print(f"Total Environment Reward (Minutely): €{total_env_reward:.2f}")
    print(f"Difference (DP - Env): €{dp_reward - total_env_reward:.2f}\n")

if __name__ == "__main__":
    debug_first_episode()
