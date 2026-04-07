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
DAYS_PER_EPISODE = 3
K_FOLDS          = 5
BATTERY_CAPACITY = 10.0   # MWh
CHARGE_RATE      = 5.0    # MW
CYCLE_COST       = 6.25   # EUR per full cycle
PPO_TIMESTEPS    = 10_000

RAW_DATA_PATH  = os.path.join(os.path.dirname(__file__), "data/2025_minute.csv")
FOLD_SAVE_PATH = os.path.join(os.path.dirname(__file__), "data/")
PLOT_DIR       = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


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
    rewards are computed identically to the PPO evaluation.

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
            if env.action_masks()[action] == 0:
                action = 0
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1

        env_rewards.append(total_reward)

    return env_rewards, dp_rewards


def evaluate_ppo_on_episodes(model, episodes, days_per_episode, capacity, rate, cost):
    """
    Evaluate a trained MaskablePPO on every episode and return per-episode
    total rewards.
    """
    rewards = []

    for ep_df in episodes:
        env = ExtendedBatteryEnv(
            battery_capacity_mwh=capacity,
            charge_discharge_rate_mw=rate,
            all_data=ep_df,
            days_per_episode=days_per_episode,
            cycle_cost_eur=cost,
        )
        scaled_env = RobustScalingWrapper(env)

        obs, _ = scaled_env.reset()
        total_reward, done = 0.0, False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = scaled_env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

    return rewards


# ═══════════════════════════════════════════════════════════════════════
#  MAIN — CROSS-VALIDATION LOOP
# ═══════════════════════════════════════════════════════════════════════
def main():
    # ── 1. Load data & generate folds ──────────────────────────────────
    print("Loading data ...")
    cleaned_df = pd.read_pickle("../data/raw_elia_data_cleaned.pkl")
    all_data   = cleaned_df[["Datetime", "Imbalance Price"]]

    folds = generate_hv_block_k_folds(
        all_data,
        k_folds=K_FOLDS,
        days_per_episode=DAYS_PER_EPISODE,
        save_path=FOLD_SAVE_PATH,
    )
    print(f"Loaded {len(folds)} folds.\n")

    # ── 2. Per-fold: train PPO, evaluate both ──────────────────────────
    results = []

    for fold_idx, (train_df, val_episodes, test_episodes) in enumerate(folds):
        print(f"\n{'=' * 60}")
        print(f"  FOLD {fold_idx + 1} / {K_FOLDS}")
        print(f"{'=' * 60}")

        # Train PPO ────────────────────────────────────────────────────
        train_env = ExtendedBatteryEnv(
            battery_capacity_mwh=BATTERY_CAPACITY,
            charge_discharge_rate_mw=CHARGE_RATE,
            all_data=train_df,
            days_per_episode=DAYS_PER_EPISODE,
            cycle_cost_eur=CYCLE_COST,
        )

        ppo = PPO(policy="MlpPolicy", env=train_env, verbose=0)
        print(f"  Training PPO for {PPO_TIMESTEPS:,} steps ...")
        ppo.learn(total_timesteps=PPO_TIMESTEPS, progress_bar=True)

        # Oracle on validation episodes ────────────────────────────────
        oracle_env_rewards, oracle_dp_rewards = evaluate_oracle_on_episodes(
            val_episodes, DAYS_PER_EPISODE,
            BATTERY_CAPACITY, CHARGE_RATE, CYCLE_COST,
        )

        # PPO on validation episodes ───────────────────────────────────
        ppo_rewards = evaluate_ppo_on_episodes(
            ppo, val_episodes, DAYS_PER_EPISODE,
            BATTERY_CAPACITY, CHARGE_RATE, CYCLE_COST,
        )

        # Aggregate ────────────────────────────────────────────────────
        oracle_total = sum(oracle_env_rewards)
        ppo_total    = sum(ppo_rewards)
        regret       = oracle_total - ppo_total
        ratio        = ppo_total / oracle_total if oracle_total > 0 else float("nan")

        results.append({
            "fold":            fold_idx + 1,
            "oracle_total":    oracle_total,
            "oracle_dp_total": sum(oracle_dp_rewards),
            "ppo_total":       ppo_total,
            "ratio":           ratio,
            "regret":          regret,
            "n_val_episodes":  len(val_episodes),
            "oracle_per_ep":   oracle_env_rewards,
            "ppo_per_ep":      ppo_rewards,
        })

        print(f"  Oracle (env) : €{oracle_total:>10.2f}  ({len(val_episodes)} episodes)")
        print(f"  Oracle (DP)  : €{sum(oracle_dp_rewards):>10.2f}  (sanity check)")
        print(f"  PPO          : €{ppo_total:>10.2f}")
        print(f"  Ratio        : {ratio:>10.2%}")
        print(f"  Regret       : €{regret:>10.2f}")

    # ── 3. Results table ───────────────────────────────────────────────
    df_results = pd.DataFrame([
        {
            "Fold":               r["fold"],
            "Val Episodes":       r["n_val_episodes"],
            "Oracle (€)":         round(r["oracle_total"], 2),
            "PPO (€)":            round(r["ppo_total"], 2),
            "Regret (€)":         round(r["regret"], 2),
            "Ratio (PPO/Oracle)": round(r["ratio"], 4),
        }
        for r in results
    ])

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(df_results.to_string(index=False))

    mean_ratio = df_results["Ratio (PPO/Oracle)"].mean()
    std_ratio  = df_results["Ratio (PPO/Oracle)"].std()
    print(f"\nMean Optimality Ratio : {mean_ratio:.2%}  ±  {std_ratio:.2%}")
    print(f"Mean Regret           : €{df_results['Regret (€)'].mean():.2f}")

    # ── 4. Bar chart — Oracle vs PPO per fold ──────────────────────────
    x     = np.arange(K_FOLDS)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, df_results["Oracle (€)"], width,
           label="Oracle (upper bound)", color="steelblue")
    ax.bar(x + width / 2, df_results["PPO (€)"],    width,
           label="PPO", color="darkorange")

    ax.set_xlabel("Fold")
    ax.set_ylabel("Total Validation Reward (€)")
    ax.set_title("PPO vs Oracle — hv-Block Cross-Validation")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i+1}" for i in x])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i, r in enumerate(df_results["Ratio (PPO/Oracle)"]):
        bar_height = df_results["PPO (€)"].iloc[i]
        ax.annotate(
            f"{r:.0%}",
            xy=(x[i] + width / 2, max(bar_height, 0)),
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    plt.tight_layout()
    bar_path = os.path.join(PLOT_DIR, "ppo_vs_oracle_cv.png")
    plt.savefig(bar_path, dpi=150)
    print(f"\nSaved bar chart  → {bar_path}")
    plt.show()

    # ── 5. Scatter — per-episode Oracle vs PPO ─────────────────────────
    all_oracle, all_ppo, all_fold_labels = [], [], []
    for r in results:
        all_oracle.extend(r["oracle_per_ep"])
        all_ppo.extend(r["ppo_per_ep"])
        all_fold_labels.extend([f"Fold {r['fold']}"] * r["n_val_episodes"])

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = plt.cm.tab10(np.arange(K_FOLDS))
    fold_colors = {f"Fold {i+1}": colors[i] for i in range(K_FOLDS)}

    for label in sorted(set(all_fold_labels)):
        mask = [l == label for l in all_fold_labels]
        ax.scatter(
            np.array(all_oracle)[mask],
            np.array(all_ppo)[mask],
            label=label, alpha=0.7, s=40, color=fold_colors[label],
        )

    # Diagonal = perfect agent
    lo = min(min(all_oracle), min(all_ppo)) - 5
    hi = max(max(all_oracle), max(all_ppo)) + 5
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="PPO = Oracle")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel("Oracle Reward (€ / episode)")
    ax.set_ylabel("PPO Reward (€ / episode)")
    ax.set_title("Per-Episode Reward: PPO vs Oracle")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal")

    plt.tight_layout()
    scatter_path = os.path.join(PLOT_DIR, "ppo_vs_oracle_scatter.png")
    plt.savefig(scatter_path, dpi=150)
    print(f"Saved scatter    → {scatter_path}")
    plt.show()


if __name__ == "__main__":
    main()

