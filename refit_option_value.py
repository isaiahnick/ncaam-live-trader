#!/usr/bin/env python3
"""
refit_option_value.py — Recalibrate the option value formula from live data.

End-to-end pipeline:
  1. Measure market noise CONDITIONED ON PROBABILITY — σ(p), μ(p)
  2. Measure autocorrelation to determine decorrelation timescale
  3. Run backward induction with PROBABILITY-DEPENDENT NOISE
  3b. Diagnostic: show OV by probability bucket (visualize asymmetry)
  4. Fit the asymmetric closed-form formula to the BI output
  5. Print updated constants for live_trader.py
  6. Calibrate microstate haircut H(t, margin) for exit threshold

KEY INSIGHT (v3):
  The v2 approach tried to force asymmetry onto BI output that was symmetric,
  because the BI used a single σ for all probability levels. In reality:
  
  - At extreme p (near 0 or 1): market and model agree the game is decided.
    Independent noise is SMALL. Favorable sell opportunities are rare.
  - At p ≈ 0.50: genuine disagreement exists. Independent noise is LARGE.
    More sell opportunities arise from market overpricing.
  
  By measuring σ(p) empirically and feeding it into the BI, the backward
  induction ITSELF will produce asymmetric OV — OTM positions get less OV
  because the noise that would create profitable exits doesn't exist for them.

USAGE:
  python3 refit_option_value.py                        # Default db path
  python3 refit_option_value.py --db /path/to/db.db    # Custom db path
  python3 refit_option_value.py --apply                 # Auto-update live_trader.py
"""

import sqlite3
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
from collections import defaultdict
import math
import argparse
import time as timer
import re
import os


# ===== MODEL PARAMETERS (must match live_trader.py) =====
BASE_STD = 17.2
TOTAL_GAME_SEC = 2400
SCORE_MIN = -50
SCORE_MAX = 50
N_SCORES = SCORE_MAX - SCORE_MIN + 1

PREGAME_PROBS = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                           0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
N_PREGAME = len(PREGAME_PROBS)


# ===================================================================
# STEP 1: Measure market noise — CONDITIONED ON PROBABILITY
# ===================================================================
def measure_noise(conn):
    """Measure bid-model residual distribution from live_snapshots,
    both globally AND conditioned on model probability.
    
    Returns:
        global_mean, global_std, n_snapshots, noise_mean_fn, noise_std_fn
        
        noise_mean_fn(p) -> mean residual at probability p
        noise_std_fn(p)  -> std of residual at probability p
    """
    print("=" * 65)
    print("STEP 1: Measuring market noise (probability-conditioned)")
    print("=" * 65)

    # Pull both home and away data with the model probability
    rows = conn.execute("""
        SELECT
            home_yes_bid,
            live_home_prob * 100 as home_model_cents,
            live_home_prob,
            away_yes_bid,
            (1.0 - live_home_prob) * 100 as away_model_cents,
            (1.0 - live_home_prob) as away_model_prob,
            time_remaining_sec
        FROM live_snapshots
        WHERE home_yes_bid IS NOT NULL
          AND live_home_prob IS NOT NULL
          AND home_yes_bid > 0
          AND live_home_prob > 0.01 AND live_home_prob < 0.99
          AND time_remaining_sec >= 240
          AND game_status = 'in'
    """).fetchall()

    if len(rows) == 0:
        print("  ⚠️ No snapshot data found!")
        return None, None, 0, None, None

    # Build (prob, residual) pairs for both sides
    prob_resid_pairs = []
    
    for r in rows:
        home_bid, home_model, home_prob, away_bid, away_model, away_prob, time_rem = r
        
        # Home observation
        prob_resid_pairs.append((home_prob, home_bid - home_model))
        
        # Away observation (if valid)
        if away_bid is not None and away_bid > 0:
            prob_resid_pairs.append((away_prob, away_bid - away_model))
    
    prob_resid_pairs = np.array(prob_resid_pairs)
    all_probs = prob_resid_pairs[:, 0]
    all_resids = prob_resid_pairs[:, 1]
    
    # Global stats
    q25, q75 = np.percentile(all_resids, [25, 75])
    global_std = (q75 - q25) / 1.35
    global_mean = np.median(all_resids)

    print(f"  Snapshots: {len(rows):,} (in-game, ≥240s)")
    print(f"  Observations: {len(prob_resid_pairs):,}")
    print(f"  Global noise mean (median): {global_mean:+.2f}¢")
    print(f"  Global noise std (IQR):     {global_std:.2f}¢")

    # === CONDITIONAL NOISE BY PROBABILITY BUCKET ===
    bucket_edges = np.arange(0.05, 1.00, 0.05)
    bucket_midpoints = []
    bucket_means = []
    bucket_stds = []
    bucket_counts = []
    
    print(f"\n  === NOISE BY PROBABILITY BUCKET ===")
    print(f"  {'Bucket':>12} {'Count':>8} {'Mean':>7} {'Std(IQR)':>9} {'Std(raw)':>9}")
    print(f"  {'-' * 50}")
    
    for i in range(len(bucket_edges) + 1):
        if i == 0:
            lo = 0.0
        else:
            lo = bucket_edges[i-1]
        hi = bucket_edges[i] if i < len(bucket_edges) else 1.0
        
        mask = (all_probs >= lo) & (all_probs < hi)
        if mask.sum() < 50:
            continue
        
        bucket_resids = all_resids[mask]
        mid = (lo + hi) / 2
        
        bq25, bq75 = np.percentile(bucket_resids, [25, 75])
        b_std_iqr = (bq75 - bq25) / 1.35
        b_mean = np.median(bucket_resids)
        b_std_raw = np.std(bucket_resids)
        
        bucket_midpoints.append(mid)
        bucket_means.append(b_mean)
        bucket_stds.append(b_std_iqr)
        bucket_counts.append(mask.sum())
        
        print(f"  {lo:.2f}-{hi:.2f}  {mask.sum():>8,} {b_mean:+6.2f}¢ {b_std_iqr:>8.2f}¢ {b_std_raw:>8.2f}¢")
    
    bucket_midpoints = np.array(bucket_midpoints)
    bucket_means = np.array(bucket_means)
    bucket_stds = np.array(bucket_stds)
    
    # Build interpolation functions
    noise_mean_fn = interp1d(bucket_midpoints, bucket_means, kind='linear',
                              fill_value=(bucket_means[0], bucket_means[-1]),
                              bounds_error=False)
    noise_std_fn = interp1d(bucket_midpoints, bucket_stds, kind='linear',
                             fill_value=(bucket_stds[0], bucket_stds[-1]),
                             bounds_error=False)
    
    # Show interpolated values
    print(f"\n  === INTERPOLATED NOISE FUNCTION ===")
    print(f"  {'Prob':>6} {'μ(p)':>7} {'σ(p)':>7} {'σ ratio vs ATM':>15}")
    print(f"  {'-' * 40}")
    sigma_50 = float(noise_std_fn(0.50))
    for p in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        mu = float(noise_mean_fn(p))
        sigma = float(noise_std_fn(p))
        ratio = sigma / sigma_50 if sigma_50 > 0 else 0
        print(f"  {p:.2f}  {mu:+6.2f}¢ {sigma:5.2f}¢ {ratio:>14.2f}x")
    
    # Overpricing frequency
    print(f"\n  === HOW OFTEN DOES MARKET OVERPRICE? ===")
    for thresh in [0, 2, 4, 6, 8, 10]:
        pct = 100 * np.mean(all_resids > thresh)
        print(f"  bid > model + {thresh}¢: {pct:5.1f}% of snapshots")

    return global_mean, global_std, len(rows), noise_mean_fn, noise_std_fn


# ===================================================================
# STEP 2: Measure autocorrelation
# ===================================================================
def measure_autocorrelation(conn):
    """Measure noise autocorrelation to determine decorrelation timescale."""
    print(f"\n{'=' * 65}")
    print("STEP 2: Measuring autocorrelation")
    print("=" * 65)

    rows = conn.execute("""
        SELECT game_id, timestamp, home_yes_bid - (live_home_prob * 100) as residual
        FROM live_snapshots
        WHERE home_yes_bid IS NOT NULL AND live_home_prob IS NOT NULL
          AND live_home_prob > 0.01 AND live_home_prob < 0.99
          AND time_remaining_sec >= 240
          AND game_status = 'in'
        ORDER BY game_id, timestamp
    """).fetchall()

    games = defaultdict(list)
    for gid, ts, r in rows:
        games[gid].append(r)

    lags = [15, 30, 60, 90, 120, 180, 240, 300]
    autocorrs = {}

    for lag_sec in lags:
        corrs = []
        for gid, resids in games.items():
            r = np.array(resids)
            if len(r) > lag_sec + 10:
                c = np.corrcoef(r[:-lag_sec], r[lag_sec:])[0, 1]
                if not np.isnan(c):
                    corrs.append(c)
        if corrs:
            autocorrs[lag_sec] = np.mean(corrs)

    print(f"  Games: {len(games)}")
    print(f"\n  {'Lag':>6}  {'Autocorr':>9}")
    print(f"  {'-' * 20}")

    for lag in lags:
        if lag in autocorrs:
            ac = autocorrs[lag]
            print(f"  {lag:4d}s   {ac:8.3f}")

    dt = 120
    print(f"\n  DT={dt}s (hardcoded)")
    return dt, autocorrs


# ===================================================================
# STEP 3: Backward induction (with probability-dependent noise)
# ===================================================================
def kalshi_fee_cents(price_decimal):
    return min(0.07 * price_decimal * (1 - price_decimal) * 100, 1.75)


def model_win_probability(score_diff, time_remaining, pregame_margin):
    time_fraction = max(0.001, min(1.0, time_remaining / TOTAL_GAME_SEC))
    expected_remaining_edge = pregame_margin * time_fraction
    expected_final_margin = score_diff + expected_remaining_edge
    remaining_std = BASE_STD * math.sqrt(time_fraction)
    if remaining_std > 0.001:
        return norm.cdf(expected_final_margin / remaining_std)
    return 1.0 if expected_final_margin > 0 else 0.0


def compute_transition_probs(pregame_margin, dt, max_delta=10):
    mu_step = pregame_margin * (dt / TOTAL_GAME_SEC)
    sigma_step = BASE_STD * math.sqrt(dt / TOTAL_GAME_SEC)
    probs = {}
    total = 0.0
    for delta in range(-max_delta, max_delta + 1):
        p = norm.cdf(delta + 0.5, mu_step, sigma_step) - norm.cdf(delta - 0.5, mu_step, sigma_step)
        if p > 1e-12:
            probs[delta] = p
            total += p
    if 0 < total < 0.9999:
        for delta in probs:
            probs[delta] /= total
    return probs


def sell_premium_closed_form(mu_sell, continuation, noise_std):
    if noise_std < 0.001:
        return max(0, mu_sell - continuation)
    d = (mu_sell - continuation) / noise_std
    premium = (mu_sell - continuation) * norm.cdf(d) + noise_std * norm.pdf(d)
    return max(0, premium)


def run_backward_induction(noise_mean_fn, noise_std_fn, dt, global_mean=None, global_std=None):
    """Run backward induction with PROBABILITY-DEPENDENT noise.
    
    At each node, the noise σ and μ depend on the model probability at that state.
    This means OTM positions (p near 0) see less noise → fewer sell opportunities → less OV.
    
    Also runs with global (flat) noise for comparison.
    """
    print(f"\n{'=' * 65}")
    print("STEP 3: Backward induction (probability-dependent noise)")
    print("=" * 65)

    n_steps = TOTAL_GAME_SEC // dt
    
    print(f"  DT={dt}s | Steps={n_steps}")
    print(f"\n  Noise at key probabilities:")
    for p in [0.10, 0.25, 0.50, 0.75, 0.90]:
        mu = float(noise_mean_fn(p))
        sigma = float(noise_std_fn(p))
        print(f"    p={p:.2f}: μ={mu:+.2f}¢, σ={sigma:.2f}¢")

    start = timer.time()
    pregame_margins = np.array([
        norm.ppf(max(0.001, min(0.999, p))) * BASE_STD for p in PREGAME_PROBS
    ])

    # === RUN 1: Probability-dependent noise ===
    V = np.zeros((N_SCORES, n_steps + 1, N_PREGAME))
    OV_cond = np.zeros((N_SCORES, n_steps + 1, N_PREGAME))

    for s_idx in range(N_SCORES):
        sd = s_idx + SCORE_MIN
        if sd > 0:
            V[s_idx, 0, :] = 100.0
        elif sd == 0:
            V[s_idx, 0, :] = 50.0

    all_trans = [compute_transition_probs(m, dt) for m in pregame_margins]

    for t_idx in range(1, n_steps + 1):
        time_remaining = t_idx * dt
        for m_idx in range(N_PREGAME):
            trans = all_trans[m_idx]
            margin = pregame_margins[m_idx]
            for s_idx in range(N_SCORES):
                score_diff = s_idx + SCORE_MIN
                cont = 0.0
                for delta, p in trans.items():
                    ns = max(0, min(N_SCORES - 1, s_idx + delta))
                    cont += p * V[ns, t_idx - 1, m_idx]

                prob = model_win_probability(score_diff, time_remaining, margin)
                prob_clipped = max(0.01, min(0.99, prob))
                avg_fee = kalshi_fee_cents(prob_clipped)
                
                # Probability-dependent noise
                mu = float(noise_mean_fn(prob_clipped))
                sigma = float(noise_std_fn(prob_clipped))
                mu_sell = prob * 100 + mu - avg_fee

                premium = sell_premium_closed_form(mu_sell, cont, sigma)
                V[s_idx, t_idx, m_idx] = cont + premium
                OV_cond[s_idx, t_idx, m_idx] = V[s_idx, t_idx, m_idx] - prob * 100

    elapsed_cond = timer.time() - start
    print(f"\n  Conditional noise BI: {elapsed_cond:.1f}s ({N_SCORES * (n_steps + 1) * N_PREGAME:,} nodes)")

    # === RUN 2: Flat noise (for comparison) ===
    OV_flat = None
    if global_mean is not None and global_std is not None:
        start2 = timer.time()
        V2 = np.zeros((N_SCORES, n_steps + 1, N_PREGAME))
        OV_flat = np.zeros((N_SCORES, n_steps + 1, N_PREGAME))
        
        for s_idx in range(N_SCORES):
            sd = s_idx + SCORE_MIN
            if sd > 0:
                V2[s_idx, 0, :] = 100.0
            elif sd == 0:
                V2[s_idx, 0, :] = 50.0
        
        for t_idx in range(1, n_steps + 1):
            time_remaining = t_idx * dt
            for m_idx in range(N_PREGAME):
                trans = all_trans[m_idx]
                margin = pregame_margins[m_idx]
                for s_idx in range(N_SCORES):
                    score_diff = s_idx + SCORE_MIN
                    cont = 0.0
                    for delta, p in trans.items():
                        ns = max(0, min(N_SCORES - 1, s_idx + delta))
                        cont += p * V2[ns, t_idx - 1, m_idx]
                    
                    prob = model_win_probability(score_diff, time_remaining, margin)
                    avg_fee = kalshi_fee_cents(max(0.01, min(0.99, prob)))
                    mu_sell = prob * 100 + global_mean - avg_fee
                    premium = sell_premium_closed_form(mu_sell, cont, global_std)
                    V2[s_idx, t_idx, m_idx] = cont + premium
                    OV_flat[s_idx, t_idx, m_idx] = V2[s_idx, t_idx, m_idx] - prob * 100
        
        elapsed_flat = timer.time() - start2
        print(f"  Flat noise BI:       {elapsed_flat:.1f}s (for comparison)")

    return OV_cond, OV_flat, pregame_margins, n_steps, dt


# ===================================================================
# STEP 3b: Diagnostic — OV by probability bucket
# ===================================================================
def diagnose_ov_asymmetry(OV_cond, OV_flat, pregame_margins, n_steps, dt):
    """Compare conditional vs flat noise BI output."""
    print(f"\n{'=' * 65}")
    print("STEP 3b: OV asymmetry diagnostic")
    print("=" * 65)
    
    prob_buckets = [
        (0.00, 0.15, "p<15%  (deep OTM)"),
        (0.15, 0.25, "15-25% (OTM)"),
        (0.25, 0.35, "25-35% (slight OTM)"),
        (0.35, 0.50, "35-50% (near ATM)"),
        (0.50, 0.65, "50-65% (near ATM)"),
        (0.65, 0.75, "65-75% (slight ITM)"),
        (0.75, 0.85, "75-85% (ITM)"),
        (0.85, 1.00, "p>85%  (deep ITM)"),
    ]
    
    time_slices = [2, 5, 10, 20, 30]

    for OV_arr, title in [(OV_cond, "CONDITIONAL NOISE"), (OV_flat, "FLAT NOISE (old)")]:
        if OV_arr is None:
            continue
        print(f"\n  Average OV (¢) — {title}:")
        print(f"  {'Bucket':<22}", end="")
        for t in time_slices:
            print(f" {t:2d}:00", end="")
        print()
        print(f"  {'-' * 52}")
        
        for p_lo, p_hi, label in prob_buckets:
            print(f"  {label:<22}", end="")
            for t_min in time_slices:
                t_sec = t_min * 60
                t_idx = min(t_sec // dt, n_steps)
                
                ovs = []
                for s_idx in range(N_SCORES):
                    score = s_idx + SCORE_MIN
                    for m_idx in range(N_PREGAME):
                        margin = pregame_margins[m_idx]
                        prob = model_win_probability(score, t_sec, margin)
                        if p_lo <= prob < p_hi:
                            ovs.append(OV_arr[s_idx, t_idx, m_idx])
                
                if ovs:
                    print(f" {np.mean(ovs):5.2f}", end="")
                else:
                    print(f"    --", end="")
            print()
    
    # ITM/OTM ratio comparison
    print(f"\n  === ITM/OTM RATIO COMPARISON ===")
    print(f"  {'Pair':<25} {'Type':<12}", end="")
    for t in time_slices:
        print(f" {t:2d}:00", end="")
    print()
    print(f"  {'-' * 65}")
    
    sym_pairs = [
        (0.80, 0.20, "p=0.80 vs p=0.20"),
        (0.70, 0.30, "p=0.70 vs p=0.30"),
        (0.60, 0.40, "p=0.60 vs p=0.40"),
    ]
    
    for OV_arr, label_type in [(OV_cond, "Conditional"), (OV_flat, "Flat")]:
        if OV_arr is None:
            continue
        for p_itm, p_otm, pair_label in sym_pairs:
            print(f"  {pair_label:<25} {label_type:<12}", end="")
            for t_min in time_slices:
                t_sec = t_min * 60
                t_idx = min(t_sec // dt, n_steps)
                
                itm_ovs = []
                otm_ovs = []
                for s_idx in range(N_SCORES):
                    score = s_idx + SCORE_MIN
                    for m_idx in range(N_PREGAME):
                        margin = pregame_margins[m_idx]
                        prob = model_win_probability(score, t_sec, margin)
                        if abs(prob - p_itm) < 0.03:
                            itm_ovs.append(OV_arr[s_idx, t_idx, m_idx])
                        elif abs(prob - p_otm) < 0.03:
                            otm_ovs.append(OV_arr[s_idx, t_idx, m_idx])
                
                if itm_ovs and otm_ovs:
                    ratio = np.mean(itm_ovs) / max(np.mean(otm_ovs), 0.01)
                    print(f" {ratio:5.2f}", end="")
                else:
                    print(f"    --", end="")
            print()


# ===================================================================
# STEP 4: Fit formula to conditional-noise BI
# ===================================================================
def fit_formula(OV_cond, OV_flat, pregame_margins, n_steps, dt):
    """Fit OV = FSP(p) + scale * N^exp * p^gamma to conditional-noise BI."""
    print(f"\n{'=' * 65}")
    print("STEP 4: Fitting formulas")
    print("=" * 65)

    # Collect points from CONDITIONAL BI
    points = []
    for s_idx in range(N_SCORES):
        score = s_idx + SCORE_MIN
        for t_idx in range(1, n_steps + 1):
            time_sec = t_idx * dt
            for m_idx in range(N_PREGAME):
                margin = pregame_margins[m_idx]
                ov = OV_cond[s_idx, t_idx, m_idx]
                prob = model_win_probability(score, time_sec, margin)
                points.append((time_sec, prob, ov))

    points = np.array(points)
    times = points[:, 0]
    probs = points[:, 1]
    ovs = points[:, 2]
    N_ind = times / dt
    probs_clipped = np.clip(probs, 0.01, 0.99)
    
    # Compute FSP
    fsps = np.array([kalshi_fee_cents(p) for p in probs_clipped])
    cv_targets = np.maximum(ovs - fsps, 0.0)
    
    print(f"  Points: {len(points):,}")
    print(f"  FSP range: {fsps.min():.3f}¢ — {fsps.max():.3f}¢ (mean {fsps.mean():.3f}¢)")
    print(f"  CV target range: {cv_targets.min():.3f}¢ — {cv_targets.max():.3f}¢")
    
    # --- FIT: Asymmetric formula to CONDITIONAL BI ---
    def objective_asym(params):
        scale, exp, gamma = params
        pred = scale * np.power(N_ind, exp) * np.power(probs_clipped, gamma)
        return np.mean((pred - cv_targets) ** 2)
    
    bounds = [(0.1, 10.0), (0.1, 0.9), (0.01, 3.0)]
    de_result = differential_evolution(objective_asym, bounds, seed=42,
                                        maxiter=500, tol=1e-10, polish=False)
    result = minimize(objective_asym, de_result.x, method='Nelder-Mead',
                      options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-12})
    scale, exp, gamma = result.x
    
    pred_asym = fsps + scale * np.power(N_ind, exp) * np.power(probs_clipped, gamma)
    errors_asym = np.abs(pred_asym - ovs)
    rmse_asym = np.sqrt(np.mean(errors_asym ** 2))
    
    print(f"\n  ── NEW: Asymmetric formula fit to CONDITIONAL BI ──")
    print(f"  OV = FSP(p) + {scale:.4f} × N^{exp:.4f} × p^{gamma:.4f}")
    print(f"  where FSP(p) = min(7 × p(1-p), 1.75)")
    print(f"  RMSE:  {rmse_asym:.4f}¢")
    print(f"  Max:   {errors_asym.max():.3f}¢")
    print(f"  95%:   {np.percentile(errors_asym, 95):.3f}¢")
    print(f"  99%:   {np.percentile(errors_asym, 99):.3f}¢")
    
    # --- Also fit old symmetric formula to CONDITIONAL BI ---
    def objective_sym(params):
        a, b, c = params
        pred = a * np.power(N_ind, b) * (1 + c * probs * (1 - probs))
        return np.mean((pred - ovs) ** 2)
    
    result_sym = minimize(objective_sym, [1.5, 0.45, -0.7], method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-10})
    a_sym, b_sym, c_sym = result_sym.x
    pred_sym = a_sym * np.power(N_ind, b_sym) * (1 + c_sym * probs * (1 - probs))
    errors_sym = np.abs(pred_sym - ovs)
    rmse_sym = np.sqrt(np.mean(errors_sym ** 2))
    
    print(f"\n  ── OLD: Symmetric formula fit to CONDITIONAL BI ──")
    print(f"  OV = {a_sym:.4f} × N^{b_sym:.4f} × (1 + {c_sym:.4f} × p(1-p))")
    print(f"  RMSE:  {rmse_sym:.4f}¢")
    print(f"  Max:   {errors_sym.max():.3f}¢")
    print(f"  95%:   {np.percentile(errors_sym, 95):.3f}¢")
    print(f"  99%:   {np.percentile(errors_sym, 99):.3f}¢")
    
    # --- RMSE BY PROBABILITY BUCKET ---
    print(f"\n  ── RMSE BY PROBABILITY BUCKET (both fit to conditional BI) ──")
    print(f"  {'Bucket':<22} {'Symmetric':>9} {'Asymmetric':>10} {'Δ':>7}  {'Better':>6}")
    print(f"  {'-' * 60}")
    
    buckets = [
        (0.00, 0.20, "p < 20% (deep OTM)"),
        (0.20, 0.35, "20-35% (OTM)"),
        (0.35, 0.50, "35-50% (slight OTM)"),
        (0.50, 0.65, "50-65% (slight ITM)"),
        (0.65, 0.80, "65-80% (ITM)"),
        (0.80, 1.00, "p > 80% (deep ITM)"),
    ]
    
    for p_lo, p_hi, label in buckets:
        mask = (probs >= p_lo) & (probs < p_hi)
        if mask.sum() == 0:
            continue
        rmse_b_sym = np.sqrt(np.mean(errors_sym[mask] ** 2))
        rmse_b_asym = np.sqrt(np.mean(errors_asym[mask] ** 2))
        delta = rmse_b_asym - rmse_b_sym
        better = "ASYM ✓" if delta < 0 else "sym"
        print(f"  {label:<22} {rmse_b_sym:7.3f}¢ {rmse_b_asym:9.3f}¢ {delta:+6.3f}¢  {better}")
    
    # --- HEAD-TO-HEAD SCENARIOS ---
    print(f"\n  ── HEAD-TO-HEAD: OV predictions (¢) against conditional BI ──")
    print(f"  {'Scenario':<35} {'BI':>6} {'Sym':>6} {'Asym':>6} {'SymErr':>7} {'AsymErr':>8}")
    print(f"  {'-' * 75}")
    
    p60_idx = list(PREGAME_PROBS).index(0.60)
    p80_idx = list(PREGAME_PROBS).index(0.80)
    p20_idx = list(PREGAME_PROBS).index(0.20)
    
    scenarios = [
        ("Losing: s=0, t=20:00, pre=20%",   0, 20, p20_idx),
        ("Losing: s=-10, t=10:00, pre=60%", -10, 10, p60_idx),
        ("Losing: s=-5, t=20:00, pre=60%",  -5, 20, p60_idx),
        ("Even: s=0, t=20:00, pre=50%",      0, 20, list(PREGAME_PROBS).index(0.50)),
        ("Even: s=0, t=10:00, pre=60%",      0, 10, p60_idx),
        ("Winning: s=+5, t=20:00, pre=60%", +5, 20, p60_idx),
        ("Winning: s=+10, t=10:00, pre=60%",+10, 10, p60_idx),
        ("Winning: s=0, t=20:00, pre=80%",   0, 20, p80_idx),
    ]
    
    for label, sd, t_min, m_idx in scenarios:
        t_sec = t_min * 60
        t_idx = min(t_sec // dt, n_steps)
        s_idx = sd - SCORE_MIN
        margin = pregame_margins[m_idx]
        
        bi_ov = OV_cond[s_idx, t_idx, m_idx]
        prob = model_win_probability(sd, t_sec, margin)
        p_clip = max(0.01, min(0.99, prob))
        ni = t_sec / dt
        
        sym_ov = a_sym * ni ** b_sym * (1 + c_sym * prob * (1 - prob))
        
        fsp = kalshi_fee_cents(p_clip)
        asym_ov = fsp + scale * ni ** exp * p_clip ** gamma
        
        sym_err = sym_ov - bi_ov
        asym_err = asym_ov - bi_ov
        
        print(f"  {label:<35} {bi_ov:5.2f} {sym_ov:5.2f} {asym_ov:5.2f} {sym_err:+6.2f} {asym_err:+7.2f}")
    
    # --- BEHAVIORAL IMPACT ---
    print(f"\n  ── BEHAVIORAL IMPACT: Min sell bid (¢) ──")
    print(f"  Lower = more willing to exit (cut losses faster)")
    print(f"  {'Scenario':<35} {'Model':>6} {'Sym':>6} {'Asym':>6} {'Diff':>6}")
    print(f"  {'-' * 65}")
    
    def compute_min_bid(ev_hold):
        if ev_hold >= 99:
            return 100.0
        bid = ev_hold
        for _ in range(5):
            fee = kalshi_fee_cents(bid / 100.0)
            bid = ev_hold + fee
        return min(99.0, max(1.0, bid))
    
    impact_scenarios = [
        ("OTM: p=15%, t=20:00", 0.15, 20),
        ("OTM: p=20%, t=15:00", 0.20, 15),
        ("OTM: p=25%, t=10:00", 0.25, 10),
        ("OTM: p=30%, t=5:00",  0.30, 5),
        ("ATM: p=50%, t=20:00", 0.50, 20),
        ("ATM: p=50%, t=10:00", 0.50, 10),
        ("ITM: p=70%, t=20:00", 0.70, 20),
        ("ITM: p=80%, t=10:00", 0.80, 10),
        ("ITM: p=85%, t=5:00",  0.85, 5),
    ]
    
    for label, prob, t_min in impact_scenarios:
        t_sec = t_min * 60
        ni = t_sec / dt
        p_clip = max(0.01, min(0.99, prob))
        
        sym_ov = a_sym * ni ** b_sym * (1 + c_sym * prob * (1 - prob))
        sym_ev = prob * 100 + sym_ov
        sym_bid = compute_min_bid(sym_ev)
        
        fsp = kalshi_fee_cents(p_clip)
        asym_ov = fsp + scale * ni ** exp * p_clip ** gamma
        asym_ev = prob * 100 + asym_ov
        asym_bid = compute_min_bid(asym_ev)
        
        diff = asym_bid - sym_bid
        
        print(f"  {label:<35} {prob*100:5.0f}¢ {sym_bid:5.1f}¢ {asym_bid:5.1f}¢ {diff:+5.1f}¢")
    
    return scale, exp, gamma, dt, rmse_asym, (a_sym, b_sym, c_sym, rmse_sym)


# ===================================================================
# STEP 6: Calibrate microstate haircut H(t, margin)
# ===================================================================
def calibrate_haircut(conn):
    """Calibrate exit haircut for unobserved microstate (possession, FT, fouls).
    
    Late in close games, market makers see microstate our model can't.
    This measures the EXCESS residual dispersion in (time, margin) buckets
    and fits a smooth function H(t, m) to add to ev_hold on exits.
    
    Returns:
        haircut_params: dict with best fit parameters, or None if insufficient data
    """
    print(f"\n{'=' * 65}")
    print("STEP 6: Calibrating microstate haircut H(t, margin)")
    print("=" * 65)
    
    # Pull residuals WITH score context (no time_remaining floor — we want late-game)
    rows = conn.execute("""
        SELECT
            time_remaining_sec,
            home_score,
            away_score,
            live_home_prob,
            home_yes_bid,
            away_yes_bid
        FROM live_snapshots
        WHERE home_yes_bid IS NOT NULL
          AND live_home_prob IS NOT NULL
          AND home_yes_bid > 0
          AND live_home_prob > 0.01 AND live_home_prob < 0.99
          AND game_status = 'in'
          AND time_remaining_sec > 0
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
    """).fetchall()
    
    if len(rows) < 1000:
        print(f"  ⚠️ Only {len(rows)} snapshots with score data — need more for haircut calibration")
        return None
    
    # Build observations: (time_remaining, abs_margin, residual)
    observations = []
    for time_rem, hs, as_, hp, hbid, abid in rows:
        abs_margin = abs(hs - as_)
        observations.append((time_rem, abs_margin, hbid - hp * 100))
        if abid is not None and abid > 0:
            observations.append((time_rem, abs_margin, abid - (1 - hp) * 100))
    
    data = np.array(observations)
    times = data[:, 0]
    margins = data[:, 1]
    resids = data[:, 2]
    
    print(f"  Observations: {len(data):,}")
    
    # === MARGIN-CONDITIONAL BASELINE ===
    # Use 20:00-40:00 (first half) dispersion for each margin bucket as baseline.
    # This removes persistent close-game model-market disagreement and isolates
    # the TIME-DEPENDENT microstate component (possession, FT, fouls).
    margin_edges = [0, 1, 2, 3, 5, 7, 10, 15, 50]
    baseline_sigmas = {}  # margin_bucket_idx -> sigma
    
    print(f"  === MARGIN-CONDITIONAL BASELINES (from 20:00-40:00) ===")
    print(f"  {'Margin':>10} {'Count':>7} {'σ_base':>8}")
    print(f"  {'-' * 30}")
    
    for j in range(len(margin_edges) - 1):
        m_lo, m_hi = margin_edges[j], margin_edges[j + 1]
        base_mask = (times >= 1200) & (margins >= m_lo) & (margins < m_hi)
        if base_mask.sum() >= 50:
            q25, q75 = np.percentile(resids[base_mask], [25, 75])
            baseline_sigmas[j] = (q75 - q25) / 1.349
        else:
            # Fallback: global early-game baseline
            safe_mask = (times > 720) & (margins > 7)
            q25, q75 = np.percentile(resids[safe_mask], [25, 75])
            baseline_sigmas[j] = (q75 - q25) / 1.349
        margin_label = f"|m|={m_lo}-{m_hi}"
        print(f"  {margin_label:>10} {base_mask.sum():>7,} {baseline_sigmas[j]:>7.2f}¢")
    
    # === BUCKET DISPERSION ===
    time_edges = [0, 60, 120, 180, 240, 360, 480, 720, 1200, 2400]
    
    bucket_data = []  # (t_mid, m_mid, excess_sigma, count)
    
    print(f"\n  {'Time':>14} {'Margin':>10} {'Count':>7} {'σ(IQR)':>8} {'σ_base':>8} {'Excess':>8}")
    print(f"  {'-' * 65}")
    
    for i in range(len(time_edges) - 1):
        t_lo, t_hi = time_edges[i], time_edges[i + 1]
        for j in range(len(margin_edges) - 1):
            m_lo, m_hi = margin_edges[j], margin_edges[j + 1]
            
            mask = (times >= t_lo) & (times < t_hi) & (margins >= m_lo) & (margins < m_hi)
            if mask.sum() < 30:
                continue
            
            bq25, bq75 = np.percentile(resids[mask], [25, 75])
            sigma = (bq75 - bq25) / 1.349
            baseline = baseline_sigmas.get(j, 2.65)
            excess = max(0, sigma - baseline)
            
            t_mid = (t_lo + t_hi) / 2
            m_mid = (m_lo + m_hi) / 2
            bucket_data.append((t_mid, m_mid, excess, mask.sum()))
            
            time_label = f"{t_lo//60}:{t_lo%60:02d}-{t_hi//60}:{t_hi%60:02d}"
            margin_label = f"|m|={m_lo}-{m_hi}"
            ex_str = f"{excess:.2f}¢" if excess > 0.3 else "  ---"
            print(f"  {time_label:>14} {margin_label:>10} {mask.sum():>7,} {sigma:>7.2f}¢ {baseline:>7.2f}¢ {ex_str:>8}")
    
    if len(bucket_data) < 5:
        print(f"  ⚠️ Not enough buckets for fitting")
        return None
    
    bd = np.array(bucket_data)
    t_arr = bd[:, 0]
    m_arr = bd[:, 1]
    excess_arr = bd[:, 2]
    weights = bd[:, 3]
    weights = weights / weights.sum()
    
    # === FIT: H = scale × exp(-t/tau) × exp(-m/decay) ===
    # Only fit to buckets where haircut matters (late + close)
    fit_mask = (t_arr < 600) & (m_arr < 12)
    if fit_mask.sum() < 4:
        print(f"  ⚠️ Not enough late-close buckets for fitting, using defaults")
        return {
            'scale': 4.0, 'tau': 90.0, 'decay': 2.5,
            'formula': 'exponential', 'rmse': None, 'baseline_sigmas': baseline_sigmas
        }
    
    t_fit = t_arr[fit_mask]
    m_fit = m_arr[fit_mask]
    ex_fit = excess_arr[fit_mask]
    w_fit = weights[fit_mask]
    w_fit = w_fit / w_fit.sum()
    
    def objective(params):
        scale, tau, decay = params
        pred = scale * np.exp(-t_fit / tau) * np.exp(-m_fit / decay)
        return np.sum(w_fit * (pred - ex_fit) ** 2)
    
    bounds = [(0.5, 20.0), (30.0, 300.0), (0.5, 10.0)]
    de_result = differential_evolution(objective, bounds, seed=42, maxiter=500)
    result = minimize(objective, de_result.x, method='Nelder-Mead',
                      options={'maxiter': 10000})
    scale, tau, decay = result.x
    
    pred = scale * np.exp(-t_fit / tau) * np.exp(-m_fit / decay)
    rmse = np.sqrt(np.mean((pred - ex_fit) ** 2))
    
    print(f"\n  === FIT: H = {scale:.2f} × exp(-t/{tau:.0f}) × exp(-|m|/{decay:.1f}) ===")
    print(f"  RMSE: {rmse:.3f}¢")
    
    # === Show haircut at key scenarios ===
    print(f"\n  === HAIRCUT VALUES (¢) ===")
    print(f"  {'Scenario':<35} {'H':>6}")
    print(f"  {'-' * 45}")
    
    scenarios = [
        ("t=0:30, |m|=0 (endgame tied)",     30, 0),
        ("t=0:30, |m|=1",                     30, 1),
        ("t=0:30, |m|=3",                     30, 3),
        ("t=1:00, |m|=0",                     60, 0),
        ("t=1:00, |m|=2",                     60, 2),
        ("t=2:00, |m|=0",                    120, 0),
        ("t=2:00, |m|=3",                    120, 3),
        ("t=3:00, |m|=0",                    180, 0),
        ("t=3:00, |m|=5",                    180, 5),
        ("t=4:00, |m|=0 (entry cutoff)",     240, 0),
        ("t=4:00, |m|=3",                    240, 3),
        ("t=8:00, |m|=0",                    480, 0),
        ("t=8:00, |m|=5",                    480, 5),
        ("t=15:00, |m|=0",                   900, 0),
        ("t=20:00, |m|=10",                 1200, 10),
    ]
    
    for label, t, m in scenarios:
        h = scale * math.exp(-t / tau) * math.exp(-m / decay)
        print(f"  {label:<35} {h:5.2f}¢")
    
    # Multiplier interpretation
    print(f"\n  === INTERPRETATION ===")
    print(f"  At k=0.5 (conservative): multiply H by 0.5")
    print(f"  At k=1.0 (full σ):       use H as-is")
    print(f"  At k=1.5 (aggressive):   multiply H by 1.5")
    print(f"  Recommend starting at k=1.0 (1 excess σ of uncertainty premium)")
    
    params_dict = {
        'scale': round(scale, 2),
        'tau': round(tau, 0),
        'decay': round(decay, 1),
        'formula': 'exponential',
        'rmse': rmse,
        'baseline_sigmas': baseline_sigmas,
    }
    
    print(f"\n  === CODE FOR live_trader.py ===")
    print(f"""
    # Microstate haircut (exits only):
    # H(t, m) = HAIRCUT_SCALE × exp(-t / HAIRCUT_TAU) × exp(-|m| / HAIRCUT_MARGIN_DECAY)
    HAIRCUT_SCALE = {params_dict['scale']}
    HAIRCUT_TAU = {params_dict['tau']:.0f}
    HAIRCUT_MARGIN_DECAY = {params_dict['decay']}

    def compute_haircut(self, time_remaining_sec: int, abs_margin: int) -> float:
        \"\"\"Microstate uncertainty premium (cents). Added to ev_hold on exits.
        Late in close games, MMs see possession/FT/fouls we can't.\"\"\"
        return self.HAIRCUT_SCALE * math.exp(-time_remaining_sec / self.HAIRCUT_TAU) * math.exp(-abs_margin / self.HAIRCUT_MARGIN_DECAY)
""")
    
    return params_dict


# ===================================================================
# STEP 5: Output
# ===================================================================
def print_results(scale, exponent, gamma, decorr_sec, global_mean, global_std, rmse, n_snapshots, old_params=None, haircut_params=None):
    """Print the final constants."""
    print(f"\n{'=' * 65}")
    print("RESULTS: Updated constants for live_trader.py")
    print("=" * 65)
    print(f"\n  # Option value: OV = FSP(p) + OV_SCALE × N^OV_EXPONENT × p^OV_GAMMA")
    print(f"  # where FSP(p) = min(7 × p(1-p), 1.75), N = time_remaining / OV_DECORR_SEC")
    print(f"  OV_SCALE = {scale:.2f}")
    print(f"  OV_EXPONENT = {exponent:.2f}")
    print(f"  OV_GAMMA = {gamma:.2f}")
    print(f"  OV_DECORR_SEC = {decorr_sec}")
    print(f"\n  Calibrated from {n_snapshots:,} snapshots (in-game, ≥240s)")
    print(f"  Global noise: μ={global_mean:+.2f}¢, σ={global_std:.2f}¢")
    print(f"  Formula RMSE: {rmse:.4f}¢")
    
    if old_params:
        a_old, b_old, c_old, rmse_old = old_params
        print(f"\n  Symmetric formula (same conditional BI): {a_old:.4f} × N^{b_old:.4f} × (1 + {c_old:.4f} × p(1-p))")
        print(f"  Symmetric RMSE: {rmse_old:.4f}¢")
        improvement = (rmse_old - rmse) / rmse_old * 100
        print(f"  Improvement: {improvement:+.1f}%")

    print(f"\n  Formula: OV = FSP(p) + {scale:.2f} × (t/{decorr_sec})^{exponent:.2f} × p^{gamma:.2f}")
    print(f"           where t = time_remaining in seconds")
    print(f"           p = model win prob for our side (clipped to [0.01, 0.99])")
    print(f"           FSP(p) = min(7 × p × (1-p), 1.75)")
    
    print(f"\n  === CODE FOR live_trader.py ===")
    print(f"""
    # Class constants:
    OV_SCALE = {scale:.2f}
    OV_EXPONENT = {exponent:.2f}
    OV_GAMMA = {gamma:.2f}
    OV_DECORR_SEC = {decorr_sec}

    def compute_option_value(self, time_remaining_sec: int, prob: float) -> float:
        \"\"\"Compute option value using asymmetric formula (calibrated from
        backward induction with probability-dependent market noise).
        
        OV = FSP(p) + OV_SCALE × N^OV_EXPONENT × p^OV_GAMMA
        
        FSP = free settlement premium (fee saved by holding to settlement)
        CV  = continuation value, weighted by p^gamma (moneyness)
              ITM positions (high p) get full CV; OTM positions get less,
              reflecting reduced independent noise at extreme probabilities.
        \"\"\"
        p_clip = max(0.01, min(0.99, prob))
        
        # Free settlement premium: fee saved by not selling early
        fsp = min(0.07 * p_clip * (1 - p_clip) * 100, 1.75)
        
        # Continuation value
        N = max(0, time_remaining_sec) / self.OV_DECORR_SEC
        if N <= 0:
            return fsp
        
        cv = self.OV_SCALE * (N ** self.OV_EXPONENT) * (p_clip ** self.OV_GAMMA)
        return fsp + cv
""")

    if haircut_params:
        hp = haircut_params
        print(f"\n  # Microstate haircut (exits only):")
        print(f"  # H(t, m) = {hp['scale']} × exp(-t/{hp['tau']:.0f}) × exp(-|m|/{hp['decay']})")
        print(f"  HAIRCUT_SCALE = {hp['scale']}")
        print(f"  HAIRCUT_TAU = {hp['tau']:.0f}")
        print(f"  HAIRCUT_MARGIN_DECAY = {hp['decay']}")
        if hp.get('rmse') is not None:
            print(f"  Haircut RMSE: {hp['rmse']:.3f}¢")


def apply_to_live_trader(scale, exponent, gamma, decorr_sec, live_trader_path, haircut_params=None):
    """Auto-update the constants in live_trader.py."""
    print(f"\n{'=' * 65}")
    print(f"Applying to {live_trader_path}")
    print("=" * 65)

    if not os.path.exists(live_trader_path):
        print(f"  ⚠️ File not found: {live_trader_path}")
        return False

    with open(live_trader_path, 'r') as f:
        content = f.read()

    replacements = [
        (r'OV_SCALE = [\d.]+', f'OV_SCALE = {scale:.2f}'),
        (r'OV_EXPONENT = [\d.]+', f'OV_EXPONENT = {exponent:.2f}'),
        (r'OV_DECORR_SEC = \d+', f'OV_DECORR_SEC = {decorr_sec}'),
    ]
    
    changes = 0
    for pattern, replacement in replacements:
        new_content, n = re.subn(pattern, replacement, content)
        if n > 0:
            content = new_content
            changes += n

    # Replace OV_PROB_COEFF with OV_GAMMA
    if 'OV_PROB_COEFF' in content:
        content, n = re.subn(r'OV_PROB_COEFF = -?[\d.]+', f'OV_GAMMA = {gamma:.2f}', content)
        if n > 0:
            changes += n
            print(f"  Renamed OV_PROB_COEFF → OV_GAMMA = {gamma:.2f}")
    elif 'OV_GAMMA' in content:
        content, n = re.subn(r'OV_GAMMA = [\d.]+', f'OV_GAMMA = {gamma:.2f}', content)
        if n > 0:
            changes += n
    
    # Update compute_option_value formula
    old_formula_pattern = (
        r'return self\.OV_SCALE \* \(N \*\* self\.OV_EXPONENT\) \* '
        r'\(1 \+ self\.OV_PROB_COEFF \* prob \* \(1 - prob\)\)'
    )
    new_formula = (
        'p_clip = max(0.01, min(0.99, prob))\n'
        '        fsp = min(0.07 * p_clip * (1 - p_clip) * 100, 1.75)\n'
        '        cv = self.OV_SCALE * (N ** self.OV_EXPONENT) * (p_clip ** self.OV_GAMMA)\n'
        '        return fsp + cv'
    )
    
    new_content, n = re.subn(old_formula_pattern, new_formula, content)
    if n > 0:
        content = new_content
        changes += n
        print(f"  Updated compute_option_value() formula")
    
    # === HAIRCUT CONSTANTS ===
    if haircut_params:
        hp = haircut_params
        haircut_replacements = [
            (r'HAIRCUT_SCALE = [\d.]+', f'HAIRCUT_SCALE = {hp["scale"]}'),
            (r'HAIRCUT_TAU = [\d.]+', f'HAIRCUT_TAU = {hp["tau"]:.0f}'),
            (r'HAIRCUT_MARGIN_DECAY = [\d.]+', f'HAIRCUT_MARGIN_DECAY = {hp["decay"]}'),
        ]
        for pattern, replacement in haircut_replacements:
            new_content, n = re.subn(pattern, replacement, content)
            if n > 0:
                content = new_content
                changes += n
        print(f"  Updated haircut constants: SCALE={hp['scale']}, TAU={hp['tau']:.0f}, DECAY={hp['decay']}")
    
    if changes > 0:
        with open(live_trader_path, 'w') as f:
            f.write(content)
        print(f"  ✓ Updated {changes} items total")
        print(f"\n  ⚠️  IMPORTANT: Manually verify compute_option_value() and")
        print(f"     compute_min_sell_bid() in live_trader.py after applying!")
        return True
    else:
        print(f"  ⚠️ No constants found to update")
        return False


# ===================================================================
# Main
# ===================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Refit option value formula from live data')
    parser.add_argument('--db', type=str, default='/root/ncaam-site/backend/compiled_stats.db')
    parser.add_argument('--apply', action='store_true',
                        help='Auto-update live_trader.py with new constants')
    parser.add_argument('--trader-path', type=str, default=None,
                        help='Path to live_trader.py (default: same dir as db)')
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)

    # Step 1: Measure noise (global + conditional on probability)
    global_mean, global_std, n_snapshots, noise_mean_fn, noise_std_fn = measure_noise(conn)
    if global_mean is None:
        print("\nAborted: no data.")
        exit(1)

    # Step 2: Autocorrelation → DT
    dt, autocorrs = measure_autocorrelation(conn)

    # Step 3: Backward induction (conditional + flat for comparison)
    OV_cond, OV_flat, margins, n_steps, dt = run_backward_induction(
        noise_mean_fn, noise_std_fn, dt,
        global_mean=global_mean, global_std=global_std
    )

    # Step 3b: Diagnostic — compare asymmetry
    diagnose_ov_asymmetry(OV_cond, OV_flat, margins, n_steps, dt)
    
    # Step 4: Fit formulas to conditional BI
    scale, exponent, gamma, decorr_sec, rmse, old_params = fit_formula(
        OV_cond, OV_flat, margins, n_steps, dt
    )

    # Step 5: Output
    # Step 6: Calibrate microstate haircut (needs conn still open)
    haircut_params = calibrate_haircut(conn)
    conn.close()

    print_results(scale, exponent, gamma, decorr_sec, global_mean, global_std, rmse, n_snapshots, old_params, haircut_params)

    # Optional: auto-apply
    if args.apply:
        trader_path = args.trader_path
        if trader_path is None:
            trader_path = os.path.join(os.path.dirname(args.db), 'live_trader.py')
        apply_to_live_trader(scale, exponent, gamma, decorr_sec, trader_path, haircut_params)