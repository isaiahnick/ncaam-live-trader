#!/usr/bin/env python3
"""
calibrate_live_win_probability.py

Calibrate LIVE/IN-GAME win probability model against historical play-by-play data.
(This is separate from calibrate_win_probability.py which handles pregame predictions)

For each historical game:
1. Sample game states at various time points
2. Calculate model's win probability at each snapshot
3. Compare to actual outcome
4. Measure calibration and optimize base_std parameter

Usage:
    # Run optimization to find best base_std
    python3 calibrate_live_win_probability.py --db compiled_stats.db --optimize
    
    # Test specific base_std value
    python3 calibrate_live_win_probability.py --db compiled_stats.db --base-std 11.0
    
    # Adjust sample interval (default 120 seconds)
    python3 calibrate_live_win_probability.py --db compiled_stats.db --optimize --sample-interval 60
    
    # Optimize lead-dependent σ (reduces variance in blowouts)
    python3 calibrate_live_win_probability.py --db compiled_stats.db --optimize-lead-sigma
    
    # Optimize half-specific σ (separate σ for 1st/2nd half)
    python3 calibrate_live_win_probability.py --db compiled_stats.db --optimize-half-sigma
"""

import sqlite3
import math
from scipy.stats import norm
from collections import defaultdict
import argparse


def calculate_win_probability(pregame_win_prob, pregame_spread, score_diff, time_remaining_sec, 
                              base_std=11.0, momentum=0.0, momentum_factor=0.0, confidence_sensitivity=0.0,
                              lead_sigma_k=0.0, half_sigma=None):
    """
    Calculate win probability for team_a given current game state.
    
    Anchors to the calibrated pregame win probability, then adjusts based on
    current score and time remaining.
    
    The key insight: as the game progresses, the current score provides more
    information and the pregame prediction matters less.
    
    Args:
        pregame_win_prob: Calibrated pregame win probability for team_a (0 to 1)
        pregame_spread: Model's predicted margin (team_a perspective)
        score_diff: Current score differential (team_a - team_b)
        time_remaining_sec: Seconds remaining in game
        base_std: Full-game standard deviation (parameter to calibrate)
        momentum: Score change in last ~2 min from team_a perspective (positive = team_a on run)
        momentum_factor: How much to weight momentum (negative = anti-momentum/mean reversion)
        confidence_sensitivity: How fast to shrink pregame edge when prediction is wrong (0 = off)
            Higher = trust pregame longer, Lower = trust score faster
        lead_sigma_k: Lead-dependent σ adjustment (0 = off). Reduces σ when one team has a
            large lead, capturing pace slowdowns and reduced variance in blowouts.
            σ_eff = base_std - k × |score_diff| / √(possessions_remaining)
        half_sigma: Tuple of (first_half_std, second_half_std) for half-specific σ.
            If provided, overrides base_std based on which half the game is in.
            None = use base_std for both halves.
    
    Returns:
        Win probability for team_a (0.0 to 1.0)
    """
    total_game_sec = 40 * 60  # 40 minute game
    time_fraction = time_remaining_sec / total_game_sec
    time_fraction = max(0.001, min(1.0, time_fraction))  # Clamp to valid range
    
    # Half-specific σ override
    if half_sigma is not None:
        first_half_std, second_half_std = half_sigma
        base_std = first_half_std if time_remaining_sec > 1200 else second_half_std
    
    # Lead-dependent σ: reduce variance when one team has a large lead
    # Rationale: winning teams slow pace, trailing teams take rushed shots → less variance
    if lead_sigma_k > 0:
        poss_remaining = max(1.0, time_remaining_sec / 36.0)  # ~36 sec per possession
        norm_lead = abs(score_diff) / math.sqrt(poss_remaining)
        base_std = max(5.0, base_std - lead_sigma_k * norm_lead)  # Floor at 5
    
    # Convert pregame win prob back to "expected margin" space for blending
    # This ensures we anchor to the calibrated probability
    # Using inverse of normal CDF: if P(win) = p, then expected_margin = std * z
    # where z = norm.ppf(p)
    if pregame_win_prob >= 0.999:
        pregame_implied_margin = 3.5 * base_std  # Cap at ~3.5 std devs
    elif pregame_win_prob <= 0.001:
        pregame_implied_margin = -3.5 * base_std
    else:
        pregame_implied_margin = norm.ppf(pregame_win_prob) * base_std
    
    # Pregame edge decays linearly as game progresses
    # With 0 time remaining, pregame edge contributes nothing
    expected_remaining_edge = pregame_implied_margin * time_fraction
    
    # Bayesian confidence update: shrink pregame edge if prediction is wrong so far
    # If pregame said +8 and we're down 10, our pregame was wrong about this matchup
    if confidence_sensitivity > 0:
        elapsed_fraction = 1 - time_fraction
        if elapsed_fraction > 0.05:  # Only apply after some game has been played
            # What score diff did pregame predict by now?
            expected_score_diff_so_far = pregame_implied_margin * elapsed_fraction
            # How wrong were we?
            prediction_error = abs(score_diff - expected_score_diff_so_far)
            # Shrink confidence based on error magnitude
            confidence = 1 / (1 + prediction_error / (base_std * confidence_sensitivity))
            expected_remaining_edge = expected_remaining_edge * confidence
    
    # Expected final margin = current lead + expected edge from remaining time
    expected_final_margin = score_diff + expected_remaining_edge
    
    # Anti-momentum adjustment: runs tend to mean-revert
    # Effect scales with (1 - time_fraction) so it's stronger late game
    if momentum != 0 and momentum_factor != 0:
        momentum_weight = (1 - time_fraction) * momentum_factor
        expected_final_margin += momentum * momentum_weight
    
    # Remaining uncertainty decreases with sqrt of time (fewer possessions = less variance)
    remaining_std = base_std * math.sqrt(time_fraction)
    
    # Win probability from normal distribution: P(final_margin > 0)
    if remaining_std > 0.001:
        z_score = expected_final_margin / remaining_std
        win_prob = norm.cdf(z_score)
    else:
        win_prob = 1.0 if expected_final_margin > 0 else 0.0
    
    return win_prob


def parse_clock(clock_str, period):
    """
    Convert clock string to seconds remaining in game.
    
    Args:
        clock_str: Clock display like "15:32" or "8:05"
        period: 1 for first half, 2 for second half, 3+ for OT
    
    Returns:
        Total seconds remaining in game, or None if unparseable
    """
    if not clock_str:
        return None
    try:
        parts = clock_str.split(':')
        if len(parts) == 2:
            minutes, seconds = int(parts[0]), int(parts[1])
        else:
            minutes, seconds = 0, int(float(parts[0]))
        
        period_seconds = minutes * 60 + seconds
        
        # Assuming 2 halves of 20 minutes each
        if period == 1:
            return period_seconds + 20 * 60  # First half + second half remaining
        elif period == 2:
            return period_seconds  # Just second half remaining
        else:
            # Overtime - treat as very end of game
            return max(0, period_seconds)
    except:
        return None


def get_calibration_data(conn, sample_interval_sec=120):
    """
    Extract calibration data from play-by-play.
    
    Samples game states at regular intervals and pairs with actual outcomes.
    Uses team_a perspective consistently (matching historical_predictions table).
    
    IMPORTANT: Derives team_a_is_home by comparing team_a_id to home_team_id
    from the schedule table (don't trust team_a_home column - it has errors).
    
    Args:
        conn: SQLite database connection
        sample_interval_sec: How often to sample (default every 2 minutes)
    
    Returns:
        List of dicts with:
            - game_id
            - pregame_win_prob (team_a's calibrated win probability)
            - pregame_spread (team_a perspective)
            - time_remaining_sec
            - score_diff (team_a - team_b)
            - team_a_won (actual outcome: 1 or 0)
            - team_a_is_home (derived from schedule table)
    """
    cursor = conn.cursor()
    
    # First, figure out which schedule table to use
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'schedule_%'")
    schedule_tables = [row[0] for row in cursor.fetchall()]
    
    calibration_data = []
    
    for schedule_table in schedule_tables:
        # Get games with both PBP and predictions
        # Join on game_id and derive team_a_is_home from schedule table directly
        # (don't trust team_a_home column in historical_predictions - it has errors)
        query = f"""
            SELECT DISTINCT 
                p.event_id, 
                hp.predicted_margin,
                hp.team_a_win_prob,
                hp.team_a_id,
                s.home_score as final_home, 
                s.away_score as final_away,
                s.home_team_id
            FROM play_by_play p
            JOIN historical_predictions hp ON p.event_id = hp.game_id
            JOIN {schedule_table} s ON p.event_id = s.game_id
            WHERE hp.predicted_margin IS NOT NULL
              AND hp.team_a_win_prob IS NOT NULL
              AND s.home_score IS NOT NULL
              AND s.away_score IS NOT NULL
        """
        
        try:
            cursor.execute(query)
            games = cursor.fetchall()
        except sqlite3.OperationalError:
            continue
        
        print(f"  {schedule_table}: {len(games)} games with PBP and predictions")
        
        for game_id, predicted_margin, team_a_win_prob, team_a_id, final_home, final_away, home_team_id in games:
            # Derive team_a_is_home by comparing team_a_id to home_team_id
            # Cast both to string for safe comparison (handles int vs text mismatches)
            team_a_is_home = 1 if str(team_a_id) == str(home_team_id) else 0
            
            # Determine if team_a won (from team_a's perspective)
            if team_a_is_home == 1:
                # team_a is home
                team_a_final = final_home
                team_b_final = final_away
            else:
                # team_a is away
                team_a_final = final_away
                team_b_final = final_home
            
            team_a_won = 1 if team_a_final > team_b_final else 0
            
            # Get play-by-play snapshots for this game
            cursor.execute("""
                SELECT sequence, period, clock, home_score, away_score
                FROM play_by_play
                WHERE event_id = ?
                  AND home_score IS NOT NULL
                  AND away_score IS NOT NULL
                ORDER BY sequence
            """, (game_id,))
            
            plays = cursor.fetchall()
            if not plays:
                continue
            
            # Parse all plays with timestamps first (for momentum lookback)
            parsed_plays = []
            for seq, period, clock, home_score, away_score in plays:
                time_remaining = parse_clock(clock, period)
                if time_remaining is not None:
                    parsed_plays.append((time_remaining, home_score, away_score))
            
            if not parsed_plays:
                continue
            
            # Sample at regular intervals
            last_sampled_time = None
            MOMENTUM_LOOKBACK_SEC = 120  # 2 minutes of game time
            
            for i, (time_remaining, home_score, away_score) in enumerate(parsed_plays):
                # Compute score differential from team_a's perspective
                if team_a_is_home == 1:
                    score_diff = home_score - away_score  # team_a is home
                else:
                    score_diff = away_score - home_score  # team_a is away
                
                # Calculate momentum: score change over last ~2 min of game time
                momentum = 0
                target_time = time_remaining + MOMENTUM_LOOKBACK_SEC
                for old_time, old_home, old_away in parsed_plays[:i]:
                    if old_time >= target_time:
                        home_run = home_score - old_home
                        away_run = away_score - old_away
                        if team_a_is_home == 1:
                            momentum = home_run - away_run
                        else:
                            momentum = away_run - home_run
                        break
                
                # Sample every N seconds of game time
                if last_sampled_time is None or (last_sampled_time - time_remaining) >= sample_interval_sec:
                    calibration_data.append({
                        'game_id': game_id,
                        'pregame_win_prob': team_a_win_prob,
                        'pregame_spread': predicted_margin,
                        'time_remaining_sec': time_remaining,
                        'score_diff': score_diff,
                        'momentum': momentum,
                        'team_a_won': team_a_won,
                        'team_a_is_home': team_a_is_home
                    })
                    last_sampled_time = time_remaining
    
    print(f"\nTotal: {len(calibration_data)} calibration snapshots")
    
    # Sanity check: team_a_won should be close to 50%
    if calibration_data:
        win_rate = sum(d['team_a_won'] for d in calibration_data) / len(calibration_data)
        print(f"Sanity check - team_a win rate: {win_rate:.1%} (should be ~50%)")
    
    return calibration_data


def calculate_calibration_metrics(calibration_data, base_std=11.0, momentum_factor=0.0, confidence_sensitivity=0.0,
                                  lead_sigma_k=0.0, half_sigma=None):
    """
    Calculate calibration metrics for a given base_std.
    
    Args:
        calibration_data: List of snapshot dicts from get_calibration_data()
        base_std: Standard deviation parameter to test
        momentum_factor: How much to weight momentum (negative = anti-momentum)
        confidence_sensitivity: How fast to shrink pregame edge when wrong (0 = off)
        lead_sigma_k: Lead-dependent σ adjustment factor (0 = off)
        half_sigma: Tuple of (first_half_std, second_half_std) or None
    
    Returns:
        - brier_score: Mean squared error of probability predictions (lower is better)
        - calibration_by_bucket: Dict of predicted prob bucket -> actual win rate
    """
    predictions = []
    actuals = []
    
    for row in calibration_data:
        prob = calculate_win_probability(
            row['pregame_win_prob'],
            row['pregame_spread'],
            row['score_diff'],
            row['time_remaining_sec'],
            base_std,
            row.get('momentum', 0.0),
            momentum_factor,
            confidence_sensitivity,
            lead_sigma_k=lead_sigma_k,
            half_sigma=half_sigma
        )
        predictions.append(prob)
        actuals.append(row['team_a_won'])
    
    # Brier score: mean squared error
    brier = sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / len(predictions)
    
    # Calibration by bucket (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
    buckets = defaultdict(lambda: {'count': 0, 'wins': 0, 'prob_sum': 0})
    for prob, actual in zip(predictions, actuals):
        bucket = int(prob * 10) / 10  # 0.0, 0.1, 0.2, ... 0.9
        bucket = min(0.9, max(0.0, bucket))
        buckets[bucket]['count'] += 1
        buckets[bucket]['wins'] += actual
        buckets[bucket]['prob_sum'] += prob
    
    calibration = {}
    for bucket in sorted(buckets.keys()):
        data = buckets[bucket]
        if data['count'] > 0:
            calibration[bucket] = {
                'predicted': data['prob_sum'] / data['count'],
                'actual': data['wins'] / data['count'],
                'count': data['count']
            }
    
    return brier, calibration


def calculate_calibration_by_time(calibration_data, base_std=11.0, momentum_factor=0.0, confidence_sensitivity=0.0,
                                  lead_sigma_k=0.0, half_sigma=None):
    """
    Calculate calibration metrics broken down by time remaining.
    
    Helps identify if model is better/worse at different game stages.
    """
    time_buckets = {
        'early_1st': (30*60, 40*60),    # First 10 min
        'late_1st': (20*60, 30*60),     # Last 10 min of 1st half
        'early_2nd': (10*60, 20*60),    # First 10 min of 2nd half
        'late_2nd': (5*60, 10*60),      # 10-5 min left
        'crunch': (0, 5*60)             # Last 5 min
    }
    
    results = {}
    
    for bucket_name, (min_time, max_time) in time_buckets.items():
        bucket_data = [
            row for row in calibration_data 
            if min_time <= row['time_remaining_sec'] < max_time
        ]
        
        if len(bucket_data) < 100:
            continue
        
        predictions = []
        actuals = []
        
        for row in bucket_data:
            prob = calculate_win_probability(
                row['pregame_win_prob'],
                row['pregame_spread'],
                row['score_diff'],
                row['time_remaining_sec'],
                base_std,
                row.get('momentum', 0.0),
                momentum_factor,
                confidence_sensitivity,
                lead_sigma_k=lead_sigma_k,
                half_sigma=half_sigma
            )
            predictions.append(prob)
            actuals.append(row['team_a_won'])
        
        brier = sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / len(predictions)
        
        results[bucket_name] = {
            'brier': brier,
            'count': len(bucket_data),
            'avg_pred': sum(predictions) / len(predictions),
            'actual_win_rate': sum(actuals) / len(actuals)
        }
    
    return results


def optimize_base_std(calibration_data, std_range=(8.0, 20.0), steps=30):
    """
    Find optimal base_std that minimizes Brier score.
    
    Args:
        calibration_data: List of snapshot dicts
        std_range: (min, max) range to search
        steps: Number of values to test
    
    Returns:
        - best_std: Optimal standard deviation
        - best_brier: Brier score at optimal
        - results: List of (std, brier) tuples for all tested values
    """
    best_std = 11.0
    best_brier = float('inf')
    
    results = []
    test_values = [std_range[0] + i * (std_range[1] - std_range[0]) / steps for i in range(steps + 1)]
    
    for std in test_values:
        brier, _ = calculate_calibration_metrics(calibration_data, std)
        results.append((std, brier))
        if brier < best_brier:
            best_brier = brier
            best_std = std
    
    return best_std, best_brier, results


def optimize_momentum(calibration_data, base_std=11.0, mom_range=(-1.0, 0.5), steps=30):
    """
    Find optimal momentum_factor that minimizes Brier score.
    
    Args:
        calibration_data: List of snapshot dicts
        base_std: Base standard deviation to use
        mom_range: (min, max) range to search (negative = anti-momentum)
        steps: Number of values to test
    
    Returns:
        - best_factor: Optimal momentum factor
        - best_brier: Brier score at optimal
        - results: List of (factor, brier) tuples for all tested values
    """
    best_factor = 0.0
    best_brier = float('inf')
    
    results = []
    test_values = [mom_range[0] + i * (mom_range[1] - mom_range[0]) / steps for i in range(steps + 1)]
    
    for factor in test_values:
        brier, _ = calculate_calibration_metrics(calibration_data, base_std, factor)
        results.append((factor, brier))
        if brier < best_brier:
            best_brier = brier
            best_factor = factor
    
    return best_factor, best_brier, results


def optimize_confidence(calibration_data, base_std=11.0, momentum_factor=0.0, conf_range=(0.5, 4.0), steps=35):
    """
    Find optimal confidence_sensitivity that minimizes Brier score.
    
    Args:
        calibration_data: List of snapshot dicts
        base_std: Base standard deviation to use
        momentum_factor: Momentum factor to use
        conf_range: (min, max) range to search
            Higher = trust pregame longer, Lower = trust score faster
        steps: Number of values to test
    
    Returns:
        - best_sensitivity: Optimal confidence sensitivity
        - best_brier: Brier score at optimal
        - results: List of (sensitivity, brier) tuples for all tested values
    """
    best_sensitivity = 0.0
    best_brier = float('inf')
    
    results = []
    # Include 0.0 (off) as first test
    test_values = [0.0] + [conf_range[0] + i * (conf_range[1] - conf_range[0]) / steps for i in range(steps + 1)]
    
    for sensitivity in test_values:
        brier, _ = calculate_calibration_metrics(calibration_data, base_std, momentum_factor, sensitivity)
        results.append((sensitivity, brier))
        if brier < best_brier:
            best_brier = brier
            best_sensitivity = sensitivity
    
    return best_sensitivity, best_brier, results


def optimize_lead_sigma(calibration_data, base_std=11.0, momentum_factor=0.0, confidence_sensitivity=0.0,
                        k_range=(0.0, 3.0), steps=30):
    """
    Find optimal lead_sigma_k that minimizes Brier score.
    
    Tests: σ_eff = base_std - k × |score_diff| / √(possessions_remaining)
    where possessions_remaining ≈ time_remaining_sec / 36
    
    Hypothesis: large leads reduce remaining scoring variance because winning 
    teams slow pace and trailing teams take rushed shots. A fixed σ overestimates
    remaining variance in blowouts, making the model too uncertain.
    
    Args:
        calibration_data: List of snapshot dicts
        base_std: Base standard deviation to use
        momentum_factor: Momentum factor to use
        confidence_sensitivity: Confidence sensitivity to use
        k_range: (min, max) range for k parameter
        steps: Number of values to test
    
    Returns:
        - best_k: Optimal lead sigma k
        - best_brier: Brier score at optimal
        - results: List of (k, brier) tuples for all tested values
    """
    best_k = 0.0
    best_brier = float('inf')
    
    results = []
    test_values = [k_range[0] + i * (k_range[1] - k_range[0]) / steps for i in range(steps + 1)]
    
    for k in test_values:
        predictions = []
        actuals = []
        for row in calibration_data:
            prob = calculate_win_probability(
                row['pregame_win_prob'],
                row['pregame_spread'],
                row['score_diff'],
                row['time_remaining_sec'],
                base_std,
                row.get('momentum', 0.0),
                momentum_factor,
                confidence_sensitivity,
                lead_sigma_k=k
            )
            predictions.append(prob)
            actuals.append(row['team_a_won'])
        
        brier = sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / len(predictions)
        results.append((k, brier))
        if brier < best_brier:
            best_brier = brier
            best_k = k
    
    return best_k, best_brier, results


def optimize_half_sigma(calibration_data, momentum_factor=0.0, confidence_sensitivity=0.0,
                        std_range=(8.0, 22.0), steps=28):
    """
    Find optimal separate σ values for 1st half and 2nd half.
    
    Hypothesis: second-half basketball is structurally different — intentional fouling,
    clock management, bonus situations change scoring variance. Fitting separate σ
    values captures this without adding complexity to the core model.
    
    Uses 2D grid search over (σ_1st_half, σ_2nd_half).
    
    Args:
        calibration_data: List of snapshot dicts
        momentum_factor: Momentum factor to use
        confidence_sensitivity: Confidence sensitivity to use
        std_range: (min, max) range to search for each half
        steps: Number of values to test per dimension
    
    Returns:
        - best_pair: (σ_1st_half, σ_2nd_half) optimal values
        - best_brier: Combined Brier score at optimal
        - results: List of ((σ1, σ2), brier) tuples
    """
    # Split data by half
    first_half_data = [r for r in calibration_data if r['time_remaining_sec'] > 1200]
    second_half_data = [r for r in calibration_data if r['time_remaining_sec'] <= 1200]
    
    # Optimize each half independently (valid since they don't interact)
    def optimize_one_half(data, label):
        best_std = 11.0
        best_brier = float('inf')
        results = []
        test_values = [std_range[0] + i * (std_range[1] - std_range[0]) / steps for i in range(steps + 1)]
        
        for std in test_values:
            predictions = []
            actuals = []
            for row in data:
                prob = calculate_win_probability(
                    row['pregame_win_prob'],
                    row['pregame_spread'],
                    row['score_diff'],
                    row['time_remaining_sec'],
                    std,
                    row.get('momentum', 0.0),
                    momentum_factor,
                    confidence_sensitivity
                )
                predictions.append(prob)
                actuals.append(row['team_a_won'])
            
            brier = sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / len(predictions)
            results.append((std, brier))
            if brier < best_brier:
                best_brier = brier
                best_std = std
        
        return best_std, best_brier, results
    
    std1, brier1, results1 = optimize_one_half(first_half_data, '1st half')
    std2, brier2, results2 = optimize_one_half(second_half_data, '2nd half')
    
    # Combined Brier is weighted average
    n1, n2 = len(first_half_data), len(second_half_data)
    combined_brier = (brier1 * n1 + brier2 * n2) / (n1 + n2)
    
    return (std1, std2), combined_brier, (results1, results2, n1, n2)


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate live/in-game win probability model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Find optimal base_std parameter
    python3 calibrate_live_win_probability.py --db compiled_stats.db --optimize
    
    # Test with specific base_std
    python3 calibrate_live_win_probability.py --db compiled_stats.db --base-std 10.5
    
    # Optimize momentum factor (anti-momentum for mean reversion)
    python3 calibrate_live_win_probability.py --db compiled_stats.db --optimize-momentum
    
    # Test specific momentum factor
    python3 calibrate_live_win_probability.py --db compiled_stats.db --momentum-factor -0.5
    
    # Optimize confidence sensitivity (Bayesian update based on prediction error)
    python3 calibrate_live_win_probability.py --db compiled_stats.db --optimize-confidence
    
    # Finer sampling (every 60 seconds instead of 120)
    python3 calibrate_live_win_probability.py --db compiled_stats.db --optimize --sample-interval 60
    
    # Optimize lead-dependent σ (reduces variance in blowouts)
    python3 calibrate_live_win_probability.py --db compiled_stats.db --optimize-lead-sigma
    
    # Optimize half-specific σ (separate σ for 1st/2nd half)
    python3 calibrate_live_win_probability.py --db compiled_stats.db --optimize-half-sigma
    
    # Run all optimizations together
    python3 calibrate_live_win_probability.py --db compiled_stats.db --optimize --optimize-lead-sigma --optimize-half-sigma
        """
    )
    parser.add_argument('--db', default='compiled_stats.db', help='Database path')
    parser.add_argument('--sample-interval', type=int, default=120, 
                        help='Sample interval in seconds (default: 120)')
    parser.add_argument('--optimize', action='store_true', 
                        help='Optimize base_std parameter')
    parser.add_argument('--optimize-momentum', action='store_true',
                        help='Optimize momentum_factor parameter')
    parser.add_argument('--optimize-confidence', action='store_true',
                        help='Optimize confidence_sensitivity parameter (Bayesian pregame update)')
    parser.add_argument('--base-std', type=float, default=11.0,
                        help='Base standard deviation to use (default: 11.0)')
    parser.add_argument('--momentum-factor', type=float, default=0.0,
                        help='Momentum factor to use (negative = anti-momentum, default: 0.0)')
    parser.add_argument('--confidence-sensitivity', type=float, default=0.0,
                        help='Confidence sensitivity (0=off, higher=trust pregame longer, default: 0.0)')
    parser.add_argument('--optimize-lead-sigma', action='store_true',
                        help='Optimize lead-dependent σ: σ_eff = base_std - k × |lead| / √(poss_remaining)')
    parser.add_argument('--lead-sigma-k', type=float, default=0.0,
                        help='Lead sigma k factor (0=off, default: 0.0)')
    parser.add_argument('--optimize-half-sigma', action='store_true',
                        help='Optimize separate σ values for 1st and 2nd half')
    args = parser.parse_args()
    
    print("=" * 70)
    print("LIVE WIN PROBABILITY MODEL CALIBRATION")
    print("=" * 70)
    
    conn = sqlite3.connect(args.db)
    
    print(f"\nExtracting calibration data (sampling every {args.sample_interval}s)...")
    cal_data = get_calibration_data(conn, args.sample_interval)
    
    if len(cal_data) < 1000:
        print(f"WARNING: Only {len(cal_data)} snapshots - may not be enough for reliable calibration")
    
    if args.optimize:
        print("\n" + "=" * 70)
        print("OPTIMIZING BASE_STD PARAMETER")
        print("=" * 70)
        
        best_std, best_brier, results = optimize_base_std(cal_data)
        
        print(f"\n{'Std Dev':<12} {'Brier Score':<15}")
        print("-" * 27)
        for std, brier in results:
            marker = " <-- BEST" if abs(std - best_std) < 0.01 else ""
            print(f"{std:<12.2f} {brier:<15.6f}{marker}")
        
        print(f"\n*** Optimal base_std: {best_std:.2f} ***")
        print(f"*** Best Brier score: {best_brier:.6f} ***")
        base_std = best_std
    else:
        base_std = args.base_std
    
    # Momentum optimization
    if args.optimize_momentum:
        print("\n" + "=" * 70)
        print("OPTIMIZING MOMENTUM FACTOR")
        print("=" * 70)
        print("(Negative = anti-momentum/mean reversion, Positive = momentum continuation)")
        
        best_factor, best_brier, results = optimize_momentum(cal_data, base_std)
        
        # Get baseline for comparison
        baseline_brier, _ = calculate_calibration_metrics(cal_data, base_std, 0.0)
        
        print(f"\n{'Factor':<12} {'Brier Score':<15} {'vs Baseline':<15}")
        print("-" * 42)
        for factor, brier in results:
            diff_pct = (brier - baseline_brier) / baseline_brier * 100
            marker = " <-- BEST" if abs(factor - best_factor) < 0.01 else ""
            print(f"{factor:<12.2f} {brier:<15.6f} {diff_pct:+.3f}%{marker}")
        
        improvement = (baseline_brier - best_brier) / baseline_brier * 100
        print(f"\n*** Optimal momentum_factor: {best_factor:.2f} ***")
        print(f"*** Best Brier score: {best_brier:.6f} ({improvement:+.2f}% vs baseline) ***")
        momentum_factor = best_factor
        
        # Show breakdown by game phase
        print("\n" + "-" * 70)
        print("MOMENTUM EFFECT BY GAME PHASE")
        print("-" * 70)
        print(f"\n{'Phase':<15} {'No Momentum':<12} {'With Momentum':<14} {'Improvement':<12}")
        print("-" * 53)
        
        time_results_baseline = calculate_calibration_by_time(cal_data, base_std, 0.0)
        time_results_momentum = calculate_calibration_by_time(cal_data, base_std, best_factor)
        
        for phase in time_results_baseline.keys():
            b_base = time_results_baseline[phase]['brier']
            b_mom = time_results_momentum[phase]['brier']
            imp = (b_base - b_mom) / b_base * 100
            print(f"{phase:<15} {b_base:<12.6f} {b_mom:<14.6f} {imp:+.2f}%")
    else:
        momentum_factor = args.momentum_factor
    
    # Confidence sensitivity optimization
    if args.optimize_confidence:
        print("\n" + "=" * 70)
        print("OPTIMIZING CONFIDENCE SENSITIVITY")
        print("=" * 70)
        print("(Bayesian update: shrink pregame edge when prediction is wrong)")
        print("(0 = off, higher = trust pregame longer, lower = trust score faster)")
        
        best_sensitivity, best_brier, results = optimize_confidence(cal_data, base_std, momentum_factor)
        
        # Get baseline for comparison (sensitivity = 0)
        baseline_brier, _ = calculate_calibration_metrics(cal_data, base_std, momentum_factor, 0.0)
        
        print(f"\n{'Sensitivity':<12} {'Brier Score':<15} {'vs Baseline':<15}")
        print("-" * 42)
        for sensitivity, brier in results:
            diff_pct = (brier - baseline_brier) / baseline_brier * 100
            marker = " <-- BEST" if abs(sensitivity - best_sensitivity) < 0.01 else ""
            print(f"{sensitivity:<12.2f} {brier:<15.6f} {diff_pct:+.3f}%{marker}")
        
        improvement = (baseline_brier - best_brier) / baseline_brier * 100
        print(f"\n*** Optimal confidence_sensitivity: {best_sensitivity:.2f} ***")
        print(f"*** Best Brier score: {best_brier:.6f} ({improvement:+.2f}% vs baseline) ***")
        confidence_sensitivity = best_sensitivity
        
        # Show breakdown by game phase
        print("\n" + "-" * 70)
        print("CONFIDENCE UPDATE EFFECT BY GAME PHASE")
        print("-" * 70)
        print(f"\n{'Phase':<15} {'No Update':<12} {'With Update':<14} {'Improvement':<12}")
        print("-" * 53)
        
        time_results_baseline = calculate_calibration_by_time(cal_data, base_std, momentum_factor, 0.0)
        time_results_conf = calculate_calibration_by_time(cal_data, base_std, momentum_factor, best_sensitivity)
        
        for phase in time_results_baseline.keys():
            b_base = time_results_baseline[phase]['brier']
            b_conf = time_results_conf[phase]['brier']
            imp = (b_base - b_conf) / b_base * 100
            print(f"{phase:<15} {b_base:<12.6f} {b_conf:<14.6f} {imp:+.2f}%")
    else:
        confidence_sensitivity = args.confidence_sensitivity
    
    # Lead-dependent σ optimization
    lead_sigma_k = args.lead_sigma_k
    if args.optimize_lead_sigma:
        print("\n" + "=" * 70)
        print("OPTIMIZING LEAD-DEPENDENT σ")
        print("=" * 70)
        print("σ_eff = base_std - k × |score_diff| / √(possessions_remaining)")
        print("(Captures reduced variance in blowouts: winning team slows pace)")
        
        best_k, best_brier, results = optimize_lead_sigma(
            cal_data, base_std, momentum_factor, confidence_sensitivity)
        
        # Get baseline for comparison (k = 0)
        baseline_brier, _ = calculate_calibration_metrics(cal_data, base_std, momentum_factor, confidence_sensitivity)
        
        print(f"\n{'k':<12} {'Brier Score':<15} {'vs Baseline':<15}")
        print("-" * 42)
        for k, brier in results:
            diff_pct = (brier - baseline_brier) / baseline_brier * 100
            marker = " <-- BEST" if abs(k - best_k) < 0.01 else ""
            print(f"{k:<12.2f} {brier:<15.6f} {diff_pct:+.3f}%{marker}")
        
        improvement = (baseline_brier - best_brier) / baseline_brier * 100
        print(f"\n*** Optimal lead_sigma_k: {best_k:.2f} ***")
        print(f"*** Best Brier score: {best_brier:.6f} ({improvement:+.2f}% vs baseline) ***")
        lead_sigma_k = best_k
        
        # Show breakdown by game phase
        print("\n" + "-" * 70)
        print("LEAD-DEPENDENT σ EFFECT BY GAME PHASE")
        print("-" * 70)
        print(f"\n{'Phase':<15} {'No Lead-σ':<12} {'With Lead-σ':<14} {'Improvement':<12}")
        print("-" * 53)
        
        time_results_baseline = calculate_calibration_by_time(cal_data, base_std, momentum_factor, confidence_sensitivity)
        time_results_lead = calculate_calibration_by_time(cal_data, base_std, momentum_factor, confidence_sensitivity,
                                                          lead_sigma_k=best_k)
        
        for phase in time_results_baseline.keys():
            b_base = time_results_baseline[phase]['brier']
            b_lead = time_results_lead[phase]['brier']
            imp = (b_base - b_lead) / b_base * 100
            print(f"{phase:<15} {b_base:<12.6f} {b_lead:<14.6f} {imp:+.2f}%")
        
        # Show example σ values
        print(f"\n  Example effective σ values (base_std = {base_std:.1f}, k = {best_k:.2f}):")
        for score_diff, time_min in [(0, 20), (5, 15), (10, 10), (15, 5), (20, 3)]:
            time_sec = time_min * 60
            poss = max(1, time_sec / 36.0)
            sigma = max(5, base_std - best_k * abs(score_diff) / math.sqrt(poss))
            print(f"    Lead={score_diff:+d}, {time_min}min left → σ_eff={sigma:.1f}")
    
    # Half-specific σ optimization
    half_sigma = None
    if args.optimize_half_sigma:
        print("\n" + "=" * 70)
        print("OPTIMIZING HALF-SPECIFIC σ")
        print("=" * 70)
        print("(Separate σ for 1st half vs 2nd half)")
        print("(Captures clock management, intentional fouling, pace changes)")
        
        best_pair, combined_brier, (results1, results2, n1, n2) = optimize_half_sigma(
            cal_data, momentum_factor, confidence_sensitivity)
        
        # Get baseline for comparison
        baseline_brier, _ = calculate_calibration_metrics(cal_data, base_std, momentum_factor, confidence_sensitivity)
        
        improvement = (baseline_brier - combined_brier) / baseline_brier * 100
        
        print(f"\n  1st Half ({n1} snapshots):")
        print(f"  {'Std Dev':<12} {'Brier Score':<15}")
        print(f"  {'-'*27}")
        for std, brier in results1:
            marker = " <-- BEST" if abs(std - best_pair[0]) < 0.01 else ""
            print(f"  {std:<12.2f} {brier:<15.6f}{marker}")
        
        print(f"\n  2nd Half ({n2} snapshots):")
        print(f"  {'Std Dev':<12} {'Brier Score':<15}")
        print(f"  {'-'*27}")
        for std, brier in results2:
            marker = " <-- BEST" if abs(std - best_pair[1]) < 0.01 else ""
            print(f"  {std:<12.2f} {brier:<15.6f}{marker}")
        
        print(f"\n*** Optimal: 1st half σ = {best_pair[0]:.2f}, 2nd half σ = {best_pair[1]:.2f} ***")
        print(f"*** Combined Brier: {combined_brier:.6f} (vs {baseline_brier:.6f} uniform, {improvement:+.2f}%) ***")
        half_sigma = best_pair
        
        # Show breakdown by game phase
        print("\n" + "-" * 70)
        print("HALF-SPECIFIC σ EFFECT BY GAME PHASE")
        print("-" * 70)
        print(f"\n{'Phase':<15} {'Uniform σ':<12} {'Half-Split σ':<14} {'Improvement':<12}")
        print("-" * 53)
        
        time_results_baseline = calculate_calibration_by_time(cal_data, base_std, momentum_factor, confidence_sensitivity)
        time_results_half = calculate_calibration_by_time(cal_data, base_std, momentum_factor, confidence_sensitivity,
                                                          half_sigma=best_pair)
        
        for phase in time_results_baseline.keys():
            b_base = time_results_baseline[phase]['brier']
            b_half = time_results_half[phase]['brier']
            imp = (b_base - b_half) / b_base * 100
            print(f"{phase:<15} {b_base:<12.6f} {b_half:<14.6f} {imp:+.2f}%")
    
    print("\n" + "=" * 70)
    print(f"CALIBRATION RESULTS (base_std = {base_std:.2f}, momentum_factor = {momentum_factor:.2f}, confidence_sensitivity = {confidence_sensitivity:.2f}, lead_sigma_k = {lead_sigma_k:.2f})")
    if half_sigma:
        print(f"  (Half-specific σ: 1st={half_sigma[0]:.2f}, 2nd={half_sigma[1]:.2f})")
    print("=" * 70)
    
    brier, calibration = calculate_calibration_metrics(cal_data, base_std, momentum_factor, confidence_sensitivity,
                                                       lead_sigma_k=lead_sigma_k, half_sigma=half_sigma)
    
    print(f"\nOverall Brier Score: {brier:.6f}")
    print(f"(Lower is better. Perfect = 0.0, Random = 0.25)")
    
    print(f"\nCalibration by probability bucket:")
    print(f"{'Bucket':<12} {'Predicted':<12} {'Actual':<12} {'Count':<10} {'Error':<10}")
    print("-" * 56)
    
    total_abs_error = 0
    total_count = 0
    for bucket in sorted(calibration.keys()):
        data = calibration[bucket]
        error = abs(data['predicted'] - data['actual'])
        total_abs_error += error * data['count']
        total_count += data['count']
        print(f"{bucket:.1f}-{bucket+0.1:.1f}      {data['predicted']:.3f}        {data['actual']:.3f}        {data['count']:<10} {error:.3f}")
    
    print(f"\nWeighted Mean Absolute Calibration Error: {total_abs_error / total_count:.4f}")
    
    # Calibration by game time
    print("\n" + "=" * 70)
    print("CALIBRATION BY GAME PHASE")
    print("=" * 70)
    
    time_results = calculate_calibration_by_time(cal_data, base_std, momentum_factor, confidence_sensitivity,
                                                 lead_sigma_k=lead_sigma_k, half_sigma=half_sigma)
    
    print(f"\n{'Phase':<15} {'Brier':<12} {'Count':<10} {'Avg Pred':<12} {'Actual Win%':<12}")
    print("-" * 61)
    for phase, data in time_results.items():
        print(f"{phase:<15} {data['brier']:<12.4f} {data['count']:<10} {data['avg_pred']:<12.3f} {data['actual_win_rate']:<12.3f}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
    Good calibration means:
    - When model says 70% win probability, that team wins ~70% of the time
    - Brier score < 0.20 is good, < 0.15 is excellent
    - Calibration error < 0.05 per bucket is well-calibrated
    
    If early game phases have higher Brier scores, the pregame spread
    predictions may need improvement. If late game phases are worse,
    the time decay or variance model may need adjustment.
    
    Momentum factor: negative = anti-momentum (runs mean-revert),
    positive = momentum continuation (runs persist).
    
    Confidence sensitivity: shrinks pregame edge when observed game
    performance diverges from prediction. Higher = trust pregame longer.
    
    Lead-dependent σ (lead_sigma_k): reduces σ when one team has a large lead.
    Captures pace slowdowns and reduced variance in blowouts. The effective
    σ = base_std - k × |score_diff| / √(possessions_remaining).
    
    Half-specific σ: allows different σ for 1st vs 2nd half to capture
    structural differences (clock management, intentional fouling, bonus).
    """)
    
    conn.close()
    
    print("\n" + "=" * 70)
    rec = f"RECOMMENDED: base_std = {base_std:.2f}, momentum_factor = {momentum_factor:.2f}, confidence_sensitivity = {confidence_sensitivity:.2f}, lead_sigma_k = {lead_sigma_k:.2f}"
    if half_sigma:
        rec += f", half_sigma = ({half_sigma[0]:.2f}, {half_sigma[1]:.2f})"
    print(rec)
    print("=" * 70)


if __name__ == "__main__":
    main()