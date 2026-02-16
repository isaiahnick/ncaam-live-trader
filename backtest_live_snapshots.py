#!/usr/bin/env python3
"""
Backtest the live trading strategy using live_snapshots data.
Replicates every decision function from live_trader.py exactly.

Usage:
    python3 backtest_live_snapshots.py --db compiled_stats.db --start-date 2026-02-04 --end-date 2026-02-07
    python3 backtest_live_snapshots.py --db compiled_stats.db --start-date 2026-02-07 --end-date 2026-02-07 --debug
    python3 backtest_live_snapshots.py --db compiled_stats.db --start-date 2026-02-04 --end-date 2026-02-07 --trace-game 401829236
"""

import argparse
import sqlite3
import math
import sys
from datetime import datetime, timedelta
from scipy.stats import norm
from typing import Optional, Dict, List, Tuple
from collections import defaultdict


# ============================================================
# CONSTANTS — exact copies from live_trader.py
# ============================================================

# Win probability model
BASE_STD = 17.2

# Entry parameters
EDGE_THRESHOLD = 0.08      # Base threshold (dynamic via get_required_edge)
MIN_TIME_REMAINING = 240   # 4:00 — don't enter with less than this (live only)
MIN_ENTRY_PRICE = 0.10     # Don't buy below 10 cents
MAX_ENTRY_PRICE = 0.90     # Don't buy above 90 cents

# Exit parameters — EV-based
EV_EXIT_SLIPPAGE_CENTS = 0

# Option value formula (fit to backward induction with DT=120, sigma=4.25, slip=0)
OV_SCALE = 0.349
OV_EXPONENT = 0.4277
OV_PROB_COEFF = 17.2926
OV_DECORR_SEC = 120

# Cooldown
COOLDOWN_AFTER_EXIT = 30   # 30 seconds game-time after any exit
COOLDOWN_AFTER_STOP = 240  # 4 minutes after stop loss (if ever re-enabled)

# Microstate haircut (exits only) — calibrated from live_snapshots
HAIRCUT_SCALE = 28.27
HAIRCUT_TAU = 112
HAIRCUT_MARGIN_DECAY = 1.4
HAIRCUT_K = 0.75
HAIRCUT_RAMP_HI = 240
HAIRCUT_RAMP_LO = 120
HAIRCUT_MARGIN_MAX = 5

# ESPN score freshness guard
MAX_ESPN_SCORE_STALE_SEC = 10


# ============================================================
# FUNCTIONS — exact copies from live_trader.py
# ============================================================

def kalshi_fee_cents(price: float) -> float:
    """Kalshi fee: 7% * P * (1-P) * 100 cents, capped at 1.75 cents.
    Args: price as decimal (0-1)
    """
    return min(0.07 * price * (1 - price) * 100, 1.75)


def poly_fee_cents(price: float) -> float:
    """Polymarket fee: 10 basis points (0.1%) taker, 0 maker. Price in 0-1 range."""
    return 0.001 * price * 100


def calculate_net_ev(prob: float, ask: float, spread: float, venue: str = 'Kalshi') -> float:
    """
    Calculate net expected value for a trade, accounting for fees and spread.

    Args:
        prob: Our model's probability (0-1)
        ask: Entry price we'd pay (0-1)
        spread: Bid-ask spread in cents
        venue: 'Kalshi' or 'Polymarket'

    Returns:
        Net EV in cents
    """
    gross_edge_cents = (prob - ask) * 100

    if venue == 'Polymarket':
        entry_fee = poly_fee_cents(ask)
        exit_fee = poly_fee_cents(prob)
    else:
        entry_fee = kalshi_fee_cents(ask)
        exit_fee = kalshi_fee_cents(prob)

    spread_cost = spread / 2

    return gross_edge_cents - entry_fee - exit_fee - spread_cost


def calculate_win_probability(pregame_win_prob: float, score_diff: float, time_remaining_sec: int) -> float:
    """Bayesian live win probability. Exact copy of LiveWinProbabilityModel."""
    total_game_sec = 40 * 60
    time_fraction = max(0.001, min(1.0, time_remaining_sec / total_game_sec))

    if pregame_win_prob >= 0.999:
        pregame_implied_margin = 3.5 * BASE_STD
    elif pregame_win_prob <= 0.001:
        pregame_implied_margin = -3.5 * BASE_STD
    else:
        pregame_implied_margin = norm.ppf(pregame_win_prob) * BASE_STD

    expected_remaining_edge = pregame_implied_margin * time_fraction
    expected_final_margin = score_diff + expected_remaining_edge
    remaining_std = BASE_STD * math.sqrt(time_fraction)

    if remaining_std > 0.001:
        return norm.cdf(expected_final_margin / remaining_std)
    return 1.0 if expected_final_margin > 0 else 0.0


def get_required_edge(entry_price: float, time_remaining: int = 2400, e_start: float = None) -> float:
    """Dynamic edge threshold — exponential ramp (k=2) from E_START to 15%."""
    E_START = e_start if e_start is not None else 0.08
    E_END = 0.15     # 15% at 4:00 remaining
    T_START = 2400   # 40:00
    T_END = 240      # 4:00
    K = 2            # Exponential factor

    if time_remaining >= T_START:
        return E_START
    elif time_remaining <= T_END:
        return E_END
    else:
        progress = (T_START - time_remaining) / (T_START - T_END)
        curved_progress = progress ** K
        return E_START + curved_progress * (E_END - E_START)


def compute_option_value(time_remaining_sec: int, prob: float) -> float:
    """Closed-form option value: OV = 0.37 * N^0.42 * (1 + 16.82*p*(1-p))."""
    N = max(0, time_remaining_sec) / OV_DECORR_SEC
    if N <= 0:
        return 0.0
    return OV_SCALE * (N ** OV_EXPONENT) * (1 + OV_PROB_COEFF * prob * (1 - prob))


def compute_haircut(time_remaining_sec: int, abs_margin: int) -> float:
    """Microstate uncertainty premium (cents) for exit threshold."""
    if abs_margin > HAIRCUT_MARGIN_MAX:
        return 0.0
    if time_remaining_sec >= HAIRCUT_RAMP_HI:
        return 0.0
    elif time_remaining_sec <= HAIRCUT_RAMP_LO:
        g = 1.0
    else:
        g = (HAIRCUT_RAMP_HI - time_remaining_sec) / (HAIRCUT_RAMP_HI - HAIRCUT_RAMP_LO)
    h_raw = HAIRCUT_SCALE * math.exp(-time_remaining_sec / HAIRCUT_TAU) * math.exp(-abs_margin / HAIRCUT_MARGIN_DECAY)
    return max(0.0, HAIRCUT_K * g * h_raw)


# ============================================================
# BACKTEST ENGINE
# ============================================================

class BacktestEngine:
    def __init__(self, db_path: str, debug: bool = False, trace_game: str = None, min_edge: float = None, venue: str = 'best'):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.debug = debug
        self.trace_game = trace_game
        self.min_edge = min_edge  # Override E_START if set
        self.venue = venue  # 'kalshi', 'polymarket', or 'best'

        # --- Per-game state ---
        self.open_positions: Dict[str, dict] = {}
        self.exit_cooldowns: Dict[tuple, tuple] = {}   # (game_id, side) -> (exit_game_time, cooldown_secs)
        self.prev_scores: Dict[str, tuple] = {}         # game_id -> (home, away)
        self.last_score_change_ts: Dict[str, datetime] = {}
        self.game_settled: set = set()

        # --- Results ---
        self.trades: List[dict] = []
        self.realized_pnl: float = 0.0
        self.capital_deployed: float = 0.0
        self.peak_capital: float = 0.0

        # --- Outcomes (from schedule) ---
        self.outcomes: Dict[str, dict] = {}

        # --- Counters ---
        self.entry_skips = {
            'no_pregame_prob': 0,
            'no_market_data': 0,
            'bogus_pregame': 0,
            'score_stale_entry': 0,
            'score_stale_exit': 0,
            'bad_score_data': 0,
        }

    def load_outcomes(self, start_date: str, end_date: str):
        """Load game outcomes from schedule_2025_26. Extends range by 2 days for late tips."""
        end_plus = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=2)).strftime('%Y-%m-%d')
        cursor = self.conn.execute("""
            SELECT game_id, home_winner, home_score, away_score
            FROM schedule_2025_26
            WHERE game_date BETWEEN ? AND ?
            AND home_score IS NOT NULL
        """, (start_date, end_plus))

        for row in cursor:
            self.outcomes[str(row['game_id'])] = {
                'home_winner': row['home_winner'],
                'home_score': row['home_score'],
                'away_score': row['away_score'],
            }

    # ----------------------------------------------------------
    # CORE: Process one snapshot
    # ----------------------------------------------------------
    def process_snapshot(self, snap) -> Optional[str]:
        """Process a single snapshot row. Returns action taken or None."""
        game_id = snap['game_id']
        status = snap['game_status']
        ts = datetime.fromisoformat(snap['timestamp'])

        # Skip if already settled
        if game_id in self.game_settled:
            return None

        home_score = snap['home_score'] or 0
        away_score = snap['away_score'] or 0
        period = snap['period'] or 0

        # === HANDLE PREGAME ===
        if status == 'pre':
            # Filter bogus pregame (finished games logged as pre with nonzero scores)
            if home_score > 0 or away_score > 0:
                self.entry_skips['bogus_pregame'] += 1
                return None
            time_remaining = 2400  # Force correct pregame value (line 1924-1925)
        elif status == 'in':
            time_remaining = snap['time_remaining_sec'] or 0
        else:
            return None  # Unknown status

        # === SCORE FRESHNESS TRACKING ===
        # Replicates live_trader.py lines 1854-1891
        if game_id in self.prev_scores:
            prev_h, prev_a = self.prev_scores[game_id]
            # Score decrease detection — ESPN API bugs (line 1858)
            if home_score < prev_h or away_score < prev_a:
                self.entry_skips['bad_score_data'] += 1
                if self.trace_game == game_id:
                    print(f"  [TRACE] {ts} BAD DATA: score {prev_a}-{prev_h} -> {away_score}-{home_score}, skip")
                return None
            # Track when SCORE specifically changes (line 1865)
            if home_score != prev_h or away_score != prev_a:
                self.last_score_change_ts[game_id] = ts
        else:
            # First time seeing this game — treat as fresh (line 1869)
            self.last_score_change_ts[game_id] = ts

        self.prev_scores[game_id] = (home_score, away_score)

        # Compute seconds since last score change (line 1891)
        score_change_ts = self.last_score_change_ts.get(game_id, ts)
        seconds_since_score_change = (ts - score_change_ts).total_seconds()

        # === BUILD GAME DICT ===
        pregame_prob = snap['pregame_home_prob']
        if not pregame_prob:
            self.entry_skips['no_pregame_prob'] += 1
            return None

        game = {
            'game_id': game_id,
            'home_team': snap['home_team'],
            'away_team': snap['away_team'],
            'home_score': home_score,
            'away_score': away_score,
            'period': period,
            'clock': snap['clock'],
            'time_remaining_sec': time_remaining,
            'status': status,
            'pregame_home_prob': pregame_prob,
            'seconds_since_score_change': seconds_since_score_change,
        }

        # Use stored live_home_prob from snapshot (exact value live trader computed)
        if status == 'in':
            stored_prob = snap['live_home_prob']
            if stored_prob is not None:
                game['live_home_prob'] = stored_prob
            else:
                # Fallback: recalculate if somehow missing
                score_diff = home_score - away_score
                game['live_home_prob'] = calculate_win_probability(pregame_prob, score_diff, time_remaining)

        # === BUILD ODDS DICT ===
        odds = {
            'home_yes_bid': snap['home_yes_bid'],
            'home_yes_ask': snap['home_yes_ask'],
            'away_yes_bid': snap['away_yes_bid'],
            'away_yes_ask': snap['away_yes_ask'],
            'home_no_bid': snap['home_no_bid'],
            'home_no_ask': snap['home_no_ask'],
            'away_no_bid': snap['away_no_bid'],
            'away_no_ask': snap['away_no_ask'],
        }

        # Polymarket prices (logged by live_trader's log_snapshot)
        poly_home_bid = snap['poly_home_bid'] if 'poly_home_bid' in snap.keys() else None
        poly_home_ask = snap['poly_home_ask'] if 'poly_home_ask' in snap.keys() else None
        poly_away_bid = snap['poly_away_bid'] if 'poly_away_bid' in snap.keys() else None
        poly_away_ask = snap['poly_away_ask'] if 'poly_away_ask' in snap.keys() else None
        poly_slug = snap['poly_slug'] if 'poly_slug' in snap.keys() else None

        odds['poly_home_bid'] = poly_home_bid
        odds['poly_home_ask'] = poly_home_ask
        odds['poly_away_bid'] = poly_away_bid
        odds['poly_away_ask'] = poly_away_ask
        odds['poly_slug'] = poly_slug
        odds['has_polymarket'] = bool(poly_home_bid or poly_home_ask or poly_away_bid or poly_away_ask)

        # Check if any market data exists (Kalshi OR Polymarket)
        has_kalshi = any(v for k, v in odds.items()
                         if k in ('home_yes_bid', 'home_yes_ask', 'away_yes_bid', 'away_yes_ask',
                                  'home_no_bid', 'home_no_ask', 'away_no_bid', 'away_no_ask')
                         and v is not None and v > 0)
        has_poly = odds['has_polymarket']

        # === CHECK SETTLEMENT === (MUST be before market data guard — markets go dark at game end)
        # Game effectively over: in-game, period >= 2, time = 0, scores exist and different
        if (status == 'in' and time_remaining == 0 and period >= 2
                and home_score > 0 and home_score != away_score):
            if game_id in self.open_positions:
                self._settle_position(game_id, game, ts)
            self.game_settled.add(game_id)
            return 'settled'

        if not has_kalshi and not has_poly:
            self.entry_skips['no_market_data'] += 1
            return None

        # === CHECK EXITS first (before entries) ===
        if game_id in self.open_positions:
            position = self.open_positions[game_id]
            exit_info = self._check_exit(game, odds, position)
            if exit_info:
                self._execute_exit(exit_info, ts)
                return 'exit'
            return None  # Have position, no exit triggered — don't check entry

        # === CHECK ENTRIES ===
        opp = self._check_entry(game, odds)
        if opp:
            self._execute_entry(opp, ts)
            return 'entry'

        return None

    # ----------------------------------------------------------
    # ENTRY LOGIC — replicates check_opportunity() exactly
    # ----------------------------------------------------------
    def _check_entry(self, game: dict, odds: dict) -> Optional[dict]:
        game_id = game['game_id']
        status = game['status']
        time_remaining = game['time_remaining_sec']

        # ESPN score freshness guard — live games only (line 2120-2124)
        if status == 'in':
            score_age = game.get('seconds_since_score_change', 0)
            if score_age > MAX_ESPN_SCORE_STALE_SEC:
                self.entry_skips['score_stale_entry'] += 1
                if self.trace_game == game_id:
                    print(f"  [TRACE] SKIP entry: score stale {score_age:.1f}s > {MAX_ESPN_SCORE_STALE_SEC}s")
                return None

        # Get probability for edge calculation (lines 2127-2137)
        if status == 'in':
            if 'live_home_prob' not in game:
                return None
            our_home = game['live_home_prob']
        elif status == 'pre':
            our_home = game['pregame_home_prob']
        else:
            return None

        def evaluate_venue(odds_dict, venue_name):
            """
            Evaluate best opportunity for a single venue, comparing YES and NO contracts.
            Exact copy of live_trader.py evaluate_venue (lines 3042-3176).

            Returns (side, edge, entry_price, spread, contract_type, venue_name) or None
            """
            home_yes_ask = odds_dict.get('home_yes_ask', 0)
            home_yes_bid = odds_dict.get('home_yes_bid', 0)
            home_no_ask = odds_dict.get('home_no_ask', 0)
            home_no_bid = odds_dict.get('home_no_bid', 0)
            away_yes_ask = odds_dict.get('away_yes_ask', 0)
            away_yes_bid = odds_dict.get('away_yes_bid', 0)
            away_no_ask = odds_dict.get('away_no_ask', 0)
            away_no_bid = odds_dict.get('away_no_bid', 0)

            best = None

            # Time-dependent spread limit (line 2164-2167)
            game_time = game.get('time_remaining_sec', 2400)
            if game_time >= 1200:  # > 20:00
                max_spread = 3
            else:
                max_spread = 2

            # ===== CHECK HOME SIDE (bet on home winning) =====
            home_candidates = []

            if home_yes_ask and home_yes_bid:
                home_candidates.append({
                    'contract_type': 'home_yes',
                    'ask': home_yes_ask,
                    'bid': home_yes_bid,
                    'spread': home_yes_ask - home_yes_bid,
                })

            if away_no_ask and away_no_bid:
                home_candidates.append({
                    'contract_type': 'away_no',
                    'ask': away_no_ask,
                    'bid': away_no_bid,
                    'spread': away_no_ask - away_no_bid,
                })

            if home_candidates:
                home_candidates.sort(key=lambda x: (x['ask'], x['spread']))
                best_home = home_candidates[0]

                entry_price = best_home['ask'] / 100.0
                spread = best_home['spread']
                home_edge = our_home - entry_price

                cooldown_key = (game_id, 'home')
                in_cooldown = False
                if cooldown_key in self.exit_cooldowns:
                    exit_game_time, cooldown_secs = self.exit_cooldowns[cooldown_key]
                    in_cooldown = (exit_game_time - time_remaining) < cooldown_secs

                required_edge = get_required_edge(entry_price, time_remaining, e_start=self.min_edge)

                if (home_edge >= required_edge and
                        spread <= max_spread and
                        entry_price >= MIN_ENTRY_PRICE and
                        entry_price <= MAX_ENTRY_PRICE and
                        (status == 'pre' or time_remaining >= MIN_TIME_REMAINING) and
                        game.get('period', 1) <= 2 and
                        not in_cooldown):
                    best = ('home', home_edge, entry_price, spread, best_home['contract_type'], venue_name)

            # ===== CHECK AWAY SIDE (bet on away winning) =====
            away_candidates = []

            if away_yes_ask and away_yes_bid:
                away_candidates.append({
                    'contract_type': 'away_yes',
                    'ask': away_yes_ask,
                    'bid': away_yes_bid,
                    'spread': away_yes_ask - away_yes_bid,
                })

            if home_no_ask and home_no_bid:
                away_candidates.append({
                    'contract_type': 'home_no',
                    'ask': home_no_ask,
                    'bid': home_no_bid,
                    'spread': home_no_ask - home_no_bid,
                })

            if away_candidates:
                away_candidates.sort(key=lambda x: (x['ask'], x['spread']))
                best_away = away_candidates[0]

                entry_price = best_away['ask'] / 100.0
                spread = best_away['spread']
                away_edge = (1 - our_home) - entry_price

                cooldown_key = (game_id, 'away')
                in_cooldown = False
                if cooldown_key in self.exit_cooldowns:
                    exit_game_time, cooldown_secs = self.exit_cooldowns[cooldown_key]
                    in_cooldown = (exit_game_time - time_remaining) < cooldown_secs

                required_edge = get_required_edge(entry_price, time_remaining, e_start=self.min_edge)

                if (away_edge >= required_edge and
                        spread <= max_spread and
                        entry_price >= MIN_ENTRY_PRICE and
                        entry_price <= MAX_ENTRY_PRICE and
                        (status == 'pre' or time_remaining >= MIN_TIME_REMAINING) and
                        game.get('period', 1) <= 2 and
                        not in_cooldown):
                    if best is None or away_edge > best[1]:
                        best = ('away', away_edge, entry_price, spread, best_away['contract_type'], venue_name)

            return best

        # === DUAL-VENUE EVALUATION: Pick best net edge after fees ===
        # Evaluate Kalshi (skip if venue is polymarket-only)
        kalshi_best = None
        if self.venue in ('kalshi', 'best'):
            kalshi_best = evaluate_venue(odds, 'Kalshi')

        # Evaluate Polymarket (skip if venue is kalshi-only)
        poly_best = None
        if self.venue in ('polymarket', 'best') and odds.get('has_polymarket'):
            poly_odds = {
                'home_yes_bid': odds.get('poly_home_bid'),
                'home_yes_ask': odds.get('poly_home_ask'),
                'away_yes_bid': odds.get('poly_away_bid'),
                'away_yes_ask': odds.get('poly_away_ask'),
                # No NO contracts on Polymarket
                'home_no_bid': None, 'home_no_ask': None,
                'away_no_bid': None, 'away_no_ask': None,
            }
            poly_best = evaluate_venue(poly_odds, 'Polymarket')

        # Pick the venue with better net EV after fees
        best = None
        if kalshi_best and poly_best:
            k_side, k_edge, k_price, k_spread = kalshi_best[0], kalshi_best[1], kalshi_best[2], kalshi_best[3]
            p_side, p_edge, p_price, p_spread = poly_best[0], poly_best[1], poly_best[2], poly_best[3]
            k_prob = our_home if k_side == 'home' else 1 - our_home
            p_prob = our_home if p_side == 'home' else 1 - our_home
            k_net_ev = calculate_net_ev(k_prob, k_price, k_spread, 'Kalshi')
            p_net_ev = calculate_net_ev(p_prob, p_price, p_spread, 'Polymarket')

            if self.trace_game == game_id:
                print(f"  [TRACE] VENUE: K:{kalshi_best[4]} {k_price*100:.0f}c spd={k_spread:.0f} ev={k_net_ev:.2f} | "
                      f"P:{poly_best[4]} {p_price*100:.0f}c spd={p_spread:.0f} ev={p_net_ev:.2f} -> "
                      f"{'POLY' if p_net_ev > k_net_ev else 'KALSHI'}")

            if p_net_ev > k_net_ev:
                best = poly_best
            else:
                best = kalshi_best
        elif kalshi_best:
            best = kalshi_best
        elif poly_best:
            best = poly_best

        if not best:
            return None

        side, edge, entry_price, spread, contract_type, market_source = best
        our_prob = our_home if side == 'home' else 1 - our_home

        # Cross-venue duplicate guard (live_trader lines 3250-3257)
        if game_id in self.open_positions:
            existing_venue = self.open_positions[game_id].get('market_source', 'Kalshi')
            if existing_venue != market_source:
                return None

        if self.trace_game == game_id:
            print(f"  [TRACE] ENTRY candidate: {side.upper()} edge={edge:.3f} "
                  f"price={entry_price:.2f} spread={spread} type={contract_type} venue={market_source}")

        return {
            'game_id': game_id,
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'side': side,
            'edge': edge,
            'entry_price': entry_price,
            'spread': spread,
            'contract_type': contract_type,
            'our_prob': our_prob,
            'status': status,
            'time_remaining_sec': time_remaining,
            'period': game.get('period', 0),
            'home_score': game.get('home_score', 0),
            'away_score': game.get('away_score', 0),
            'pregame_home_prob': game.get('pregame_home_prob'),
            'live_home_prob': game.get('live_home_prob'),
            'market_source': market_source,
        }

    # ----------------------------------------------------------
    # EXIT LOGIC — replicates check_exit() exactly
    # ----------------------------------------------------------
    def _check_exit(self, game: dict, odds: dict, position: dict) -> Optional[dict]:
        side = position['side']
        entry_price = position['entry_price']
        status = game['status']
        contract_type = position['contract_type']
        time_remaining = game['time_remaining_sec']
        market_source = position.get('market_source', 'Kalshi')

        # Only EV-exit during live games with model prob (line 2438)
        if status != 'in' or 'live_home_prob' not in game:
            return None

        # ESPN score freshness guard for exits (line 2443-2446)
        score_age = game.get('seconds_since_score_change', 0)
        if score_age > MAX_ESPN_SCORE_STALE_SEC:
            self.entry_skips['score_stale_exit'] += 1
            if self.trace_game == game['game_id']:
                print(f"  [TRACE] SKIP exit check: score stale {score_age:.1f}s")
            return None

        # Get bid for our contract type — venue-specific
        if market_source == 'Polymarket':
            # Polymarket: only YES contracts (home_yes, away_yes)
            # If we bet home via short, our "bid" is poly_home_bid
            # If we bet away via long, our "bid" is poly_away_bid
            if side == 'home':
                current_bid_cents = odds.get('poly_home_bid', 0) or 0
            else:
                current_bid_cents = odds.get('poly_away_bid', 0) or 0
        else:
            # Kalshi: use contract-type-specific bid
            bid_map = {
                'home_yes': 'home_yes_bid',
                'home_no': 'home_no_bid',
                'away_yes': 'away_yes_bid',
                'away_no': 'away_no_bid',
            }
            bid_key = bid_map.get(contract_type, f'{side}_yes_bid')
            current_bid_cents = odds.get(bid_key, 0) or 0

        if not current_bid_cents or current_bid_cents <= 0:
            return None

        # Model probability for our side (line 2463)
        our_prob = game['live_home_prob'] if side == 'home' else 1 - game['live_home_prob']

        # EV-based exit (lines 2466-2480)
        ov = compute_option_value(time_remaining, our_prob)
        abs_margin = abs(game.get('home_score', 0) - game.get('away_score', 0))
        haircut = compute_haircut(time_remaining, abs_margin)
        ev_hold = our_prob * 100 + ov + haircut

        bid_decimal = current_bid_cents / 100.0
        if market_source == 'Polymarket':
            exit_fee = poly_fee_cents(bid_decimal)
        else:
            exit_fee = kalshi_fee_cents(bid_decimal)
        ev_exit = current_bid_cents - exit_fee - EV_EXIT_SLIPPAGE_CENTS

        if ev_exit > ev_hold:
            if self.trace_game == game['game_id']:
                print(f"  [TRACE] EV EXIT ({market_source}): ev_exit={ev_exit:.1f} > ev_hold={ev_hold:.1f} "
                      f"(prob={our_prob:.3f} ov={ov:.1f} H={haircut:.1f} bid={current_bid_cents}c)")

            return {
                'game_id': game['game_id'],
                'side': side,
                'entry_price': entry_price,
                'exit_price': bid_decimal,
                'exit_reason': 'EV_EXIT',
                'time_remaining_sec': time_remaining,
                'contract_type': contract_type,
                'home_score': game.get('home_score', 0),
                'away_score': game.get('away_score', 0),
                'live_home_prob': game.get('live_home_prob'),
                'our_prob': our_prob,
                'ev_exit': ev_exit,
                'ev_hold': ev_hold,
                'ov': ov,
                'haircut': haircut,
                'bid_cents': current_bid_cents,
                'market_source': market_source,
            }

        return None

    # ----------------------------------------------------------
    # EXECUTION
    # ----------------------------------------------------------
    def _execute_entry(self, opp: dict, ts: datetime):
        game_id = opp['game_id']
        market_source = opp.get('market_source', 'Kalshi')
        position = {
            'game_id': game_id,
            'side': opp['side'],
            'entry_price': opp['entry_price'],
            'contract_type': opp['contract_type'],
            'entry_time': ts,
            'edge': opp['edge'],
            'our_prob': opp['our_prob'],
            'home_team': opp['home_team'],
            'away_team': opp['away_team'],
            'entry_status': opp['status'],
            'entry_time_remaining': opp['time_remaining_sec'],
            'entry_period': opp.get('period', 0),
            'entry_home_score': opp.get('home_score', 0),
            'entry_away_score': opp.get('away_score', 0),
            'pregame_home_prob': opp.get('pregame_home_prob'),
            'live_home_prob': opp.get('live_home_prob'),
            'market_source': market_source,
        }
        self.open_positions[game_id] = position
        self.capital_deployed += opp['entry_price'] * 100
        self.peak_capital = max(self.peak_capital, self.capital_deployed)

        team = opp['home_team'] if opp['side'] == 'home' else opp['away_team']
        venue_tag = f" [{market_source[:1]}]" if market_source != 'Kalshi' else ""
        if self.debug or self.trace_game == game_id:
            time_str = 'PRE' if opp['status'] == 'pre' else f"{opp['time_remaining_sec']}s"
            print(f"  >> ENTRY{venue_tag}: {opp['side'].upper()} {team[:30]} @ {opp['entry_price']*100:.0f}c "
                  f"| edge={opp['edge']:.3f} | {opp['contract_type']} | {time_str} | {ts.strftime('%H:%M:%S')}")

    def _execute_exit(self, exit_info: dict, ts: datetime):
        game_id = exit_info['game_id']
        position = self.open_positions[game_id]
        market_source = position.get('market_source', 'Kalshi')

        entry_price = position['entry_price']
        exit_price = exit_info['exit_price']

        gross_pnl = (exit_price - entry_price) * 100

        # Venue-specific fees
        if market_source == 'Polymarket':
            # Polymarket SHORT fee: fees are on raw CLOB price, not home-equivalent
            # For home bets (short away), raw CLOB entry = 1 - entry_price
            if position['side'] == 'home':
                entry_fee = poly_fee_cents(1 - entry_price)
                exit_fee = poly_fee_cents(1 - exit_price)
            else:
                entry_fee = poly_fee_cents(entry_price)
                exit_fee = poly_fee_cents(exit_price)
        else:
            entry_fee = kalshi_fee_cents(entry_price)
            exit_fee = kalshi_fee_cents(exit_price)

        net_pnl = gross_pnl - entry_fee - exit_fee

        trade = self._build_trade_record(position, exit_info, ts, gross_pnl, entry_fee, exit_fee, net_pnl)
        self.trades.append(trade)
        self.realized_pnl += net_pnl
        self.capital_deployed -= entry_price * 100

        # Set cooldown — game-time based (line 2692-2695)
        exit_game_time = exit_info.get('time_remaining_sec', 0)
        cooldown_key = (game_id, exit_info['side'])
        self.exit_cooldowns[cooldown_key] = (exit_game_time, COOLDOWN_AFTER_EXIT)

        del self.open_positions[game_id]

        team = position['home_team'] if exit_info['side'] == 'home' else position['away_team']
        venue_tag = f" [{market_source[:1]}]" if market_source != 'Kalshi' else ""
        if self.debug or self.trace_game == game_id:
            print(f"  << EV EXIT{venue_tag}: {exit_info['side'].upper()} {team[:30]} | "
                  f"Entry: {entry_price*100:.0f}c -> Exit: {exit_price*100:.0f}c | "
                  f"Gross: {gross_pnl:+.1f}c | Fees: {entry_fee+exit_fee:.1f}c | Net: {net_pnl:+.1f}c "
                  f"| {ts.strftime('%H:%M:%S')}")

    def _settle_position(self, game_id: str, game: dict, ts: datetime):
        position = self.open_positions[game_id]
        side = position['side']
        entry_price = position['entry_price']
        market_source = position.get('market_source', 'Kalshi')

        # Use schedule outcomes (source of truth)
        outcome = self.outcomes.get(game_id)
        if outcome:
            home_won = outcome['home_winner'] == 1
        else:
            # Fallback to snapshot scores
            home_won = game['home_score'] > game['away_score']

        we_won = (side == 'home' and home_won) or (side == 'away' and not home_won)
        exit_price = 1.0 if we_won else 0.0

        gross_pnl = (exit_price - entry_price) * 100

        # Venue-specific entry fee; settlement = no exit fee on both venues
        if market_source == 'Polymarket':
            if side == 'home':
                entry_fee = poly_fee_cents(1 - entry_price)
            else:
                entry_fee = poly_fee_cents(entry_price)
        else:
            entry_fee = kalshi_fee_cents(entry_price)

        exit_fee = 0.0  # Settlement = no exit fee
        net_pnl = gross_pnl - entry_fee

        exit_info = {
            'game_id': game_id,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': f"SETTLEMENT:{'WON' if we_won else 'LOST'}",
            'time_remaining_sec': 0,
            'contract_type': position['contract_type'],
            'home_score': game.get('home_score', 0),
            'away_score': game.get('away_score', 0),
            'market_source': market_source,
        }

        trade = self._build_trade_record(position, exit_info, ts, gross_pnl, entry_fee, exit_fee, net_pnl)
        self.trades.append(trade)
        self.realized_pnl += net_pnl
        self.capital_deployed -= entry_price * 100

        # Cooldown
        cooldown_key = (game_id, side)
        self.exit_cooldowns[cooldown_key] = (0, COOLDOWN_AFTER_EXIT)

        del self.open_positions[game_id]

        team = position['home_team'] if side == 'home' else position['away_team']
        emoji = "W" if we_won else "L"
        venue_tag = f" [{market_source[:1]}]" if market_source != 'Kalshi' else ""
        if self.debug or self.trace_game == game_id:
            print(f"  ** SETTLE [{emoji}]{venue_tag}: {side.upper()} {team[:30]} | "
                  f"Entry: {entry_price*100:.0f}c -> {'$1' if we_won else '$0'} | "
                  f"Gross: {gross_pnl:+.1f}c | Fee: {entry_fee:.1f}c | Net: {net_pnl:+.1f}c")

    def _build_trade_record(self, position, exit_info, ts, gross_pnl, entry_fee, exit_fee, net_pnl):
        return {
            'game_id': exit_info['game_id'],
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_info['exit_price'],
            'gross_pnl': gross_pnl,
            'entry_fee': entry_fee,
            'exit_fee': exit_fee,
            'net_pnl': net_pnl,
            'exit_reason': exit_info['exit_reason'],
            'entry_time': position['entry_time'],
            'exit_time': ts,
            'home_team': position.get('home_team', ''),
            'away_team': position.get('away_team', ''),
            'entry_status': position.get('entry_status'),
            'contract_type': position['contract_type'],
            'entry_edge': position.get('edge'),
            'entry_time_remaining': position.get('entry_time_remaining'),
            'entry_period': position.get('entry_period', 0),
            'entry_home_score': position.get('entry_home_score', 0),
            'entry_away_score': position.get('entry_away_score', 0),
            'pregame_home_prob': position.get('pregame_home_prob'),
            'live_home_prob': position.get('live_home_prob'),
            'market_source': position.get('market_source', 'Kalshi'),
        }

    def settle_remaining(self):
        """Force-settle any positions still open at end of backtest."""
        for game_id in list(self.open_positions.keys()):
            outcome = self.outcomes.get(game_id)
            position = self.open_positions[game_id]
            side = position['side']
            entry_price = position['entry_price']
            market_source = position.get('market_source', 'Kalshi')

            if outcome:
                home_won = outcome['home_winner'] == 1
                we_won = (side == 'home' and home_won) or (side == 'away' and not home_won)
                exit_price = 1.0 if we_won else 0.0
                reason = f"SETTLEMENT(force):{'WON' if we_won else 'LOST'}"
            else:
                exit_price = position['entry_price']  # Flat close
                reason = "UNRESOLVED"
                we_won = False

            gross_pnl = (exit_price - entry_price) * 100

            # Venue-specific entry fee; settlement = no exit fee
            if market_source == 'Polymarket':
                if side == 'home':
                    entry_fee = poly_fee_cents(1 - entry_price)
                else:
                    entry_fee = poly_fee_cents(entry_price)
            else:
                entry_fee = kalshi_fee_cents(entry_price)

            net_pnl = gross_pnl - entry_fee

            trade = {
                'game_id': game_id,
                'side': side,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'gross_pnl': gross_pnl,
                'entry_fee': entry_fee,
                'exit_fee': 0.0,
                'net_pnl': net_pnl,
                'exit_reason': reason,
                'entry_time': position['entry_time'],
                'exit_time': None,
                'home_team': position.get('home_team', ''),
                'away_team': position.get('away_team', ''),
                'entry_status': position.get('entry_status'),
                'contract_type': position['contract_type'],
                'entry_edge': position.get('edge'),
                'entry_time_remaining': position.get('entry_time_remaining'),
                'entry_period': position.get('entry_period', 0),
                'entry_home_score': position.get('entry_home_score', 0),
                'entry_away_score': position.get('entry_away_score', 0),
                'pregame_home_prob': position.get('pregame_home_prob'),
                'live_home_prob': position.get('live_home_prob'),
                'market_source': market_source,
            }
            self.trades.append(trade)
            self.realized_pnl += net_pnl

            team = position['home_team'] if side == 'home' else position['away_team']
            venue_tag = f" [{market_source[:1]}]" if market_source != 'Kalshi' else ""
            print(f"  !! FORCE SETTLE{venue_tag}: {side.upper()} {team[:30]} -> {reason} | Net: {net_pnl:+.1f}c")

            del self.open_positions[game_id]

    # ----------------------------------------------------------
    # MAIN RUN LOOP
    # ----------------------------------------------------------
    def run(self, start_date: str, end_date: str):
        """Run full backtest over date range."""
        print(f"\n{'='*70}")
        print(f"  BACKTEST: {start_date} to {end_date}")
        print(f"{'='*70}")

        # Load outcomes
        self.load_outcomes(start_date, end_date)
        print(f"  Game outcomes loaded: {len(self.outcomes)}")

        # Ensure index exists for performance
        try:
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON live_snapshots(timestamp)")
            self.conn.commit()
        except Exception:
            pass

        # Count snapshots
        count = self.conn.execute("""
            SELECT COUNT(*) FROM live_snapshots
            WHERE date(timestamp) BETWEEN ? AND ?
        """, (start_date, end_date)).fetchone()[0]

        print(f"  Snapshots to process: {count:,}")
        print(f"  Score freshness guard: {MAX_ESPN_SCORE_STALE_SEC}s")
        print(f"  Venue: {self.venue}")
        e_start = self.min_edge if self.min_edge is not None else 0.08
        print(f"  Edge: {get_required_edge(0.5, 2400, e_start=self.min_edge)*100:.0f}% pregame -> {get_required_edge(0.5, 240, e_start=self.min_edge)*100:.0f}% @ 4:00")
        print(f"  Spread: <=3c early, <=2c late | Price: {MIN_ENTRY_PRICE*100:.0f}-{MAX_ENTRY_PRICE*100:.0f}c")
        print(f"  OV: {OV_SCALE} x N^{OV_EXPONENT} x (1 + {OV_PROB_COEFF}*p*(1-p))")
        print(f"  Cooldown: {COOLDOWN_AFTER_EXIT}s game-time after exit")
        if self.min_edge is not None:
            print(f"  *** CUSTOM EDGE START: {self.min_edge:.0%} ***")
        print(f"{'='*70}\n")

        # Process snapshots in chronological order
        cursor = self.conn.execute("""
            SELECT * FROM live_snapshots
            WHERE date(timestamp) BETWEEN ? AND ?
            ORDER BY timestamp, game_id
        """, (start_date, end_date))

        processed = 0
        last_progress = 0
        start_time = datetime.now()

        for snap in cursor:
            self.process_snapshot(snap)
            processed += 1

            if processed - last_progress >= 500000:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = processed / elapsed if elapsed > 0 else 0
                pct = processed / count * 100 if count > 0 else 0
                eta = (count - processed) / rate if rate > 0 else 0
                print(f"  Progress: {processed:,}/{count:,} ({pct:.0f}%) | "
                      f"{rate:,.0f} rows/sec | ETA: {eta:.0f}s | "
                      f"Trades: {len(self.trades)} | Open: {len(self.open_positions)}")
                last_progress = processed

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n  Processed {processed:,} snapshots in {elapsed:.1f}s ({processed/elapsed:,.0f}/sec)")

        # Force-settle remaining positions
        remaining = len(self.open_positions)
        if remaining > 0:
            print(f"\n  Force-settling {remaining} remaining positions...")
            self.settle_remaining()

    # ----------------------------------------------------------
    # RESULTS OUTPUT
    # ----------------------------------------------------------
    def print_results(self):
        if not self.trades:
            print("\n  No trades executed.")
            return

        edge_label = f" (EDGE={self.min_edge:.0%})" if self.min_edge is not None else ""
        print(f"\n{'='*70}")
        print(f"  RESULTS SUMMARY{edge_label}")
        print(f"{'='*70}")

        total = len(self.trades)
        settlements = [t for t in self.trades if 'SETTLEMENT' in t['exit_reason']]
        ev_exits = [t for t in self.trades if t['exit_reason'] == 'EV_EXIT']
        winners = [t for t in self.trades if t['net_pnl'] > 0]
        losers = [t for t in self.trades if t['net_pnl'] < 0]
        flat = [t for t in self.trades if abs(t['net_pnl']) < 0.01]

        total_gross = sum(t['gross_pnl'] for t in self.trades)
        total_entry_fees = sum(t['entry_fee'] for t in self.trades)
        total_exit_fees = sum(t['exit_fee'] for t in self.trades)
        total_fees = total_entry_fees + total_exit_fees
        total_net = sum(t['net_pnl'] for t in self.trades)

        print(f"\n  Total trades:     {total}")
        print(f"  Winners:          {len(winners)} ({len(winners)/total*100:.1f}%)")
        print(f"  Losers:           {len(losers)} ({len(losers)/total*100:.1f}%)")
        if flat:
            print(f"  Flat:             {len(flat)}")
        print(f"  Peak capital:     {self.peak_capital:.0f}c")

        print(f"\n  Gross P/L:        {total_gross:+.1f}c")
        print(f"  Entry fees:       -{total_entry_fees:.1f}c")
        print(f"  Exit fees:        -{total_exit_fees:.1f}c")
        print(f"  Total fees:       -{total_fees:.1f}c")
        print(f"  ----------------------------")
        print(f"  Net P/L:          {total_net:+.1f}c")
        if self.peak_capital > 0:
            print(f"  ROI (peak cap):   {total_net / self.peak_capital * 100:+.1f}%")

        # --- By exit type ---
        print(f"\n  BY EXIT TYPE:")
        for label, subset in [("Settlements", settlements), ("EV Exits", ev_exits)]:
            if subset:
                sub_net = sum(t['net_pnl'] for t in subset)
                sub_winners = len([t for t in subset if t['net_pnl'] > 0])
                sub_gross = sum(t['gross_pnl'] for t in subset)
                sub_fees = sum(t['entry_fee'] + t['exit_fee'] for t in subset)
                print(f"    {label:15s}: {len(subset):3d} trades | Win: {sub_winners}/{len(subset)} "
                      f"({sub_winners/len(subset)*100:.0f}%) | Gross: {sub_gross:+.1f}c | "
                      f"Fees: {sub_fees:.1f}c | Net: {sub_net:+.1f}c")

        # --- Settlement wins/losses breakdown ---
        settle_wins = [t for t in settlements if 'WON' in t['exit_reason']]
        settle_losses = [t for t in settlements if 'LOST' in t['exit_reason']]
        if settlements:
            print(f"\n  SETTLEMENT DETAIL:")
            if settle_wins:
                avg_entry = sum(t['entry_price'] for t in settle_wins) / len(settle_wins)
                print(f"    Wins:  {len(settle_wins):3d} | Avg entry: {avg_entry*100:.0f}c | "
                      f"Net: {sum(t['net_pnl'] for t in settle_wins):+.1f}c")
            if settle_losses:
                avg_entry = sum(t['entry_price'] for t in settle_losses) / len(settle_losses)
                print(f"    Losses: {len(settle_losses):3d} | Avg entry: {avg_entry*100:.0f}c | "
                      f"Net: {sum(t['net_pnl'] for t in settle_losses):+.1f}c")

        # --- By entry type ---
        print(f"\n  BY ENTRY TYPE:")
        pregame_trades = [t for t in self.trades if t['entry_status'] == 'pre']
        live_trades = [t for t in self.trades if t['entry_status'] == 'in']
        for label, subset in [("Pregame", pregame_trades), ("Live", live_trades)]:
            if subset:
                sub_net = sum(t['net_pnl'] for t in subset)
                sub_winners = len([t for t in subset if t['net_pnl'] > 0])
                sub_gross = sum(t['gross_pnl'] for t in subset)
                sub_fees = sum(t['entry_fee'] + t['exit_fee'] for t in subset)
                avg_edge = sum(t['entry_edge'] or 0 for t in subset) / len(subset)
                print(f"    {label:15s}: {len(subset):3d} trades | Win: {sub_winners}/{len(subset)} "
                      f"({sub_winners/len(subset)*100:.0f}%) | Gross: {sub_gross:+.1f}c | "
                      f"Fees: {sub_fees:.1f}c | Net: {sub_net:+.1f}c | Avg edge: {avg_edge:.3f}")

        # --- By contract type ---
        print(f"\n  BY CONTRACT TYPE:")
        contract_types = set(t['contract_type'] for t in self.trades)
        for ct in sorted(contract_types):
            subset = [t for t in self.trades if t['contract_type'] == ct]
            sub_net = sum(t['net_pnl'] for t in subset)
            sub_winners = len([t for t in subset if t['net_pnl'] > 0])
            print(f"    {ct:15s}: {len(subset):3d} trades | Win: {sub_winners}/{len(subset)} | Net: {sub_net:+.1f}c")

        # --- By side ---
        print(f"\n  BY SIDE:")
        for side_label in ['home', 'away']:
            subset = [t for t in self.trades if t['side'] == side_label]
            if subset:
                sub_net = sum(t['net_pnl'] for t in subset)
                sub_winners = len([t for t in subset if t['net_pnl'] > 0])
                print(f"    {side_label:15s}: {len(subset):3d} trades | Win: {sub_winners}/{len(subset)} "
                      f"({sub_winners/len(subset)*100:.0f}%) | Net: {sub_net:+.1f}c")

        # --- By venue ---
        venues = set(t.get('market_source', 'Kalshi') for t in self.trades)
        if len(venues) > 1 or 'Polymarket' in venues:
            print(f"\n  BY VENUE:")
            for venue_label in sorted(venues):
                subset = [t for t in self.trades if t.get('market_source', 'Kalshi') == venue_label]
                if subset:
                    sub_net = sum(t['net_pnl'] for t in subset)
                    sub_gross = sum(t['gross_pnl'] for t in subset)
                    sub_fees = sum(t['entry_fee'] + t['exit_fee'] for t in subset)
                    sub_winners = len([t for t in subset if t['net_pnl'] > 0])
                    avg_edge = sum(t['entry_edge'] or 0 for t in subset) / len(subset)
                    print(f"    {venue_label:15s}: {len(subset):3d} trades | Win: {sub_winners}/{len(subset)} "
                          f"({sub_winners/len(subset)*100:.0f}%) | Gross: {sub_gross:+.1f}c | "
                          f"Fees: {sub_fees:.1f}c | Net: {sub_net:+.1f}c | Avg edge: {avg_edge:.3f}")

        # --- Daily breakdown ---
        print(f"\n  DAILY BREAKDOWN:")
        daily = {}
        for t in self.trades:
            if t['entry_time']:
                day = t['entry_time'].strftime('%Y-%m-%d')
            else:
                day = 'unknown'
            if day not in daily:
                daily[day] = {'trades': 0, 'net': 0, 'gross': 0, 'fees': 0, 'winners': 0,
                              'kalshi': 0, 'poly': 0}
            daily[day]['trades'] += 1
            daily[day]['net'] += t['net_pnl']
            daily[day]['gross'] += t['gross_pnl']
            daily[day]['fees'] += t['entry_fee'] + t['exit_fee']
            if t['net_pnl'] > 0:
                daily[day]['winners'] += 1
            if t.get('market_source', 'Kalshi') == 'Polymarket':
                daily[day]['poly'] += 1
            else:
                daily[day]['kalshi'] += 1

        for day in sorted(daily.keys()):
            d = daily[day]
            wr = d['winners'] / d['trades'] * 100 if d['trades'] > 0 else 0
            venue_str = f" (K:{d['kalshi']} P:{d['poly']})" if d['poly'] > 0 else ""
            print(f"    {day}: {d['trades']:3d} trades{venue_str} | Win: {wr:.0f}% | "
                  f"Gross: {d['gross']:+.1f}c | Fees: {d['fees']:.1f}c | Net: {d['net']:+.1f}c")

        # --- Skip reasons ---
        print(f"\n  SNAPSHOT SKIP COUNTS:")
        for reason, cnt in sorted(self.entry_skips.items(), key=lambda x: -x[1]):
            if cnt > 0:
                print(f"    {reason:25s}: {cnt:>10,}")

    # ----------------------------------------------------------
    # ANALYSIS: Edge buckets × Time buckets × Edge thresholds
    # ----------------------------------------------------------
    def print_analysis(self):
        """Break down win rate and P/L by edge bucket, entry time bucket, and their cross-tab."""
        if not self.trades:
            print("\n  No trades to analyze.")
            return

        print(f"\n{'='*70}")
        print(f"  EDGE / TIME / THRESHOLD ANALYSIS")
        print(f"{'='*70}")

        # ---- Define buckets ----
        edge_buckets = [
            ('6-8%',   0.06, 0.08),
            ('8-10%',  0.08, 0.10),
            ('10-12%', 0.10, 0.12),
            ('12-15%', 0.12, 0.15),
            ('15-20%', 0.15, 0.20),
            ('20%+',   0.20, 9.99),
        ]

        time_buckets = [
            ('Pregame',       2400, 2401),   # exactly 2400 = pregame
            ('H1 early (30-40m)', 1800, 2400),
            ('H1 late (20-30m)',  1200, 1800),
            ('H2 early (10-20m)',  600, 1200),
            ('H2 late (4-10m)',    240,  600),
        ]

        def _bucket_stats(subset, label):
            """Print one row of stats."""
            if not subset:
                return
            n = len(subset)
            winners = len([t for t in subset if t['net_pnl'] > 0])
            losers = len([t for t in subset if t['net_pnl'] < 0])
            net = sum(t['net_pnl'] for t in subset)
            gross = sum(t['gross_pnl'] for t in subset)
            fees = sum(t['entry_fee'] + t['exit_fee'] for t in subset)
            avg_edge = sum(t['entry_edge'] or 0 for t in subset) / n
            avg_price = sum(t['entry_price'] for t in subset) / n
            wr = winners / n * 100
            print(f"    {label:22s}: {n:3d} trades | W/L: {winners}/{losers} "
                  f"({wr:4.0f}%) | Net: {net:+7.1f}c | Avg edge: {avg_edge:.3f} | "
                  f"Avg price: {avg_price*100:.0f}c")

        # ---- 1) BY EDGE BUCKET ----
        print(f"\n  BY ENTRY EDGE:")
        for label, lo, hi in edge_buckets:
            subset = [t for t in self.trades if t['entry_edge'] is not None and lo <= t['entry_edge'] < hi]
            _bucket_stats(subset, label)

        # ---- 2) BY ENTRY TIME BUCKET ----
        print(f"\n  BY ENTRY TIME:")
        for label, lo_sec, hi_sec in time_buckets:
            subset = [t for t in self.trades
                      if t.get('entry_time_remaining') is not None
                      and lo_sec <= t['entry_time_remaining'] < hi_sec]
            _bucket_stats(subset, label)

        # Special: pregame exact match (entry_time_remaining == 2400 or entry_status == 'pre')
        pregame = [t for t in self.trades if t.get('entry_status') == 'pre']
        live = [t for t in self.trades if t.get('entry_status') == 'in']
        print(f"\n  PREGAME vs LIVE (clean split):")
        _bucket_stats(pregame, 'Pregame entries')
        _bucket_stats(live, 'Live entries')

        # ---- 3) BY ENTRY PRICE BUCKET ----
        print(f"\n  BY ENTRY PRICE:")
        price_buckets = [
            ('10-30c',  0.10, 0.30),
            ('30-50c',  0.30, 0.50),
            ('50-70c',  0.50, 0.70),
            ('70-90c',  0.70, 0.91),
        ]
        for label, lo, hi in price_buckets:
            subset = [t for t in self.trades if lo <= t['entry_price'] < hi]
            _bucket_stats(subset, label)

        # ---- 4) CROSS-TAB: Edge × Time ----
        print(f"\n  CROSS-TAB: EDGE × TIME")
        print(f"    {'':22s}", end='')
        time_labels_short = ['Pregame', 'H1early', 'H1late', 'H2early', 'H2late']
        for tl in time_labels_short:
            print(f" | {tl:>12s}", end='')
        print(f" | {'TOTAL':>12s}")
        print(f"    {'='*22}", end='')
        for _ in time_labels_short:
            print(f" | {'='*12}", end='')
        print(f" | {'='*12}")

        for elabel, elo, ehi in edge_buckets:
            edge_trades = [t for t in self.trades if t['entry_edge'] is not None and elo <= t['entry_edge'] < ehi]
            print(f"    {elabel:22s}", end='')

            for i, (tlabel, tlo, thi) in enumerate(time_buckets):
                cell = [t for t in edge_trades
                        if t.get('entry_time_remaining') is not None
                        and tlo <= t['entry_time_remaining'] < thi]
                if cell:
                    n = len(cell)
                    w = len([t for t in cell if t['net_pnl'] > 0])
                    net = sum(t['net_pnl'] for t in cell)
                    print(f" | {w}/{n} {net:+.0f}c", end='')
                    # Pad to 12
                    used = len(f"{w}/{n} {net:+.0f}c")
                    print(' ' * max(0, 12 - used), end='')
                else:
                    print(f" | {'--':>12s}", end='')

            # Row total
            if edge_trades:
                n = len(edge_trades)
                w = len([t for t in edge_trades if t['net_pnl'] > 0])
                net = sum(t['net_pnl'] for t in edge_trades)
                print(f" | {w}/{n} {net:+.0f}c", end='')
            else:
                print(f" | {'--':>12s}", end='')
            print()

        # Column totals
        print(f"    {'TOTAL':22s}", end='')
        for i, (tlabel, tlo, thi) in enumerate(time_buckets):
            cell = [t for t in self.trades
                    if t.get('entry_time_remaining') is not None
                    and tlo <= t['entry_time_remaining'] < thi]
            if cell:
                n = len(cell)
                w = len([t for t in cell if t['net_pnl'] > 0])
                net = sum(t['net_pnl'] for t in cell)
                s = f"{w}/{n} {net:+.0f}c"
                print(f" | {s:>12s}", end='')
            else:
                print(f" | {'--':>12s}", end='')
        # Grand total
        n = len(self.trades)
        w = len([t for t in self.trades if t['net_pnl'] > 0])
        net = sum(t['net_pnl'] for t in self.trades)
        print(f" | {w}/{n} {net:+.0f}c")

        # ---- 5) CUMULATIVE EDGE THRESHOLD SWEEP ----
        print(f"\n  EDGE THRESHOLD SWEEP (only trades >= threshold):")
        print(f"    {'Threshold':12s} {'Trades':>7s} {'Win%':>6s} {'Net P/L':>9s} {'Avg Edge':>9s} {'$/Trade':>9s}")
        print(f"    {'='*55}")
        thresholds = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
        for thresh in thresholds:
            subset = [t for t in self.trades if t['entry_edge'] is not None and t['entry_edge'] >= thresh]
            if not subset:
                continue
            n = len(subset)
            w = len([t for t in subset if t['net_pnl'] > 0])
            net = sum(t['net_pnl'] for t in subset)
            avg_edge = sum(t['entry_edge'] for t in subset) / n
            per_trade = net / n
            wr = w / n * 100
            print(f"    {thresh:>10.0%}   {n:>7d} {wr:>5.1f}% {net:>+8.1f}c {avg_edge:>8.3f} {per_trade:>+8.2f}c")

        # ---- 6) TIME REMAINING SWEEP (live only) ----
        print(f"\n  TIME REMAINING SWEEP (live entries only, entered at or before cutoff):")
        print(f"    {'Max time':12s} {'Trades':>7s} {'Win%':>6s} {'Net P/L':>9s} {'$/Trade':>9s}")
        print(f"    {'='*47}")
        time_cutoffs = [2400, 2100, 1800, 1500, 1200, 900, 600, 240]
        for cutoff in time_cutoffs:
            subset = [t for t in self.trades
                      if t.get('entry_status') == 'in'
                      and t.get('entry_time_remaining') is not None
                      and t['entry_time_remaining'] <= cutoff]
            if not subset:
                continue
            n = len(subset)
            w = len([t for t in subset if t['net_pnl'] > 0])
            net = sum(t['net_pnl'] for t in subset)
            per_trade = net / n
            wr = w / n * 100
            mins = cutoff // 60
            secs = cutoff % 60
            label = f"<= {mins}:{secs:02d}"
            print(f"    {label:>10s}   {n:>7d} {wr:>5.1f}% {net:>+8.1f}c {per_trade:>+8.2f}c")

    def print_trade_log(self, max_trades: int = 100):
        """Print detailed trade log."""
        print(f"\n{'='*70}")
        print(f"  TRADE LOG ({min(len(self.trades), max_trades)} of {len(self.trades)})")
        print(f"{'='*70}")

        print(f"  {'#':>3} {'SIDE':5} {'TEAM':28} {'WHEN':8} {'ENTRY':>6} {'EXIT':>6} "
              f"{'GROSS':>7} {'FEES':>6} {'NET':>7} {'VENUE':5} {'REASON':22}")
        print(f"  {'='*113}")

        for i, t in enumerate(self.trades[:max_trades]):
            team = t['home_team'] if t['side'] == 'home' else t['away_team']
            team = (team[:28] if team else '???')
            entry_type = 'PRE' if t['entry_status'] == 'pre' else f"{t.get('entry_time_remaining', '?')}s"
            fees = t['entry_fee'] + t['exit_fee']
            reason = t['exit_reason'][:22]
            marker = "W" if t['net_pnl'] > 0 else "L" if t['net_pnl'] < 0 else "-"
            venue = t.get('market_source', 'Kalshi')[:1]  # K or P

            print(f"  {i+1:>3} {t['side']:5} {team:28} {entry_type:8} "
                  f"{t['entry_price']*100:5.0f}c {t['exit_price']*100:5.0f}c "
                  f"{t['gross_pnl']:+6.1f}c {fees:5.1f}c {t['net_pnl']:+6.1f}c [{marker}] {venue:1s} {reason}")


# ============================================================
# DIAGNOSTICS ENGINE — Model calibration & structural analysis
# Uses ALL snapshots (not just traded games) to avoid selection bias
# Streams game-by-game to avoid OOM on large datasets
# ============================================================

class DiagnosticsEngine:
    """Runs post-mortem diagnostics on live_snapshots data.
    
    Answers the binary question: Is this system structurally negative EV,
    or is it positive EV being destroyed by execution friction?
    
    Processes one game at a time to stay within memory limits.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

        # game_id -> {'home_winner': 0/1, 'home_score': int, 'away_score': int}
        self.outcomes: Dict[str, dict] = {}

        # Game IDs to process (populated during run)
        self.game_ids: List[str] = []

        # Accumulated results across games (lightweight — just numbers/small dicts)
        # Phase 1: Calibration
        self.cal_buckets = defaultdict(lambda: {'count': 0, 'sum_model': 0.0, 'sum_market': 0.0,
                                                 'wins': 0, 'has_market': 0, 'sum_market_wins': 0})
        self.cal_games_used = 0
        self.cal_games_skipped_no_outcome = 0
        self.cal_games_skipped_no_model = 0

        # Phase 1B: Brier
        self.brier_model_sum = 0.0
        self.brier_market_sum = 0.0
        self.brier_n_model = 0
        self.brier_n_market = 0

        # Phase 2: Hold-to-settlement entries (kept in memory — small, one per game per threshold)
        self.thresholds = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
        self.h2s_entries: Dict[float, List[dict]] = {t: [] for t in self.thresholds}

        # Phase 6: Adverse selection accumulators
        self.adv_thresholds = [0.06, 0.10, 0.15]
        self.adv_lookaheads = [5, 10, 30, 60]
        # {thresh: {lookahead: [move1, move2, ...]}}
        self.adv_movements: Dict[float, Dict[int, List[float]]] = {
            t: {la: [] for la in self.adv_lookaheads} for t in self.adv_thresholds
        }

        # Phase 7: Stale edge / edge decay test
        self.stale_thresholds = [0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
        self.stale_lookaheads = [2, 5, 10]
        # 7A: Edge decay — {thresh: {lookahead: [edge_at_future_tick, ...]}}
        #   edge_at_future = model_prob_at_signal - market_ask_at_future / 100
        self.edge_at_signal: Dict[float, List[float]] = {t: [] for t in self.stale_thresholds}
        self.edge_decay: Dict[float, Dict[int, List[float]]] = {
            t: {la: [] for la in self.stale_lookaheads} for t in self.stale_thresholds
        }
        # Also track fully-updated edge (model AND market at future tick)
        self.edge_updated: Dict[float, Dict[int, List[float]]] = {
            t: {la: [] for la in self.stale_lookaheads} for t in self.stale_thresholds
        }
        # 7B: Score proximity — {thresh: {'within_5': n, 'within_10': n, 'within_20': n, 'total': n}}
        self.score_proximity: Dict[float, Dict[str, int]] = {
            t: {'within_5': 0, 'within_10': 0, 'within_20': 0, 'total': 0, 'no_change': 0}
            for t in self.stale_thresholds
        }
        # 7C: Hold-to-settlement filtered (exclude entries near score changes)
        #   {thresh: {filter_secs: [entry_dicts]}}
        self.h2s_filtered: Dict[float, Dict[int, List[dict]]] = {
            t: {5: [], 10: [], 20: []} for t in self.stale_thresholds
        }

    # ----------------------------------------------------------
    # DATA LOADING
    # ----------------------------------------------------------
    def load_outcomes(self, start_date: str, end_date: str):
        """Load game outcomes from schedule_2025_26."""
        end_plus = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=2)).strftime('%Y-%m-%d')
        cursor = self.conn.execute("""
            SELECT game_id, home_winner, home_score, away_score
            FROM schedule_2025_26
            WHERE game_date BETWEEN ? AND ?
            AND home_score IS NOT NULL
        """, (start_date, end_plus))

        for row in cursor:
            self.outcomes[str(row['game_id'])] = {
                'home_winner': row['home_winner'],
                'home_score': row['home_score'],
                'away_score': row['away_score'],
            }

    def load_game_ids(self, start_date: str, end_date: str):
        """Get list of distinct game_ids in the date range."""
        cursor = self.conn.execute("""
            SELECT DISTINCT game_id FROM live_snapshots
            WHERE date(timestamp) BETWEEN ? AND ?
            ORDER BY game_id
        """, (start_date, end_date))
        self.game_ids = [str(row['game_id']) for row in cursor]

    def load_game_snapshots(self, game_id: str, start_date: str, end_date: str) -> List[dict]:
        """Load all snapshots for a single game. Returns list of dicts."""
        cursor = self.conn.execute("""
            SELECT game_id, game_status, timestamp, time_remaining_sec,
                   home_score, away_score, period,
                   pregame_home_prob, live_home_prob,
                   home_yes_bid, home_yes_ask, away_yes_bid, away_yes_ask,
                   home_no_bid, home_no_ask, away_no_bid, away_no_ask
            FROM live_snapshots
            WHERE game_id = ? AND date(timestamp) BETWEEN ? AND ?
            ORDER BY timestamp
        """, (game_id, start_date, end_date))

        snaps = []
        for row in cursor:
            snaps.append({
                'game_id': str(row['game_id']),
                'status': row['game_status'],
                'timestamp': row['timestamp'],
                'time_remaining_sec': row['time_remaining_sec'],
                'home_score': row['home_score'] or 0,
                'away_score': row['away_score'] or 0,
                'period': row['period'] or 0,
                'pregame_home_prob': row['pregame_home_prob'],
                'live_home_prob': row['live_home_prob'],
                'home_yes_bid': row['home_yes_bid'],
                'home_yes_ask': row['home_yes_ask'],
                'away_yes_bid': row['away_yes_bid'],
                'away_yes_ask': row['away_yes_ask'],
                'home_no_bid': row['home_no_bid'],
                'home_no_ask': row['home_no_ask'],
                'away_no_bid': row['away_no_bid'],
                'away_no_ask': row['away_no_ask'],
            })
        return snaps

    # ----------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------
    @staticmethod
    def _get_best_ask_for_side(snap: dict, side: str) -> Optional[float]:
        """Get cheapest entry ask (cents) for a side, replicating YES/NO arbitrage."""
        if side == 'home':
            candidates = []
            if snap.get('home_yes_ask') and snap['home_yes_ask'] > 0:
                candidates.append(snap['home_yes_ask'])
            if snap.get('away_no_ask') and snap['away_no_ask'] > 0:
                candidates.append(snap['away_no_ask'])
            return min(candidates) if candidates else None
        else:
            candidates = []
            if snap.get('away_yes_ask') and snap['away_yes_ask'] > 0:
                candidates.append(snap['away_yes_ask'])
            if snap.get('home_no_ask') and snap['home_no_ask'] > 0:
                candidates.append(snap['home_no_ask'])
            return min(candidates) if candidates else None

    @staticmethod
    def _get_best_bid_for_side(snap: dict, side: str) -> Optional[float]:
        """Get best exit bid (cents) for a side."""
        if side == 'home':
            candidates = []
            if snap.get('home_yes_bid') and snap['home_yes_bid'] > 0:
                candidates.append(snap['home_yes_bid'])
            if snap.get('away_no_bid') and snap['away_no_bid'] > 0:
                candidates.append(snap['away_no_bid'])
            return max(candidates) if candidates else None
        else:
            candidates = []
            if snap.get('away_yes_bid') and snap['away_yes_bid'] > 0:
                candidates.append(snap['away_yes_bid'])
            if snap.get('home_no_bid') and snap['home_no_bid'] > 0:
                candidates.append(snap['home_no_bid'])
            return max(candidates) if candidates else None

    @staticmethod
    def _get_spread_for_side(snap: dict, side: str) -> Optional[int]:
        """Get tightest spread (cents) for a side."""
        if side == 'home':
            spreads = []
            if snap.get('home_yes_ask') and snap.get('home_yes_bid'):
                spreads.append(snap['home_yes_ask'] - snap['home_yes_bid'])
            if snap.get('away_no_ask') and snap.get('away_no_bid'):
                spreads.append(snap['away_no_ask'] - snap['away_no_bid'])
            return min(spreads) if spreads else None
        else:
            spreads = []
            if snap.get('away_yes_ask') and snap.get('away_yes_bid'):
                spreads.append(snap['away_yes_ask'] - snap['away_yes_bid'])
            if snap.get('home_no_ask') and snap.get('home_no_bid'):
                spreads.append(snap['home_no_ask'] - snap['home_no_bid'])
            return min(spreads) if spreads else None

    def _get_market_mid_home(self, snap: dict) -> Optional[float]:
        """Get market mid for home win probability (decimal 0-1)."""
        best_ask = self._get_best_ask_for_side(snap, 'home')
        best_bid = self._get_best_bid_for_side(snap, 'home')
        if best_ask is not None and best_bid is not None:
            return (best_ask + best_bid) / 200.0
        return None

    # ----------------------------------------------------------
    # PER-GAME PROCESSING — runs all phases on one game's snapshots
    # ----------------------------------------------------------
    def process_game(self, game_id: str, snaps: List[dict]):
        """Run all phase computations for a single game's snapshots."""
        outcome = self.outcomes.get(game_id)

        # ---- Phase 1A: Calibration (one data point per game) ----
        if not outcome:
            self.cal_games_skipped_no_outcome += 1
        else:
            home_won = outcome['home_winner'] == 1
            model_prob = None
            market_mid = None

            # Find first live snapshot with model prob
            for s in snaps:
                if s['status'] == 'in' and s['live_home_prob'] is not None:
                    model_prob = s['live_home_prob']
                    market_mid = self._get_market_mid_home(s)
                    break

            # Fallback to pregame
            if model_prob is None:
                for s in snaps:
                    if s['pregame_home_prob'] is not None:
                        model_prob = s['pregame_home_prob']
                        market_mid = self._get_market_mid_home(s)
                        break

            if model_prob is None:
                self.cal_games_skipped_no_model += 1
            else:
                self.cal_games_used += 1

                bucket_idx = max(0, min(19, int(model_prob * 20)))
                bucket_lo = bucket_idx * 0.05
                bucket_key = f"{bucket_lo:.2f}-{bucket_lo+0.05:.2f}"

                self.cal_buckets[bucket_key]['count'] += 1
                self.cal_buckets[bucket_key]['sum_model'] += model_prob
                if home_won:
                    self.cal_buckets[bucket_key]['wins'] += 1

                if market_mid is not None:
                    self.cal_buckets[bucket_key]['has_market'] += 1
                    self.cal_buckets[bucket_key]['sum_market'] += market_mid
                    if home_won:
                        self.cal_buckets[bucket_key]['sum_market_wins'] += 1

                # Phase 1B: Brier
                home_won_f = 1.0 if home_won else 0.0
                self.brier_model_sum += (model_prob - home_won_f) ** 2
                self.brier_n_model += 1

                if market_mid is not None:
                    self.brier_market_sum += (market_mid - home_won_f) ** 2
                    self.brier_n_market += 1

        # ---- Phase 2: Hold-to-settlement ----
        if outcome:
            home_won = outcome['home_winner'] == 1

            for thresh in self.thresholds:
                entered_sides = set()

                for s in snaps:
                    status = s['status']

                    if status == 'in':
                        model_home = s['live_home_prob']
                        time_remaining = s['time_remaining_sec'] or 0
                    elif status == 'pre':
                        model_home = s['pregame_home_prob']
                        time_remaining = 2400
                    else:
                        continue

                    if model_home is None:
                        continue

                    for side in ['home', 'away']:
                        if side in entered_sides:
                            continue

                        our_prob = model_home if side == 'home' else 1 - model_home
                        best_ask_cents = self._get_best_ask_for_side(s, side)
                        if best_ask_cents is None:
                            continue

                        entry_price = best_ask_cents / 100.0
                        if entry_price < MIN_ENTRY_PRICE or entry_price > MAX_ENTRY_PRICE:
                            continue

                        edge = our_prob - entry_price
                        required = get_required_edge(entry_price, time_remaining, e_start=thresh)
                        if edge < required:
                            continue

                        spread = self._get_spread_for_side(s, side)
                        if spread is None:
                            continue
                        max_spread = 3 if time_remaining >= 1200 else 2
                        if spread > max_spread:
                            continue

                        if status == 'in' and time_remaining < MIN_TIME_REMAINING:
                            continue
                        if s['period'] > 2:
                            continue

                        we_won = (side == 'home' and home_won) or (side == 'away' and not home_won)
                        settlement = 1.0 if we_won else 0.0
                        entry_fee = kalshi_fee_cents(entry_price)
                        gross_pnl = (settlement - entry_price) * 100
                        net_pnl = gross_pnl - entry_fee

                        margin = abs(s['home_score'] - s['away_score'])

                        self.h2s_entries[thresh].append({
                            'game_id': game_id,
                            'side': side,
                            'entry_price': entry_price,
                            'our_prob': our_prob,
                            'edge': edge,
                            'time_remaining': time_remaining,
                            'status': status,
                            'margin': margin,
                            'we_won': we_won,
                            'gross_pnl': gross_pnl,
                            'entry_fee': entry_fee,
                            'net_pnl': net_pnl,
                            'home_score': s['home_score'],
                            'away_score': s['away_score'],
                        })
                        entered_sides.add(side)

                    if len(entered_sides) >= 2:
                        break

        # ---- Phase 6: Adverse selection ----
        live_snaps = [s for s in snaps if s['status'] == 'in'
                      and s['live_home_prob'] is not None]

        for thresh in self.adv_thresholds:
            for i, s in enumerate(live_snaps):
                model_home = s['live_home_prob']
                time_remaining = s['time_remaining_sec'] or 0

                if time_remaining < MIN_TIME_REMAINING:
                    continue

                for side in ['home', 'away']:
                    our_prob = model_home if side == 'home' else 1 - model_home
                    ask_cents = self._get_best_ask_for_side(s, side)
                    if ask_cents is None or ask_cents <= 0:
                        continue

                    entry_price = ask_cents / 100.0
                    if entry_price < MIN_ENTRY_PRICE or entry_price > MAX_ENTRY_PRICE:
                        continue

                    edge = our_prob - entry_price
                    if edge < thresh:
                        continue

                    for la in self.adv_lookaheads:
                        if i + la < len(live_snaps):
                            future_snap = live_snaps[i + la]
                            future_ask = self._get_best_ask_for_side(future_snap, side)
                            if future_ask is not None and future_ask > 0:
                                move = future_ask - ask_cents
                                self.adv_movements[thresh][la].append(move)

        # ---- Phase 7: Stale edge / edge decay ----
        # Precompute: for each live snapshot index, how many ticks until next score change?
        if live_snaps:
            ticks_to_score_change = [None] * len(live_snaps)
            last_change_idx = None
            # Walk backwards to find next score change for each position
            for i in range(len(live_snaps) - 1, -1, -1):
                if i < len(live_snaps) - 1:
                    s_cur = live_snaps[i]
                    s_next = live_snaps[i + 1]
                    if (s_next['home_score'] != s_cur['home_score'] or
                            s_next['away_score'] != s_cur['away_score']):
                        last_change_idx = i + 1
                if last_change_idx is not None:
                    ticks_to_score_change[i] = last_change_idx - i
                # else remains None (no future score change in this game's data)

            for thresh in self.stale_thresholds:
                entered_sides_7 = set()  # One entry per side per game for H2S filtered

                for i, s in enumerate(live_snaps):
                    model_home = s['live_home_prob']
                    time_remaining = s['time_remaining_sec'] or 0

                    if time_remaining < MIN_TIME_REMAINING:
                        continue

                    for side in ['home', 'away']:
                        our_prob = model_home if side == 'home' else 1 - model_home
                        ask_cents = self._get_best_ask_for_side(s, side)
                        if ask_cents is None or ask_cents <= 0:
                            continue

                        entry_price = ask_cents / 100.0
                        if entry_price < MIN_ENTRY_PRICE or entry_price > MAX_ENTRY_PRICE:
                            continue

                        edge = our_prob - entry_price
                        required = get_required_edge(entry_price, time_remaining, e_start=thresh)
                        if edge < required:
                            continue

                        spread = self._get_spread_for_side(s, side)
                        if spread is None:
                            continue
                        max_spread = 3 if time_remaining >= 1200 else 2
                        if spread > max_spread:
                            continue

                        if s['period'] > 2:
                            continue

                        # --- 7A: Edge decay ---
                        self.edge_at_signal[thresh].append(edge)

                        for la in self.stale_lookaheads:
                            if i + la < len(live_snaps):
                                fut = live_snaps[i + la]
                                # Same model prob, future market price
                                fut_ask = self._get_best_ask_for_side(fut, side)
                                if fut_ask is not None and fut_ask > 0:
                                    edge_decay = our_prob - fut_ask / 100.0
                                    self.edge_decay[thresh][la].append(edge_decay)

                                    # Fully updated: future model AND future market
                                    fut_model_home = fut['live_home_prob']
                                    if fut_model_home is not None:
                                        fut_our_prob = fut_model_home if side == 'home' else 1 - fut_model_home
                                        edge_updated = fut_our_prob - fut_ask / 100.0
                                        self.edge_updated[thresh][la].append(edge_updated)

                        # --- 7B: Score proximity ---
                        self.score_proximity[thresh]['total'] += 1
                        ttsc = ticks_to_score_change[i]
                        if ttsc is None:
                            self.score_proximity[thresh]['no_change'] += 1
                        else:
                            if ttsc <= 5:
                                self.score_proximity[thresh]['within_5'] += 1
                            if ttsc <= 10:
                                self.score_proximity[thresh]['within_10'] += 1
                            if ttsc <= 20:
                                self.score_proximity[thresh]['within_20'] += 1

                        # --- 7C: H2S filtered (first entry per side only) ---
                        if outcome and side not in entered_sides_7:
                            home_won = outcome['home_winner'] == 1
                            we_won = (side == 'home' and home_won) or (side == 'away' and not home_won)
                            settlement = 1.0 if we_won else 0.0
                            entry_fee = kalshi_fee_cents(entry_price)
                            gross_pnl = (settlement - entry_price) * 100
                            net_pnl = gross_pnl - entry_fee
                            margin = abs(s['home_score'] - s['away_score'])

                            entry_dict = {
                                'game_id': game_id,
                                'side': side,
                                'entry_price': entry_price,
                                'our_prob': our_prob,
                                'edge': edge,
                                'time_remaining': time_remaining,
                                'status': 'in',
                                'margin': margin,
                                'we_won': we_won,
                                'gross_pnl': gross_pnl,
                                'entry_fee': entry_fee,
                                'net_pnl': net_pnl,
                            }

                            for filter_secs in [5, 10, 20]:
                                if ttsc is None or ttsc > filter_secs:
                                    # No imminent score change — include this entry
                                    self.h2s_filtered[thresh][filter_secs].append(entry_dict)

                            entered_sides_7.add(side)

    # ----------------------------------------------------------
    # PRINT RESULTS
    # ----------------------------------------------------------
    def print_calibration(self):
        """Print Phase 1A: Reliability Curve."""
        print(f"\n{'='*70}")
        print(f"  PHASE 1A: MODEL CALIBRATION (Reliability Curve)")
        print(f"{'='*70}")

        print(f"\n  Games used: {self.cal_games_used} | No outcome: {self.cal_games_skipped_no_outcome} | No model: {self.cal_games_skipped_no_model}")
        print(f"\n  {'Bucket':12s} {'Count':>6s} {'Avg Model':>10s} {'Actual Win%':>12s} {'Diff':>8s} {'Avg Market':>11s} {'Mkt Diff':>9s}")
        print(f"  {'='*70}")

        for key in sorted(self.cal_buckets.keys()):
            b = self.cal_buckets[key]
            if b['count'] == 0:
                continue
            avg_model = b['sum_model'] / b['count']
            actual_wr = b['wins'] / b['count']
            diff = actual_wr - avg_model

            mkt_str = ""
            mkt_diff_str = ""
            if b['has_market'] > 0:
                avg_mkt = b['sum_market'] / b['has_market']
                mkt_actual = b['sum_market_wins'] / b['has_market']
                mkt_diff = mkt_actual - avg_mkt
                mkt_str = f"{avg_mkt:.3f}"
                mkt_diff_str = f"{mkt_diff:+.3f}"

            print(f"  {key:12s} {b['count']:>6d} {avg_model:>10.3f} {actual_wr:>11.1%} {diff:>+8.3f} {mkt_str:>11s} {mkt_diff_str:>9s}")

    def print_brier(self):
        """Print Phase 1B: Brier Scores."""
        print(f"\n  --- Per-Game Brier Scores ---")
        if self.brier_n_model > 0:
            brier_model = self.brier_model_sum / self.brier_n_model
            print(f"  Model Brier:  {brier_model:.4f}  (n={self.brier_n_model})")
        if self.brier_n_market > 0:
            brier_market = self.brier_market_sum / self.brier_n_market
            print(f"  Market Brier: {brier_market:.4f}  (n={self.brier_n_market})")
        if self.brier_n_model > 0 and self.brier_n_market > 0:
            brier_model = self.brier_model_sum / self.brier_n_model
            brier_market = self.brier_market_sum / self.brier_n_market
            diff = brier_model - brier_market
            print(f"  Difference:   {diff:+.4f}  ({'MODEL WORSE' if diff > 0 else 'MODEL BETTER'})")
            if diff > 0:
                print(f"  *** Market is more accurate than model. Edge may not exist. ***")
            else:
                print(f"  Model has informational edge of {-diff:.4f} Brier points.")

    def print_hold_to_settlement(self):
        """Print Phase 2: Hold-to-Settlement."""
        print(f"\n{'='*70}")
        print(f"  PHASE 2: HOLD-TO-SETTLEMENT SIMULATION")
        print(f"{'='*70}")

        print(f"\n  {'Threshold':>10s} {'Trades':>7s} {'Win%':>7s} {'Gross':>9s} {'Fees':>7s} {'Net P/L':>9s} {'$/Trade':>9s} {'Avg Edge':>9s} {'Avg Price':>10s}")
        print(f"  {'='*80}")
        for thresh in self.thresholds:
            entries = self.h2s_entries[thresh]
            if not entries:
                print(f"  {thresh:>9.0%}   {'---':>7s}")
                continue
            n = len(entries)
            wins = sum(1 for e in entries if e['we_won'])
            gross = sum(e['gross_pnl'] for e in entries)
            fees = sum(e['entry_fee'] for e in entries)
            net = sum(e['net_pnl'] for e in entries)
            avg_edge = sum(e['edge'] for e in entries) / n
            avg_price = sum(e['entry_price'] for e in entries) / n
            wr = wins / n * 100
            per_trade = net / n
            print(f"  {thresh:>9.0%}   {n:>7d} {wr:>6.1f}% {gross:>+8.1f}c {fees:>6.1f}c {net:>+8.1f}c {per_trade:>+8.2f}c {avg_edge:>8.3f} {avg_price*100:>8.0f}c")

    def print_time_segment(self, entries: List[dict]):
        """Print Phase 3: Time-Segment ROI."""
        print(f"\n{'='*70}")
        print(f"  PHASE 3: TIME-SEGMENT ROI (Hold-to-Settlement)")
        print(f"{'='*70}")

        time_buckets = [
            ('Pregame',            2400, 2401),
            ('H1 early (30-40m)',  1800, 2400),
            ('H1 late (20-30m)',   1200, 1800),
            ('H2 early (10-20m)',   600, 1200),
            ('H2 late (4-10m)',     240,  600),
        ]

        print(f"\n  {'Time Bucket':22s} {'Trades':>7s} {'Win%':>7s} {'Net P/L':>9s} {'$/Trade':>9s} {'Avg Edge':>9s}")
        print(f"  {'='*66}")

        for label, lo, hi in time_buckets:
            if label == 'Pregame':
                subset = [e for e in entries if e['status'] == 'pre']
            else:
                subset = [e for e in entries if e['status'] == 'in'
                          and lo <= e['time_remaining'] < hi]

            if not subset:
                print(f"  {label:22s} {'---':>7s}")
                continue

            n = len(subset)
            wins = sum(1 for e in subset if e['we_won'])
            net = sum(e['net_pnl'] for e in subset)
            avg_edge = sum(e['edge'] for e in subset) / n
            wr = wins / n * 100
            per_trade = net / n
            print(f"  {label:22s} {n:>7d} {wr:>6.1f}% {net:>+8.1f}c {per_trade:>+8.2f}c {avg_edge:>8.3f}")

    def print_margin_conditioning(self, entries: List[dict]):
        """Print Phase 4: Margin Conditioning."""
        print(f"\n{'='*70}")
        print(f"  PHASE 4: MARGIN CONDITIONING (Hold-to-Settlement)")
        print(f"{'='*70}")

        margin_buckets = [
            ('Tight (0-2)',         0,   3),
            ('Medium (3-5)',        3,   6),
            ('Comfortable (6-10)',  6,  11),
            ('Blowout (11+)',      11, 999),
        ]

        print(f"\n  {'Margin Bucket':22s} {'Trades':>7s} {'Win%':>7s} {'Net P/L':>9s} {'$/Trade':>9s} {'Avg Edge':>9s}")
        print(f"  {'='*66}")

        for label, lo, hi in margin_buckets:
            subset = [e for e in entries if lo <= e['margin'] < hi]
            if not subset:
                print(f"  {label:22s} {'---':>7s}")
                continue
            n = len(subset)
            wins = sum(1 for e in subset if e['we_won'])
            net = sum(e['net_pnl'] for e in subset)
            avg_edge = sum(e['edge'] for e in subset) / n
            wr = wins / n * 100
            per_trade = net / n
            print(f"  {label:22s} {n:>7d} {wr:>6.1f}% {net:>+8.1f}c {per_trade:>+8.2f}c {avg_edge:>8.3f}")

        # Cross-tab
        print(f"\n  CROSS-TAB: MARGIN × TIME (net P/L per cell)")
        time_buckets = [
            ('Pre',  2400, 2401),
            ('H1e',  1800, 2400),
            ('H1l',  1200, 1800),
            ('H2e',   600, 1200),
            ('H2l',   240,  600),
        ]

        print(f"  {'':22s}", end='')
        for tl, _, _ in time_buckets:
            print(f" | {tl:>14s}", end='')
        print(f" | {'TOTAL':>14s}")
        print(f"  {'='*22}", end='')
        for _ in time_buckets:
            print(f" | {'='*14}", end='')
        print(f" | {'='*14}")

        for mlabel, mlo, mhi in margin_buckets:
            margin_subset = [e for e in entries if mlo <= e['margin'] < mhi]
            print(f"  {mlabel:22s}", end='')

            for tlabel, tlo, thi in time_buckets:
                if tlabel == 'Pre':
                    cell = [e for e in margin_subset if e['status'] == 'pre']
                else:
                    cell = [e for e in margin_subset if e['status'] == 'in'
                            and tlo <= e['time_remaining'] < thi]

                if cell:
                    n = len(cell)
                    w = sum(1 for e in cell if e['we_won'])
                    net = sum(e['net_pnl'] for e in cell)
                    s = f"{w}/{n} {net:+.0f}c"
                    print(f" | {s:>14s}", end='')
                else:
                    print(f" | {'--':>14s}", end='')

            if margin_subset:
                n = len(margin_subset)
                w = sum(1 for e in margin_subset if e['we_won'])
                net = sum(e['net_pnl'] for e in margin_subset)
                s = f"{w}/{n} {net:+.0f}c"
                print(f" | {s:>14s}", end='')
            print()

    def print_adverse_selection(self):
        """Print Phase 6: Adverse Selection."""
        print(f"\n{'='*70}")
        print(f"  PHASE 6: ADVERSE SELECTION (Price Movement After Signal)")
        print(f"{'='*70}")

        for thresh in self.adv_thresholds:
            print(f"\n  --- Edge >= {thresh:.0%} ---")
            print(f"  {'Lookahead':>12s} {'Signals':>8s} {'Avg Move':>10s} {'Median':>8s} {'% Against':>10s}")
            print(f"  {'='*52}")

            for la in self.adv_lookaheads:
                mvs = self.adv_movements[thresh][la]
                if not mvs:
                    continue
                avg_mv = sum(mvs) / len(mvs)
                sorted_mvs = sorted(mvs)
                median_mv = sorted_mvs[len(sorted_mvs) // 2]
                pct_against = sum(1 for m in mvs if m > 0) / len(mvs) * 100
                marker = "ADVERSE" if avg_mv > 0.3 else "OK" if avg_mv < -0.3 else "FLAT"
                print(f"  {'+' + str(la) + ' ticks':>12s} {len(mvs):>8d} {avg_mv:>+9.2f}c {median_mv:>+7.1f}c {pct_against:>9.1f}%  [{marker}]")

    def print_stale_edge(self):
        """Print Phase 7: Stale Edge / Edge Decay analysis."""
        print(f"\n{'='*70}")
        print(f"  PHASE 7A: EDGE DECAY (Is 'edge' stale state?)")
        print(f"  Same model prob at signal, future market price at +k ticks")
        print(f"{'='*70}")

        for thresh in self.stale_thresholds:
            signals = self.edge_at_signal[thresh]
            if not signals:
                continue
            avg_e0 = sum(signals) / len(signals)

            print(f"\n  --- Edge >= {thresh:.0%} ({len(signals):,} signals) | Avg E₀ = {avg_e0:.3f} ---")
            print(f"  {'Tick':>8s} {'Signals':>8s} {'Avg Edge':>10s} {'Δ from E₀':>10s} {'Collapse%':>10s} {'Updated E':>10s}")
            print(f"  {'='*60}")

            for la in self.stale_lookaheads:
                decay_vals = self.edge_decay[thresh][la]
                updated_vals = self.edge_updated[thresh][la]
                if not decay_vals:
                    continue

                avg_decay = sum(decay_vals) / len(decay_vals)
                delta = avg_decay - avg_e0
                collapse_pct = (1 - avg_decay / avg_e0) * 100 if avg_e0 > 0 else 0

                upd_str = ""
                if updated_vals:
                    avg_upd = sum(updated_vals) / len(updated_vals)
                    upd_str = f"{avg_upd:.3f}"

                print(f"  {'+' + str(la):>8s} {len(decay_vals):>8d} {avg_decay:>10.3f} {delta:>+10.3f} {collapse_pct:>9.1f}% {upd_str:>10s}")

        # 7B: Score proximity
        print(f"\n{'='*70}")
        print(f"  PHASE 7B: SCORE PROXIMITY (Do signals cluster near score changes?)")
        print(f"{'='*70}")

        print(f"\n  {'Threshold':>10s} {'Total':>8s} {'<5 ticks':>10s} {'<10 ticks':>11s} {'<20 ticks':>11s} {'No change':>10s}")
        print(f"  {'='*64}")

        for thresh in self.stale_thresholds:
            sp = self.score_proximity[thresh]
            total = sp['total']
            if total == 0:
                continue
            w5 = sp['within_5']
            w10 = sp['within_10']
            w20 = sp['within_20']
            nc = sp['no_change']
            print(f"  {thresh:>9.0%}  {total:>8d} {w5:>6d} ({w5/total*100:4.1f}%) "
                  f"{w10:>6d} ({w10/total*100:4.1f}%) "
                  f"{w20:>6d} ({w20/total*100:4.1f}%) "
                  f"{nc:>6d} ({nc/total*100:4.1f}%)")

        # 7C: H2S filtered
        print(f"\n{'='*70}")
        print(f"  PHASE 7C: HOLD-TO-SETTLEMENT — EXCLUDING IMMINENT SCORE CHANGES")
        print(f"  (Live entries only. If profitability vanishes, 'alpha' was stale state)")
        print(f"{'='*70}")

        for filter_secs in [5, 10, 20]:
            print(f"\n  --- Filter: exclude entries with score change within {filter_secs} ticks ---")
            print(f"  {'Threshold':>10s} {'Trades':>7s} {'Win%':>7s} {'Net P/L':>9s} {'$/Trade':>9s} {'Avg Edge':>9s}")
            print(f"  {'='*55}")

            for thresh in self.stale_thresholds:
                entries = self.h2s_filtered[thresh][filter_secs]
                if not entries:
                    print(f"  {thresh:>9.0%}   {'---':>7s}")
                    continue
                n = len(entries)
                wins = sum(1 for e in entries if e['we_won'])
                net = sum(e['net_pnl'] for e in entries)
                avg_edge = sum(e['edge'] for e in entries) / n
                wr = wins / n * 100
                per_trade = net / n
                print(f"  {thresh:>9.0%}   {n:>7d} {wr:>6.1f}% {net:>+8.1f}c {per_trade:>+8.2f}c {avg_edge:>8.3f}")

    # ----------------------------------------------------------
    # RUN ALL DIAGNOSTICS
    # ----------------------------------------------------------
    def run(self, start_date: str, end_date: str):
        """Run complete diagnostic suite, streaming game-by-game."""
        print(f"\n{'='*70}")
        print(f"  LIVE TRADING DIAGNOSTICS: {start_date} to {end_date}")
        print(f"  Is this system structurally -EV, or +EV destroyed by friction?")
        print(f"{'='*70}")

        self.load_outcomes(start_date, end_date)
        print(f"  Outcomes loaded: {len(self.outcomes)}")

        self.load_game_ids(start_date, end_date)
        print(f"  Games to process: {len(self.game_ids)}")

        # Ensure index for per-game queries
        try:
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_game_ts ON live_snapshots(game_id, timestamp)")
            self.conn.commit()
        except Exception:
            pass

        # Process each game
        total_snaps = 0
        for i, game_id in enumerate(self.game_ids):
            snaps = self.load_game_snapshots(game_id, start_date, end_date)
            total_snaps += len(snaps)
            self.process_game(game_id, snaps)

            if (i + 1) % 50 == 0 or (i + 1) == len(self.game_ids):
                print(f"  Processed {i+1}/{len(self.game_ids)} games ({total_snaps:,} snapshots)")

        # Print all results
        self.print_calibration()
        self.print_brier()
        self.print_hold_to_settlement()

        base_entries = self.h2s_entries.get(0.06, [])
        if base_entries:
            print(f"\n  (Phases 3-4 use {len(base_entries)} entries at 6% threshold)")
            self.print_time_segment(base_entries)
            self.print_margin_conditioning(base_entries)
        else:
            print(f"\n  No entries at 6% threshold — skipping Phases 3-4")

        self.print_adverse_selection()
        self.print_stale_edge()

        # Summary
        print(f"\n{'='*70}")
        print(f"  DIAGNOSTIC SUMMARY")
        print(f"{'='*70}")
        print(f"  Review the above phases in order:")
        print(f"  1. Calibration: Is the model systematically over/under-confident?")
        print(f"  2. Hold-to-settlement: Is the entry signal itself +EV?")
        print(f"  3. Time segments: Where in the game does edge exist/die?")
        print(f"  4. Margin conditioning: Are tight late games killing us?")
        print(f"  5. Edge threshold: Are we overtrading small edges?")
        print(f"  6. Adverse selection: Is 'edge' just stale ESPN data?")
        print(f"  7. Stale edge: Does 'edge' collapse after score updates? Is alpha fake?")
        print(f"  Compare Phase 2 results with regular backtest to isolate exit damage.")
        print(f"{'='*70}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Backtest live trading strategy against live_snapshots')
    parser.add_argument('--db', required=True, help='Path to compiled_stats.db')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--debug', action='store_true', help='Print every trade as it happens')
    parser.add_argument('--trace-game', type=str, help='Print detailed trace for specific game_id')
    parser.add_argument('--trade-log', action='store_true', help='Print detailed trade log')
    parser.add_argument('--max-trades', type=int, default=100, help='Max trades to show in trade log')
    parser.add_argument('--min-edge', type=float, default=None,
                        help='Override starting edge threshold (e.g. 0.08 = 8%%)')
    parser.add_argument('--venue', choices=['kalshi', 'polymarket', 'best'], default='best',
                        help='Venue to trade on: kalshi, polymarket, or best (default: best)')
    parser.add_argument('--analysis', action='store_true',
                        help='Print edge/time/threshold analysis breakdown')
    parser.add_argument('--diagnostics', action='store_true',
                        help='Run model calibration & structural diagnostics (no trading simulation)')

    args = parser.parse_args()

    # Diagnostics mode — completely separate code path
    if args.diagnostics:
        diag = DiagnosticsEngine(args.db)
        diag.run(args.start_date, args.end_date)
        return

    engine = BacktestEngine(args.db, debug=args.debug, trace_game=args.trace_game, min_edge=args.min_edge, venue=args.venue)
    engine.run(args.start_date, args.end_date)
    engine.print_results()

    if args.analysis:
        engine.print_analysis()

    if args.trade_log:
        engine.print_trade_log(max_trades=args.max_trades)


if __name__ == '__main__':
    main()