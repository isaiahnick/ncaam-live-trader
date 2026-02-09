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


# ============================================================
# CONSTANTS — exact copies from live_trader.py
# ============================================================

# Win probability model
BASE_STD = 17.2

# Entry parameters
EDGE_THRESHOLD = 0.10      # Base threshold (dynamic via get_required_edge)
MIN_TIME_REMAINING = 240   # 4:00 — don't enter with less than this (live only)
MIN_ENTRY_PRICE = 0.10     # Don't buy below 10 cents
MAX_ENTRY_PRICE = 0.90     # Don't buy above 90 cents

# Exit parameters — EV-based
EV_EXIT_SLIPPAGE_CENTS = 0

# Option value formula (fit to backward induction with DT=120, sigma=4.25, slip=0)
OV_SCALE = 0.39
OV_EXPONENT = 0.42
OV_PROB_COEFF = 16.12
OV_DECORR_SEC = 120

# Cooldown
COOLDOWN_AFTER_EXIT = 30   # 30 seconds game-time after any exit
COOLDOWN_AFTER_STOP = 240  # 4 minutes after stop loss (if ever re-enabled)

# ESPN score freshness guard
MAX_ESPN_SCORE_STALE_SEC = 15


# ============================================================
# FUNCTIONS — exact copies from live_trader.py
# ============================================================

def kalshi_fee_cents(price: float) -> float:
    """Kalshi fee: 7% * P * (1-P) * 100 cents, capped at 1.75 cents.
    Args: price as decimal (0-1)
    """
    return min(0.07 * price * (1 - price) * 100, 1.75)


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


def get_required_edge(entry_price: float, time_remaining: int = 2400) -> float:
    """Dynamic edge threshold — exponential ramp (k=2) from 5% to 15%."""
    E_START = 0.05   # 5% at game start / pregame
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
    """Closed-form option value: OV = 1.75 * N^0.43 * (1 - 0.75*p(1-p))."""
    N = max(0, time_remaining_sec) / OV_DECORR_SEC
    if N <= 0:
        return 0.0
    return OV_SCALE * (N ** OV_EXPONENT) * (1 + OV_PROB_COEFF * prob * (1 - prob))


# ============================================================
# BACKTEST ENGINE
# ============================================================

class BacktestEngine:
    def __init__(self, db_path: str, debug: bool = False, trace_game: str = None):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.debug = debug
        self.trace_game = trace_game

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

        # Check if any market data exists
        has_market = any(v for v in odds.values() if v is not None and v > 0)
        if not has_market:
            self.entry_skips['no_market_data'] += 1
            return None

        # === CHECK SETTLEMENT ===
        # Game effectively over: in-game, period >= 2, time = 0, scores exist and different
        if (status == 'in' and time_remaining == 0 and period >= 2
                and home_score > 0 and home_score != away_score):
            if game_id in self.open_positions:
                self._settle_position(game_id, game, ts)
            self.game_settled.add(game_id)
            return 'settled'

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

        # Time-dependent spread limit (line 2164-2167)
        if time_remaining >= 1200:  # > 20:00
            max_spread = 3
        else:
            max_spread = 2

        best = None

        # ===== HOME SIDE (bet on home winning) =====
        # Option A: Buy HOME YES @ home_yes_ask
        # Option B: Buy AWAY NO @ away_no_ask (also pays $1 if home wins)
        home_candidates = []

        if odds.get('home_yes_ask') and odds.get('home_yes_bid'):
            home_candidates.append({
                'contract_type': 'home_yes',
                'ask': odds['home_yes_ask'],
                'bid': odds['home_yes_bid'],
                'spread': odds['home_yes_ask'] - odds['home_yes_bid'],
            })

        if odds.get('away_no_ask') and odds.get('away_no_bid'):
            home_candidates.append({
                'contract_type': 'away_no',
                'ask': odds['away_no_ask'],
                'bid': odds['away_no_bid'],
                'spread': odds['away_no_ask'] - odds['away_no_bid'],
            })

        if home_candidates:
            # Pick cheapest ask; if tie, pick smallest spread (line 2194)
            home_candidates.sort(key=lambda x: (x['ask'], x['spread']))
            best_home = home_candidates[0]

            entry_price = best_home['ask'] / 100.0
            spread = best_home['spread']
            home_edge = our_home - entry_price

            # Cooldown check — game-time based (lines 2201-2206)
            cooldown_key = (game_id, 'home')
            in_cooldown = False
            if cooldown_key in self.exit_cooldowns:
                exit_game_time, cooldown_secs = self.exit_cooldowns[cooldown_key]
                in_cooldown = (exit_game_time - time_remaining) < cooldown_secs

            required_edge = get_required_edge(entry_price, time_remaining)

            # Entry filter (exact lines 2211-2217)
            if (home_edge >= required_edge and
                    spread <= max_spread and
                    entry_price >= MIN_ENTRY_PRICE and
                    entry_price <= MAX_ENTRY_PRICE and
                    (status == 'pre' or time_remaining >= MIN_TIME_REMAINING) and
                    game.get('period', 1) <= 2 and
                    not in_cooldown):
                best = ('home', home_edge, entry_price, spread, best_home['contract_type'])

                if self.trace_game == game_id:
                    print(f"  [TRACE] HOME candidate: edge={home_edge:.3f} req={required_edge:.3f} "
                          f"price={entry_price:.2f} spread={spread} type={best_home['contract_type']}")

        # ===== AWAY SIDE (bet on away winning) =====
        away_candidates = []

        if odds.get('away_yes_ask') and odds.get('away_yes_bid'):
            away_candidates.append({
                'contract_type': 'away_yes',
                'ask': odds['away_yes_ask'],
                'bid': odds['away_yes_bid'],
                'spread': odds['away_yes_ask'] - odds['away_yes_bid'],
            })

        if odds.get('home_no_ask') and odds.get('home_no_bid'):
            away_candidates.append({
                'contract_type': 'home_no',
                'ask': odds['home_no_ask'],
                'bid': odds['home_no_bid'],
                'spread': odds['home_no_ask'] - odds['home_no_bid'],
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

            required_edge = get_required_edge(entry_price, time_remaining)

            if (away_edge >= required_edge and
                    spread <= max_spread and
                    entry_price >= MIN_ENTRY_PRICE and
                    entry_price <= MAX_ENTRY_PRICE and
                    (status == 'pre' or time_remaining >= MIN_TIME_REMAINING) and
                    game.get('period', 1) <= 2 and
                    not in_cooldown):
                # Compare with existing best — pick higher edge (line 2270)
                if best is None or away_edge > best[1]:
                    best = ('away', away_edge, entry_price, spread, best_away['contract_type'])

                    if self.trace_game == game_id:
                        print(f"  [TRACE] AWAY candidate: edge={away_edge:.3f} req={required_edge:.3f} "
                              f"price={entry_price:.2f} spread={spread} type={best_away['contract_type']}")

        if not best:
            return None

        side, edge, entry_price, spread, contract_type = best
        our_prob = our_home if side == 'home' else 1 - our_home

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

        # Get bid for our contract type (lines 2449-2457)
        bid_map = {
            'home_yes': 'home_yes_bid',
            'home_no': 'home_no_bid',
            'away_yes': 'away_yes_bid',
            'away_no': 'away_no_bid',
        }
        bid_key = bid_map.get(contract_type, f'{side}_yes_bid')
        current_bid_cents = odds.get(bid_key, 0)

        if not current_bid_cents or current_bid_cents <= 0:
            return None

        # Model probability for our side (line 2463)
        our_prob = game['live_home_prob'] if side == 'home' else 1 - game['live_home_prob']

        # EV-based exit (lines 2466-2480)
        ov = compute_option_value(time_remaining, our_prob)
        ev_hold = our_prob * 100 + ov

        bid_decimal = current_bid_cents / 100.0
        exit_fee = kalshi_fee_cents(bid_decimal)
        ev_exit = current_bid_cents - exit_fee - EV_EXIT_SLIPPAGE_CENTS

        if ev_exit > ev_hold:
            if self.trace_game == game['game_id']:
                print(f"  [TRACE] EV EXIT: ev_exit={ev_exit:.1f} > ev_hold={ev_hold:.1f} "
                      f"(prob={our_prob:.3f} ov={ov:.1f} bid={current_bid_cents}c)")

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
                'bid_cents': current_bid_cents,
            }

        return None

    # ----------------------------------------------------------
    # EXECUTION
    # ----------------------------------------------------------
    def _execute_entry(self, opp: dict, ts: datetime):
        game_id = opp['game_id']
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
        }
        self.open_positions[game_id] = position
        self.capital_deployed += opp['entry_price'] * 100
        self.peak_capital = max(self.peak_capital, self.capital_deployed)

        team = opp['home_team'] if opp['side'] == 'home' else opp['away_team']
        if self.debug or self.trace_game == game_id:
            time_str = 'PRE' if opp['status'] == 'pre' else f"{opp['time_remaining_sec']}s"
            print(f"  >> ENTRY: {opp['side'].upper()} {team[:30]} @ {opp['entry_price']*100:.0f}c "
                  f"| edge={opp['edge']:.3f} | {opp['contract_type']} | {time_str} | {ts.strftime('%H:%M:%S')}")

    def _execute_exit(self, exit_info: dict, ts: datetime):
        game_id = exit_info['game_id']
        position = self.open_positions[game_id]

        entry_price = position['entry_price']
        exit_price = exit_info['exit_price']

        gross_pnl = (exit_price - entry_price) * 100
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
        if self.debug or self.trace_game == game_id:
            print(f"  << EV EXIT: {exit_info['side'].upper()} {team[:30]} | "
                  f"Entry: {entry_price*100:.0f}c -> Exit: {exit_price*100:.0f}c | "
                  f"Gross: {gross_pnl:+.1f}c | Fees: {entry_fee+exit_fee:.1f}c | Net: {net_pnl:+.1f}c "
                  f"| {ts.strftime('%H:%M:%S')}")

    def _settle_position(self, game_id: str, game: dict, ts: datetime):
        position = self.open_positions[game_id]
        side = position['side']
        entry_price = position['entry_price']

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
        if self.debug or self.trace_game == game_id:
            print(f"  ** SETTLE [{emoji}]: {side.upper()} {team[:30]} | "
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
        }

    def settle_remaining(self):
        """Force-settle any positions still open at end of backtest."""
        for game_id in list(self.open_positions.keys()):
            outcome = self.outcomes.get(game_id)
            position = self.open_positions[game_id]
            side = position['side']
            entry_price = position['entry_price']

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
            }
            self.trades.append(trade)
            self.realized_pnl += net_pnl

            team = position['home_team'] if side == 'home' else position['away_team']
            print(f"  !! FORCE SETTLE: {side.upper()} {team[:30]} -> {reason} | Net: {net_pnl:+.1f}c")

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
        print(f"  Edge: {get_required_edge(0.5, 2400)*100:.0f}% pregame -> {get_required_edge(0.5, 240)*100:.0f}% @ 4:00")
        print(f"  Spread: <=3c early, <=2c late | Price: {MIN_ENTRY_PRICE*100:.0f}-{MAX_ENTRY_PRICE*100:.0f}c")
        print(f"  OV: {OV_SCALE} x N^{OV_EXPONENT} x (1 + {OV_PROB_COEFF}*p*(1-p))")
        print(f"  Cooldown: {COOLDOWN_AFTER_EXIT}s game-time after exit")
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

        print(f"\n{'='*70}")
        print(f"  RESULTS SUMMARY")
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

        # --- Daily breakdown ---
        print(f"\n  DAILY BREAKDOWN:")
        daily = {}
        for t in self.trades:
            if t['entry_time']:
                day = t['entry_time'].strftime('%Y-%m-%d')
            else:
                day = 'unknown'
            if day not in daily:
                daily[day] = {'trades': 0, 'net': 0, 'gross': 0, 'fees': 0, 'winners': 0}
            daily[day]['trades'] += 1
            daily[day]['net'] += t['net_pnl']
            daily[day]['gross'] += t['gross_pnl']
            daily[day]['fees'] += t['entry_fee'] + t['exit_fee']
            if t['net_pnl'] > 0:
                daily[day]['winners'] += 1

        for day in sorted(daily.keys()):
            d = daily[day]
            wr = d['winners'] / d['trades'] * 100 if d['trades'] > 0 else 0
            print(f"    {day}: {d['trades']:3d} trades | Win: {wr:.0f}% | "
                  f"Gross: {d['gross']:+.1f}c | Fees: {d['fees']:.1f}c | Net: {d['net']:+.1f}c")

        # --- Skip reasons ---
        print(f"\n  SNAPSHOT SKIP COUNTS:")
        for reason, cnt in sorted(self.entry_skips.items(), key=lambda x: -x[1]):
            if cnt > 0:
                print(f"    {reason:25s}: {cnt:>10,}")

    def print_trade_log(self, max_trades: int = 100):
        """Print detailed trade log."""
        print(f"\n{'='*70}")
        print(f"  TRADE LOG ({min(len(self.trades), max_trades)} of {len(self.trades)})")
        print(f"{'='*70}")

        print(f"  {'#':>3} {'SIDE':5} {'TEAM':28} {'WHEN':8} {'ENTRY':>6} {'EXIT':>6} "
              f"{'GROSS':>7} {'FEES':>6} {'NET':>7} {'REASON':22}")
        print(f"  {'='*108}")

        for i, t in enumerate(self.trades[:max_trades]):
            team = t['home_team'] if t['side'] == 'home' else t['away_team']
            team = (team[:28] if team else '???')
            entry_type = 'PRE' if t['entry_status'] == 'pre' else f"{t.get('entry_time_remaining', '?')}s"
            fees = t['entry_fee'] + t['exit_fee']
            reason = t['exit_reason'][:22]
            marker = "W" if t['net_pnl'] > 0 else "L" if t['net_pnl'] < 0 else "-"

            print(f"  {i+1:>3} {t['side']:5} {team:28} {entry_type:8} "
                  f"{t['entry_price']*100:5.0f}c {t['exit_price']*100:5.0f}c "
                  f"{t['gross_pnl']:+6.1f}c {fees:5.1f}c {t['net_pnl']:+6.1f}c [{marker}] {reason}")


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

    args = parser.parse_args()

    engine = BacktestEngine(args.db, debug=args.debug, trace_game=args.trace_game)
    engine.run(args.start_date, args.end_date)
    engine.print_results()

    if args.trade_log:
        engine.print_trade_log(max_trades=args.max_trades)


if __name__ == '__main__':
    main()