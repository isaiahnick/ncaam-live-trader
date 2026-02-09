#!/usr/bin/env python3
"""
live_trader.py - Kalshi Live Trading System for NCAA Basketball

Automated trading system that monitors live college basketball games, calculates 
model-implied win probabilities, compares against Kalshi market prices, and 
executes trades when sufficient edge is detected.

FEATURES:
- Real-time Kalshi API integration for odds fetching and order execution
- Bayesian live win probability model (BASE_STD=17.2)
- YES/NO contract optimization (automatically picks cheaper entry)
- Position persistence across restarts (loads from database)
- Configurable edge thresholds, spread limits, and position sizing
- Parallel API fetching for low-latency updates (~1 second cycles)

MODES:
- Paper Mode: Logs trades to database without executing (for backtesting)
- Live Mode: Places real orders on Kalshi with real money

USAGE:
    # Paper trading (monitor and log, no real orders)
    python3 live_trader.py --db compiled_stats.db
    
    # Live trading with 10 contracts per trade (DEFAULT)
    python3 live_trader.py --db compiled_stats.db --live
    
    # Live trading with custom position size
    python3 live_trader.py --db compiled_stats.db --live --contracts 5
    
    # Faster/slower polling interval (default: 1 second)
    python3 live_trader.py --db compiled_stats.db --live --interval 2

REQUIREMENTS:
- kalshi_live_client.py in same directory (for live trading)
- Kalshi API key configured in the script
- compiled_stats.db with future_predictions table populated

SAFETY:
- Live mode requires typing 'LIVE' to confirm
- Configurable max position size and order value limits
- Minimum time remaining check prevents late-game entries
"""

import sqlite3
import json
import time
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from dataclasses import dataclass
from typing import Optional, List
from scipy.stats import norm
import math
import sys
import os
import numpy as np

# ===== LIVE TRADING IMPORT =====
try:
    from kalshi_live_client import KalshiLiveClient, OrderSide, OrderAction, TimeInForce
    KALSHI_LIVE_AVAILABLE = True
except ImportError:
    KALSHI_LIVE_AVAILABLE = False

# ===== WEBSOCKET ORDERBOOK IMPORT =====
try:
    from kalshi_ws_orderbook import KalshiWSOrderbook
    KALSHI_WS_AVAILABLE = True
except ImportError:
    KALSHI_WS_AVAILABLE = False

# ===== FEE CALCULATIONS =====
def kalshi_fee_cents(price: float) -> float:
    """Kalshi fee: 7% × P × (1-P) × 100 cents, capped at 1.75¢"""
    return min(0.07 * price * (1 - price) * 100, 1.75)


def calculate_net_ev(prob: float, ask: float, spread: float) -> float:
    """
    Calculate net expected value for a Kalshi trade, accounting for fees and spread.
    
    Args:
        prob: Our model's probability (0-1)
        ask: Entry price we'd pay (0-1)
        spread: Bid-ask spread in cents
    
    Returns:
        Net EV in cents
    
    Formula:
        net_EV = gross_edge - entry_fee - expected_exit_fee - (spread / 2)
    
    We estimate exit at prob (fair value) since ~80% of trades exit via edge collapse.
    Spread/2 accounts for exit slippage.
    """
    gross_edge_cents = (prob - ask) * 100
    entry_fee = kalshi_fee_cents(ask)
    exit_fee = kalshi_fee_cents(prob)  # Estimate exit at fair value
    spread_cost = spread / 2  # Half spread as exit slippage estimate
    
    return gross_edge_cents - entry_fee - exit_fee - spread_cost


# ESPN API
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?groups=50&limit=200"

# Kalshi API - College Basketball Games
KALSHI_CBB_URL = "https://api.elections.kalshi.com/trade-api/v2/events?series_ticker=KXNCAAMBGAME&status=open&with_nested_markets=true&limit=200"

# Hardcoded team name mappings: ESPN (lowercase, no mascot) -> Kalshi format
TEAM_NAME_MAP = {
    # A
    "abilene christian": "abilene christian",
    "air force": "air force",
    "akron": "akron",
    "alabama": "alabama",
    "alabama a&m": "alabama a&m",
    "alabama state": "alabama st",
    "albany": "albany",
    "ualbany": "albany",
    "alcorn state": "alcorn st",
    "american": "american",
    "american university": "american",
    "app state": "appalachian st",
    "appalachian state": "appalachian st",
    "arizona": "arizona",
    "arizona state": "arizona st",
    "arkansas": "arkansas",
    "arkansas state": "arkansas st",
    "arkansas-pine bluff": "arkansas-pine bluff",
    "army": "army",
    "auburn": "auburn",
    "austin peay": "austin peay",
    
    # B
    "ball state": "ball st",
    "baylor": "baylor",
    "bellarmine": "bellarmine",
    "belmont": "belmont",
    "bethune-cookman": "bethune-cookman",
    "binghamton": "binghamton",
    "boise state": "boise st",
    "boston college": "boston college",
    "boston university": "boston university",
    "boston": "boston university",
    "bowling green": "bowling green",
    "bradley": "bradley",
    "brown": "brown",
    "bryant": "bryant",
    "bucknell": "bucknell",
    "buffalo": "buffalo",
    "butler": "butler",
    "byu": "byu",
    
    # C
    "cal poly": "cal poly",
    "cal state bakersfield": "cal state bakersfield",
    "bakersfield": "cal state bakersfield",
    "cal state fullerton": "cal state fullerton",
    "cal state northridge": "cal state northridge",
    "california": "california",
    "california baptist": "california baptist",
    "campbell": "campbell",
    "canisius": "canisius",
    "central arkansas": "central arkansas",
    "central connecticut": "central connecticut st",
    "central michigan": "central michigan",
    "charleston": "charleston",
    "charleston southern": "charleston southern",
    "charlotte": "charlotte",
    "chattanooga": "chattanooga",
    "chicago state": "chicago st",
    "cincinnati": "cincinnati",
    "citadel": "the citadel",
    "the citadel": "the citadel",
    "clemson": "clemson",
    "cleveland state": "cleveland st",
    "coastal carolina": "coastal carolina",
    "colgate": "colgate",
    "colorado": "colorado",
    "colorado state": "colorado st",
    "columbia": "columbia",
    "coppin state": "coppin st",
    "cornell": "cornell",
    "creighton": "creighton",
    
    # D
    "dartmouth": "dartmouth",
    "davidson": "davidson",
    "dayton": "dayton",
    "delaware": "delaware",
    "delaware fightin": "delaware",
    "delaware state": "delaware st",
    "denver": "denver",
    "depaul": "depaul",
    "detroit mercy": "detroit mercy",
    "detroit": "detroit mercy",
    "detroit titans": "detroit mercy",
    "drake": "drake",
    "drexel": "drexel",
    "duke": "duke",
    "duquesne": "duquesne",
    
    # E
    "east carolina": "east carolina",
    "east tennessee state": "east tennessee st",
    "east texas a&m": "east texas a&m",
    "eastern illinois": "eastern illinois",
    "eastern kentucky": "eastern kentucky",
    "eastern michigan": "eastern michigan",
    "eastern washington": "eastern washington",
    "elon": "elon",
    "evansville": "evansville",
    
    # F
    "fairfield": "fairfield",
    "fairleigh dickinson": "fdu",
    "florida": "florida",
    "florida a&m": "florida a&m",
    "florida atlantic": "florida atlantic",
    "florida gulf coast": "florida gulf coast",
    "florida international": "florida international",
    "florida state": "florida st",
    "fordham": "fordham",
    "fresno state": "fresno st",
    "furman": "furman",
    
    # G
    "gardner-webb": "gardner-webb",
    "george mason": "george mason",
    "george washington": "george washington",
    "georgetown": "georgetown",
    "georgia": "georgia",
    "georgia southern": "georgia southern",
    "georgia state": "georgia st",
    "georgia tech": "georgia tech",
    "gonzaga": "gonzaga",
    "grambling": "grambling st",
    "grand canyon": "grand canyon",
    "green bay": "green bay",
    
    # H
    "hampton": "hampton",
    "harvard": "harvard",
    "hawai'i": "hawai'i",
    "hawaii": "hawai'i",
    "high point": "high point",
    "hofstra": "hofstra",
    "holy cross": "holy cross",
    "houston": "houston",
    "houston christian": "houston christian",
    "howard": "howard",
    
    # I
    "idaho": "idaho",
    "idaho state": "idaho st",
    "illinois": "illinois",
    "illinois state": "illinois st",
    "incarnate word": "incarnate word",
    "indiana": "indiana",
    "indiana state": "indiana st",
    "iona": "iona",
    "iowa": "iowa",
    "iowa state": "iowa st",
    "iu indianapolis": "iu indy",
    
    # J
    "jackson state": "jackson st",
    "jacksonville": "jacksonville",
    "jacksonville state": "jacksonville st",
    "james madison": "james madison",
    
    # K
    "kansas": "kansas",
    "kansas city": "kansas city",
    "kansas state": "kansas st",
    "kennesaw state": "kennesaw st",
    "kent state": "kent st",
    "kentucky": "kentucky",
    
    # L
    "la salle": "la salle",
    "lafayette": "lafayette",
    "lamar": "lamar",
    "le moyne": "le moyne",
    "lehigh": "lehigh",
    "liberty": "liberty",
    "lindenwood": "lindenwood",
    "lipscomb": "lipscomb",
    "little rock": "little rock",
    "liu": "liu",
    "long island": "liu",
    "long island university": "liu",
    "long beach state": "long beach st",
    "longwood": "longwood",
    "louisiana": "louisiana",
    "louisiana-lafayette": "louisiana",
    "louisiana ragin cajuns": "louisiana",
    "louisiana tech": "louisiana tech",
    "ul monroe": "louisiana-monroe",
    "louisiana-monroe": "louisiana-monroe",
    "louisville": "louisville",
    "loyola chicago": "loyola chicago",
    "loyola marymount": "loyola marymount",
    "loyola maryland": "loyola maryland",
    "lsu": "lsu",
    
    # M
    "maine": "maine",
    "manhattan": "manhattan",
    "marist": "marist",
    "marquette": "marquette",
    "marshall": "marshall",
    "maryland": "maryland",
    "maryland eastern shore": "maryland-eastern shore",
    "massachusetts": "umass",
    "umass": "umass",
    "mcneese": "mcneese",
    "mcneese state": "mcneese",
    "memphis": "memphis",
    "mercer": "mercer",
    "mercyhurst": "mercyhurst",
    "merrimack": "merrimack",
    "miami": "miami (fl)",
    "miami (oh)": "miami (oh)",
    "michigan": "michigan",
    "michigan state": "michigan st",
    "middle tennessee": "middle tennessee",
    "milwaukee": "milwaukee",
    "minnesota": "minnesota",
    "mississippi state": "mississippi st",
    "mississippi valley state": "mississippi valley st",
    "missouri": "missouri",
    "missouri state": "missouri st",
    "monmouth": "monmouth",
    "montana": "montana",
    "montana state": "montana st",
    "morehead state": "morehead st",
    "morgan state": "morgan st",
    "mount st. mary's": "mount st. mary's",
    "murray state": "murray st",
    
    # N
    "navy": "navy",
    "nc state": "north carolina st",
    "nebraska": "nebraska",
    "nevada": "nevada",
    "new hampshire": "new hampshire",
    "new haven": "new haven",
    "new mexico": "new mexico",
    "new mexico state": "new mexico st",
    "new orleans": "new orleans",
    "niagara": "niagara",
    "nicholls": "nicholls st",
    "njit": "njit",
    "norfolk state": "norfolk st",
    "north alabama": "north alabama",
    "north carolina": "north carolina",
    "north carolina a&t": "north carolina a&t",
    "north carolina central": "north carolina central",
    "north carolina state": "north carolina st",
    "north dakota": "north dakota",
    "north dakota state": "north dakota st",
    "north florida": "north florida",
    "north texas": "north texas",
    "northeastern": "northeastern",
    "northern arizona": "northern arizona",
    "northern colorado": "northern colorado",
    "northern illinois": "northern illinois",
    "northern iowa": "northern iowa",
    "northern kentucky": "northern kentucky",
    "northwestern": "northwestern",
    "northwestern state": "northwestern st",
    "notre dame": "notre dame",
    
    # O
    "oakland": "oakland",
    "ohio": "ohio",
    "ohio state": "ohio st",
    "oklahoma": "oklahoma",
    "oklahoma state": "oklahoma st",
    "old dominion": "old dominion",
    "ole miss": "ole miss",
    "omaha": "omaha",
    "oral roberts": "oral roberts",
    "oregon": "oregon",
    "oregon state": "oregon st",
    
    # P
    "pacific": "pacific",
    "penn": "penn",
    "pennsylvania": "penn",
    "penn state": "penn st",
    "pepperdine": "pepperdine",
    "pittsburgh": "pittsburgh",
    "portland": "portland",
    "portland state": "portland st",
    "prairie view a&m": "prairie view a&m",
    "presbyterian": "presbyterian",
    "princeton": "princeton",
    "providence": "providence",
    "purdue": "purdue",
    "purdue fort wayne": "purdue fort wayne",
    
    # Q
    "queens": "queens university",
    "queens university": "queens university",
    "queens (nc)": "queens university",
    "quinnipiac": "quinnipiac",
    
    # R
    "radford": "radford",
    "rhode island": "rhode island",
    "rice": "rice",
    "richmond": "richmond",
    "rider": "rider",
    "robert morris": "robert morris",
    "rutgers": "rutgers",
    
    # S
    "sacramento state": "sacramento st",
    "sacred heart": "sacred heart",
    "saint francis": "st francis (pa)",
    "st. francis": "st francis (pa)",
    "st francis": "st francis (pa)",
    "saint joseph's": "saint joseph's",
    "st. joseph's": "saint joseph's",
    "saint louis": "saint louis",
    "st. louis": "saint louis",
    "saint mary's": "saint mary's",
    "st. mary's": "saint mary's",
    "saint peter's": "saint peter's",
    "st. peter's": "saint peter's",
    "sam houston": "sam houston",
    "samford": "samford",
    "san diego": "san diego",
    "san diego state": "san diego st",
    "san francisco": "san francisco",
    "san jose state": "san jose st",
    "san josé state": "san jose st",
    "santa clara": "santa clara",
    "seattle": "seattle",
    "seattle u": "seattle",
    "seton hall": "seton hall",
    "siena": "siena",
    "siu edwardsville": "siu edwardsville",
    "smu": "smu",
    "south alabama": "south alabama",
    "south carolina": "south carolina",
    "south carolina state": "south carolina st",
    "south carolina upstate": "usc upstate",
    "south dakota": "south dakota",
    "south dakota state": "south dakota st",
    "south florida": "south florida",
    "southeast missouri state": "southeast missouri st",
    "southeastern louisiana": "southeastern louisiana",
    "se louisiana": "southeastern louisiana",
    "southern": "southern university",
    "southern illinois": "southern illinois",
    "southern indiana": "southern indiana",
    "southern miss": "southern miss",
    "southern utah": "southern utah",
    "st. bonaventure": "st. bonaventure",
    "st. john's": "st. john's",
    "st thomas-minnesota": "st thomas",
    "stanford": "stanford",
    "stephen f. austin": "stephen f. austin",
    "stetson": "stetson",
    "stonehill": "stonehill",
    "stony brook": "stony brook",
    "syracuse": "syracuse",
    
    # T
    "tarleton state": "tarleton st",
    "tcu": "tcu",
    "temple": "temple",
    "tennessee": "tennessee",
    "tennessee state": "tennessee st",
    "tennessee tech": "tennessee tech",
    "texas": "texas",
    "texas a&m": "texas a&m",
    "texas a&m-corpus christi": "texas a&m-corpus christi",
    "texas southern": "texas southern",
    "texas state": "texas st",
    "texas tech": "texas tech",
    "toledo": "toledo",
    "towson": "towson",
    "troy": "troy",
    "tulane": "tulane",
    "tulsa": "tulsa",
    
    # U
    "uab": "uab",
    "uc davis": "uc davis",
    "uc irvine": "uc irvine",
    "uc riverside": "uc riverside",
    "uc san diego": "uc san diego",
    "uc santa barbara": "uc santa barbara",
    "ucf": "ucf",
    "ucla": "ucla",
    "uconn": "uconn",
    "connecticut": "uconn",
    "uic": "uic",
    "umass lowell": "umass lowell",
    "massachusetts-lowell": "umass lowell",
    "umbc": "umbc",
    "unc asheville": "unc asheville",
    "unc greensboro": "unc greensboro",
    "uncg": "unc greensboro",
    "unc wilmington": "unc wilmington",
    "unlv": "nevada las vegas",
    "usc": "usc",
    "ut arlington": "ut arlington",
    "ut martin": "tennessee-martin",
    "ut rio grande valley": "ut rio grande valley",
    "texas-rio grande valley": "ut rio grande valley",
    "utah": "utah",
    "utah state": "utah st",
    "utah tech": "utah tech",
    "utah valley": "utah valley",
    "utep": "utep",
    "utsa": "utsa",
    
    # V
    "valparaiso": "valparaiso",
    "vanderbilt": "vanderbilt",
    "vcu": "vcu",
    "vermont": "vermont",
    "villanova": "villanova",
    "virginia": "virginia",
    "virginia tech": "virginia tech",
    "vmi": "vmi",
    
    # W
    "wagner": "wagner",
    "wake forest": "wake forest",
    "washington": "washington",
    "washington state": "washington st",
    "weber state": "weber st",
    "west georgia": "west georgia",
    "west virginia": "west virginia",
    "western carolina": "western carolina",
    "western illinois": "western illinois",
    "western kentucky": "western kentucky",
    "western michigan": "western michigan",
    "wichita state": "wichita st",
    "william & mary": "william & mary",
    "winthrop": "winthrop",
    "wisconsin": "wisconsin",
    "wofford": "wofford",
    "wright state": "wright st",
    "wyoming": "wyoming",
    
    # X-Y-Z
    "xavier": "xavier",
    "yale": "yale",
    "youngstown state": "youngstown st",

    # State for ESPN -> Kalshi
    # Kalshi "St." identity mappings
    "alabama st": "alabama st",
    "alcorn st": "alcorn st",
    "appalachian st": "appalachian st",
    "arizona st": "arizona st",
    "arkansas st": "arkansas st",
    "ball st": "ball st",
    "boise st": "boise st",
    "chicago st": "chicago st",
    "cleveland st": "cleveland st",
    "colorado st": "colorado st",
    "coppin st": "coppin st",
    "delaware st": "delaware st",
    "florida st": "florida st",
    "fresno st": "fresno st",
    "georgia st": "georgia st",
    "grambling st": "grambling st",
    "idaho st": "idaho st",
    "illinois st": "illinois st",
    "indiana st": "indiana st",
    "iowa st": "iowa st",
    "jackson st": "jackson st",
    "jacksonville st": "jacksonville st",
    "kansas st": "kansas st",
    "kennesaw st": "kennesaw st",
    "kent st": "kent st",
    "long beach st": "long beach st",
    "michigan st": "michigan st",
    "mississippi st": "mississippi st",
    "mississippi valley st": "mississippi valley st",
    "missouri st": "missouri st",
    "montana st": "montana st",
    "morehead st": "morehead st",
    "morgan st": "morgan st",
    "murray st": "murray st",
    "new mexico st": "new mexico st",
    "nicholls st": "nicholls st",
    "norfolk st": "norfolk st",
    "north carolina st": "north carolina st",
    "north dakota st": "north dakota st",
    "northwestern st": "northwestern st",
    "ohio st": "ohio st",
    "oklahoma st": "oklahoma st",
    "oregon st": "oregon st",
    "penn st": "penn st",
    "portland st": "portland st",
    "sacramento st": "sacramento st",
    "san diego st": "san diego st",
    "san jose st": "san jose st",
    "south carolina st": "south carolina st",
    "south dakota st": "south dakota st",
    "southeast missouri st": "southeast missouri st",
    "tarleton st": "tarleton st",
    "tennessee st": "tennessee st",
    "texas st": "texas st",
    "utah st": "utah st",
    "washington st": "washington st",
    "weber st": "weber st",
    "wichita st": "wichita st",
    "wright st": "wright st",
    "youngstown st": "youngstown st",

    # Other Kalshi-specific formats
    "iu indy": "iu indy",
    "central connecticut st": "central connecticut st",
    "east tennessee st": "east tennessee st",
    "niagara purple": "niagara",
    "niagara purple eagle": "niagara",
}

MASCOTS = [
    'wildcats', 'bulldogs', 'tigers', 'eagles', 'bears', 'lions', 'panthers', 'hawks', 'falcons',
    'cardinals', 'blue devils', 'demon deacons', 'tar heels', 'wolfpack', 'crimson tide',
    'fighting irish', 'hoosiers', 'buckeyes', 'wolverines', 'spartans', 'gophers', 'hawkeyes',
    'cyclones', 'jayhawks', 'sooners', 'longhorns', 'aggies', 'red raiders', 'horned frogs',
    'mustangs', 'cowboys', 'buffaloes', 'rebels', 'commodores', 'volunteers', 'gamecocks',
    'razorbacks', 'seminoles', 'hurricanes', 'cavaliers', 'hokies', 'yellow jackets', 'terrapins',
    'nittany lions', 'orange', 'red storm', 'hoyas', 'friars', 'musketeers', 'bluejays', 'shockers',
    'phoenix', 'billikens', 'braves', 'salukis', 'sycamores', 'redbirds', 'leathernecks', 'huskies',
    'flames', 'ramblers', 'golden griffins', 'purple eagles', 'bonnies', 'explorers', 'dukes', 'antelopes',
    'rams', 'owls', 'quakers', 'leopards', 'crusaders', 'patriots', 'colonials', 'spiders',
    'keydets', 'tribe', 'monarchs', 'pirates', 'chanticleers', 'paladins', 'mocs', 'catamounts',
    'terriers', 'buccaneers', 'seahawks', 'thundering herd', 'mountaineers', 'mean green',
    'roadrunners', 'golden eagles', "ragin' cajuns", 'warhawks', 'red wolves', 'trojans',
    'sun devils', 'lumberjacks', 'hornets', 'anteaters', 'gauchos', 'tritons', 'matadors',
    'highlanders', 'beach', 'toreros', 'dons', 'broncos', 'aztecs', 'lobos', 'utes', 'cougars',
    'beavers', 'ducks', 'pilots', 'waves', 'gaels', 'zags', 'saints', 'running rebels', 'runnin rebels', 'wolf pack', 'ragin cajuns', 'coyotes',
    'miners', 'islanders', 'vaqueros', 'bobcats', 'bengals', 'vandals', 'governors', 'skyhawks',
    'racers', 'colonels', 'hilltoppers', 'toppers', 'blue hose', 'fighting camels', 'jaspers',
    'red foxes', 'peacocks', 'purple aces', 'beacons', 'golden flashes', 'rockets', 'chippewas',
    'redhawks', 'penguins', 'raiders', 'flyers', 'bison', 'big green', 'crimson', 'privateers',
    'delta devils', 'jaguars', 'rattlers', 'blazers', 'bearkats', "runnin' bulldogs", 'hatters',
    'ospreys', 'dolphins', 'bisons', 'lancers', 'bruins', 'titans', '49ers', 'golden gophers',
    'badgers', 'boilermakers', 'fighting illini', 'scarlet knights', 'knights', 'golden hurricane',
    'green wave', 'blue demons', 'red flash', 'retrievers', 'river hawks', 'great danes',
    'seawolves', 'black bears', 'minutemen', 'royals', 'mastodons', 'tommies', 'warriors',
    'screaming eagles', 'norse', 'wolves', 'chargers', 'sharks', 'zips', 'thunderbirds',
    'trailblazers', 'bearcats', 'dragons', 'pride', 'lakers', 'revolutionaries',
    'golden lions', 'black knights', 'bulls', 'golden bears', 'vikings', 'big red', 
    'blue hens', 'pioneers', 'stags', 'gators', 'lopes', 'rainbow warriors', 'roos',
    'mountain hawks', 'greyhounds', 'blue raiders', 'grizzlies', 'midshipmen', 
    'cornhuskers', 'fighting hawks', 'golden grizzlies', 'demons', 'mavericks', 
    'broncs', 'jackrabbits', 'cardinal', 'texans', 'purple eagle', 'ragin cajuns', 'coyotes',
]

# Pre-sort mascots by length descending for matching (longer first)
_SORTED_MASCOTS = sorted(MASCOTS, key=len, reverse=True)

def strip_mascot(name: str) -> str:
    """Strip mascot from team name for display, preserving original case.
    E.g., 'Cal State Bakersfield Roadrunners' -> 'Cal State Bakersfield'
    """
    lower = name.lower().replace("'", "").replace(".", "")
    for mascot in _SORTED_MASCOTS:
        if lower.endswith(' ' + mascot):
            return name[:len(name) - len(mascot) - 1].rstrip(". '")
    return name

def normalize_team_name(name: str) -> str:
    """Normalize team name: lowercase, strip mascot, map to Kalshi format"""
    name = name.lower().strip()
    
    # Remove apostrophes and periods BEFORE mascot check
    name = name.replace("'", "").replace(".", "")
    
    # Strip mascot (using pre-sorted list, longest first)
    for mascot in _SORTED_MASCOTS:
        if name.endswith(' ' + mascot):
            name = name[:-len(mascot)-1].strip()
            break
    
    # Look up in map, return as-is if not found
    return TEAM_NAME_MAP.get(name, name)


def fetch_kalshi_odds() -> dict:
    """Fetch today's CBB odds from Kalshi using markets endpoint with pagination"""
    # If before 4am, use yesterday's date (games run late)
    now = datetime.now()
    if now.hour < 4:
        effective_date = now - timedelta(days=1)
    else:
        effective_date = now
    today_str = effective_date.strftime('%y%b%d').upper()  # e.g., '26JAN29'
    
    # Paginate through ALL markets using requests (like the working test)
    base_url = "https://api.elections.kalshi.com/trade-api/v2/markets"
    params = {"series_ticker": "KXNCAAMBGAME", "limit": 200}
    all_markets = []
    cursor = None
    
    print(f"  Fetching Kalshi markets (date={today_str})...", end='', flush=True)
    
    while True:
        try:
            if cursor:
                params["cursor"] = cursor
            resp = requests.get(base_url, params=params, timeout=15)
            data = resp.json()
            
            markets = data.get('markets', [])
            all_markets.extend(markets)
            
            cursor = data.get('cursor')
            if not cursor:
                break
                
        except Exception as e:
            print(f" error: {e}")
            break
    
    print(f" {len(all_markets)} total...", end='', flush=True)
    
    # Filter to today's markets
    today_markets = [m for m in all_markets if today_str in m.get('ticker', '')]
    print(f" {len(today_markets)} today...", end='', flush=True)
    
    # Group by event stem
    events = {}
    for m in today_markets:
        ticker = m.get('ticker', '')
        stem = ticker.rsplit('-', 1)[0]  # Remove team suffix
        events.setdefault(stem, []).append(m)
    
    print(f" {len(events)} games...", end='', flush=True)
    
    odds = {}
    matched = 0
    for event_stem, markets in events.items():
        if len(markets) != 2:
            continue
        
        # Get title from market
        title = markets[0].get('title', '')
        if ' at ' not in title:
            continue
        
        parts = title.replace(' Winner?', '').split(' at ')
        if len(parts) != 2:
            continue
        
        kalshi_away = parts[0].strip()
        kalshi_home = parts[1].strip()
        kalshi_away_norm = normalize_team_name(kalshi_away)
        kalshi_home_norm = normalize_team_name(kalshi_home)
        
        # Match markets to teams using yes_sub_title
        home_market = None
        away_market = None
        
        for m in markets:
            sub = normalize_team_name(m.get('yes_sub_title', ''))
            # Exact match only - substring matching causes Michigan/Michigan St. confusion
            if sub == kalshi_home_norm:
                home_market = m
            elif sub == kalshi_away_norm:
                away_market = m
        
        if not home_market or not away_market:
            continue
        
        # Extract prices directly from market data (YES and NO contracts)
        home_yes_bid = home_market.get('yes_bid', 0)
        home_yes_ask = home_market.get('yes_ask', 0)
        home_no_bid = home_market.get('no_bid', 0)
        home_no_ask = home_market.get('no_ask', 0)
        away_yes_bid = away_market.get('yes_bid', 0)
        away_yes_ask = away_market.get('yes_ask', 0)
        away_no_bid = away_market.get('no_bid', 0)
        away_no_ask = away_market.get('no_ask', 0)
        
        # Skip if no valid prices
        if not (home_yes_bid > 0 or home_yes_ask > 0 or away_yes_bid > 0 or away_yes_ask > 0):
            continue
        
        # Calculate mid
        home_mid = (home_yes_bid + home_yes_ask) / 200.0 if home_yes_bid and home_yes_ask else None
        away_mid = (away_yes_bid + away_yes_ask) / 200.0 if away_yes_bid and away_yes_ask else None
        
        if home_mid and not away_mid:
            away_mid = 1 - home_mid
        elif away_mid and not home_mid:
            home_mid = 1 - away_mid
        
        game_odds = {
            'kalshi_away': kalshi_away,
            'kalshi_home': kalshi_home,
            'kalshi_away_norm': kalshi_away_norm,
            'kalshi_home_norm': kalshi_home_norm,
            'away_prob': away_mid,
            'home_prob': home_mid,
            'home_yes_bid': home_yes_bid,
            'home_yes_ask': home_yes_ask,
            'home_no_bid': home_no_bid,
            'home_no_ask': home_no_ask,
            'away_yes_bid': away_yes_bid,
            'away_yes_ask': away_yes_ask,
            'away_no_bid': away_no_bid,
            'away_no_ask': away_no_ask,
            'home_ticker': home_market.get('ticker'),
            'away_ticker': away_market.get('ticker'),
            'title': title
        }
        
        odds[kalshi_away_norm] = game_odds
        odds[kalshi_home_norm] = game_odds
        matched += 1
    
    print(f" done ({matched} with prices)")
    return odds



def setup_database(conn):
    """Create paper trading tables"""
    cursor = conn.cursor()
    
    # Log every snapshot we take (including market data for backtesting)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS live_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            game_id TEXT NOT NULL,
            home_team TEXT,
            away_team TEXT,
            home_score INTEGER,
            away_score INTEGER,
            period INTEGER,
            clock TEXT,
            time_remaining_sec INTEGER,
            game_status TEXT,  -- 'pre', 'in', 'post'
            
            -- Model probabilities
            pregame_home_prob REAL,
            live_home_prob REAL,
            prob_change REAL,
            
            -- Kalshi YES market data
            home_yes_bid REAL,
            home_yes_ask REAL,
            away_yes_bid REAL,
            away_yes_ask REAL,
            home_spread REAL,
            away_spread REAL,
            
            -- Kalshi NO market data (for YES/NO arbitrage)
            home_no_bid REAL,
            home_no_ask REAL,
            away_no_bid REAL,
            away_no_ask REAL,
            
            -- Best tradeable Kalshi prices (factoring in YES vs NO)
            best_home_ask REAL,  -- min(home_yes_ask, 100-away_no_bid) - entry price
            best_home_bid REAL,  -- max(home_yes_bid, 100-away_no_ask) - exit price
            best_away_ask REAL,  -- min(away_yes_ask, 100-home_no_bid) - entry price
            best_away_bid REAL,  -- max(away_yes_bid, 100-home_no_ask) - exit price
            
            -- Calculated edges (using best tradeable prices)
            kalshi_home_edge REAL,  -- model - best_home_ask
            kalshi_away_edge REAL,  -- model - best_away_ask
            
            -- Legacy columns (kept for backwards compatibility)
            home_edge REAL,
            away_edge REAL
        )
    """)
    
    # Add new columns to existing live_snapshots table (if they don't exist)
    new_columns = [
        ('game_status', 'TEXT'),
        ('best_home_ask', 'REAL'),
        ('best_home_bid', 'REAL'),
        ('best_away_ask', 'REAL'),
        ('best_away_bid', 'REAL'),
        ('kalshi_home_edge', 'REAL'),
        ('kalshi_away_edge', 'REAL'),
    ]
    for col_name, col_type in new_columns:
        try:
            cursor.execute(f"ALTER TABLE live_snapshots ADD COLUMN {col_name} {col_type}")
        except:
            pass  # Column already exists
    
    # Create indexes for faster queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_live_snapshots_game_id ON live_snapshots(game_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_live_snapshots_timestamp ON live_snapshots(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_live_snapshots_game_ts ON live_snapshots(game_id, timestamp)")
    
    # Log trades (entries and exits)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            game_id TEXT NOT NULL,
            home_team TEXT,
            away_team TEXT,
            
            -- Game state at trade time
            home_score INTEGER,
            away_score INTEGER,
            time_remaining_sec INTEGER,
            game_status TEXT,  -- 'pre' or 'in'
            
            -- Model probabilities at trade time
            pregame_home_prob REAL,
            live_home_prob REAL,
            
            -- Market at trade time (home team perspective)
            market_home_prob REAL,
            market_source TEXT,
            
            -- Trade details
            trade_type TEXT,  -- 'ENTRY' or 'EXIT'
            side TEXT,  -- 'home' or 'away'
            
            -- Entry fields
            entry_price REAL,  -- ASK price we paid
            entry_spread REAL,  -- Spread at entry
            entry_edge REAL,  -- Edge at entry
            market_ticker TEXT,
            
            -- Exit fields (NULL for entries)
            exit_price REAL,  -- BID price we got
            exit_reason TEXT,
            pnl REAL,  -- Profit/loss in decimal (0.05 = 5¢)
            
            -- Linking
            entry_trade_id INTEGER,  -- For exits, references the entry
            
            -- Notes
            notes TEXT
        )
    """)
    
    # Add columns if they don't exist (for existing databases)
    new_columns = [
        ("paper_trades", "entry_price", "REAL"),
        ("paper_trades", "market_ticker", "TEXT"),
        ("paper_trades", "trade_type", "TEXT"),
        ("paper_trades", "entry_spread", "REAL"),
        ("paper_trades", "entry_edge", "REAL"),
        ("paper_trades", "exit_price", "REAL"),
        ("paper_trades", "exit_reason", "TEXT"),
        ("paper_trades", "pnl", "REAL"),
        ("paper_trades", "entry_trade_id", "INTEGER"),
        ("paper_trades", "game_status", "TEXT"),
        ("live_snapshots", "home_yes_bid", "REAL"),
        ("live_snapshots", "home_yes_ask", "REAL"),
        ("live_snapshots", "away_yes_bid", "REAL"),
        ("live_snapshots", "away_yes_ask", "REAL"),
        ("live_snapshots", "home_spread", "REAL"),
        ("live_snapshots", "away_spread", "REAL"),
        ("live_snapshots", "home_edge", "REAL"),
        ("live_snapshots", "away_edge", "REAL"),
        # Kalshi NO contract columns (for hedge analysis)
        ("live_snapshots", "home_no_bid", "REAL"),
        ("live_snapshots", "home_no_ask", "REAL"),
        ("live_snapshots", "away_no_bid", "REAL"),
        ("live_snapshots", "away_no_ask", "REAL"),
        # Contract type for position persistence
        ("paper_trades", "contract_type", "TEXT"),
    ]
    for table, col, dtype in new_columns:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {dtype}")
        except:
            pass
    
    conn.commit()
    print("✓ Database tables ready")


class LiveWinProbabilityModel:
    """Calibrated live win probability model"""
    BASE_STD = 17.2
    
    def calculate_win_probability(self, pregame_win_prob: float, score_diff: float, time_remaining_sec: int) -> float:
        total_game_sec = 40 * 60
        time_fraction = max(0.001, min(1.0, time_remaining_sec / total_game_sec))
        
        if pregame_win_prob >= 0.999:
            pregame_implied_margin = 3.5 * self.BASE_STD
        elif pregame_win_prob <= 0.001:
            pregame_implied_margin = -3.5 * self.BASE_STD
        else:
            pregame_implied_margin = norm.ppf(pregame_win_prob) * self.BASE_STD
        
        expected_remaining_edge = pregame_implied_margin * time_fraction
        expected_final_margin = score_diff + expected_remaining_edge
        remaining_std = self.BASE_STD * math.sqrt(time_fraction)
        
        if remaining_std > 0.001:
            return norm.cdf(expected_final_margin / remaining_std)
        return 1.0 if expected_final_margin > 0 else 0.0


class PaperTrader:
    """Paper trading system with logging"""
    
    # Entry parameters (OPTIMIZED 2026-02-02 via 3-day backtest)
    EDGE_THRESHOLD = 0.10  # 10% edge to trigger entry
    MAX_SPREAD_CENTS = 3   # Only trade tight spreads (was 4¢)
    MIN_TIME_REMAINING = 240  # Don't enter with less than 4 minutes remaining
    MIN_ENTRY_PRICE = 0.10   # Don't enter below 10¢
    MAX_ENTRY_PRICE = 0.90   # Don't enter above 90¢
    
    # Exit parameters - EV-BASED (2026-02-04)
    # Exit when: EV_exit > EV_hold
    # EV_hold = model_prob × 100
    # EV_exit = current_bid - exit_fee - slippage
    EV_EXIT_SLIPPAGE_CENTS = 0  # IOC fills at best available price; fee already covers transaction cost
    
    # Option value formula (fit to backward induction with DT=120, σ=4.25¢, slip=0)
    # OV = OV_SCALE * N^OV_EXPONENT * (1 + OV_PROB_COEFF * p * (1-p))
    # where N = time_remaining / OV_DECORR_SEC (independent sell opportunities)
    OV_SCALE = 1.71
    OV_EXPONENT = 0.43
    OV_PROB_COEFF = -0.76
    OV_DECORR_SEC = 120  # Noise decorrelation timescale (empirical autocorrelation)
    
    # Stop loss and cooldown
    STOP_LOSS_CENTS = None   # No stop loss - EV-based exits handle this
    PREGAME_STOP_LOSS_CENTS = None  # No stop loss for pregame entries either
    COOLDOWN_AFTER_EXIT = 30  # 30 second cooldown after any exit
    COOLDOWN_AFTER_STOP = 240  # 4 minutes after stop loss (if ever re-enabled)
    
    # ESPN score freshness guard
    # Only enter/exit when ESPN score data is recently updated.
    # Prevents phantom edges from stale ESPN scores after Kalshi has already priced in a basket.
    # Exception: if game clock is also stale (timeout/halftime), scores aren't changing so data is fine.
    MAX_ESPN_SCORE_STALE_SEC = 15  # Max seconds since last ESPN score change during active play

    def get_required_edge(self, entry_price: float, time_remaining: int = 2400) -> float:
        """Dynamic edge threshold based on time remaining (exponential ramp).
        
        Exponential (k=2) ramp from 5% at game start to 15% at 4:00 remaining.
        Gentle early (when MMs still calibrating), steep late (info asymmetry explodes).
        
        Pregame (time_remaining >= 2400) uses 5%
        """
        E_START = 0.05  # 5% at game start / pregame
        E_END = 0.15    # 15% at 4:00 remaining (MIN_TIME_REMAINING)
        T_START = 2400  # 40:00 (full game)
        T_END = 240     # 4:00 (MIN_TIME_REMAINING cutoff)
        K = 2           # Exponential factor (k=2 = squared, convex curve)
        
        if time_remaining >= T_START:
            return E_START
        elif time_remaining <= T_END:
            return E_END
        else:
            # Exponential interpolation (convex: gentle early, steep late)
            progress = (T_START - time_remaining) / (T_START - T_END)  # 0 to 1
            curved_progress = progress ** K  # Squaring makes it convex
            return E_START + curved_progress * (E_END - E_START)
    
    def __init__(self, db_path: str, live_mode: bool = False, live_contracts: int = 10):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.model = LiveWinProbabilityModel()
        self.predictions = {}
        self.kalshi_odds = {}
        self.open_positions = {}  # game_id -> position info
        self.exit_cooldowns = {}  # (game_id, side) -> (exit_game_time, cooldown_secs, reason)
        self.just_exited = set()  # game_ids that exited this tick (prevent instant re-entry)
        self.last_snapshot = {}
        self.realized_pnl = 0.0  # Running total of closed trade P/L
        self.trade_count = 0  # Number of completed trades
        self.current_capital_deployed = 0.0  # Sum of entry prices for open positions (in cents)
        self.peak_capital_deployed = 0.0  # Max capital deployed at any moment (for real ROI calc)
        self.previous_scores = {}  # game_id -> (home_score, away_score) - for bad data detection
        self.last_score_change_time = {}  # game_id -> time.time() of last ESPN score change
        
        # === ANTI-ADVERSE-SELECTION STATE ===
        self.last_espn_update = {}  # game_id -> (timestamp, home_score, away_score, time_remaining) - detect stale data
        self.edge_confirmations = {}  # (game_id, side) -> (edge, timestamp) - two-tick confirmation
        
        # === LIVE TRADING STATE ===
        self.live_mode = live_mode
        self.live_contracts = live_contracts
        self.kalshi_client = None
        self.kalshi_starting_balance = 0.0  # Portfolio value at session start
        self.kalshi_starting_cash = 0.0     # Cash at session start
        
        if self.live_mode:
            if not KALSHI_LIVE_AVAILABLE:
                print("❌ kalshi_live_client.py not found - cannot enable live trading")
                sys.exit(1)
            
            print("\n" + "=" * 70)
            print("🔴 LIVE TRADING MODE - REAL MONEY 🔴")
            print("=" * 70)
            try:
                self.kalshi_client = KalshiLiveClient(
                    api_key_id=os.environ['KALSHI_API_KEY_ID'],
                    private_key_path=os.environ.get('KALSHI_PRIVATE_KEY_PATH', 'kalshi-key.key'),
                    use_demo=False,
                    max_position_size=100,
                    max_order_value_cents=5000
                )
                balance = self.kalshi_client.get_balance()
                positions = self.kalshi_client.get_positions()
                self.kalshi_positions = positions  # Store for reconciliation
                # Calculate position value - negative yes_count = NO position
                position_value = 0.0
                for p in positions.values():
                    if p.yes_count != 0:
                        # yes_avg_price is always the actual price paid
                        position_value += p.yes_avg_price * abs(p.yes_count) / 100.0
                self.kalshi_starting_cash = balance['balance']
                self.kalshi_starting_balance = balance['balance'] + position_value
                print(f"✓ Connected to Kalshi")
                print(f"  Cash: ${balance['balance']:.2f} | Positions: ${position_value:.2f} | Total: ${self.kalshi_starting_balance:.2f}")
                print(f"  Contracts per trade: {self.live_contracts}")
                print("=" * 70 + "\n")
            except Exception as e:
                print(f"❌ Failed to connect to Kalshi: {e}")
                sys.exit(1)
        
        # ===== WEBSOCKET ORDERBOOK =====
        self.ws_book = None
        self.use_ws = False
        self._ws_subscribed_tickers = set()
        
        if KALSHI_WS_AVAILABLE and self.live_mode:
            try:
                self.ws_book = KalshiWSOrderbook(
                    api_key=os.environ['KALSHI_API_KEY_ID'],
                    private_key_path=os.environ.get('KALSHI_PRIVATE_KEY_PATH', 'kalshi-key.key'),
                    verbose=False
                )
                self.ws_book.start()
                time.sleep(1)  # Wait for connection
                
                if self.ws_book.connected:
                    self.use_ws = True
                    print("  ✓ Kalshi WebSocket orderbook connected")
                else:
                    print("  ⚠️ WebSocket failed to connect, using REST fallback")
            except Exception as e:
                print(f"  ⚠️ WebSocket init error: {e}, using REST fallback")
        
        setup_database(self.conn)
        self._load_predictions()
        self._load_kalshi_odds()
        self._load_open_positions()  # In live mode, builds from Kalshi API; in paper mode, from DB
        self._load_realized_pnl()    # Restore P/L from today's completed trades
    
    def compute_option_value(self, time_remaining_sec: int, prob: float) -> float:
        """Compute option value using closed-form formula.
        
        Fit to backward induction (DT=120, σ_noise=4.25¢, slip=0) using
        274K+ live snapshots. Autocorrelation analysis shows noise decorrelates
        at ~120s intervals, giving N = time/120 independent sell opportunities.
        
        OV = 1.75 × N^0.43 × (1 - 0.75 × p(1-p))
        
        Args:
            time_remaining_sec: seconds remaining in game
            prob: model win probability for our side (0-1)
            
        Returns:
            Option value in cents
        """
        N = max(0, time_remaining_sec) / self.OV_DECORR_SEC
        if N <= 0:
            return 0.0
        return self.OV_SCALE * (N ** self.OV_EXPONENT) * (1 + self.OV_PROB_COEFF * prob * (1 - prob))
    
    def compute_min_sell_bid(self, time_remaining_sec: int, our_prob: float) -> float:
        """Compute minimum bid (in cents) needed to trigger an EV exit.
        
        Solves: bid - fee(bid) = our_prob * 100 + OV
        Uses iterative Newton steps from initial guess bid₀ = ev_hold.
        
        Returns:
            Minimum bid in cents to justify selling, or 100 if hold-to-settlement.
        """
        ov = self.compute_option_value(time_remaining_sec, our_prob)
        ev_hold = our_prob * 100 + ov
        
        if ev_hold >= 99:
            return 100.0  # Effectively "hold forever"
        
        # Newton: bid - fee(bid) = ev_hold
        # Start with bid₀ = ev_hold, iterate: bid_{n+1} = ev_hold + fee(bid_n)
        bid = ev_hold
        for _ in range(3):
            fee = kalshi_fee_cents(bid / 100.0)
            bid = ev_hold + fee
        
        return min(99.0, max(1.0, bid))
    
    def _load_predictions(self):
        """Load today's predictions (uses yesterday if before 4am)"""
        cursor = self.conn.cursor()
        # If before 4am, use yesterday's date (games run late)
        now = datetime.now()
        if now.hour < 4:
            effective_date = now - timedelta(days=1)
        else:
            effective_date = now
        today = effective_date.strftime('%Y-%m-%d')
        
        cursor.execute("""
            SELECT game_id, team_a_win_prob, team_a_id, team_b_id, team_a_home
            FROM future_predictions
            WHERE game_date = ?
        """, (today,))
        
        for row in cursor.fetchall():
            game_id, win_prob, team_a_id, team_b_id, team_a_home = row
            # team_a_win_prob is from team_a's perspective
            # We need home team's perspective for live tracking
            if team_a_home == 1:
                # team_a is home, so team_a_win_prob = home_prob
                home_prob = win_prob
            else:
                # team_a is away, so home_prob = 1 - team_a_win_prob
                home_prob = 1 - win_prob if win_prob else None
            
            self.predictions[str(game_id)] = {
                'pregame_home_prob': home_prob,
                'team_a_id': str(team_a_id),
                'team_b_id': str(team_b_id),
                'team_a_home': team_a_home
            }
        
        print(f"✓ Loaded {len(self.predictions)} predictions for {today}")
    
    def _load_kalshi_odds(self):
        """Load current Kalshi odds for today's games"""
        self.kalshi_odds = fetch_kalshi_odds()
        print(f"✓ Loaded {len(self.kalshi_odds) // 2} Kalshi markets")  # div 2 because indexed by both teams
    
    def _load_open_positions(self):
        """Load open positions - uses Kalshi API as source of truth in live mode.
        
        LIVE MODE: Builds positions entirely from Kalshi API. No paper_trades lookup.
        PAPER MODE: Uses paper_trades table (legacy behavior).
        """
        if self.live_mode:
            self._build_positions_from_kalshi()
        else:
            self._load_positions_from_db()
    
    def _build_positions_from_kalshi(self):
        """Build open_positions dict entirely from Kalshi API (LIVE MODE).
        
        Kalshi is the ONLY source of truth:
        - Contract counts from yes_count
        - Entry prices from yes_avg_price  
        - Side/contract_type inferred from ticker + position sign
        
        No paper_trades lookup. No reconciliation needed.
        """
        if not self.kalshi_client:
            print("✗ Cannot build positions - no Kalshi client")
            return
        
        try:
            positions = self.kalshi_client.get_positions()
        except Exception as e:
            print(f"✗ Failed to fetch Kalshi positions: {e}")
            return
        
        # Build reverse index: ticker -> kalshi_odds entry
        ticker_to_odds = {}
        for team_key, odds in self.kalshi_odds.items():
            if odds.get('home_ticker'):
                ticker_to_odds[odds['home_ticker']] = ('home', odds)
            if odds.get('away_ticker'):
                ticker_to_odds[odds['away_ticker']] = ('away', odds)
        
        loaded = 0
        skipped = 0
        
        for ticker, pos in positions.items():
            if pos.yes_count == 0:
                continue  # No position
            
            # Kalshi uses signed yes_count: positive = YES, negative = NO
            is_no_position = pos.yes_count < 0
            contract_count = abs(pos.yes_count)
            entry_price_cents = pos.yes_avg_price
            
            # Find the kalshi_odds entry for this ticker
            if ticker not in ticker_to_odds:
                print(f"  ⚠️ Unknown ticker (no matching market): {ticker}")
                skipped += 1
                continue
            
            ticker_team, odds = ticker_to_odds[ticker]  # 'home' or 'away'
            
            # Determine our betting side and contract type
            # ticker_team = which team's contract this ticker is for
            # is_no_position = whether we bought NO (negative yes_count)
            #
            # If we own HOME YES: we're betting home wins, side='home', contract='home_yes'
            # If we own HOME NO: we're betting home loses (away wins), side='away', contract='home_no'
            # If we own AWAY YES: we're betting away wins, side='away', contract='away_yes'
            # If we own AWAY NO: we're betting away loses (home wins), side='home', contract='away_no'
            
            if ticker_team == 'home':
                if is_no_position:
                    side = 'away'  # Betting against home = betting for away
                    contract_type = 'home_no'
                else:
                    side = 'home'
                    contract_type = 'home_yes'
            else:  # ticker_team == 'away'
                if is_no_position:
                    side = 'home'  # Betting against away = betting for home
                    contract_type = 'away_no'
                else:
                    side = 'away'
                    contract_type = 'away_yes'
            
            home_team = odds.get('kalshi_home', 'Unknown')
            away_team = odds.get('kalshi_away', 'Unknown')
            
            # Try to find game_id by matching to predictions
            # We'll use ticker as temporary key if no match found
            game_id = self._find_game_id_for_teams(home_team, away_team)
            if not game_id:
                # Use ticker as fallback key - will match later when ESPN data comes in
                game_id = f"kalshi:{ticker}"
                print(f"  ⚠️ No game_id found for {away_team} @ {home_team}, using ticker key")
            
            # Build position dict with all Kalshi data
            self.open_positions[game_id] = {
                'trade_id': None,  # No DB entry in live mode
                'side': side,
                'entry_price': entry_price_cents / 100.0,  # Convert to decimal
                'entry_time': None,  # Unknown - position existed before this session
                'entry_status': 'unknown',  # Unknown if pregame or in-game
                'home_team': home_team,
                'away_team': away_team,
                'market_ticker': ticker,
                'market_source': 'Kalshi',
                'contract_type': contract_type,
                'live_fill_count': contract_count,
                'live_avg_price': entry_price_cents,
                'live_order_id': None  # Unknown - position existed before this session
            }
            loaded += 1
        
        if loaded > 0:
            print(f"✓ Loaded {loaded} positions from Kalshi API")
            total_value = 0
            for game_id, pos in self.open_positions.items():
                team = pos['home_team'] if pos['side'] == 'home' else pos['away_team']
                contract = pos['contract_type'].replace('_', ' ').upper()
                entry_cents = pos['entry_price'] * 100
                count = pos['live_fill_count']
                position_value = entry_cents * count / 100
                total_value += position_value
                print(f"  └─ {contract} {team[:20]} @ {entry_cents:.0f}¢ x{count} = ${position_value:.2f}")
            print(f"  Total position value (cost basis): ${total_value:.2f}")
        elif skipped > 0:
            print(f"✓ No active positions ({skipped} unmatched tickers)")
        else:
            print(f"✓ No open positions on Kalshi")
    
    def _find_game_id_for_teams(self, home_team: str, away_team: str) -> Optional[str]:
        """Try to find ESPN game_id for a pair of team names.
        
        Searches predictions which are keyed by game_id.
        Returns game_id if found, None otherwise.
        """
        home_norm = normalize_team_name(home_team)
        away_norm = normalize_team_name(away_team)
        
        # Query schedules table for today's games with team names
        cursor = self.conn.cursor()
        now = datetime.now()
        if now.hour < 4:
            effective_date = now - timedelta(days=1)
        else:
            effective_date = now
        today = effective_date.strftime('%Y-%m-%d')
        
        # Try to find in schedules table (if it has team names)
        try:
            cursor.execute("""
                SELECT game_id, home_team, away_team 
                FROM schedules 
                WHERE game_date = ?
            """, (today,))
            
            for row in cursor.fetchall():
                db_game_id, db_home, db_away = row
                db_home_norm = normalize_team_name(db_home) if db_home else ''
                db_away_norm = normalize_team_name(db_away) if db_away else ''
                
                # Check for match (allow some flexibility)
                home_match = home_norm == db_home_norm or home_norm in db_home_norm or db_home_norm in home_norm
                away_match = away_norm == db_away_norm or away_norm in db_away_norm or db_away_norm in away_norm
                
                if home_match and away_match:
                    return str(db_game_id)
        except Exception:
            pass  # schedules table might not have team names
        
        return None
    
    def _find_position_for_game(self, game_id: str, odds: Optional[dict]) -> Optional[str]:
        """Find position key for a game, handling both game_id and kalshi: prefix keys.
        
        In live mode, positions might be keyed by 'kalshi:TICKER' if game_id wasn't found
        at startup. This method also checks for ticker matches and migrates keys if found.
        
        Returns the key in open_positions, or None if no position exists.
        """
        # Direct lookup by game_id
        if game_id in self.open_positions:
            return game_id
        
        # Check for kalshi: prefixed positions that match this game's tickers
        if odds:
            home_ticker = odds.get('home_ticker')
            away_ticker = odds.get('away_ticker')
            
            for pos_key in list(self.open_positions.keys()):
                if pos_key.startswith('kalshi:'):
                    pos = self.open_positions[pos_key]
                    pos_ticker = pos.get('market_ticker')
                    
                    if pos_ticker and (pos_ticker == home_ticker or pos_ticker == away_ticker):
                        # Found a match! Migrate to proper game_id key
                        print(f"  ℹ️ Matched position {pos_key} -> {game_id}")
                        self.open_positions[game_id] = self.open_positions.pop(pos_key)
                        return game_id
        
        return None
    
    def _load_positions_from_db(self):
        """Load positions from paper_trades table (PAPER MODE only).
        
        Legacy behavior for non-live mode where we track positions in the database.
        """
        cursor = self.conn.cursor()
        now = datetime.now()
        if now.hour < 4:
            effective_date = now - timedelta(days=1)
        else:
            effective_date = now
        today = effective_date.strftime('%Y-%m-%d')
        
        # Find all ENTRY records that don't have a matching EXIT
        cursor.execute("""
            SELECT 
                p.id as trade_id,
                p.game_id,
                p.side,
                p.entry_price,
                p.timestamp as entry_time,
                p.game_status as entry_status,
                p.home_team,
                p.away_team,
                p.market_ticker,
                p.market_source,
                p.contract_type
            FROM paper_trades p
            WHERE p.trade_type LIKE 'ENTRY%'
            AND p.id NOT IN (
                SELECT entry_trade_id 
                FROM paper_trades 
                WHERE trade_type = 'EXIT' 
                AND entry_trade_id IS NOT NULL
            )
            AND date(p.timestamp) = ?
        """, (today,))
        
        restored = 0
        skipped = 0
        
        for row in cursor.fetchall():
            trade_id, game_id, side, entry_price, entry_time, entry_status, \
                home_team, away_team, market_ticker, market_source, contract_type = row
            
            # Only restore if we have predictions for this game (it's still active)
            if game_id not in self.predictions:
                skipped += 1
                continue
            
            self.open_positions[game_id] = {
                'trade_id': trade_id,
                'side': side,
                'entry_price': entry_price,
                'entry_time': datetime.fromisoformat(entry_time) if entry_time else datetime.now(),
                'entry_status': entry_status or 'in',
                'home_team': home_team,
                'away_team': away_team,
                'market_ticker': market_ticker,
                'market_source': market_source or 'Kalshi',
                'contract_type': contract_type or f'{side}_yes',
                'live_fill_count': 0,
                'live_avg_price': None
            }
            restored += 1
        
        if restored > 0:
            print(f"✓ Restored {restored} open positions from DB")
            for game_id, pos in self.open_positions.items():
                team = pos['home_team'] if pos['side'] == 'home' else pos['away_team']
                contract = pos['contract_type'].replace('_', ' ').upper()[:2]
                print(f"  └─ #{pos['trade_id']}: {contract} {team[:20]} @ {pos['entry_price']*100:.0f}¢")
        elif skipped > 0:
            print(f"✓ No positions to restore ({skipped} orphaned from finished games)")
        else:
            print(f"✓ No open positions to restore")
    
    # NOTE: _reconcile_kalshi_positions removed - in live mode, Kalshi IS the source of truth
    # No reconciliation needed since we build positions directly from Kalshi API

    def _load_realized_pnl(self):
        """Restore realized P/L and trade count from today's completed trades.
        
        This allows the trader to restart without losing the running P/L total.
        """
        cursor = self.conn.cursor()
        # If before 4am, use yesterday's date (games run late)
        now = datetime.now()
        if now.hour < 4:
            effective_date = now - timedelta(days=1)
        else:
            effective_date = now
        today = effective_date.strftime('%Y-%m-%d')
        
        # Sum up P/L from all EXIT trades today
        cursor.execute("""
            SELECT 
                COALESCE(SUM(x.pnl), 0) as total_pnl,
                COUNT(*) as trade_count
            FROM paper_trades x
            WHERE x.trade_type = 'EXIT'
            AND date(x.timestamp) = ?
            AND x.pnl IS NOT NULL
        """, (today,))
        
        row = cursor.fetchone()
        if row:
            total_pnl, trade_count = row
            self.realized_pnl = (total_pnl or 0) * 100  # Convert to cents
            self.trade_count = trade_count or 0
            
            # Set current capital from open positions
            self.current_capital_deployed = sum(
                pos['entry_price'] * 100 for pos in self.open_positions.values()
            )
            # Initialize peak to current (will update as we trade)
            self.peak_capital_deployed = self.current_capital_deployed if self.current_capital_deployed > 0 else 1.0
            
            if self.trade_count > 0:
                roi_pct = (self.realized_pnl / self.peak_capital_deployed * 100) if self.peak_capital_deployed > 0 else 0
                print(f"✓ Restored Net P/L: {self.realized_pnl:+.0f}¢ ({roi_pct:+.1f}% ROI) from {self.trade_count} trades today")
            else:
                print(f"✓ No completed trades yet today")
        else:
            print(f"✓ No completed trades yet today")

    def _match_kalshi_odds(self, espn_home: str, espn_away: str) -> Optional[dict]:
        """Match ESPN team names to Kalshi odds, returning full odds from ESPN's perspective"""
        espn_home_norm = normalize_team_name(espn_home)
        espn_away_norm = normalize_team_name(espn_away)
        
        best_match = None
        best_score = 0
        best_orientation = None
        
        # Track which games we've already checked (since each game is indexed twice)
        checked_titles = set()
        
        for kalshi_key, odds in self.kalshi_odds.items():
            title = odds['title']
            if title in checked_titles:
                continue
            checked_titles.add(title)
            
            kalshi_home_norm = odds['kalshi_home_norm']
            kalshi_away_norm = odds['kalshi_away_norm']
            
            # Score the match quality for both orientations
            # Same orientation: ESPN home = Kalshi home, ESPN away = Kalshi away
            same_home_score = self._name_match_score(espn_home_norm, kalshi_home_norm)
            same_away_score = self._name_match_score(espn_away_norm, kalshi_away_norm)
            same_total = same_home_score + same_away_score
            
            # Flipped orientation: ESPN home = Kalshi away, ESPN away = Kalshi home
            flip_home_score = self._name_match_score(espn_home_norm, kalshi_away_norm)
            flip_away_score = self._name_match_score(espn_away_norm, kalshi_home_norm)
            flip_total = flip_home_score + flip_away_score
            
            # Require BOTH teams to have at least some match (score > 0)
            if same_total > best_score and same_home_score > 0 and same_away_score > 0:
                best_score = same_total
                best_match = odds
                best_orientation = 'same'
            
            if flip_total > best_score and flip_home_score > 0 and flip_away_score > 0:
                best_score = flip_total
                best_match = odds
                best_orientation = 'flipped'
        
        # Require minimum match quality (2 = both teams exact match)
        if best_match and best_score >= 1.5:
            if best_orientation == 'same':
                return best_match
            else:
                # Flipped - swap home/away fields (including NO contracts)
                return {
                    'kalshi_home': best_match['kalshi_away'],
                    'kalshi_away': best_match['kalshi_home'],
                    'kalshi_home_norm': best_match['kalshi_away_norm'],
                    'kalshi_away_norm': best_match['kalshi_home_norm'],
                    'home_prob': best_match['away_prob'],
                    'away_prob': best_match['home_prob'],
                    'home_yes_bid': best_match['away_yes_bid'],
                    'home_yes_ask': best_match['away_yes_ask'],
                    'home_no_bid': best_match['away_no_bid'],
                    'home_no_ask': best_match['away_no_ask'],
                    'away_yes_bid': best_match['home_yes_bid'],
                    'away_yes_ask': best_match['home_yes_ask'],
                    'away_no_bid': best_match['home_no_bid'],
                    'away_no_ask': best_match['home_no_ask'],
                    'home_ticker': best_match['away_ticker'],
                    'away_ticker': best_match['home_ticker'],
                    'title': best_match['title']
                }
        
        return None
    
    def _name_match_score(self, name1: str, name2: str) -> float:
        """Score how well two normalized team names match. Returns 0-1."""
        if name1 == name2:
            return 1.0
        
        # Check if one is a subset of the other, but penalize partial matches
        if len(name1) >= 4 and len(name2) >= 4:
            # Only allow substring if it's a significant portion
            if name1 in name2:
                return len(name1) / len(name2) if len(name1) / len(name2) > 0.6 else 0
            if name2 in name1:
                return len(name2) / len(name1) if len(name2) / len(name1) > 0.6 else 0
        
        return 0
    
    def _get_kalshi_portfolio(self) -> dict:
        """Get current Kalshi portfolio state for display.
        
        Returns dict with: cash, position_value, total, position_count, session_pnl
        Uses current market bid prices for accurate position valuation.
        """
        if not self.live_mode or not self.kalshi_client:
            return None
        
        try:
            balance = self.kalshi_client.get_balance()
            positions = self.kalshi_client.get_positions()
            
            # Calculate position value using CURRENT market prices, not cost basis
            position_value = 0.0
            cost_basis = 0.0
            position_count = 0
            
            for ticker, pos in positions.items():
                if pos.yes_count != 0:
                    count = abs(pos.yes_count)
                    entry_price = pos.yes_avg_price  # Actual price paid (cents)
                    
                    # Negative yes_count = NO position, Positive = YES position
                    is_no_position = pos.yes_count < 0
                    
                    position_count += 1
                    cost_basis += entry_price * count / 100.0
                    
                    # Look up current bid from our live odds data
                    current_bid = None
                    for team_key, odds in self.kalshi_odds.items():
                        if odds.get('home_ticker') == ticker:
                            current_bid = odds.get('home_no_bid') if is_no_position else odds.get('home_yes_bid')
                            break
                        elif odds.get('away_ticker') == ticker:
                            current_bid = odds.get('away_no_bid') if is_no_position else odds.get('away_yes_bid')
                            break
                    
                    if current_bid:
                        position_value += current_bid * count / 100.0
                    else:
                        # Fallback to cost basis if we can't find current price
                        position_value += entry_price * count / 100.0
                
            
            cash = balance['balance']
            total = cash + position_value
            session_pnl = total - self.kalshi_starting_balance
            
            return {
                'cash': cash,
                'position_value': position_value,
                'cost_basis': cost_basis,
                'total': total,
                'position_count': position_count,
                'session_pnl': session_pnl,
                'positions': positions
            }
        except Exception as e:
            print(f"  ⚠️ Kalshi portfolio fetch error: {e}")
            return None
    
    def _ws_subscribe_markets(self, home_ticker: str, away_ticker: str):
        """Subscribe to WebSocket orderbook updates for a game's markets"""
        if not self.use_ws or not self.ws_book:
            return
        
        tickers = []
        if home_ticker and home_ticker not in self._ws_subscribed_tickers:
            tickers.append(home_ticker)
            self._ws_subscribed_tickers.add(home_ticker)
        if away_ticker and away_ticker not in self._ws_subscribed_tickers:
            tickers.append(away_ticker)
            self._ws_subscribed_tickers.add(away_ticker)
        
        if tickers:
            self.ws_book.subscribe(tickers)
    
    def _get_best_market_odds(self, espn_home: str, espn_away: str) -> Optional[dict]:
        """Get Kalshi odds for a game, preferring WebSocket data over REST.
        
        Returns Kalshi odds dict with market_source set to 'Kalshi'.
        """
        kalshi = self._match_kalshi_odds(espn_home, espn_away)
        
        if not kalshi:
            return None
        
        kalshi['market_source'] = 'Kalshi'
        kalshi['data_source'] = 'rest'  # Default to REST
        kalshi['ws_age_ms'] = None
        
        # If WebSocket is available, try to get fresher prices
        if self.use_ws and self.ws_book:
            home_ticker = kalshi.get('home_ticker')
            away_ticker = kalshi.get('away_ticker')
            
            # Subscribe if not already
            self._ws_subscribe_markets(home_ticker, away_ticker)
            
            # Get WS prices
            home_ws = self.ws_book.get_prices(home_ticker) if home_ticker else None
            away_ws = self.ws_book.get_prices(away_ticker) if away_ticker else None
            
            # If we have fresh WS data, use it instead of REST
            if home_ws or away_ws:
                kalshi['data_source'] = 'websocket'
                
                if home_ws and home_ws.get('yes_bid') is not None:
                    kalshi['home_yes_bid'] = home_ws['yes_bid']
                    kalshi['home_yes_ask'] = home_ws['yes_ask']
                    kalshi['home_no_bid'] = home_ws['no_bid']
                    kalshi['home_no_ask'] = home_ws['no_ask']
                    kalshi['ws_age_ms'] = home_ws['age_ms']
                
                if away_ws and away_ws.get('yes_bid') is not None:
                    kalshi['away_yes_bid'] = away_ws['yes_bid']
                    kalshi['away_yes_ask'] = away_ws['yes_ask']
                    kalshi['away_no_bid'] = away_ws['no_bid']
                    kalshi['away_no_ask'] = away_ws['no_ask']
                    if kalshi['ws_age_ms'] is None:
                        kalshi['ws_age_ms'] = away_ws['age_ms']
                    else:
                        kalshi['ws_age_ms'] = max(kalshi['ws_age_ms'], away_ws['age_ms'])
        
        return kalshi
    
    def _fetch_scoreboard(self) -> dict:
        """Fetch ESPN scoreboard for today's games"""
        try:
            # Add date filter to get TODAY's games, not yesterday's
            # Use same date logic as predictions (yesterday if before 4am)
            now = datetime.now()
            if now.hour < 4:
                effective_date = now - timedelta(days=1)
            else:
                effective_date = now
            date_str = effective_date.strftime('%Y%m%d')  # ESPN format: YYYYMMDD
            
            url = f"{SCOREBOARD_URL}&dates={date_str}"
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=15) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print(f"Error fetching scoreboard: {e}")
            return {}
    
    def _parse_clock(self, clock_str: str, period: int) -> int:
        """Convert clock to seconds remaining"""
        if not clock_str:
            return 0
        try:
            parts = clock_str.split(':')
            minutes = int(parts[0]) if len(parts) >= 1 else 0
            seconds = int(parts[1]) if len(parts) >= 2 else 0
            period_seconds = minutes * 60 + seconds
            
            if period == 1:
                return period_seconds + 20 * 60
            return period_seconds
        except:
            return 0
    
    def scan_games(self) -> List[dict]:
        """Scan all games and return current states"""
        data = self._fetch_scoreboard()
        if not data:
            return []
        
        now = time.time()
        games = []
        for event in data.get('events', []):
            try:
                game = self._parse_game(event)
                if game:
                    game_id = game['game_id']
                    home_score = game['home_score']
                    away_score = game['away_score']
                    time_remaining = game.get('time_remaining_sec', 2400)
                    
                    # Validate scores can only increase (detect ESPN bugs)
                    if game_id in self.previous_scores:
                        prev_home, prev_away = self.previous_scores[game_id]
                        
                        # Scores should never decrease in basketball
                        if home_score < prev_home or away_score < prev_away:
                            # Bad data! Skip this update, don't act on it
                            print(f"  ⚠️ BAD DATA: {game['away_team'][:15]} @ {game['home_team'][:15]} "
                                  f"score went {prev_away}-{prev_home} → {away_score}-{home_score}, ignoring")
                            continue  # Skip this game entirely this cycle
                        
                        # Track when SCORE specifically changes (not clock)
                        if home_score != prev_home or away_score != prev_away:
                            self.last_score_change_time[game_id] = now
                    else:
                        # First time seeing this game - treat scores as fresh
                        self.last_score_change_time[game_id] = now
                    
                    # Track ESPN data freshness - did the game state actually change?
                    current_state = (home_score, away_score, time_remaining)
                    if game_id in self.last_espn_update:
                        last_ts, last_state = self.last_espn_update[game_id]
                        if current_state != last_state:
                            # State changed - update timestamp
                            self.last_espn_update[game_id] = (now, current_state)
                            game['espn_update_age'] = 0.0  # Fresh data
                        else:
                            # Same state - data might be stale
                            game['espn_update_age'] = now - last_ts
                    else:
                        # First time seeing this game
                        self.last_espn_update[game_id] = (now, current_state)
                        game['espn_update_age'] = 0.0
                    
                    # Store current scores as valid
                    self.previous_scores[game_id] = (home_score, away_score)
                    
                    # Add score freshness info for adverse selection guard
                    game['seconds_since_score_change'] = now - self.last_score_change_time.get(game_id, now)
                    
                    games.append(game)
            except Exception as e:
                continue
        
        return games
    
    def _parse_game(self, event: dict) -> Optional[dict]:
        """Parse ESPN event"""
        game_id = str(event.get('id'))
        competition = event.get('competitions', [{}])[0]
        competitors = competition.get('competitors', [])
        
        if len(competitors) != 2:
            return None
        
        home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
        away = next((c for c in competitors if c.get('homeAway') == 'away'), None)
        
        if not home or not away:
            return None
        
        status = event.get('status', {})
        state = status.get('type', {}).get('state', 'pre')
        period = status.get('period', 0)
        clock = status.get('displayClock', '20:00')
        
        home_score = int(home.get('score', 0) or 0)
        away_score = int(away.get('score', 0) or 0)
        time_remaining = self._parse_clock(clock, period)
        
        # Pregame games should have full game time (40 minutes)
        if state == 'pre':
            time_remaining = 2400
        
        home_team = home.get('team', {})
        away_team = away.get('team', {})
        
        game = {
            'game_id': game_id,
            'home_team': home_team.get('displayName', 'Unknown'),
            'away_team': away_team.get('displayName', 'Unknown'),
            'home_team_id': str(home_team.get('id')),
            'away_team_id': str(away_team.get('id')),
            'home_score': home_score,
            'away_score': away_score,
            'period': period,
            'clock': clock,
            'time_remaining_sec': time_remaining,
            'status': state
        }
        
        # Add predictions if we have them
        pred = self.predictions.get(game_id)
        if pred:
            game['pregame_home_prob'] = pred['pregame_home_prob']
            
            if state == 'in':
                score_diff = home_score - away_score
                live_prob = self.model.calculate_win_probability(
                    pred['pregame_home_prob'], score_diff, time_remaining
                )
                game['live_home_prob'] = live_prob
                game['prob_change'] = live_prob - pred['pregame_home_prob']
        
        return game
    
    def log_snapshot(self, game: dict, kalshi_odds: dict = None, commit: bool = True):
        """Log a game snapshot with market data"""
        cursor = self.conn.cursor()
        
        # Initialize all market data fields
        home_yes_bid = None
        home_yes_ask = None
        away_yes_bid = None
        away_yes_ask = None
        home_no_bid = None
        home_no_ask = None
        away_no_bid = None
        away_no_ask = None
        home_spread = None
        away_spread = None
        
        # Best tradeable prices
        best_home_ask = None
        best_home_bid = None
        best_away_ask = None
        best_away_bid = None
        
        # Edges
        kalshi_home_edge = None
        kalshi_away_edge = None
        home_edge = None  # Legacy
        away_edge = None  # Legacy
        
        # Get model probability
        our_home = game.get('live_home_prob') or game.get('pregame_home_prob')
        
        if kalshi_odds:
            # YES contracts
            home_yes_bid = kalshi_odds.get('home_yes_bid')
            home_yes_ask = kalshi_odds.get('home_yes_ask')
            away_yes_bid = kalshi_odds.get('away_yes_bid')
            away_yes_ask = kalshi_odds.get('away_yes_ask')
            
            # NO contracts
            home_no_bid = kalshi_odds.get('home_no_bid')
            home_no_ask = kalshi_odds.get('home_no_ask')
            away_no_bid = kalshi_odds.get('away_no_bid')
            away_no_ask = kalshi_odds.get('away_no_ask')
            
            # Kalshi spreads (YES contracts)
            if home_yes_bid and home_yes_ask:
                home_spread = home_yes_ask - home_yes_bid
            if away_yes_bid and away_yes_ask:
                away_spread = away_yes_ask - away_yes_bid
            
            # Calculate BEST tradeable prices (YES vs NO equivalence)
            # To bet HOME: buy home_yes OR buy away_no (both pay $1 if home wins)
            # To bet AWAY: buy away_yes OR buy home_no (both pay $1 if away wins)
            # Best ENTRY = min of direct YES ask vs opposite NO ask
            home_asks = [x for x in [home_yes_ask, away_no_ask] if x is not None]
            away_asks = [x for x in [away_yes_ask, home_no_ask] if x is not None]
            best_home_ask = min(home_asks) if home_asks else None
            best_away_ask = min(away_asks) if away_asks else None
            
            # Best EXIT = max of direct YES bid vs opposite NO bid
            # (But note: you must exit via same contract you entered)
            home_bids = [x for x in [home_yes_bid, away_no_bid] if x is not None]
            away_bids = [x for x in [away_yes_bid, home_no_bid] if x is not None]
            best_home_bid = max(home_bids) if home_bids else None
            best_away_bid = max(away_bids) if away_bids else None
            
            # Calculate edges using BEST tradeable prices
            if our_home:
                if best_home_ask:
                    kalshi_home_edge = our_home - (best_home_ask / 100.0)
                if best_away_ask:
                    kalshi_away_edge = (1 - our_home) - (best_away_ask / 100.0)
                
                # Legacy edge (for backwards compatibility - uses YES ask only)
                if home_yes_ask:
                    home_edge = our_home - (home_yes_ask / 100.0)
                if away_yes_ask:
                    away_edge = (1 - our_home) - (away_yes_ask / 100.0)
        
        cursor.execute("""
            INSERT INTO live_snapshots 
            (timestamp, game_id, home_team, away_team, home_score, away_score,
             period, clock, time_remaining_sec, game_status, pregame_home_prob, live_home_prob,
             prob_change, home_yes_bid, home_yes_ask, away_yes_bid, away_yes_ask,
             home_spread, away_spread,
             home_no_bid, home_no_ask, away_no_bid, away_no_ask,
             best_home_ask, best_home_bid, best_away_ask, best_away_bid,
             kalshi_home_edge, kalshi_away_edge,
             home_edge, away_edge)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            game['game_id'],
            game['home_team'],
            game['away_team'],
            game.get('home_score'),
            game.get('away_score'),
            game.get('period'),
            game.get('clock'),
            game.get('time_remaining_sec'),
            game.get('status'),  # 'pre', 'in', 'post'
            game.get('pregame_home_prob'),
            game.get('live_home_prob'),
            game.get('prob_change'),
            home_yes_bid,
            home_yes_ask,
            away_yes_bid,
            away_yes_ask,
            home_spread,
            away_spread,
            home_no_bid,
            home_no_ask,
            away_no_bid,
            away_no_ask,
            best_home_ask,
            best_home_bid,
            best_away_ask,
            best_away_bid,
            kalshi_home_edge,
            kalshi_away_edge,
            home_edge,
            away_edge
        ))
        if commit:
            self.conn.commit()
    
    def check_opportunity(self, game: dict, kalshi_odds: dict) -> Optional[dict]:
        """Check if game presents a trading opportunity based on edge vs Kalshi market.
        Works for both pregame and live games.
        """
        if kalshi_odds is None:
            return None
        
        game_id = game['game_id']
        status = game['status']
        
        # Don't re-enter on same tick we exited
        if game_id in self.just_exited:
            return None
        
        time_remaining = game.get('time_remaining_sec', 2400)
        
        # === KALSHI DATA FRESHNESS GUARD ===
        # If using WebSocket, require recent Kalshi data before trading
        # This prevents phantom edges from stale REST data
        if self.use_ws and kalshi_odds and status == 'in':
            kalshi_age_ms = kalshi_odds.get('ws_age_ms')
            
            if kalshi_age_ms is not None:
                # Time-dependent threshold: stricter late game
                if time_remaining > 480:  # > 8:00
                    MAX_KALSHI_STALE_MS = 8000  # 8 seconds early/mid game
                else:
                    MAX_KALSHI_STALE_MS = 5000  # 5 seconds late game
                
                if kalshi_age_ms > MAX_KALSHI_STALE_MS:
                    return None  # Kalshi data too stale, skip
        
        # === ESPN SCORE FRESHNESS GUARD ===
        # Only enter when ESPN score data is recently updated.
        # Prevents phantom edges: score happens → Kalshi adjusts → ESPN lags → we see fake edge.
        if status == 'in':
            score_age = game.get('seconds_since_score_change', 0)
            
            if score_age > self.MAX_ESPN_SCORE_STALE_SEC:
                return None  # Score data too stale, possible adverse selection
        
        # Get the appropriate probability based on game state
        if status == 'in':
            if 'live_home_prob' not in game:
                return None
            our_home = game['live_home_prob']
        elif status == 'pre':
            if 'pregame_home_prob' not in game:
                return None
            our_home = game['pregame_home_prob']
        else:
            # Game finished or other status
            return None
        
        def evaluate_venue(odds_dict, venue_name):
            """
            Evaluate best opportunity for a single venue, comparing YES and NO contracts.
            
            To bet HOME wins: compare home_yes_ask vs away_no_ask
            To bet AWAY wins: compare away_yes_ask vs home_no_ask
            
            Returns (side, edge, entry_price, spread, ticker, venue, contract_type) or None
            """
            # Extract all prices
            home_yes_ask = odds_dict.get('home_yes_ask', 0)
            home_yes_bid = odds_dict.get('home_yes_bid', 0)
            home_no_ask = odds_dict.get('home_no_ask', 0)
            home_no_bid = odds_dict.get('home_no_bid', 0)
            away_yes_ask = odds_dict.get('away_yes_ask', 0)
            away_yes_bid = odds_dict.get('away_yes_bid', 0)
            away_no_ask = odds_dict.get('away_no_ask', 0)
            away_no_bid = odds_dict.get('away_no_bid', 0)
            home_ticker = odds_dict.get('home_ticker')
            away_ticker = odds_dict.get('away_ticker')
            
            best = None
            
            # Time-dependent spread limit: more lenient early, stricter late
            game_time = game.get('time_remaining_sec', 2400)
            if game_time >= 1200:  # > 20:00
                max_spread = 3  # Early game: allow 3¢ spreads
            else:
                max_spread = 2  # Late game: allow 2¢ spreads
            
            # ===== CHECK HOME SIDE (bet on home team winning) =====
            # Option A: Buy HOME YES @ home_yes_ask, exit at home_yes_bid
            # Option B: Buy AWAY NO @ away_no_ask, exit at away_no_bid
            home_candidates = []
            
            if home_yes_ask and home_yes_bid:
                home_candidates.append({
                    'contract_type': 'home_yes',
                    'ask': home_yes_ask,
                    'bid': home_yes_bid,
                    'spread': home_yes_ask - home_yes_bid,
                    'ticker': home_ticker
                })
            
            if away_no_ask and away_no_bid:
                home_candidates.append({
                    'contract_type': 'away_no',
                    'ask': away_no_ask,
                    'bid': away_no_bid,
                    'spread': away_no_ask - away_no_bid,
                    'ticker': away_ticker  # NO contract on away team
                })
            
            if home_candidates:
                # Pick the cheaper ask; if tie, pick smaller spread
                home_candidates.sort(key=lambda x: (x['ask'], x['spread']))
                best_home = home_candidates[0]
                
                entry_price = best_home['ask'] / 100.0
                spread = best_home['spread']
                home_edge = our_home - entry_price
                
                cooldown_key = (game_id, 'home')
                in_cooldown = False
                if cooldown_key in self.exit_cooldowns:
                    exit_game_time, cooldown_secs, _ = self.exit_cooldowns[cooldown_key]
                    current_game_time = game.get('time_remaining_sec', 2400)
                    in_cooldown = (exit_game_time - current_game_time) < cooldown_secs
                
                time_remaining = game.get('time_remaining_sec', 2400)
                required_edge = self.get_required_edge(entry_price, time_remaining)
                
                if (home_edge >= required_edge and 
                    spread <= max_spread and
                    entry_price >= self.MIN_ENTRY_PRICE and
                    entry_price <= self.MAX_ENTRY_PRICE and
                    (game.get('status') == 'pre' or time_remaining >= self.MIN_TIME_REMAINING) and
                    game.get('period', 1) <= 2 and
                    not in_cooldown):
                    best = ('home', home_edge, entry_price, spread, best_home['ticker'], venue_name, best_home['contract_type'])
            
            # ===== CHECK AWAY SIDE (bet on away team winning) =====
            # Option A: Buy AWAY YES @ away_yes_ask, exit at away_yes_bid
            # Option B: Buy HOME NO @ home_no_ask, exit at home_no_bid
            away_candidates = []
            
            if away_yes_ask and away_yes_bid:
                away_candidates.append({
                    'contract_type': 'away_yes',
                    'ask': away_yes_ask,
                    'bid': away_yes_bid,
                    'spread': away_yes_ask - away_yes_bid,
                    'ticker': away_ticker
                })
            
            if home_no_ask and home_no_bid:
                away_candidates.append({
                    'contract_type': 'home_no',
                    'ask': home_no_ask,
                    'bid': home_no_bid,
                    'spread': home_no_ask - home_no_bid,
                    'ticker': home_ticker  # NO contract on home team
                })
            
            if away_candidates:
                # Pick the cheaper ask; if tie, pick smaller spread
                away_candidates.sort(key=lambda x: (x['ask'], x['spread']))
                best_away = away_candidates[0]
                
                entry_price = best_away['ask'] / 100.0
                spread = best_away['spread']
                away_edge = (1 - our_home) - entry_price
                
                cooldown_key = (game_id, 'away')
                in_cooldown = False
                if cooldown_key in self.exit_cooldowns:
                    exit_game_time, cooldown_secs, _ = self.exit_cooldowns[cooldown_key]
                    current_game_time = game.get('time_remaining_sec', 2400)
                    in_cooldown = (exit_game_time - current_game_time) < cooldown_secs
                
                time_remaining = game.get('time_remaining_sec', 2400)
                required_edge = self.get_required_edge(entry_price, time_remaining)
                
                if (away_edge >= required_edge and 
                    spread <= max_spread and
                    entry_price >= self.MIN_ENTRY_PRICE and
                    entry_price <= self.MAX_ENTRY_PRICE and
                    (game.get('status') == 'pre' or time_remaining >= self.MIN_TIME_REMAINING) and
                    game.get('period', 1) <= 2 and
                    not in_cooldown):
                    # Compare with existing best (home side)
                    if best is None or away_edge > best[1]:
                        best = ('away', away_edge, entry_price, spread, best_away['ticker'], venue_name, best_away['contract_type'])
            
            return best
        
        # Determine the primary venue (always Kalshi now)
        primary_venue = 'Kalshi'
        
        # Evaluate Kalshi - pass full odds dict
        best = evaluate_venue(kalshi_odds, primary_venue)
        
        if not best:
            # No opportunity - clear any pending confirmation for this game
            for side in ['home', 'away']:
                self.edge_confirmations.pop((game_id, side), None)
            return None
        
        best_side, best_edge, best_entry_price, best_spread, market_ticker, market_source, contract_type = best
        our_prob = our_home if best_side == 'home' else 1 - our_home
        
        # === TWO-TICK CONFIRMATION ===
        # Only needed when using REST (stale data). WebSocket provides real-time prices.
        if status == 'in' and not self.use_ws:
            # For REST: require edge to persist for 1 second to filter phantom edges
            MIN_CONFIRMATION_SECONDS = 1.0
            BIG_EDGE_MARGIN = 0.06  # +6% extra edge bypasses confirmation
            now = time.time()
            confirmation_key = (game_id, best_side)
            
            time_remaining = game.get('time_remaining_sec', 2400)
            required_edge = self.get_required_edge(best_entry_price, time_remaining)
            
            # Fast-path: huge edges skip confirmation
            if best_edge >= required_edge + BIG_EDGE_MARGIN:
                self.edge_confirmations.pop(confirmation_key, None)
                self.edge_confirmations.pop((game_id, 'away' if best_side == 'home' else 'home'), None)
            elif confirmation_key in self.edge_confirmations:
                prev_edge, prev_ts = self.edge_confirmations[confirmation_key]
                time_since_first = now - prev_ts
                
                if time_since_first >= MIN_CONFIRMATION_SECONDS:
                    self.edge_confirmations.pop(confirmation_key, None)
                    self.edge_confirmations.pop((game_id, 'away' if best_side == 'home' else 'home'), None)
                else:
                    self.edge_confirmations[confirmation_key] = (best_edge, prev_ts)
                    return None
            else:
                self.edge_confirmations[confirmation_key] = (best_edge, now)
                self.edge_confirmations.pop((game_id, 'away' if best_side == 'home' else 'home'), None)
                return None
        
        return {
            'game_id': game_id,
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'home_score': game.get('home_score', 0),
            'away_score': game.get('away_score', 0),
            'time_remaining_sec': game.get('time_remaining_sec', 2400),
            'period': game.get('period', 0),
            'clock': game.get('clock', 'PRE'),
            'pregame_home_prob': game.get('pregame_home_prob'),
            'live_home_prob': game.get('live_home_prob', our_home),
            'market_home_prob': kalshi_odds.get('home_prob'),
            'entry_price': best_entry_price,
            'spread': best_spread,
            'market_ticker': market_ticker,
            'edge': best_edge,
            'side': best_side,
            'our_prob': our_prob,
            'status': status,
            'market_source': market_source,  # Track which venue we're entering on
            'contract_type': contract_type   # Track which contract: home_yes, away_no, away_yes, home_no
        }
    
    def log_opportunity(self, opp: dict, trade_type: str = 'ENTRY'):
        """Log a trade entry"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO paper_trades
            (timestamp, game_id, home_team, away_team, home_score, away_score,
             time_remaining_sec, game_status, pregame_home_prob, live_home_prob,
             market_home_prob, market_source, trade_type, side,
             entry_price, entry_spread, entry_edge, market_ticker, contract_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            opp['game_id'],
            opp['home_team'],
            opp['away_team'],
            opp.get('home_score', 0),
            opp.get('away_score', 0),
            opp.get('time_remaining_sec', 2400),
            opp.get('status', 'in'),
            opp.get('pregame_home_prob'),
            opp.get('live_home_prob'),
            opp.get('market_home_prob'),
            opp.get('market_source', 'Kalshi'),  # Use venue from opportunity
            trade_type,
            opp['side'],
            opp.get('entry_price'),
            opp.get('spread'),
            opp.get('edge'),
            opp.get('market_ticker'),
            opp.get('contract_type', f"{opp['side']}_yes")  # Store contract type for persistence
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def check_exit(self, game: dict, kalshi_odds: dict, position: dict) -> Optional[dict]:
        """Check if we should exit an open position using EV-based logic.
        
        Exit triggers:
        1. SETTLEMENT: Game is over (status='post') - settle at $0 or $1
        2. EV-BASED: Exit when EV_exit > EV_hold
           - EV_hold = model_prob × 100 (what we expect at settlement)
           - EV_exit = current_bid - exit_fee - slippage (what we get if we sell now)
        
        This is symmetric and rational:
        - If market is overpaying relative to our model → EXIT (take the gift)
        - If our model still likes the position → HOLD (let it ride)
        - Works for both winners AND losers
        
        IMPORTANT: Must use bid from the SAME contract type we entered on.
        """
        side = position['side']
        entry_price = position['entry_price']
        game_status = game.get('status', 'in')
        market_source = position.get('market_source', 'Kalshi')
        contract_type = position.get('contract_type', f'{side}_yes')
        
        # Get scores and time
        home_score = game.get('home_score', 0)
        away_score = game.get('away_score', 0)
        period = game.get('period', 0)
        time_remaining = game.get('time_remaining_sec', 2400)
        
        # 1. SETTLEMENT: Game is over - settle at $0 or $1 (no fees!)
        if game_status == 'post' and home_score > 0:
            home_won = home_score > away_score
            we_won = (side == 'home' and home_won) or (side == 'away' and not home_won)
            exit_price = 1.0 if we_won else 0.0
            exit_reason = f"SETTLEMENT: {'WON' if we_won else 'LOST'} ({home_score}-{away_score})"
            
            return {
                'game_id': game['game_id'],
                'home_team': game.get('home_team', 'Unknown'),
                'away_team': game.get('away_team', 'Unknown'),
                'side': side,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': exit_price - entry_price,
                'exit_reason': exit_reason,
                'home_score': home_score,
                'away_score': away_score,
                'time_remaining_sec': 0,
                'period': period,
                'live_home_prob': game.get('live_home_prob'),
                'market_home_prob': None,
                'market_source': market_source,
                'contract_type': contract_type,
                'market_ticker': position.get('market_ticker'),
                'trade_id': position.get('trade_id')
            }
        
        # For non-settlement exits, we need odds and live probability
        if kalshi_odds is None:
            return None
        
        if game_status != 'in' or 'live_home_prob' not in game:
            return None  # Only EV-exit during live games with model prob
        
        # === ESPN SCORE FRESHNESS GUARD (exits) ===
        # Don't EV-exit on stale data — our model prob might be wrong, causing premature exits.
        score_age = game.get('seconds_since_score_change', 0)
        
        if score_age > self.MAX_ESPN_SCORE_STALE_SEC:
            return None  # Wait for fresh ESPN data before deciding to exit
        
        # Get current BID based on contract_type
        bid_map = {
            'home_yes': 'home_yes_bid',
            'home_no': 'home_no_bid',
            'away_yes': 'away_yes_bid',
            'away_no': 'away_no_bid'
        }
        
        bid_key = bid_map.get(contract_type, f'{side}_yes_bid')
        current_bid_cents = kalshi_odds.get(bid_key, 0)
        
        if not current_bid_cents or current_bid_cents <= 0:
            return None
        
        # Calculate our model's probability for our side
        our_prob = game['live_home_prob'] if side == 'home' else 1 - game['live_home_prob']
        
        # 2. EV-BASED EXIT
        # EV_hold = model_prob × 100 + option_value
        # Option value from closed-form formula (accounts for free settlement + market noise)
        time_remaining = game.get('time_remaining_sec', 0) or 0
        
        ov = self.compute_option_value(time_remaining, our_prob)
        ev_hold = our_prob * 100 + ov
        
        # EV_exit = current_bid - exit_fee - slippage
        bid_decimal = current_bid_cents / 100.0
        exit_fee = kalshi_fee_cents(bid_decimal)
        ev_exit = current_bid_cents - exit_fee - self.EV_EXIT_SLIPPAGE_CENTS
        
        exit_reason = None
        
        if ev_exit > ev_hold:
            ev_diff = ev_exit - ev_hold
            exit_reason = f"EV EXIT: Sell@{current_bid_cents:.0f}¢ (EV={ev_exit:.1f}) > Hold (EV={ev_hold:.1f}) | +{ev_diff:.1f}¢ EV"
        
        if exit_reason:
            return {
                'game_id': game['game_id'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'home_score': home_score,
                'away_score': away_score,
                'time_remaining_sec': time_remaining,
                'side': side,
                'entry_price': entry_price,
                'exit_price': bid_decimal,
                'pnl': bid_decimal - entry_price,
                'exit_reason': exit_reason,
                'live_home_prob': game.get('live_home_prob'),
                'market_home_prob': kalshi_odds.get('home_prob'),
                'market_source': market_source,
                'contract_type': contract_type
            }
        
        return None
    
    def _handle_exit(self, exit_info: dict):
        """Handle exiting a position"""
        game_id = exit_info['game_id']
        side = exit_info['side']
        position = self.open_positions.get(game_id)
        if not position:
            return
        
        # === LIVE TRADING: Sell position FIRST ===
        is_settlement = 'SETTLEMENT' in exit_info['exit_reason'] or exit_info['exit_price'] in [0.0, 1.0]
        live_fill_count = position.get('live_fill_count', 0)
        
        # If live_fill_count not tracked (e.g., after restart), get from Kalshi
        ticker = position.get('market_ticker')
        if live_fill_count == 0 and self.live_mode and self.kalshi_client and ticker:
            try:
                kalshi_positions = self.kalshi_client.get_positions()
                if ticker in kalshi_positions:
                    live_fill_count = abs(kalshi_positions[ticker].yes_count)
            except Exception as e:
                print(f"  ⚠️ Could not get Kalshi position count: {e}")
        
        if self.live_mode and self.kalshi_client and live_fill_count > 0 and not is_settlement:
            # Only sell if we have a live position and it's not a settlement (settlements auto-settle)
            ticker = position.get('market_ticker')
            if ticker:
                contract_type = position.get('contract_type', f'{side}_yes')
                
                # === SAFETY CHECK: Verify actual position side from Kalshi ===
                # This prevents selling the WRONG side which would ADD exposure!
                try:
                    kalshi_positions = self.kalshi_client.get_positions()
                    if ticker in kalshi_positions:
                        actual_yes_count = kalshi_positions[ticker].yes_count
                        if actual_yes_count == 0:
                            print(f"  ⚠️ NO POSITION on Kalshi for {ticker} - checking sibling ticker...")
                            # Check sibling ticker before removing
                            ticker_stem = ticker.rsplit('-', 1)[0]
                            found_sibling = False
                            for other_ticker, other_pos in kalshi_positions.items():
                                if other_ticker != ticker and other_pos.yes_count != 0 and other_ticker.startswith(ticker_stem):
                                    remaining = abs(other_pos.yes_count)
                                    other_is_no = other_pos.yes_count < 0
                                    if 'home' in contract_type:
                                        new_ct = 'away_no' if other_is_no else 'away_yes'
                                    else:
                                        new_ct = 'home_no' if other_is_no else 'home_yes'
                                    print(f"  ⚠️ REMAINING: {remaining} contracts on {other_ticker} ({new_ct})")
                                    position['market_ticker'] = other_ticker
                                    position['contract_type'] = new_ct
                                    position['live_fill_count'] = remaining
                                    found_sibling = True
                                    break
                            if not found_sibling:
                                self.open_positions.pop(game_id, None)
                                self.just_exited.add(game_id)
                                cooldown_key = (game_id, side)
                                self.exit_cooldowns[cooldown_key] = (exit_info.get('time_remaining_sec', 0), self.COOLDOWN_AFTER_EXIT, "Position already closed on Kalshi")
                            return
                        
                        # Determine ACTUAL side from Kalshi (positive = YES, negative = NO)
                        actual_is_no = actual_yes_count < 0
                        tracked_is_no = '_no' in contract_type
                        
                        if actual_is_no != tracked_is_no:
                            print(f"  ⚠️ SIDE MISMATCH! Tracked as {'NO' if tracked_is_no else 'YES'} but Kalshi shows {'NO' if actual_is_no else 'YES'}")
                            print(f"     Correcting contract_type and retrying...")
                            # Fix the contract_type
                            if actual_is_no:
                                contract_type = 'home_no' if 'home' in contract_type else 'away_no'
                            else:
                                contract_type = 'home_yes' if 'home' in contract_type else 'away_yes'
                            # Update tracked position
                            position['contract_type'] = contract_type
                        
                        # Also use actual count from Kalshi
                        live_fill_count = abs(actual_yes_count)
                    else:
                        print(f"  ⚠️ Ticker {ticker} not in Kalshi positions - checking sibling ticker...")
                        # Check sibling ticker before removing
                        ticker_stem = ticker.rsplit('-', 1)[0]
                        found_sibling = False
                        for other_ticker, other_pos in kalshi_positions.items():
                            if other_ticker != ticker and other_pos.yes_count != 0 and other_ticker.startswith(ticker_stem):
                                remaining = abs(other_pos.yes_count)
                                other_is_no = other_pos.yes_count < 0
                                if 'home' in contract_type:
                                    new_ct = 'away_no' if other_is_no else 'away_yes'
                                else:
                                    new_ct = 'home_no' if other_is_no else 'home_yes'
                                print(f"  ⚠️ REMAINING: {remaining} contracts on {other_ticker} ({new_ct})")
                                position['market_ticker'] = other_ticker
                                position['contract_type'] = new_ct
                                position['live_fill_count'] = remaining
                                found_sibling = True
                                break
                        if not found_sibling:
                            self.open_positions.pop(game_id, None)
                            self.just_exited.add(game_id)
                            cooldown_key = (game_id, side)
                            self.exit_cooldowns[cooldown_key] = (exit_info.get('time_remaining_sec', 0), self.COOLDOWN_AFTER_EXIT, "Ticker not in Kalshi positions")
                        return
                except Exception as e:
                    print(f"  ⚠️ Could not verify position side: {e}")
                    # Continue with tracked side - risky but order might still work
                
                # For IOC exits, accept slightly worse price to guarantee fill
                # Bid can move between when we read it and when order arrives
                EXIT_PRICE_BUFFER_CENTS = 3  # Accept up to 3¢ less than current bid
                
                kalshi_side = OrderSide.NO if '_no' in contract_type else OrderSide.YES
                kalshi_action = OrderAction.SELL
                raw_price_cents = round(exit_info['exit_price'] * 100)  # Use round() to avoid floating point truncation
                price_cents = max(1, raw_price_cents - EXIT_PRICE_BUFFER_CENTS)  # Floor at 1¢
                
                print(f"\n  🔴 LIVE SELL: {ticker} {kalshi_side.value.upper()} @ {price_cents}¢ (bid was {raw_price_cents}¢) x {live_fill_count}")
                
                try:
                    result = self.kalshi_client.place_order(
                        ticker=ticker,
                        side=kalshi_side,
                        action=kalshi_action,
                        count=live_fill_count,
                        price_cents=price_cents,
                        time_in_force=TimeInForce.IOC
                    )
                    
                    if result.success and result.fill_count > 0:
                        actual_exit_price = result.avg_price or price_cents  # Fall back to requested price
                        
                        # Kalshi returns NO contract fills as YES-equivalent price - convert back
                        if '_no' in contract_type:
                            actual_exit_price = 100 - actual_exit_price
                        
                        exit_info['exit_price'] = actual_exit_price / 100.0  # Use actual fill price
                        print(f"  ✓ SOLD {result.fill_count} @ {actual_exit_price:.0f}¢ (fees: ${result.taker_fees:.4f})")
                        
                        # VERIFY: Query Kalshi to confirm position is actually gone
                        try:
                            import time
                            time.sleep(0.3)  # Brief delay for Kalshi to update
                            verify_positions = self.kalshi_client.get_positions()
                            if ticker in verify_positions and verify_positions[ticker].yes_count != 0:
                                remaining = abs(verify_positions[ticker].yes_count)
                                print(f"  ⚠️ PARTIAL FILL: Sold {result.fill_count}, {remaining} remaining on Kalshi")
                                print(f"     Will retry on next cycle")
                                position['live_fill_count'] = remaining  # Update tracking to actual remaining
                                return  # Stay open, retry next cycle
                            else:
                                print(f"  ✓ VERIFIED: Position closed on Kalshi")
                                
                                # Check for remaining contracts on OTHER ticker for this game
                                # (e.g. had 14 JMU NO + 30 Toledo YES, just sold 14 JMU NO)
                                ticker_stem = ticker.rsplit('-', 1)[0]
                                for other_ticker, other_pos in verify_positions.items():
                                    if other_ticker != ticker and other_pos.yes_count != 0 and other_ticker.startswith(ticker_stem):
                                        remaining = abs(other_pos.yes_count)
                                        other_is_no = other_pos.yes_count < 0
                                        # Current ticker is home or away? Other is the opposite
                                        if 'home' in contract_type:
                                            new_ct = 'away_no' if other_is_no else 'away_yes'
                                        else:
                                            new_ct = 'home_no' if other_is_no else 'home_yes'
                                        print(f"  ⚠️ REMAINING: {remaining} contracts on {other_ticker} ({new_ct})")
                                        print(f"     Switching position to track remaining contracts")
                                        position['market_ticker'] = other_ticker
                                        position['contract_type'] = new_ct
                                        position['live_fill_count'] = remaining
                                        return  # Keep position open for remaining contracts
                        except Exception as ve:
                            print(f"  ⚠️ Could not verify position closure: {ve}")
                            # Continue anyway - order said it filled
                    else:
                        print(f"  ⚠️ SELL NOT FILLED: {result.error or 'No fills'} - keeping position tracked")
                        return  # Don't remove position if sell failed
                except Exception as e:
                    print(f"  ⚠️ SELL ERROR: {e} - keeping position tracked")
                    return  # Don't remove position if sell errored
        
        # Only remove position after successful exit (or settlement)
        self.open_positions.pop(game_id, None)
        
        # Prevent re-entry on same tick
        self.just_exited.add(game_id)
        
        # Set cooldown based on exit reason - keyed by (game_id, side)
        is_stop_loss = 'STOP LOSS' in exit_info['exit_reason']
        cooldown_secs = self.COOLDOWN_AFTER_STOP if is_stop_loss else self.COOLDOWN_AFTER_EXIT
        exit_game_time = exit_info['time_remaining_sec']
        cooldown_key = (game_id, side)
        self.exit_cooldowns[cooldown_key] = (exit_game_time, cooldown_secs, exit_info['exit_reason'])
        
        # Calculate P/L with fees
        entry_price = position['entry_price']
        exit_price = exit_info['exit_price']
        venue = exit_info.get('market_source', position.get('market_source', 'Kalshi'))
        
        gross_pnl_cents = (exit_price - entry_price) * 100
        
        # Calculate Kalshi fees (settlement = no exit fee)
        entry_fee = kalshi_fee_cents(entry_price)
        exit_fee = 0 if is_settlement else kalshi_fee_cents(exit_price)
        
        total_fees = entry_fee + exit_fee
        net_pnl_cents = gross_pnl_cents - total_fees
        
        # For live trades, multiply P/L by contract count
        if live_fill_count > 0:
            net_pnl_cents *= live_fill_count
            gross_pnl_cents *= live_fill_count
            total_fees *= live_fill_count
        
        entry_trade_id = position.get('trade_id')
        entry_price_cents = entry_price * 100
        
        # Track running totals (use NET P/L)
        self.realized_pnl += net_pnl_cents
        self.trade_count += 1
        self.current_capital_deployed -= entry_price_cents  # Remove closed position's capital
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO paper_trades
            (timestamp, game_id, home_team, away_team, home_score, away_score,
             time_remaining_sec, live_home_prob, market_home_prob, market_source,
             trade_type, side, entry_price, exit_price, exit_reason, pnl, entry_trade_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            game_id,
            exit_info['home_team'],
            exit_info['away_team'],
            exit_info['home_score'],
            exit_info['away_score'],
            exit_info['time_remaining_sec'],
            exit_info['live_home_prob'],
            exit_info['market_home_prob'],
            venue,
            'EXIT',
            side,
            entry_price,  # Original entry price
            exit_price,   # Exit bid
            exit_info['exit_reason'],
            net_pnl_cents / 100,  # Store NET P/L in dollars
            entry_trade_id
        ))
        self.conn.commit()
        
        team = exit_info['home_team'] if side == 'home' else exit_info['away_team']
        emoji = "💰" if net_pnl_cents > 0 else "🔻"
        live_str = f" 🔴x{live_fill_count}" if live_fill_count > 0 else ""
        print(f"\n  {emoji} EXIT: {side.upper()} ({team[:20]}) @ {venue}{live_str}")
        print(f"     Entry: {entry_price*100:.0f}¢ → Exit: {exit_price*100:.0f}¢")
        print(f"     Gross: {gross_pnl_cents:+.1f}¢ | Fees: {total_fees:.1f}¢ | Net: {net_pnl_cents:+.1f}¢")
        print(f"     Reason: {exit_info['exit_reason']}")
        print(f"     Cooldown: {cooldown_secs}s game time on {side} side\n")
    
    def run(self, poll_interval: int = 1):
        """Main trading loop"""
        print("\n" + "═"*80)
        print("  📈 KALSHI TRADER │ Live Market Trading")
        print("═"*80)
        print(f"  Entry: 5%→15% edge (exp k=2) │ Spread: 3¢/2¢")
        print(f"  Exit: EV-based (OV={self.OV_SCALE}×N^{self.OV_EXPONENT}) │ Cooldown: {self.COOLDOWN_AFTER_EXIT}s game-time")
        ws_str = "WS" if KALSHI_WS_AVAILABLE else "REST"
        print(f"  Data: {ws_str}")
        if self.live_mode:
            print(f"  Contracts: {self.live_contracts}")
        print("═"*80 + "\n")
        
        try:
            while True:
                cycle_start = time.time()
                
                # Clear just_exited from previous tick
                self.just_exited.clear()
                
                # Fetch all data in PARALLEL for lower latency
                kalshi_portfolio = None
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = {
                        executor.submit(self._load_kalshi_odds): 'kalshi',
                        executor.submit(self.scan_games): 'espn',
                    }
                    # Add portfolio fetch for live mode
                    if self.live_mode:
                        futures[executor.submit(self._get_kalshi_portfolio)] = 'portfolio'
                    
                    games = []
                    for future in as_completed(futures):
                        name = futures[future]
                        try:
                            result = future.result()
                            if name == 'espn':
                                games = result
                            elif name == 'portfolio':
                                kalshi_portfolio = result
                        except Exception as e:
                            print(f"  ⚠️ {name} fetch error: {e}")
                
                live_games = [g for g in games if g['status'] == 'in' and 'live_home_prob' in g]
                upcoming_games = [g for g in games if g['status'] == 'pre' and 'pregame_home_prob' in g]
                finished_games = [g for g in games if g['status'] == 'post']
                
                # Check for settlements on finished games
                for game in finished_games:
                    game_id = game['game_id']
                    odds = self._get_best_market_odds(game['home_team'], game['away_team'])
                    pos_key = self._find_position_for_game(game_id, odds)
                    if pos_key:
                        # Game finished - settle the position
                        pos = self.open_positions[pos_key]
                        exit_info = self.check_exit(game, odds, pos)
                        if exit_info:
                            self._handle_exit(exit_info)
                
                # Clear screen
                print("\033[2J\033[H", end="")
                
                # Clean up cooldowns for finished games
                active_game_ids = {g['game_id'] for g in live_games + upcoming_games}
                expired = [key for key in self.exit_cooldowns.keys() if key[0] not in active_game_ids]
                for key in expired:
                    del self.exit_cooldowns[key]
                
                # Helper to get bid for a contract type (Kalshi only)
                def get_contract_bid(odds, contract_type, market_source):
                    """Get the bid price for a specific Kalshi contract type"""
                    bid_map = {
                        'home_yes': 'home_yes_bid',
                        'home_no': 'home_no_bid',
                        'away_yes': 'away_yes_bid',
                        'away_no': 'away_no_bid'
                    }
                    bid_key = bid_map.get(contract_type, 'home_yes_bid')
                    return odds.get(bid_key, 0) / 100.0 if odds.get(bid_key) else 0
                
                # Calculate unrealized P/L from open positions
                unrealized_pnl = 0.0
                for game_id, pos in self.open_positions.items():
                    current_game = next((g for g in live_games if g['game_id'] == game_id), None)
                    if not current_game:
                        current_game = next((g for g in upcoming_games if g['game_id'] == game_id), None)
                    
                    # If game_id is kalshi:TICKER, try to find by ticker
                    if not current_game and game_id.startswith('kalshi:'):
                        ticker = pos.get('market_ticker')
                        if ticker:
                            for team_key, odds in self.kalshi_odds.items():
                                if odds.get('home_ticker') == ticker or odds.get('away_ticker') == ticker:
                                    kalshi_home = odds.get('kalshi_home_norm', '')
                                    kalshi_away = odds.get('kalshi_away_norm', '')
                                    for g in live_games + upcoming_games:
                                        espn_home_norm = normalize_team_name(g['home_team'])
                                        espn_away_norm = normalize_team_name(g['away_team'])
                                        if (espn_home_norm == kalshi_home or kalshi_home in espn_home_norm) and \
                                           (espn_away_norm == kalshi_away or kalshi_away in espn_away_norm):
                                            current_game = g
                                            break
                                    break
                    
                    if current_game:
                        kalshi = self._get_best_market_odds(current_game['home_team'], current_game['away_team'])
                        if kalshi:
                            market_source = pos.get('market_source', 'Kalshi')
                            contract_type = pos.get('contract_type', f"{pos['side']}_yes")
                            current_bid = get_contract_bid(kalshi, contract_type, market_source)
                            unrealized_pnl += (current_bid - pos['entry_price']) * 100
                
                # kalshi_portfolio already fetched in parallel above
                
                print(f"\n{'═'*120}")
                # Real ROI = profit / peak capital deployed
                roi_pct = (self.realized_pnl / self.peak_capital_deployed * 100) if self.peak_capital_deployed > 0 else 0
                kalshi_count = len(self.kalshi_odds) // 2
                
                # Compact header with ROI
                time_str = datetime.now().strftime('%H:%M:%S')
                
                if self.live_mode:
                    # Show actual Kalshi portfolio data
                    ws_str = " │ WS" if self.use_ws else ""
                    if kalshi_portfolio:
                        session_pnl = kalshi_portfolio['session_pnl']
                        pnl_str = f"+${session_pnl:.2f}" if session_pnl >= 0 else f"-${abs(session_pnl):.2f}"
                        print(f"  🔴 LIVE │ {time_str} │ Portfolio: ${kalshi_portfolio['total']:.2f} │ Cash: ${kalshi_portfolio['cash']:.2f} │ Positions: ${kalshi_portfolio['position_value']:.2f} ({kalshi_portfolio['position_count']}) │ Session: {pnl_str}{ws_str}")
                    else:
                        print(f"  🔴 LIVE │ {time_str} │ K:{kalshi_count} │ (portfolio fetch failed){ws_str}")
                else:
                    roi_str = f" ({roi_pct:+.1f}% ROI)" if self.trade_count > 0 else ""
                    print(f"  PAPER TRADER │ {time_str} │ K:{kalshi_count} │ {len(self.open_positions)} pos │ Net: {self.realized_pnl:+.0f}¢{roi_str} ({self.trade_count} trades) │ GrossUnreal: {unrealized_pnl:+.0f}¢")
                print(f"{'═'*120}")
                
                # Show open positions first
                live_positions = []
                pregame_positions = []
                unmatched_positions = []  # Positions we can't find games for
                if self.open_positions:
                    
                    for game_id, pos in list(self.open_positions.items()):
                        current_game = next((g for g in live_games if g['game_id'] == game_id), None)
                        if not current_game:
                            current_game = next((g for g in upcoming_games if g['game_id'] == game_id), None)
                        
                        # If game_id is kalshi:TICKER, try to find by ticker match
                        if not current_game and game_id.startswith('kalshi:'):
                            ticker = pos.get('market_ticker')
                            if ticker:
                                # Try to find game via kalshi_odds ticker
                                for team_key, odds in self.kalshi_odds.items():
                                    if odds.get('home_ticker') == ticker or odds.get('away_ticker') == ticker:
                                        # Found the odds, now find matching ESPN game
                                        kalshi_home = odds.get('kalshi_home_norm', '')
                                        kalshi_away = odds.get('kalshi_away_norm', '')
                                        for g in live_games + upcoming_games:
                                            espn_home_norm = normalize_team_name(g['home_team'])
                                            espn_away_norm = normalize_team_name(g['away_team'])
                                            if (espn_home_norm == kalshi_home or kalshi_home in espn_home_norm) and \
                                               (espn_away_norm == kalshi_away or kalshi_away in espn_away_norm):
                                                current_game = g
                                                # Migrate the key
                                                print(f"  ℹ️ Matched position {game_id} -> {g['game_id']}")
                                                self.open_positions[g['game_id']] = self.open_positions.pop(game_id)
                                                game_id = g['game_id']
                                                break
                                        break
                        
                        if current_game:
                            odds = self._get_best_market_odds(current_game['home_team'], current_game['away_team'])
                            if odds:
                                market_source = pos.get('market_source', 'Kalshi')
                                contract_type = pos.get('contract_type', f"{pos['side']}_yes")
                                current_bid = get_contract_bid(odds, contract_type, market_source)
                                if current_bid == 0:
                                    continue
                                
                                pnl_cents = (current_bid - pos['entry_price']) * 100
                                game_status = current_game.get('status', 'pre')
                                
                                pos_info = {
                                    'game_id': game_id,
                                    'pos': pos,
                                    'game': current_game,
                                    'odds': odds,
                                    'current_bid': current_bid,
                                    'pnl_cents': pnl_cents,
                                    'market_source': market_source,
                                    'contract_type': contract_type
                                }
                                
                                if game_status == 'pre':
                                    pregame_positions.append(pos_info)
                                else:
                                    live_positions.append(pos_info)
                        else:
                            # Position with no matching ESPN game
                            unmatched_positions.append((game_id, pos))
                    
                # === UPCOMING SUMMARY (at top) ===
                upcoming_with_edge = []
                for game in upcoming_games:
                    odds = self._get_best_market_odds(game['home_team'], game['away_team'])
                    best_edge = None
                    best_side = None
                    
                    if odds:
                        our_home = game['pregame_home_prob']
                        
                        k_home_yes = odds.get('home_yes_ask')
                        k_away_no = odds.get('away_no_ask')
                        k_home_ask = min(filter(None, [k_home_yes, k_away_no]), default=None)
                        
                        k_away_yes = odds.get('away_yes_ask')
                        k_home_no = odds.get('home_no_ask')
                        k_away_ask = min(filter(None, [k_away_yes, k_home_no]), default=None)
                        
                        home_ask = k_home_ask
                        away_ask = k_away_ask
                        
                        if home_ask:
                            home_edge = our_home - (home_ask / 100.0)
                            if home_edge > 0 and (best_edge is None or home_edge > best_edge):
                                best_edge = home_edge
                                best_side = 'home'
                        if away_ask:
                            away_edge = (1 - our_home) - (away_ask / 100.0)
                            if away_edge > 0 and (best_edge is None or away_edge > best_edge):
                                best_edge = away_edge
                                best_side = 'away'
                        
                        if best_edge is None:
                            if home_ask and away_ask:
                                home_e = our_home - (home_ask / 100.0)
                                away_e = (1 - our_home) - (away_ask / 100.0)
                                if home_e > away_e:
                                    best_edge = home_e
                                    best_side = 'home'
                                else:
                                    best_edge = away_e
                                    best_side = 'away'
                    
                    upcoming_with_edge.append((game, odds, best_edge, best_side))
                
                upcoming_with_edge.sort(key=lambda x: x[2] if x[2] else -999, reverse=True)
                
                # Show upcoming summary line
                positive_edges = [e for _, _, e, _ in upcoming_with_edge if e and e > 0]
                if positive_edges:
                    avg_edge = sum(positive_edges) / len(positive_edges) * 100
                    print(f"\n  🎯 UPCOMING: {len(upcoming_games)} games | avg: {avg_edge:+.1f}%")
                else:
                    print(f"\n  🎯 UPCOMING: {len(upcoming_games)} games | no positive edges")
                    
                if not live_games:
                    print("\n  No live games with predictions right now.\n")
                else:
                    print(f"\n  🏀 LIVE ({len(live_games)} games)")
                    print(f"  {'MATCHUP':<40} {'SCORE':^8} {'TIME':>8} {'MODEL':^7} {'──KALSHI──':^11} {'EDGE':>7}")
                    print(f"  {'─'*78}")
                    
                    games_with_edge = []
                    for game in live_games:
                        odds = self._get_best_market_odds(game['home_team'], game['away_team'])
                        best_edge = None
                        best_side = None
                        
                        if odds:
                            our_home = game['live_home_prob']
                            
                            # Get best ask for HOME (min of home_yes_ask, away_no_ask)
                            k_home_yes = odds.get('home_yes_ask')
                            k_away_no = odds.get('away_no_ask')
                            k_home_ask = min(filter(None, [k_home_yes, k_away_no]), default=None)
                            
                            # Get best ask for AWAY (min of away_yes_ask, home_no_ask)
                            k_away_yes = odds.get('away_yes_ask')
                            k_home_no = odds.get('home_no_ask')
                            k_away_ask = min(filter(None, [k_away_yes, k_home_no]), default=None)
                            
                            # Use Kalshi prices only
                            home_ask = k_home_ask
                            away_ask = k_away_ask
                            
                            if home_ask:
                                home_edge = our_home - (home_ask / 100.0)
                                if home_edge > 0 and (best_edge is None or home_edge > best_edge):
                                    best_edge = home_edge
                                    best_side = 'home'
                            if away_ask:
                                away_edge = (1 - our_home) - (away_ask / 100.0)
                                if away_edge > 0 and (best_edge is None or away_edge > best_edge):
                                    best_edge = away_edge
                                    best_side = 'away'
                            
                            # If no positive edge, show the better one
                            if best_edge is None:
                                if home_ask and away_ask:
                                    home_e = our_home - (home_ask / 100.0)
                                    away_e = (1 - our_home) - (away_ask / 100.0)
                                    if home_e > away_e:
                                        best_edge = home_e
                                        best_side = 'home'
                                    else:
                                        best_edge = away_e
                                        best_side = 'away'
                        
                        games_with_edge.append((game, odds, best_edge, best_side))
                    
                    # Sort by edge
                    games_with_edge.sort(key=lambda x: x[2] if x[2] else -999, reverse=True)
                    
                    for game, odds, edge, edge_side in games_with_edge:
                        self.log_snapshot(game, odds, commit=False)
                        
                        # Format matchup
                        away = strip_mascot(game['away_team'])[:18]
                        home = strip_mascot(game['home_team'])[:18]
                        matchup = f"{away:<18} @ {home}"
                        
                        score = f"{game['away_score']}-{game['home_score']}"
                        time_str = f"P{game['period']} {game['clock']:>5}"
                        
                        # Model probability for edge side
                        home_prob = game['live_home_prob']
                        if edge_side == 'home':
                            model_pct = home_prob * 100
                            side_label = "H"
                        else:
                            model_pct = (1 - home_prob) * 100
                            side_label = "A"
                        model_str = f"{model_pct:3.0f}%{side_label}"
                        
                        game_id = game['game_id']
                        
                        if odds:
                            # Get prices for the relevant side (best of YES vs NO)
                            if edge_side == 'home':
                                # Betting home: compare home_yes_ask vs away_no_ask
                                k_yes_ask = odds.get('home_yes_ask')
                                k_no_ask = odds.get('away_no_ask')
                                k_yes_bid = odds.get('home_yes_bid')
                                k_no_bid = odds.get('away_no_bid')
                                
                                # Pick better Kalshi price
                                if k_yes_ask and k_no_ask:
                                    if k_no_ask < k_yes_ask:
                                        k_ask, k_bid = k_no_ask, k_no_bid
                                    else:
                                        k_ask, k_bid = k_yes_ask, k_yes_bid
                                elif k_yes_ask:
                                    k_ask, k_bid = k_yes_ask, k_yes_bid
                                elif k_no_ask:
                                    k_ask, k_bid = k_no_ask, k_no_bid
                                else:
                                    k_ask, k_bid = None, None
                            else:
                                # Betting away: compare away_yes_ask vs home_no_ask
                                k_yes_ask = odds.get('away_yes_ask')
                                k_no_ask = odds.get('home_no_ask')
                                k_yes_bid = odds.get('away_yes_bid')
                                k_no_bid = odds.get('home_no_bid')
                                
                                # Pick better Kalshi price
                                if k_yes_ask and k_no_ask:
                                    if k_no_ask < k_yes_ask:
                                        k_ask, k_bid = k_no_ask, k_no_bid
                                    else:
                                        k_ask, k_bid = k_yes_ask, k_yes_bid
                                elif k_yes_ask:
                                    k_ask, k_bid = k_yes_ask, k_yes_bid
                                elif k_no_ask:
                                    k_ask, k_bid = k_no_ask, k_no_bid
                                else:
                                    k_ask, k_bid = None, None
                            
                            # Format Kalshi column
                            if k_ask and k_bid:
                                k_str = f"{k_ask:2.0f}¢({k_ask-k_bid:.0f})"
                            else:
                                k_str = "  --"
                            
                            # Edge calculation
                            if edge is not None:
                                edge_str = f"{edge*100:+5.1f}%"
                            else:
                                edge_str = "   --"
                        else:
                            k_str = "  --"
                            edge_str = "   --"
                        
                        print(f"  {matchup:<40} {score:^8} {time_str:>8} {model_str:^7} {k_str:^11} {edge_str:>7}")
                        
                        # Check for exit on open positions (handles kalshi: prefix keys too)
                        pos_key = self._find_position_for_game(game_id, odds)
                        if pos_key:
                            exit_info = self.check_exit(game, odds, self.open_positions[pos_key])
                            if exit_info:
                                self._handle_exit(exit_info)
                        
                        # Check for new entry or top-up if under-filled
                        if not pos_key:
                            opp = self.check_opportunity(game, odds)
                            if opp:
                                self._handle_entry(opp)
                        elif self.live_mode and pos_key in self.open_positions and self.open_positions[pos_key].get('live_fill_count', 0) < self.live_contracts:
                            opp = self.check_opportunity(game, odds)
                            if opp and opp.get('contract_type') == self.open_positions[pos_key].get('contract_type'):
                                self._handle_entry(opp)

                # Process upcoming entry/exit logic
                for game, odds, edge, side in upcoming_with_edge:
                    if odds:
                        self.log_snapshot(game, odds, commit=False)
                    game_id = game['game_id']
                    pos_key = self._find_position_for_game(game_id, odds)
                    if pos_key and odds:
                        exit_info = self.check_exit(game, odds, self.open_positions[pos_key])
                        if exit_info:
                            self._handle_exit(exit_info)
                    if not pos_key and odds:
                        opp = self.check_opportunity(game, odds)
                        if opp:
                            self._handle_entry(opp)
                    elif pos_key and odds and self.live_mode and pos_key in self.open_positions and self.open_positions[pos_key].get('live_fill_count', 0) < self.live_contracts:
                        opp = self.check_opportunity(game, odds)
                        if opp and opp.get('contract_type') == self.open_positions[pos_key].get('contract_type'):
                            self._handle_entry(opp)
                
                # Show live positions individually (at bottom)
                if live_positions:
                    print(f"\n  📊 LIVE POSITIONS ({len(live_positions)})")
                    print(f"  {'GAME':<40} {'TYPE':<8} {'ENTRY':>6} {'BID':>6} {'SELL@':>6} {'GROSS':>6} {'STATUS':<10}")
                    print(f"  {'-'*90}")
                    
                    for p in live_positions:
                        pos = p['pos']
                        current_game = p['game']
                        pnl_cents = p['pnl_cents']
                        
                        if self.STOP_LOSS_CENTS and pnl_cents <= -self.STOP_LOSS_CENTS:
                            status = "⚠️ STOP"
                        elif pnl_cents > 5:
                            status = "✅ UP"
                        elif pnl_cents < -5:
                            status = "❌ DOWN"
                        else:
                            status = "⬜ FLAT"
                        
                        away = strip_mascot(current_game['away_team'])[:18]
                        home = strip_mascot(current_game['home_team'])[:18]
                        matchup = f"{away:<18} @ {home}"
                        
                        type_labels = {'home_yes': 'HY', 'home_no': 'HN', 'away_yes': 'AY', 'away_no': 'AN'}
                        type_label = type_labels.get(p['contract_type'], 'HY')
                        venue = "ᴷ" if p['market_source'] == 'Kalshi' else "ᴾ"
                        
                        # Compute min sell bid
                        side = pos['side']
                        our_prob = current_game.get('live_home_prob', 0.5)
                        if side == 'away':
                            our_prob = 1 - our_prob
                        time_rem = current_game.get('time_remaining_sec', 2400)
                        min_sell = self.compute_min_sell_bid(time_rem, our_prob)
                        
                        print(f"  {matchup:<40} {type_label}{venue:<5} {pos['entry_price']*100:5.0f}¢ {p['current_bid']*100:5.0f}¢ {min_sell:5.0f}¢ {pnl_cents:+5.0f}¢ {status:<10}")
                
                # Show pregame positions as condensed summary
                if pregame_positions:
                    pregame_pnl = sum(p['pnl_cents'] for p in pregame_positions)
                    print(f"\n  📦 PREGAME POSITIONS: {len(pregame_positions)} entries | {pregame_pnl:+.0f}¢ gross unrealized")
                
                # Show unmatched positions (Kalshi positions with no ESPN game)
                if unmatched_positions:
                    print(f"\n  ⚠️ UNMATCHED POSITIONS ({len(unmatched_positions)}):")
                    for gid, pos in unmatched_positions:
                        team = pos['home_team'] if pos['side'] == 'home' else pos['away_team']
                        contract = pos['contract_type'].replace('_', ' ').upper()
                        count = pos.get('live_fill_count', 0)
                        print(f"     {contract} {team[:20]} @ {pos['entry_price']*100:.0f}¢ x{count} (no ESPN match)")

                
                # Batch commit all snapshots from this cycle
                self.conn.commit()
                
                print(f"\n{'═'*120}")
                # Real ROI = profit / peak capital deployed
                roi_pct = (self.realized_pnl / self.peak_capital_deployed * 100) if self.peak_capital_deployed > 0 else 0
                roi_str = f" ({roi_pct:+.1f}% ROI)" if self.trade_count > 0 else ""
                
                # Smart sleep: only wait the remainder of poll_interval
                elapsed = time.time() - cycle_start
                sleep_time = max(0, poll_interval - elapsed)
                cycle_info = f"cycle: {elapsed:.1f}s"
                
                if self.live_mode and kalshi_portfolio:
                    session_pnl = kalshi_portfolio['session_pnl']
                    pnl_str = f"+${session_pnl:.2f}" if session_pnl >= 0 else f"-${abs(session_pnl):.2f}"
                    print(f"  📊 Portfolio: ${kalshi_portfolio['total']:.2f} │ Session: {pnl_str} │ {kalshi_portfolio['position_count']} positions │ {cycle_info} │ Ctrl+C to stop")
                else:
                    print(f"  📊 Net: {self.realized_pnl:+.0f}¢{roi_str} realized | {len(self.open_positions)} open ({unrealized_pnl:+.0f}¢ gross) | {self.trade_count} trades | {cycle_info} | Ctrl+C to stop")
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            # Clean up WebSocket
            if self.ws_book:
                self.ws_book.stop()
            print("\n\nStopped. Use --review to see logged trades.")
    
    def _handle_entry(self, opp: dict):
        """Handle entry into a new position"""
        game_id = opp['game_id']
        side = opp['side']
        entry_price = opp['entry_price']
        status = opp.get('status', 'in')
        market_source = opp.get('market_source', 'Kalshi')
        contract_type = opp.get('contract_type', f'{side}_yes')  # e.g., home_yes, away_no
        
        # === LIVE TRADING: Place order FIRST ===
        live_fill_count = 0
        live_avg_price = None
        live_order_id = None
        
        if self.live_mode and self.kalshi_client and market_source == 'Kalshi':
            ticker = opp.get('market_ticker')
            if ticker:
                # === SAFETY CHECK: Verify position and calculate order size ===
                contracts_to_order = self.live_contracts
                
                # Cap based on tracked position (handles cross-contract top-ups)
                if game_id in self.open_positions:
                    existing_pos = self.open_positions[game_id]
                    # Safety: only top-up with same contract type (don't mix home_yes with away_no)
                    existing_ct = existing_pos.get('contract_type', '')
                    if existing_ct and existing_ct != contract_type:
                        return  # Wrong contract type for this position
                    existing_tracked = existing_pos.get('live_fill_count', 0)
                    contracts_to_order = max(0, self.live_contracts - existing_tracked)
                    if contracts_to_order <= 0:
                        return  # Already at full size across all contracts
                    print(f"  📈 TOP-UP: {existing_tracked}/{self.live_contracts} filled, ordering {contracts_to_order} more")
                
                try:
                    kalshi_positions = self.kalshi_client.get_positions()
                    if ticker in kalshi_positions and kalshi_positions[ticker].yes_count != 0:
                        existing_count = abs(kalshi_positions[ticker].yes_count)
                        if existing_count >= self.live_contracts:
                            print(f"\n  ⚠️ BLOCKED: Already have {existing_count} contracts on {ticker}")
                            # Add to open_positions to prevent future attempts
                            if game_id not in self.open_positions:
                                self.open_positions[game_id] = {
                                    'trade_id': None,
                                    'side': side,
                                    'entry_price': kalshi_positions[ticker].yes_avg_price / 100.0,
                                    'entry_time': None,
                                    'entry_status': 'unknown',
                                    'home_team': opp['home_team'],
                                    'away_team': opp['away_team'],
                                    'market_ticker': ticker,
                                    'market_source': 'Kalshi',
                                    'contract_type': contract_type,
                                    'live_fill_count': existing_count,
                                    'live_avg_price': kalshi_positions[ticker].yes_avg_price,
                                    'live_order_id': None
                                }
                            return  # Already at full size!
                        else:
                            ticker_remaining = self.live_contracts - existing_count
                            contracts_to_order = min(contracts_to_order, ticker_remaining)
                            print(f"  📈 TOP-UP: Have {existing_count} on ticker + {self.live_contracts - contracts_to_order - existing_count} other, ordering {contracts_to_order} more")
                except Exception as e:
                    print(f"  ⚠️ Could not verify existing position: {e}")
                
                kalshi_side = OrderSide.NO if '_no' in contract_type else OrderSide.YES
                
                # Smart price walk-up: if IOC at ask won't fill, try 1¢ above if edge still clears
                raw_price_cents = round(entry_price * 100)
                time_remaining = opp.get('time_remaining_sec', 2400)
                required_edge = self.get_required_edge(entry_price, time_remaining)
                current_edge = opp['our_prob'] - entry_price
                
                buffer = 0
                if current_edge - 0.01 >= required_edge:
                    buffer = 1  # Edge survives +1¢
                    if current_edge - 0.02 >= required_edge:
                        buffer = 2  # Edge survives +2¢
                
                price_cents = min(99, raw_price_cents + buffer)
                
                buf_str = f" (ask={raw_price_cents}¢ +{buffer}¢ buffer)" if buffer > 0 else ""
                print(f"\n  🔴 LIVE BUY: {ticker} {kalshi_side.value.upper()} @ {price_cents}¢ x {contracts_to_order}{buf_str}")
                
                try:
                    result = self.kalshi_client.place_order(
                        ticker=ticker,
                        side=kalshi_side,
                        action=OrderAction.BUY,
                        count=contracts_to_order,
                        price_cents=price_cents,
                        time_in_force=TimeInForce.IOC
                    )
                    
                    if result.success and result.fill_count > 0:
                        live_fill_count = result.fill_count
                        live_avg_price = result.avg_price or (entry_price * 100)  # Fall back to requested price
                        
                        # Kalshi returns NO contract fills as YES-equivalent price - convert back
                        if '_no' in contract_type:
                            live_avg_price = 100 - live_avg_price
                        
                        live_order_id = result.order_id
                        entry_price = live_avg_price / 100.0  # Use actual fill price
                        opp['entry_price'] = entry_price  # Update opp so log_opportunity uses actual fill price
                        print(f"  ✓ FILLED {live_fill_count} @ {live_avg_price:.0f}¢ (fees: ${result.taker_fees:.4f})")
                    else:
                        print(f"  ❌ NO FILL: {result.error or 'Order not filled'}")
                        return  # Don't track position if live order failed
                except Exception as e:
                    print(f"  ❌ ORDER ERROR: {e}")
                    return  # Don't track position if live order failed
        
        # Log the entry to database
        trade_id = self.log_opportunity(opp, f'ENTRY ({status.upper()})')
        
        # Track the position
        if game_id in self.open_positions and live_fill_count > 0:
            # Top-up: update existing position with weighted average
            existing = self.open_positions[game_id]
            old_count = existing.get('live_fill_count', 0)
            old_avg = existing.get('live_avg_price', 0) or 0
            new_total = old_count + live_fill_count
            if new_total > 0 and old_avg > 0:
                existing['live_avg_price'] = (old_avg * old_count + live_avg_price * live_fill_count) / new_total
                existing['entry_price'] = existing['live_avg_price'] / 100.0
            existing['live_fill_count'] = new_total
        else:
            self.open_positions[game_id] = {
                'trade_id': trade_id,
                'side': side,
                'entry_price': entry_price,
                'entry_time': datetime.now(),
                'entry_status': status,  # 'pre' or 'in'
                'home_team': opp['home_team'],
                'away_team': opp['away_team'],
                'market_ticker': opp.get('market_ticker'),
                'market_source': market_source,  # Track which venue we entered on
                'contract_type': contract_type,   # Track which contract: home_yes, away_no, etc.
                # Live trading fields
                'live_fill_count': live_fill_count,
                'live_avg_price': live_avg_price,
                'live_order_id': live_order_id
            }
        
        # Track peak capital deployed for accurate ROI calculation
        self.current_capital_deployed = sum(pos['entry_price'] * 100 for pos in self.open_positions.values())
        self.peak_capital_deployed = max(self.peak_capital_deployed, self.current_capital_deployed)
        
        team = opp['home_team'] if side == 'home' else opp['away_team']
        spread_str = f" | Spread: {opp.get('spread', 0):.0f}¢" if opp.get('spread') else ""
        status_str = "PREGAME " if status == 'pre' else ""
        venue_str = f" [{market_source}]" if market_source != 'Kalshi' else ""
        total_count = self.open_positions.get(game_id, {}).get('live_fill_count', live_fill_count)
        if live_fill_count > 0 and total_count > live_fill_count:
            live_str = f" 🔴+{live_fill_count}={total_count}"
        elif live_fill_count > 0:
            live_str = f" 🔴x{live_fill_count}"
        else:
            live_str = ""
        
        # Show contract type (YES vs NO)
        contract_label = contract_type.replace('_', ' ').upper()  # e.g., "HOME YES" or "AWAY NO"
        
        print(f"\n  🎯 {status_str}ENTRY #{trade_id}: BUY {contract_label} ({team[:20]}) @ {entry_price*100:.0f}¢ | Model: {opp['our_prob']*100:.0f}% | Edge: {opp['edge']*100:.1f}%{spread_str}{venue_str}{live_str}\n")
    
    def _print_opportunity_inline(self, opp: dict):
        """Print compact opportunity alert"""
        if opp['side'] == 'home':
            team = opp['home_team'][:20]
            our_prob = opp['live_home_prob']
        else:
            team = opp['away_team'][:20]
            our_prob = 1 - opp['live_home_prob']
        
        trade_id = self.log_opportunity(opp)
        print(f"  └─ 🚨 #{trade_id}: BUY {opp['side'].upper()} ({team}) @ {our_prob:.1%} - {opp['thesis']}")
    
    def _print_game(self, game: dict):
        """Print game state"""
        prob_change = game.get('prob_change', 0)
        arrow = "↑" if prob_change > 0.02 else "↓" if prob_change < -0.02 else "→"
        
        print(f"{game['away_team']} {game['away_score']} @ {game['home_team']} {game['home_score']}")
        print(f"  P{game['period']} {game['clock']} | {game['time_remaining_sec']//60}:{game['time_remaining_sec']%60:02d} left")
        print(f"  Pregame: Home {game['pregame_home_prob']:.1%} / Away {1-game['pregame_home_prob']:.1%}")
        print(f"  LIVE:    Home {game['live_home_prob']:.1%} / Away {1-game['live_home_prob']:.1%} {arrow} ({prob_change:+.1%})")
        print()
    
    def _print_opportunity(self, opp: dict):
        """Print opportunity alert"""
        print(f"  🚨 OPPORTUNITY DETECTED")
        print(f"  {opp['thesis']}")
        
        if opp['side'] == 'home':
            our_prob = opp['live_home_prob']
            print(f"  Model says: {opp['home_team']} = {our_prob:.1%}")
        else:
            our_prob = 1 - opp['live_home_prob']
            print(f"  Model says: {opp['away_team']} = {our_prob:.1%}")
        
        print()
        
        # Log it
        trade_id = self.log_opportunity(opp)
        print(f"  Logged as trade #{trade_id}")
        print(f"  To add market price (home prob): sqlite3 compiled_stats.db \"UPDATE paper_trades SET market_home_prob=0.XX WHERE id={trade_id};\"")
        print()
    
    def review_opportunities(self):
        """Review logged opportunities"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, home_team, away_team, 
                   home_score, away_score, time_remaining_sec,
                   live_home_prob, prob_swing, market_home_prob, edge, side
            FROM paper_trades
            ORDER BY timestamp DESC
            LIMIT 20
        """)
        
        print("\n" + "="*80)
        print("RECENT PAPER TRADES")
        print("="*80 + "\n")
        
        rows = cursor.fetchall()
        if not rows:
            print("No opportunities logged yet.")
            return
        
        for row in rows:
            id, ts, home, away, hs, as_, tr, prob, swing, mkt, edge, side = row
            ts_short = ts[11:19] if ts else ""
            
            print(f"#{id} | {ts_short} | {away} @ {home} ({as_}-{hs})")
            print(f"     Live Home: {prob:.1%} | Swing: {swing:+.1%} | Rec: {side.upper()}")
            if mkt:
                print(f"     Market Home: {mkt:.1%} | Edge: {edge:+.1%}")
            else:
                print(f"     Market: NOT ENTERED")
                print(f"     sqlite3 compiled_stats.db \"UPDATE paper_trades SET market_home_prob=0.XX WHERE id={id};\"")
            print()
    
    def analyze_results(self):
        """Analyze completed paper trades"""
        cursor = self.conn.cursor()
        
        # Get trades with market prices
        cursor.execute("""
            SELECT id, game_id, home_team, away_team, live_home_prob, 
                   market_home_prob, edge, side, would_have_won
            FROM paper_trades
            WHERE market_home_prob IS NOT NULL
        """)
        
        rows = cursor.fetchall()
        
        print("\n" + "="*80)
        print("PAPER TRADE ANALYSIS")
        print("="*80 + "\n")
        
        if not rows:
            print("No trades with market prices logged yet.")
            print("Add market prices: sqlite3 compiled_stats.db \"UPDATE paper_trades SET market_home_prob=0.XX WHERE id=N;\"")
            return
        
        total_edge = 0
        count = 0
        
        for row in rows:
            id, gid, home, away, model, market, edge, side, won = row
            print(f"#{id}: {away} @ {home}")
            print(f"     Model Home: {model:.1%} | Market Home: {market:.1%} | Edge: {edge:+.1%} | Rec: {side.upper()}")
            if won is not None:
                print(f"     Result: {'WON' if won else 'LOST'}")
            total_edge += edge if edge else 0
            count += 1
        
        print(f"\n{'='*80}")
        print(f"Total trades with market data: {count}")
        print(f"Average edge: {total_edge/count:.2%}" if count > 0 else "")


def main():
    parser = argparse.ArgumentParser(description='Live Paper Trading System')
    parser.add_argument('--db', default='compiled_stats.db', help='Database path')
    parser.add_argument('--interval', type=int, default=1, help='Poll interval seconds')
    parser.add_argument('--review', action='store_true', help='Review logged opportunities')
    parser.add_argument('--analyze', action='store_true', help='Analyze completed trades')
    parser.add_argument('--live', action='store_true', help='Enable LIVE trading on Kalshi (real money!)')
    parser.add_argument('--contracts', type=int, default=10, help='Number of contracts per trade in live mode (default: 10)')
    args = parser.parse_args()
    
    # Safety confirmation for live mode
    if args.live:
        print("⚠️  WARNING: LIVE TRADING MODE ⚠️")
        print("This will place REAL orders with REAL money on Kalshi!")
        print(f"Position size: {args.contracts} contracts per trade")
        confirm = input("\nType 'live' to confirm: ")
        if confirm != 'live':
            print("Cancelled.")
            sys.exit(0)
    
    trader = PaperTrader(args.db, live_mode=args.live, live_contracts=args.contracts)
    
    if args.review:
        trader.review_opportunities()
    elif args.analyze:
        trader.analyze_results()
    else:
        trader.run(poll_interval=args.interval)


if __name__ == "__main__":
    main()