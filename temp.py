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
import shutil
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

# ===== POLYMARKET IMPORT =====
try:
    from polymarket_us import PolymarketUS
    POLYMARKET_AVAILABLE = True
except ImportError:
    POLYMARKET_AVAILABLE = False

# ===== POLYMARKET WEBSOCKET IMPORT =====
try:
    from polymarket_ws_orderbook import PolymarketWSOrderbook
    POLY_WS_AVAILABLE = True
except ImportError:
    POLY_WS_AVAILABLE = False

# ===== API CREDENTIALS (from environment variables) =====
KALSHI_API_KEY = os.environ.get("KALSHI_API_KEY")
KALSHI_KEY_PATH = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "/root/kalshi-key.key")
POLY_KEY_ID = os.environ.get("POLY_KEY_ID")
POLY_SECRET_KEY = os.environ.get("POLY_SECRET_KEY")

# ===== FEE CALCULATIONS =====
def kalshi_fee_cents(price: float) -> float:
    """Kalshi fee: 7% × P × (1-P) × 100 cents, capped at 1.75¢"""
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


def get_terminal_width() -> int:
    """Get current terminal width with fallback"""
    try:
        return shutil.get_terminal_size((120, 40)).columns
    except:
        return 120

def get_layout():
    """Return layout parameters based on terminal width.
    Desktop (>=100): full wide layout
    Mobile (<100): compact layout for Termius/phone"""
    w = get_terminal_width()
    compact = w < 100
    return {
        'width': w,
        'compact': compact,
        'sep_width': min(w - 2, 120) if not compact else w - 2,
        'team_width': 12 if compact else 18,
        'matchup_width': 28 if compact else 40,
    }


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


def fetch_polymarket_odds(poly_client) -> dict:
    """Fetch today's CBB moneyline odds from Polymarket US.
    
    Uses series 7 (CBB 2025). Slug format: aec-cbb-{away}-{home}-{YYYY-MM-DD}
    BBO represents the outcome team (away team for CBB, ordering='away').
    
    Returns dict keyed by normalized team name, matching Kalshi odds format.
    """
    if not poly_client:
        return {}
    
    # Use same date logic as predictions (yesterday if before 4am)
    now = datetime.now()
    if now.hour < 4:
        effective_date = now - timedelta(days=1)
    else:
        effective_date = now
    today_str = effective_date.strftime('%Y-%m-%d')
    
    print(f"  Fetching Polymarket markets...", end='', flush=True)
    
    try:
        ev_list = []
        offset = 0
        while True:
            events = poly_client.events.list({
                "seriesId": [7],
                "active": True,
                "closed": False,
                "limit": 100,
                "offset": offset,
            })
            batch = events.get('events', [])
            ev_list.extend(batch)
            if len(batch) < 100:
                break
            offset += 100
    except Exception as e:
        print(f" error: {e}")
        return {}
    
    # Filter to today's events by slug date
    today_events = [e for e in ev_list if today_str in e.get('slug', '')]
    print(f" {len(today_events)} today...", end='', flush=True)
    
    odds = {}
    matched = 0
    
    for ev in today_events:
        title = ev.get('title', '')
        slug = ev.get('slug', '')
        
        if ' vs. ' not in title:
            continue
        
        # Parse team names from title: "Away vs. Home" (ordering='away' for CBB)
        parts = title.split(' vs. ')
        if len(parts) != 2:
            continue
        
        away_name = parts[0].strip()
        home_name = parts[1].strip()
        away_norm = normalize_team_name(away_name)
        home_norm = normalize_team_name(home_name)
        
        # Fetch BBO for this market
        try:
            bbo = poly_client.markets.bbo(slug)
            md = bbo.get('marketData', {})
            best_bid = md.get('bestBid', {})
            best_ask = md.get('bestAsk', {})
            
            if not best_bid or not best_ask:
                continue
            
            bid_val = float(best_bid.get('value', 0))
            ask_val = float(best_ask.get('value', 0))
            
            if bid_val <= 0.01 or ask_val <= 0.01:
                continue
            
            # BBO is for outcome team (away in CBB)
            away_bid_cents = bid_val * 100
            away_ask_cents = ask_val * 100
            
            # Home prices are inverted (short side)
            home_bid_cents = 100 - away_ask_cents
            home_ask_cents = 100 - away_bid_cents
            
            game_odds = {
                'poly_away': away_name,
                'poly_home': home_name,
                'poly_away_norm': away_norm,
                'poly_home_norm': home_norm,
                'home_prob': (home_bid_cents + home_ask_cents) / 200.0,
                'away_prob': (away_bid_cents + away_ask_cents) / 200.0,
                'home_yes_bid': home_bid_cents,
                'home_yes_ask': home_ask_cents,
                'away_yes_bid': away_bid_cents,
                'away_yes_ask': away_ask_cents,
                # No NO contracts on Polymarket
                'home_no_bid': None,
                'home_no_ask': None,
                'away_no_bid': None,
                'away_no_ask': None,
                'home_ticker': None,
                'away_ticker': None,
                'title': title,
                'slug': slug,
                'source': 'Polymarket'
            }
            
            odds[away_norm] = game_odds
            odds[home_norm] = game_odds
            matched += 1
            
        except Exception:
            continue
    
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
        # Polymarket price columns
        ("live_snapshots", "poly_home_bid", "REAL"),
        ("live_snapshots", "poly_home_ask", "REAL"),
        ("live_snapshots", "poly_away_bid", "REAL"),
        ("live_snapshots", "poly_away_ask", "REAL"),
        ("live_snapshots", "poly_home_edge", "REAL"),
        ("live_snapshots", "poly_away_edge", "REAL"),
        ("live_snapshots", "poly_slug", "TEXT"),
        ("live_snapshots", "chosen_venue", "TEXT"),
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
    OV_SCALE = 0.349
    OV_EXPONENT = 0.4277
    OV_PROB_COEFF = 17.2926
    OV_DECORR_SEC = 120  # Noise decorrelation timescale (empirical autocorrelation)
    
    # Stop loss and cooldown
    STOP_LOSS_CENTS = None   # No stop loss - EV-based exits handle this
    PREGAME_STOP_LOSS_CENTS = None  # No stop loss for pregame entries either
    COOLDOWN_AFTER_EXIT = 30  # 30 second cooldown after any exit
    COOLDOWN_AFTER_STOP = 240  # 4 minutes after stop loss (if ever re-enabled)
    
    # Microstate haircut (exits only) — calibrated from live_snapshots
    # Late in close games, MMs see possession/FT/fouls our model can't.
    # H_eff = K × g(t) × SCALE × exp(-t/TAU) × exp(-|m|/MARGIN_DECAY) if |m| <= MARGIN_MAX
    # where g(t) ramps linearly from 0 at RAMP_HI to 1 at RAMP_LO
    HAIRCUT_SCALE = 28.27     # Peak excess dispersion (cents) at t=0, m=0
    HAIRCUT_TAU = 112         # Time decay constant (seconds)
    HAIRCUT_MARGIN_DECAY = 1.4  # Margin decay constant (points)
    HAIRCUT_K = 0.75          # Conservative scale factor (tunable)
    HAIRCUT_RAMP_HI = 240     # g(t)=0 above this (seconds)
    HAIRCUT_RAMP_LO = 120     # g(t)=1 below this (seconds)
    HAIRCUT_MARGIN_MAX = 5    # No haircut if |margin| > this
    
    # ESPN score freshness guard
    # Only enter/exit when ESPN score data is recently updated.
    # Prevents phantom edges from stale ESPN scores after Kalshi has already priced in a basket.
    # Exception: if game clock is also stale (timeout/halftime), scores aren't changing so data is fine.
    MAX_ESPN_SCORE_STALE_SEC = 10  # Max seconds since last ESPN score change during active play

    def get_required_edge(self, entry_price: float, time_remaining: int = 2400) -> float:
        """Dynamic edge threshold based on time remaining (exponential ramp).
        
        Exponential (k=2) ramp from 5% at game start to 15% at 4:00 remaining.
        Gentle early (when MMs still calibrating), steep late (info asymmetry explodes).
        
        Pregame (time_remaining >= 2400) uses 5%
        """
        E_START = 0.08  # 8% at game start / pregame
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
    
    def __init__(self, db_path: str, live_mode: bool = False, live_contracts: int = 10, venue: str = 'best'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.model = LiveWinProbabilityModel()
        self.predictions = {}
        self.kalshi_odds = {}
        self.polymarket_odds = {}
        self.poly_client = None  # Polymarket US SDK client
        self.venue = venue  # 'kalshi', 'polymarket', or 'best'
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
        self._cached_kalshi_positions = None
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
                    api_key_id=KALSHI_API_KEY,
                    private_key_path=KALSHI_KEY_PATH,
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
                    api_key=KALSHI_API_KEY,
                    private_key_path=KALSHI_KEY_PATH,
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
        
        # ===== POLYMARKET CLIENT =====
        if self.venue in ('polymarket', 'best') and POLYMARKET_AVAILABLE:
            try:
                self.poly_client = PolymarketUS(
                    key_id=POLY_KEY_ID,
                    secret_key=POLY_SECRET_KEY
                )
                bal = self.poly_client.account.balances()
                poly_balance = bal.get('balances', [{}])[0].get('currentBalance', 0)
                print(f"  ✓ Polymarket connected (balance: ${poly_balance:.2f})")
            except Exception as e:
                print(f"  ⚠️ Polymarket init error: {e} — Kalshi only")
                self.poly_client = None
        elif self.venue == 'kalshi':
            print("  ℹ️ Venue: Kalshi only (--venue kalshi)")
        else:
            print("  ⚠️ polymarket-us not installed — Kalshi only")
        
        # ===== POLYMARKET WEBSOCKET =====
        self.poly_ws_book = None
        self.use_poly_ws = False
        self._poly_ws_subscribed_slugs = set()
        
        if self.poly_client and POLY_WS_AVAILABLE and self.live_mode:
            try:
                self.poly_ws_book = PolymarketWSOrderbook(
                    key_id=POLY_KEY_ID,
                    secret_key=POLY_SECRET_KEY,
                    verbose=False
                )
                self.poly_ws_book.start()
                time.sleep(1)
                
                if self.poly_ws_book.connected:
                    self.use_poly_ws = True
                    print("  ✓ Polymarket WebSocket connected")
                else:
                    print("  ⚠️ Polymarket WebSocket failed, using REST fallback")
            except Exception as e:
                print(f"  ⚠️ Polymarket WebSocket init error: {e}, using REST fallback")
        
        setup_database(self.conn)
        self._load_predictions()
        self._load_kalshi_odds()
        if self.poly_client:
            self._load_polymarket_odds()
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
    
    def compute_min_sell_bid(self, time_remaining_sec: int, our_prob: float, abs_margin: int = 0, venue: str = 'Kalshi') -> float:
        """Compute minimum bid (in cents) needed to trigger an EV exit.
        
        Solves: bid - fee(bid) = our_prob * 100 + OV + haircut
        Uses iterative Newton steps from initial guess bid₀ = ev_hold.
        
        Args:
            venue: 'Kalshi' or 'Polymarket' — determines fee model used.
        
        Returns:
            Minimum bid in cents to justify selling, or 100 if hold-to-settlement.
        """
        ov = self.compute_option_value(time_remaining_sec, our_prob)
        haircut = self.compute_haircut(time_remaining_sec, abs_margin)
        ev_hold = our_prob * 100 + ov + haircut
        
        if ev_hold >= 99:
            return 100.0  # Effectively "hold forever"
        
        fee_fn = poly_fee_cents if venue == 'Polymarket' else kalshi_fee_cents
        
        # Newton: bid - fee(bid) = ev_hold
        # Start with bid₀ = ev_hold, iterate: bid_{n+1} = ev_hold + fee(bid_n)
        bid = ev_hold
        for _ in range(3):
            fee = fee_fn(bid / 100.0)
            bid = ev_hold + fee
        
        return min(99.0, max(1.0, bid))
    
    def compute_haircut(self, time_remaining_sec: int, abs_margin: int) -> float:
        """Compute microstate uncertainty premium (cents) for exit threshold.
        
        Late in close games, market makers observe microstate (possession, FT,
        foul count, shot clock) that our model is blind to. This raises ev_hold
        so we only sell when market overpays enough to overcome our info gap.
        
        Gated: only applies when |margin| <= 5 and time < 4:00, ramping in
        from 4:00 to full strength at 2:00.
        
        Calibrated from excess residual dispersion in (time, margin) buckets
        using margin-conditional baselines from first-half data.
        """
        # Margin gate: no haircut if game isn't close
        if abs_margin > self.HAIRCUT_MARGIN_MAX:
            return 0.0
        
        # Time ramp: 0 above RAMP_HI, linear ramp, 1 below RAMP_LO
        if time_remaining_sec >= self.HAIRCUT_RAMP_HI:
            return 0.0
        elif time_remaining_sec <= self.HAIRCUT_RAMP_LO:
            g = 1.0
        else:
            g = (self.HAIRCUT_RAMP_HI - time_remaining_sec) / (self.HAIRCUT_RAMP_HI - self.HAIRCUT_RAMP_LO)
        
        # Raw exponential surface
        h_raw = self.HAIRCUT_SCALE * math.exp(-time_remaining_sec / self.HAIRCUT_TAU) * math.exp(-abs_margin / self.HAIRCUT_MARGIN_DECAY)
        
        return max(0.0, self.HAIRCUT_K * g * h_raw)
    
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
    
    def _load_polymarket_odds(self):
        """Load current Polymarket odds for today's games"""
        self.polymarket_odds = fetch_polymarket_odds(self.poly_client)
        if self.polymarket_odds:
            print(f"✓ Loaded {len(self.polymarket_odds) // 2} Polymarket markets")
    
    def _load_open_positions(self):
        """Load open positions - uses venue APIs as source of truth in live mode.
        
        LIVE MODE: Builds positions from Kalshi API + Polymarket API. No paper_trades lookup.
        PAPER MODE: Uses paper_trades table (legacy behavior).
        """
        if self.live_mode:
            self._build_positions_from_kalshi()
            if self.poly_client:
                self._build_positions_from_polymarket()
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
        
        # Track loaded positions by event stem to detect duplicates
        stem_to_game_id = {}  # event_stem -> game_id key used in open_positions
        
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
            
            # Check for duplicate position on same event (e.g., home_no + away_yes)
            event_stem = self._ticker_event_stem(ticker)
            if event_stem and event_stem in stem_to_game_id:
                existing_key = stem_to_game_id[event_stem]
                existing_pos = self.open_positions.get(existing_key)
                if existing_pos:
                    existing_ct = existing_pos.get('contract_type', '')
                    print(f"  ⚠️ DUPLICATE POSITION on same event!")
                    print(f"     Existing: {existing_ct} x{existing_pos.get('live_fill_count')} @ {existing_pos['entry_price']*100:.0f}¢")
                    print(f"     New:      {contract_type} x{contract_count} @ {entry_price_cents:.0f}¢")
                    print(f"     Both bet {side.upper()} — keeping first, tracking second as sibling")
                    # Don't create a second position — the first one already covers this game
                    # The duplicate will still be on Kalshi but won't trigger new entries
                    loaded += 1
                    continue
            
            # Try to find game_id by matching to predictions
            # We'll use ticker as temporary key if no match found
            game_id = self._find_game_id_for_teams(home_team, away_team)
            if not game_id:
                # Use ticker as fallback key - will match later when ESPN data comes in
                game_id = f"kalshi:{ticker}"
                print(f"  ⚠️ No game_id found for {away_team} @ {home_team}, using ticker key")
            
            # Track by event stem for duplicate detection
            if event_stem:
                stem_to_game_id[event_stem] = game_id
            
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
    
    # ===== POLYMARKET POSITION MANAGEMENT =====
    
    def _get_poly_positions(self) -> Optional[dict]:
        """Query current Polymarket positions/holdings from API.
        
        Returns dict keyed by slug -> {quantity, avg_price, side, net_position, ...} or None on failure.
        
        SDK response: {'positions': {'slug': {'netPosition': '20', 'cost': {'value': '10.58', ...}, ...}}}
        netPosition positive = LONG, negative = SHORT.
        """
        if not self.poly_client:
            return None
        
        try:
            response = self.poly_client.portfolio.positions()
            positions_dict = response.get('positions', {})
            
            # positions is a dict keyed by slug, not a list
            if not isinstance(positions_dict, dict):
                return {}
            
            result = {}
            for slug, pos in positions_dict.items():
                net_pos = int(float(pos.get('netPosition', '0')))
                if net_pos == 0:
                    continue
                
                quantity = abs(net_pos)
                side = 'LONG' if net_pos > 0 else 'SHORT'
                
                # Avg price from cost / quantity
                cost_val = float(pos.get('cost', {}).get('value', '0'))
                avg_price = cost_val / quantity if quantity > 0 else 0
                
                result[slug] = {
                    'quantity': quantity,
                    'avg_price': avg_price,
                    'side': side,
                    'net_position': net_pos,
                    'team_ordering': pos.get('marketMetadata', {}).get('team', {}).get('ordering', ''),
                    'team_name': pos.get('marketMetadata', {}).get('team', {}).get('safeName', ''),
                    'raw': pos
                }
            
            return result
            
        except Exception as e:
            print(f"  ⚠️ Polymarket positions API error: {e}")
            return None
    
    def _build_positions_from_polymarket(self):
        """Build open_positions entries from Polymarket API (LIVE MODE).
        
        Polymarket is a secondary source of truth alongside Kalshi.
        Only adds positions not already covered by Kalshi.
        """
        poly_positions = self._get_poly_positions()
        if poly_positions is None:
            print("  ⚠️ Cannot build Polymarket positions - API unavailable")
            return
        
        if not poly_positions:
            print(f"✓ No open positions on Polymarket")
            return
        
        loaded = 0
        skipped = 0
        
        for slug, pos_data in poly_positions.items():
            quantity = pos_data['quantity']
            avg_price = pos_data['avg_price']
            side_hint = pos_data.get('side', '')
            
            # Try to find matching Polymarket odds to get team names
            matched_odds = None
            for poly_key, odds in self.polymarket_odds.items():
                if odds.get('slug') == slug:
                    matched_odds = odds
                    break
            
            if not matched_odds:
                print(f"  ⚠️ Unknown Polymarket slug (no matching market): {slug}")
                skipped += 1
                continue
            
            # Determine side from intent/side field
            # BUY_LONG on away team = betting away wins, side='away'
            # BUY_SHORT on away team = betting home wins, side='home'
            poly_away = matched_odds.get('poly_away', 'Unknown')
            poly_home = matched_odds.get('poly_home', 'Unknown')
            
            if 'LONG' in side_hint.upper() or 'long' in side_hint.lower():
                side = 'away'  # Long on outcome (away) team
                contract_type = 'away_yes'
            elif 'SHORT' in side_hint.upper() or 'short' in side_hint.lower():
                side = 'home'  # Short on outcome team = bet on home
                contract_type = 'home_yes'
            else:
                # Fallback: try to infer from price
                # If avg_price < 50, likely bought a low-priced contract (underdog)
                print(f"  ⚠️ Cannot determine side for {slug}, skipping")
                skipped += 1
                continue
            
            # Try to find game_id
            game_id = self._find_game_id_for_teams(poly_home, poly_away)
            if not game_id:
                game_id = f"poly:{slug}"
                print(f"  ⚠️ No game_id found for {poly_away} @ {poly_home}, using slug key")
            
            # Check if this game already has a Kalshi position
            if game_id in self.open_positions:
                existing = self.open_positions[game_id]
                existing_venue = existing.get('market_source', 'Kalshi')
                print(f"  ⚠️ Game {game_id} already has {existing_venue} position, skipping Polymarket")
                skipped += 1
                continue
            
            # Build position dict
            # API cost/quantity already represents actual cost paid:
            #   LONG: paid 30¢ for away token → entry = 0.30 (away_yes)
            #   SHORT: paid 20¢ to short away → entry = 0.20 (home_yes, home-equivalent)
            entry_price_decimal = avg_price if avg_price <= 1.0 else avg_price / 100.0
            
            self.open_positions[game_id] = {
                'trade_id': None,
                'side': side,
                'entry_price': entry_price_decimal,
                'entry_time': None,
                'entry_status': 'unknown',
                'home_team': poly_home,
                'away_team': poly_away,
                'market_ticker': slug,  # For Poly, market_ticker holds the slug
                'market_source': 'Polymarket',
                'contract_type': contract_type,
                'live_fill_count': quantity,
                'live_avg_price': round(entry_price_decimal * 100, 1),
                'live_order_id': None
            }
            loaded += 1
        
        if loaded > 0:
            print(f"✓ Loaded {loaded} positions from Polymarket API")
            for game_id, pos in self.open_positions.items():
                if pos.get('market_source') != 'Polymarket':
                    continue
                team = pos['home_team'] if pos['side'] == 'home' else pos['away_team']
                contract = pos['contract_type'].replace('_', ' ').upper()
                entry_cents = pos['entry_price'] * 100
                count = pos['live_fill_count']
                position_value = entry_cents * count / 100
                print(f"  └─ {contract} {team[:20]} @ {entry_cents:.0f}¢ x{count} = ${position_value:.2f} [Poly]")
        elif skipped > 0:
            print(f"✓ No new Polymarket positions ({skipped} skipped/unmatched)")
    
    def _verify_poly_position(self, slug: str, expected_change: int, action: str = 'buy') -> Optional[int]:
        """Verify a Polymarket position changed as expected after a trade.
        
        Args:
            slug: Market slug
            expected_change: Expected change in quantity (positive for buys, negative for sells)
            action: 'buy' or 'sell' for logging
            
        Returns:
            Actual quantity held after trade, or None if verification failed.
        """
        if not self.poly_client:
            return None
        
        try:
            time.sleep(0.5)  # Brief delay for Polymarket to update
            poly_positions = self._get_poly_positions()
            
            if poly_positions is None:
                print(f"  ⚠️ Could not verify Poly position (API unavailable)")
                return None
            
            if slug in poly_positions:
                actual_qty = poly_positions[slug]['quantity']
                print(f"  ✓ VERIFIED: {slug} → {actual_qty} contracts on Polymarket")
                return actual_qty
            else:
                if action == 'sell':
                    print(f"  ✓ VERIFIED: Position closed on Polymarket")
                    return 0
                else:
                    print(f"  ⚠️ Position not found on Polymarket after {action}")
                    return None
        except Exception as e:
            print(f"  ⚠️ Poly verification error: {e}")
            return None
    
    def _get_polymarket_portfolio(self) -> Optional[dict]:
        """Get current Polymarket portfolio state for display.
        
        Returns dict with: cash, position_value, position_count, session_pnl
        Mirrors _get_kalshi_portfolio() structure.
        """
        if not self.poly_client:
            return None
        
        try:
            # Get cash balance
            bal = self.poly_client.account.balances()
            cash = 0.0
            balances = bal.get('balances', [])
            if balances:
                cash = float(balances[0].get('currentBalance', 0))
            
            # Get positions
            poly_positions = self._get_poly_positions()
            position_value = 0.0
            cost_basis = 0.0
            position_count = 0
            
            if poly_positions:
                for slug, pos_data in poly_positions.items():
                    qty = pos_data['quantity']
                    avg = pos_data['avg_price']
                    
                    position_count += 1
                    cost_basis += avg * qty
                    
                    # Use cashValue from API if available (most accurate)
                    raw = pos_data.get('raw', {})
                    cash_value = raw.get('cashValue', {}).get('value')
                    if cash_value:
                        position_value += float(cash_value)
                    else:
                        # Fallback: try current bid from live odds
                        current_bid = None
                        for team_key, odds in self.polymarket_odds.items():
                            if odds.get('slug') == slug:
                                if pos_data.get('side') == 'LONG':
                                    current_bid = odds.get('away_yes_bid', 0) / 100.0
                                else:
                                    current_bid = odds.get('home_yes_bid', 0) / 100.0
                                break
                        
                        if current_bid and current_bid > 0:
                            position_value += current_bid * qty
                        else:
                            position_value += avg * qty  # Last resort: cost basis
            
            # currentBalance from Poly API INCLUDES position cost basis,
            # so true free cash = currentBalance - cost_basis
            free_cash = cash - cost_basis
            total = free_cash + position_value
            
            return {
                'cash': free_cash,
                'position_value': position_value,
                'cost_basis': cost_basis,
                'total': total,
                'position_count': position_count,
            }
        except Exception as e:
            print(f"  ⚠️ Polymarket portfolio fetch error: {e}")
            return None
    
    @staticmethod
    def _ticker_event_stem(ticker: str) -> Optional[str]:
        """Extract the event stem from a Kalshi ticker.
        
        Tickers look like: KXNCAAMBGAME-26FEB09-CENTRALARK-NALABAMA-NALABAMA
        The event stem is everything up to the last dash: KXNCAAMBGAME-26FEB09-CENTRALARK-NALABAMA
        Both teams' tickers for the same game share this stem.
        """
        if not ticker:
            return None
        parts = ticker.rsplit('-', 1)
        return parts[0] if len(parts) == 2 else None
    
    def _find_position_for_game(self, game_id: str, odds: Optional[dict]) -> Optional[str]:
        """Find position key for a game, handling game_id, kalshi:, and poly: prefix keys.
        
        In live mode, positions might be keyed by 'kalshi:TICKER' or 'poly:SLUG' if game_id 
        wasn't found at startup. This method checks for ticker matches, event stem matches,
        AND Polymarket slug matches, and migrates keys if found.
        
        Returns the key in open_positions, or None if no position exists.
        """
        # Direct lookup by game_id
        if game_id in self.open_positions:
            return game_id
        
        # Build set of event stems for this game's tickers
        game_stems = set()
        if odds:
            home_ticker = odds.get('home_ticker')
            away_ticker = odds.get('away_ticker')
            if home_ticker:
                stem = self._ticker_event_stem(home_ticker)
                if stem:
                    game_stems.add(stem)
            if away_ticker:
                stem = self._ticker_event_stem(away_ticker)
                if stem:
                    game_stems.add(stem)
        
        # Get Polymarket slug for this game (if available)
        game_poly_slug = None
        if odds:
            game_poly_slug = odds.get('poly_slug') or odds.get('slug')
        
        # Check all positions for ticker, event stem, or slug match
        for pos_key in list(self.open_positions.keys()):
            pos = self.open_positions[pos_key]
            pos_ticker = pos.get('market_ticker')
            
            if not pos_ticker:
                continue
            
            matched = False
            
            # Exact ticker match (Kalshi)
            if odds and (pos_ticker == odds.get('home_ticker') or pos_ticker == odds.get('away_ticker')):
                matched = True
            
            # Event stem match (catches home_no vs away_yes on same Kalshi event)
            if not matched and game_stems:
                pos_stem = self._ticker_event_stem(pos_ticker)
                if pos_stem and pos_stem in game_stems:
                    matched = True
            
            # Polymarket slug match (for poly: prefixed keys or slug-based market_ticker)
            if not matched and game_poly_slug and pos_ticker == game_poly_slug:
                matched = True
            
            # Also check poly: prefix keys against slug
            if not matched and game_poly_slug and pos_key == f"poly:{game_poly_slug}":
                matched = True
            
            if matched:
                if pos_key != game_id:
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
    
    def _reconcile_positions(self):
        """Periodic reconciliation: compare tracked open_positions against live API state.
        
        Called every ~60 cycles (~1 minute). Detects and corrects:
        - Phantom positions (tracked but not on venue)
        - Orphan positions (on venue but not tracked)
        - Count drift (tracked count != actual count)
        
        Uses cached positions from this cycle to avoid extra API calls.
        """
        if not self.live_mode:
            return
        
        drift_found = False
        
        # --- Kalshi reconciliation ---
        if self.kalshi_client and self._cached_kalshi_positions is not None:
            kalshi_tickers_held = set()
            for ticker, pos in self._cached_kalshi_positions.items():
                if pos.yes_count != 0:
                    kalshi_tickers_held.add(ticker)
            
            for game_id, pos in list(self.open_positions.items()):
                if pos.get('market_source') != 'Kalshi':
                    continue
                ticker = pos.get('market_ticker')
                if not ticker:
                    continue
                
                if ticker not in kalshi_tickers_held:
                    # Check sibling tickers (same event stem)
                    stem = self._ticker_event_stem(ticker) if ticker else None
                    found_sibling = False
                    if stem:
                        for k_ticker in kalshi_tickers_held:
                            if self._ticker_event_stem(k_ticker) == stem:
                                found_sibling = True
                                break
                    
                    if not found_sibling:
                        print(f"  🔄 RECONCILE: Removing phantom Kalshi position {game_id} (not on Kalshi API)")
                        self.open_positions.pop(game_id, None)
                        drift_found = True
                else:
                    # Check count drift
                    actual_count = abs(self._cached_kalshi_positions[ticker].yes_count)
                    tracked_count = pos.get('live_fill_count', 0)
                    if actual_count != tracked_count and tracked_count > 0:
                        print(f"  🔄 RECONCILE: Kalshi count drift on {game_id}: tracked={tracked_count}, actual={actual_count}")
                        pos['live_fill_count'] = actual_count
                        drift_found = True
        
        # --- Polymarket reconciliation ---
        if self.poly_client and self._cached_poly_positions is not None:
            poly_slugs_held = set(self._cached_poly_positions.keys())
            
            for game_id, pos in list(self.open_positions.items()):
                if pos.get('market_source') != 'Polymarket':
                    continue
                slug = pos.get('market_ticker')
                if not slug:
                    continue
                
                if slug not in poly_slugs_held:
                    print(f"  🔄 RECONCILE: Removing phantom Poly position {game_id} (not on Polymarket API)")
                    self.open_positions.pop(game_id, None)
                    drift_found = True
                else:
                    # Check count drift
                    actual_count = self._cached_poly_positions[slug]['quantity']
                    tracked_count = pos.get('live_fill_count', 0)
                    if actual_count != tracked_count and tracked_count > 0:
                        print(f"  🔄 RECONCILE: Poly count drift on {game_id}: tracked={tracked_count}, actual={actual_count}")
                        pos['live_fill_count'] = actual_count
                        drift_found = True
        
        if drift_found:
            print(f"  🔄 Reconciliation complete — drift corrected")

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
    
    def _match_polymarket_odds(self, espn_home: str, espn_away: str) -> Optional[dict]:
        """Match ESPN team names to Polymarket odds, returning in Kalshi-compatible format.
        
        Polymarket titles are "Away vs. Home" with outcome=away team.
        Returns dict with home_yes_bid/ask, away_yes_bid/ask (in cents) + slug.
        """
        espn_home_norm = normalize_team_name(espn_home)
        espn_away_norm = normalize_team_name(espn_away)
        
        best_match = None
        best_score = 0
        best_orientation = None  # 'team0_is_home' or 'team1_is_home'
        
        checked_titles = set()
        
        for poly_key, odds in self.polymarket_odds.items():
            title = odds['title']
            if title in checked_titles:
                continue
            checked_titles.add(title)
            
            # poly_away = first team in title (outcome team), poly_home = second
            t0_norm = odds['poly_away_norm']  # outcome team
            t1_norm = odds['poly_home_norm']
            
            # Check if poly_away = ESPN away AND poly_home = ESPN home
            t0_away_score = self._name_match_score(espn_away_norm, t0_norm)
            t1_home_score = self._name_match_score(espn_home_norm, t1_norm)
            orientation1_total = t0_away_score + t1_home_score
            
            # Check if poly_away = ESPN home AND poly_home = ESPN away (flipped)
            t0_home_score = self._name_match_score(espn_home_norm, t0_norm)
            t1_away_score = self._name_match_score(espn_away_norm, t1_norm)
            orientation2_total = t0_home_score + t1_away_score
            
            if orientation1_total > best_score and t0_away_score > 0 and t1_home_score > 0:
                best_score = orientation1_total
                best_match = odds
                best_orientation = 'normal'  # poly_away=espn_away, poly_home=espn_home
            
            if orientation2_total > best_score and t0_home_score > 0 and t1_away_score > 0:
                best_score = orientation2_total
                best_match = odds
                best_orientation = 'flipped'  # poly_away=espn_home, poly_home=espn_away
        
        if best_match and best_score >= 1.5:
            if best_orientation == 'normal':
                # Poly away = ESPN away, Poly home = ESPN home → prices already correct
                return {
                    'home_yes_bid': best_match['home_yes_bid'],
                    'home_yes_ask': best_match['home_yes_ask'],
                    'away_yes_bid': best_match['away_yes_bid'],
                    'away_yes_ask': best_match['away_yes_ask'],
                    'home_no_bid': None, 'home_no_ask': None,
                    'away_no_bid': None, 'away_no_ask': None,
                    'home_ticker': None, 'away_ticker': None,
                    'home_prob': best_match['home_prob'],
                    'away_prob': best_match['away_prob'],
                    'title': best_match['title'],
                    'slug': best_match['slug'],
                    'source': 'Polymarket'
                }
            else:
                # Poly away = ESPN home → swap prices
                return {
                    'home_yes_bid': best_match['away_yes_bid'],
                    'home_yes_ask': best_match['away_yes_ask'],
                    'away_yes_bid': best_match['home_yes_bid'],
                    'away_yes_ask': best_match['home_yes_ask'],
                    'home_no_bid': None, 'home_no_ask': None,
                    'away_no_bid': None, 'away_no_ask': None,
                    'home_ticker': None, 'away_ticker': None,
                    'home_prob': best_match['away_prob'],
                    'away_prob': best_match['home_prob'],
                    'title': best_match['title'],
                    'slug': best_match['slug'],
                    'source': 'Polymarket'
                }
        
        return None
    
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
    
    def _poly_ws_subscribe(self, slug: str):
        """Subscribe to Polymarket WebSocket BBO updates for a market slug"""
        if not self.use_poly_ws or not self.poly_ws_book or not slug:
            return
        
        if slug not in self._poly_ws_subscribed_slugs:
            self._poly_ws_subscribed_slugs.add(slug)
            self.poly_ws_book.subscribe([slug])
    
    def _get_best_market_odds(self, espn_home: str, espn_away: str) -> Optional[dict]:
        """Get odds from both Kalshi and Polymarket, returning merged dict.
        
        Kalshi is always the primary dict (it has YES/NO + tickers).
        Polymarket data is tagged on as poly_home_bid/ask etc + has_polymarket flag.
        If Kalshi not available but Poly is, returns Poly-only dict.
        """
        kalshi = self._match_kalshi_odds(espn_home, espn_away)
        poly = self._match_polymarket_odds(espn_home, espn_away)
        
        if not kalshi and not poly:
            return None
        
        if not kalshi:
            # Polymarket only — use as primary
            poly['market_source'] = 'Polymarket'
            poly['data_source'] = 'rest'
            poly['ws_age_ms'] = None
            poly['has_polymarket'] = True
            poly['poly_home_bid'] = poly['home_yes_bid']
            poly['poly_home_ask'] = poly['home_yes_ask']
            poly['poly_away_bid'] = poly['away_yes_bid']
            poly['poly_away_ask'] = poly['away_yes_ask']
            poly['poly_slug'] = poly.get('slug')
            
            # Override with WS if available
            slug = poly.get('slug')
            if slug and self.use_poly_ws and self.poly_ws_book:
                self._poly_ws_subscribe(slug)
                ws_prices = self.poly_ws_book.get_prices(slug)
                if ws_prices and ws_prices.get('away_bid') is not None:
                    poly['home_yes_bid'] = ws_prices['home_bid']
                    poly['home_yes_ask'] = ws_prices['home_ask']
                    poly['away_yes_bid'] = ws_prices['away_bid']
                    poly['away_yes_ask'] = ws_prices['away_ask']
                    poly['poly_home_bid'] = ws_prices['home_bid']
                    poly['poly_home_ask'] = ws_prices['home_ask']
                    poly['poly_away_bid'] = ws_prices['away_bid']
                    poly['poly_away_ask'] = ws_prices['away_ask']
                    poly['data_source'] = 'websocket'
                    poly['ws_age_ms'] = ws_prices['age_ms']
            
            return poly
        
        # Kalshi is primary
        kalshi['market_source'] = 'Kalshi'
        kalshi['data_source'] = 'rest'
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
        
        # Tag Polymarket data onto Kalshi dict for dual-venue evaluation
        if poly:
            kalshi['has_polymarket'] = True
            kalshi['poly_home_bid'] = poly['home_yes_bid']
            kalshi['poly_home_ask'] = poly['home_yes_ask']
            kalshi['poly_away_bid'] = poly['away_yes_bid']
            kalshi['poly_away_ask'] = poly['away_yes_ask']
            kalshi['poly_slug'] = poly.get('slug')
            
            # If Polymarket WS available, override REST with fresher prices
            slug = poly.get('slug')
            if slug and self.use_poly_ws and self.poly_ws_book:
                self._poly_ws_subscribe(slug)
                ws_prices = self.poly_ws_book.get_prices(slug)
                if ws_prices and ws_prices.get('away_bid') is not None:
                    kalshi['poly_away_bid'] = ws_prices['away_bid']
                    kalshi['poly_away_ask'] = ws_prices['away_ask']
                    kalshi['poly_home_bid'] = ws_prices['home_bid']
                    kalshi['poly_home_ask'] = ws_prices['home_ask']
                    kalshi['poly_data_source'] = 'websocket'
                    kalshi['poly_ws_age_ms'] = ws_prices['age_ms']
                else:
                    kalshi['poly_data_source'] = 'rest'
                    kalshi['poly_ws_age_ms'] = None
        else:
            kalshi['has_polymarket'] = False
        
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
                        # First time seeing this game — do NOT mark as fresh.
                        # In live mode, we must observe an actual score change before
                        # trusting data (prevents stale-data entries on restart).
                        # last_score_change_time stays unset → score_age will be large.
                        pass
                    
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
                    # Default to 0 (epoch) if no score change seen yet → score_age will be huge → blocks entry
                    game['seconds_since_score_change'] = now - self.last_score_change_time.get(game_id, 0)
                    
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
            # Verify team alignment — ESPN and predictions may disagree
            # on home/away for neutral site games. If so, flip ESPN to match.
            if pred['team_a_home'] == 1:
                pred_home_id = pred['team_a_id']
            else:
                pred_home_id = pred['team_b_id']
            
            if game['home_team_id'] != pred_home_id:
                # ESPN has teams flipped vs predictions — swap everything
                game['home_team'], game['away_team'] = game['away_team'], game['home_team']
                game['home_team_id'], game['away_team_id'] = game['away_team_id'], game['home_team_id']
                game['home_score'], game['away_score'] = game['away_score'], game['home_score']
                home_score, away_score = game['home_score'], game['away_score']
            
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
        
        # Polymarket prices
        poly_home_bid = None
        poly_home_ask = None
        poly_away_bid = None
        poly_away_ask = None
        poly_home_edge = None
        poly_away_edge = None
        poly_slug = None
        chosen_venue = None
        
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
            
            # Polymarket prices (tagged onto odds dict by _get_best_market_odds)
            if kalshi_odds.get('has_polymarket'):
                poly_home_bid = kalshi_odds.get('poly_home_bid')
                poly_home_ask = kalshi_odds.get('poly_home_ask')
                poly_away_bid = kalshi_odds.get('poly_away_bid')
                poly_away_ask = kalshi_odds.get('poly_away_ask')
                poly_slug = kalshi_odds.get('poly_slug')
                
                if our_home:
                    if poly_home_ask:
                        poly_home_edge = our_home - (poly_home_ask / 100.0)
                    if poly_away_ask:
                        poly_away_edge = (1 - our_home) - (poly_away_ask / 100.0)
            
            # Determine which venue has best entry for snapshot logging
            if our_home and best_home_ask and poly_home_ask:
                k_best = min(best_home_ask, best_away_ask) if best_away_ask else best_home_ask
                p_best = min(poly_home_ask, poly_away_ask) if poly_away_ask else poly_home_ask
                chosen_venue = 'Polymarket' if p_best < k_best else 'Kalshi'
            elif poly_home_ask and not best_home_ask:
                chosen_venue = 'Polymarket'
            elif best_home_ask:
                chosen_venue = 'Kalshi'
        
        cursor.execute("""
            INSERT INTO live_snapshots 
            (timestamp, game_id, home_team, away_team, home_score, away_score,
             period, clock, time_remaining_sec, game_status, pregame_home_prob, live_home_prob,
             prob_change, home_yes_bid, home_yes_ask, away_yes_bid, away_yes_ask,
             home_spread, away_spread,
             home_no_bid, home_no_ask, away_no_bid, away_no_ask,
             best_home_ask, best_home_bid, best_away_ask, best_away_bid,
             kalshi_home_edge, kalshi_away_edge,
             home_edge, away_edge,
             poly_home_bid, poly_home_ask, poly_away_bid, poly_away_ask,
             poly_home_edge, poly_away_edge,
             poly_slug, chosen_venue)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            away_edge,
            poly_home_bid,
            poly_home_ask,
            poly_away_bid,
            poly_away_ask,
            poly_home_edge,
            poly_away_edge,
            poly_slug,
            chosen_venue
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
        
        # === POLYMARKET DATA FRESHNESS GUARD ===
        # Same posture as Kalshi: if WS is active, require recent Poly data before trading Poly
        if self.poly_ws_book and kalshi_odds and status == 'in' and kalshi_odds.get('has_polymarket'):
            poly_age_ms = kalshi_odds.get('poly_ws_age_ms')
            
            if poly_age_ms is not None:
                if time_remaining > 480:
                    MAX_POLY_STALE_MS = 10000  # 10 seconds (Poly WS can be slower)
                else:
                    MAX_POLY_STALE_MS = 6000  # 6 seconds late game
                
                if poly_age_ms > MAX_POLY_STALE_MS:
                    # Poly data stale — only block Poly venue, not Kalshi
                    # We handle this by nullifying Poly prices so evaluate_venue skips it
                    kalshi_odds['poly_home_bid'] = None
                    kalshi_odds['poly_home_ask'] = None
                    kalshi_odds['poly_away_bid'] = None
                    kalshi_odds['poly_away_ask'] = None
        
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
        
        # === DUAL-VENUE EVALUATION: Pick best net edge after fees ===
        is_kalshi_primary = kalshi_odds.get('market_source') != 'Polymarket'
        
        # Evaluate Kalshi (skip if venue is polymarket-only)
        kalshi_best = None
        if self.venue in ('kalshi', 'best') and is_kalshi_primary:
            kalshi_best = evaluate_venue(kalshi_odds, 'Kalshi')
        
        # Evaluate Polymarket (skip if venue is kalshi-only)
        poly_best = None
        if self.venue in ('polymarket', 'best'):
            if kalshi_odds.get('has_polymarket'):
                poly_odds = {
                    'home_yes_bid': kalshi_odds.get('poly_home_bid'),
                    'home_yes_ask': kalshi_odds.get('poly_home_ask'),
                    'away_yes_bid': kalshi_odds.get('poly_away_bid'),
                    'away_yes_ask': kalshi_odds.get('poly_away_ask'),
                    'home_no_bid': None, 'home_no_ask': None,
                    'away_no_bid': None, 'away_no_ask': None,
                    'home_ticker': None, 'away_ticker': None,
                }
                poly_best = evaluate_venue(poly_odds, 'Polymarket')
            elif not is_kalshi_primary:
                # Poly-only match (no Kalshi) — evaluate from primary dict
                poly_best = evaluate_venue(kalshi_odds, 'Polymarket')
        
        # Pick the venue with better net EV after fees
        best = None
        if kalshi_best and poly_best:
            k_side, k_edge, k_price, k_spread = kalshi_best[0], kalshi_best[1], kalshi_best[2], kalshi_best[3]
            p_side, p_edge, p_price, p_spread = poly_best[0], poly_best[1], poly_best[2], poly_best[3]
            k_prob = our_home if k_side == 'home' else 1 - our_home
            p_prob = our_home if p_side == 'home' else 1 - our_home
            k_net_ev = calculate_net_ev(k_prob, k_price, k_spread, 'Kalshi')
            p_net_ev = calculate_net_ev(p_prob, p_price, p_spread, 'Polymarket')
            
            # Log venue comparison
            k_ct = kalshi_best[6]
            p_ct = poly_best[6]
            winner = "POLY" if p_net_ev > k_net_ev else "KALSHI"
            away_short = strip_mascot(game.get('away_team', ''))[:12]
            home_short = strip_mascot(game.get('home_team', ''))[:12]
            print(f"  🔀 VENUE: {away_short}@{home_short} | K:{k_ct} {k_price*100:.0f}¢ spd={k_spread:.0f} ev={k_net_ev:.2f} | P:{p_ct} {p_price*100:.0f}¢ spd={p_spread:.0f} ev={p_net_ev:.2f} → {winner}")
            
            if p_net_ev > k_net_ev:
                best = poly_best
                best = (best[0], best[1], best[2], best[3], kalshi_odds.get('poly_slug'), 'Polymarket', best[6])
            else:
                best = kalshi_best
        elif kalshi_best:
            best = kalshi_best
            away_short = strip_mascot(game.get('away_team', ''))[:12]
            home_short = strip_mascot(game.get('home_team', ''))[:12]
            has_poly = kalshi_odds.get('has_polymarket', False)
            print(f"  🔀 VENUE: {away_short}@{home_short} | K-only ({kalshi_best[6]} {kalshi_best[2]*100:.0f}¢) | Poly={'no match' if not has_poly else 'no edge'}")
        elif poly_best:
            best = poly_best
            slug = kalshi_odds.get('poly_slug') or kalshi_odds.get('slug')
            best = (best[0], best[1], best[2], best[3], slug, 'Polymarket', best[6])
            away_short = strip_mascot(game.get('away_team', ''))[:12]
            home_short = strip_mascot(game.get('home_team', ''))[:12]
            print(f"  🔀 VENUE: {away_short}@{home_short} | P-only ({poly_best[6]} {poly_best[2]*100:.0f}¢) | Kalshi={'no match' if not is_kalshi_primary else 'no edge'}")
        
        if not best:
            # No opportunity - clear any pending confirmation for this game
            for side in ['home', 'away']:
                self.edge_confirmations.pop((game_id, side), None)
            return None
        
        best_side, best_edge, best_entry_price, best_spread, market_ticker, market_source, contract_type = best
        our_prob = our_home if best_side == 'home' else 1 - our_home
        
        # === CROSS-VENUE DUPLICATE GUARD ===
        # Prevent placing a trade on Poly if same game already has Kalshi exposure, and vice versa.
        # Unless the position is already on the same venue (top-up case handled in _handle_entry).
        if game_id in self.open_positions:
            existing_venue = self.open_positions[game_id].get('market_source', 'Kalshi')
            if existing_venue != market_source:
                # Same game, different venue — block to avoid cross-venue duplication
                return None
        
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
        
        # Get current BID based on contract_type and venue
        if market_source == 'Polymarket':
            # Use Polymarket bids (poly_home_bid / poly_away_bid)
            poly_bid_map = {
                'home_yes': 'poly_home_bid',
                'away_yes': 'poly_away_bid',
            }
            bid_key = poly_bid_map.get(contract_type)
            if not bid_key:
                return None  # No NO contracts on Polymarket
            current_bid_cents = kalshi_odds.get(bid_key, 0)
            # If no poly_ fields (Poly-only dict), fall back to standard keys
            if not current_bid_cents:
                bid_key = f'{side}_yes_bid'
                current_bid_cents = kalshi_odds.get(bid_key, 0)
        else:
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
        haircut = self.compute_haircut(time_remaining, abs(home_score - away_score))
        ev_hold = our_prob * 100 + ov + haircut
        
        # EV_exit = current_bid - exit_fee - slippage
        bid_decimal = current_bid_cents / 100.0
        exit_fee = poly_fee_cents(bid_decimal) if market_source == 'Polymarket' else kalshi_fee_cents(bid_decimal)
        ev_exit = current_bid_cents - exit_fee - self.EV_EXIT_SLIPPAGE_CENTS
        
        exit_reason = None
        
        if ev_exit > ev_hold:
            ev_diff = ev_exit - ev_hold
            exit_reason = f"EV EXIT: Sell@{current_bid_cents:.0f}¢ (EV={ev_exit:.1f}) > Hold (EV={ev_hold:.1f}, H={haircut:.1f}¢) | +{ev_diff:.1f}¢ EV"
        
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
        market_source = position.get('market_source', 'Kalshi')
        
        # If live_fill_count not tracked (e.g., after restart), get from Kalshi
        ticker = position.get('market_ticker')
        if live_fill_count == 0 and self.live_mode and self.kalshi_client and ticker and market_source == 'Kalshi':
            try:
                kalshi_positions = self.kalshi_client.get_positions()
                if ticker in kalshi_positions:
                    live_fill_count = abs(kalshi_positions[ticker].yes_count)
            except Exception as e:
                print(f"  ⚠️ Could not get Kalshi position count: {e}")
        
        if self.live_mode and self.kalshi_client and live_fill_count > 0 and not is_settlement and market_source == 'Kalshi':
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
                    print(f"  🛡️ Could not verify position side: {e} — BLOCKING sell for safety")
                    return  # If we can't verify, don't risk opening a short
                
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
        
        # === LIVE TRADING: Polymarket sell ===
        if self.live_mode and self.poly_client and live_fill_count > 0 and not is_settlement and market_source == 'Polymarket':
            slug = position.get('market_ticker')  # For Poly, market_ticker holds the slug
            if slug:
                contract_type = position.get('contract_type', f'{side}_yes')
                
                # === PRE-SELL SAFETY CHECK: Verify we actually hold this position ===
                try:
                    poly_positions = self._get_poly_positions()
                    if poly_positions is None:
                        print(f"  🛡️ Could not verify Poly position — BLOCKING sell for safety")
                        return
                    
                    if slug not in poly_positions or poly_positions[slug]['quantity'] == 0:
                        print(f"  ⚠️ NO POSITION on Polymarket for {slug} — removing from tracking")
                        self.open_positions.pop(game_id, None)
                        self.just_exited.add(game_id)
                        cooldown_key = (game_id, side)
                        self.exit_cooldowns[cooldown_key] = (exit_info.get('time_remaining_sec', 0), self.COOLDOWN_AFTER_EXIT, "Position already closed on Polymarket")
                        return
                    
                    # Use actual count from API
                    actual_qty = poly_positions[slug]['quantity']
                    if actual_qty != live_fill_count:
                        print(f"  ℹ️ Poly position count drift: tracked={live_fill_count}, actual={actual_qty}")
                        live_fill_count = actual_qty
                        position['live_fill_count'] = actual_qty
                        
                except Exception as e:
                    print(f"  🛡️ Could not verify Poly position: {e} — BLOCKING sell for safety")
                    return
                
                # Map to sell intent (opposite of entry)
                if 'away' in contract_type:
                    sell_intent = 'ORDER_INTENT_SELL_LONG'
                else:
                    sell_intent = 'ORDER_INTENT_SELL_SHORT'
                
                EXIT_PRICE_BUFFER_CENTS = 3
                home_exit_cents = exit_info['exit_price'] * 100
                
                if 'away' in contract_type:
                    # SELL_LONG: selling away token, price = sell floor, lower = more aggressive
                    raw_price_cents = home_exit_cents
                    price_cents = max(1, raw_price_cents - EXIT_PRICE_BUFFER_CENTS)
                else:
                    # SELL_SHORT: SDK sends BUY on away token to close short
                    # Complement price: home bid 60¢ → buy away at 40¢ ceiling
                    # Higher ceiling = more aggressive for buys
                    raw_price_cents = 100 - home_exit_cents
                    price_cents = min(99, raw_price_cents + EXIT_PRICE_BUFFER_CENTS)
                
                price_decimal = price_cents / 100.0
                
                print(f"\n  🟣 POLY SELL: {slug} {sell_intent} @ {price_cents:.1f}¢ (home_bid={home_exit_cents:.1f}¢) x {live_fill_count}")
                
                try:
                    result = self.poly_client.orders.create({
                        "marketSlug": slug,
                        "intent": sell_intent,
                        "type": "ORDER_TYPE_LIMIT",
                        "price": {"value": f"{price_cents / 100:.4f}", "currency": "USD"},
                        "quantity": live_fill_count,
                        "tif": "TIME_IN_FORCE_IMMEDIATE_OR_CANCEL",
                    })
                    
                    fills = result.get('executions', []) or result.get('fills', [])
                    fill_count = sum(int(float(f.get('quantity', 0))) if isinstance(f.get('quantity', 0), str) else f.get('quantity', 0) for f in fills)
                    
                    if fill_count > 0:
                        # Exit price in home-probability terms from our known limit.
                        # Same logic as entry: IOC fills at limit, avoid SDK ambiguity.
                        if sell_intent == 'ORDER_INTENT_SELL_SHORT':
                            # We sent BUY away @ price_cents. Home equiv = complement.
                            avg_exit = 1.0 - (price_cents / 100.0)
                        else:
                            avg_exit = price_cents / 100.0
                        
                        # Fees on raw CLOB price
                        fee_cents = poly_fee_cents(price_cents / 100.0) * fill_count
                        
                        exit_info['exit_price'] = avg_exit
                        print(f"  ✓ SOLD {fill_count} @ {avg_exit*100:.0f}¢ (fees: ~{fee_cents:.2f}¢)")
                        
                        # === POST-TRADE VERIFICATION ===
                        verified_qty = self._verify_poly_position(slug, -fill_count, action='sell')
                        if verified_qty is not None and verified_qty > 0:
                            remaining = verified_qty
                            print(f"  ⚠️ PARTIAL EXIT: {remaining} contracts still on Polymarket")
                            position['live_fill_count'] = remaining
                            return  # Stay open, retry next cycle
                        elif verified_qty == 0:
                            print(f"  ✓ VERIFIED: Position fully closed on Polymarket")
                        
                        if fill_count < live_fill_count:
                            remaining = live_fill_count - fill_count
                            print(f"  ⚠️ PARTIAL FILL: Sold {fill_count}, {remaining} remaining")
                            position['live_fill_count'] = remaining
                            return  # Retry next cycle
                    else:
                        # SDK may report 0 fills even when order executed — verify via positions API
                        import time
                        time.sleep(0.5)
                        verify_positions = self._get_poly_positions()
                        if verify_positions is not None and (slug not in verify_positions or verify_positions[slug]['quantity'] == 0):
                            print(f"  ✓ VERIFIED: Position closed on Polymarket (SDK reported 0 fills but position gone)")
                            # Use the limit price as exit price since we can't get actual fill price
                            if sell_intent == 'ORDER_INTENT_SELL_SHORT':
                                avg_exit = 1.0 - (price_cents / 100.0)
                            else:
                                avg_exit = price_cents / 100.0
                            fee_cents = poly_fee_cents(price_cents / 100.0) * live_fill_count
                            exit_info['exit_price'] = avg_exit
                            print(f"  ✓ SOLD {live_fill_count} @ ~{avg_exit*100:.0f}¢ (fees: ~{fee_cents:.2f}¢)")
                        else:
                            print(f"  ⚠️ POLY SELL NOT FILLED - keeping position tracked")
                            return
                except Exception as e:
                    print(f"  ⚠️ POLY SELL ERROR: {e} - keeping position tracked")
                    return
        
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
        
        # Calculate fees based on venue
        if venue == 'Polymarket':
            # SHORT positions store complemented home prices; fees are on raw CLOB (away) price
            ct = position.get('contract_type', f'{side}_yes')
            raw_entry = (1.0 - entry_price) if ct == 'home_yes' else entry_price
            raw_exit = (1.0 - exit_price) if ct == 'home_yes' else exit_price
            entry_fee = poly_fee_cents(raw_entry)
            exit_fee = 0 if is_settlement else poly_fee_cents(raw_exit)
        else:
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
        _w = get_terminal_width()
        _sw = min(_w - 2, 80)
        print("\n" + "═"*_sw)
        print("  📈 LIVE TRADER │ Dual-Venue Market Trading")
        print("═"*_sw)
        print(f"  Entry: 5%→15% edge (exp k=2) │ Spread: 3¢/2¢")
        print(f"  Exit: EV-based (OV={self.OV_SCALE}×N^{self.OV_EXPONENT}) │ Cooldown: {self.COOLDOWN_AFTER_EXIT}s game-time")
        print(f"  Haircut: k={self.HAIRCUT_K} × H(t,|m|), ramp {self.HAIRCUT_RAMP_HI//60}:{self.HAIRCUT_RAMP_HI%60:02d}→{self.HAIRCUT_RAMP_LO//60}:{self.HAIRCUT_RAMP_LO%60:02d}, |m|≤{self.HAIRCUT_MARGIN_MAX}")
        ws_str = "WS" if KALSHI_WS_AVAILABLE else "REST"
        poly_str = "WS" if self.use_poly_ws else ("REST" if self.poly_client else "✗")
        print(f"  Data: Kalshi={ws_str} │ Polymarket={poly_str} │ Venue: {self.venue}")
        if self.live_mode:
            print(f"  Contracts: {self.live_contracts}")
        print("═"*_sw + "\n")
        
        try:
            cycle_number = 0
            RECONCILE_EVERY_N_CYCLES = 60  # ~1 minute at 1-second polling
            
            while True:
                cycle_start = time.time()
                cycle_number += 1
                
                # Clear just_exited from previous tick
                self.just_exited.clear()
                
                # === WS HEALTH CHECK: Connection-level freshness ===
                WS_DEAD_SEC = 15  # No messages at all in 15s = connection dead
                
                if self.use_ws and self.ws_book:
                    age = time.time() - self.ws_book.last_any_message_ts if self.ws_book.last_any_message_ts > 0 else float('inf')
                    if age > WS_DEAD_SEC:
                        print(f"  🔄 Kalshi WS dead — no messages for {age:.0f}s — forcing reconnect...")
                        try:
                            self.ws_book.stop()
                            time.sleep(1)
                            self.ws_book.start()
                            time.sleep(1)
                            if self.ws_book.connected:
                                print("  ✓ Kalshi WS reconnected")
                                self._ws_subscribed_tickers.clear()
                            else:
                                print("  ⚠️ Kalshi WS reconnect failed — will retry next cycle")
                        except Exception as e:
                            print(f"  ⚠️ Kalshi WS restart error: {e}")
                
                if self.use_poly_ws and self.poly_ws_book:
                    age = time.time() - self.poly_ws_book.last_any_message_ts if self.poly_ws_book.last_any_message_ts > 0 else float('inf')
                    if age > WS_DEAD_SEC:
                        print(f"  🔄 Poly WS dead — no messages for {age:.0f}s — forcing reconnect...")
                        try:
                            self.poly_ws_book.stop()
                            time.sleep(1)
                            self.poly_ws_book.start()
                            time.sleep(1)
                            if self.poly_ws_book.connected:
                                print("  ✓ Poly WS reconnected")
                                self._poly_ws_subscribed_slugs.clear()
                            else:
                                print("  ⚠️ Poly WS reconnect failed — will retry next cycle")
                        except Exception as e:
                            print(f"  ⚠️ Poly WS restart error: {e}")
                
                # Fetch all data in PARALLEL for lower latency
                kalshi_portfolio = None
                poly_portfolio = None
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {
                        executor.submit(self.scan_games): 'espn',
                    }
                    # Kalshi: use REST if WS is dead, or every 10 min for new market discovery
                    kalshi_ws_alive = self.use_ws and self.ws_book and self.ws_book.connected
                    if not kalshi_ws_alive or cycle_number % 600 == 0:
                        futures[executor.submit(self._load_kalshi_odds)] = 'kalshi'
                    # Polymarket: use REST if WS is dead, or every 10 min for new market discovery
                    poly_ws_alive = self.use_poly_ws and self.poly_ws_book and self.poly_ws_book.connected
                    if self.poly_client and (not poly_ws_alive or cycle_number % 600 == 0):
                        futures[executor.submit(self._load_polymarket_odds)] = 'polymarket'
                    # Add portfolio fetches for live mode
                    if self.live_mode:
                        futures[executor.submit(self._get_kalshi_portfolio)] = 'portfolio'
                        if self.poly_client:
                            futures[executor.submit(self._get_polymarket_portfolio)] = 'poly_portfolio'
                    
                    games = []
                    for future in as_completed(futures):
                        name = futures[future]
                        try:
                            result = future.result()
                            if name == 'espn':
                                games = result
                            elif name == 'portfolio':
                                kalshi_portfolio = result
                            elif name == 'poly_portfolio':
                                poly_portfolio = result
                        except Exception as e:
                            print(f"  ⚠️ {name} fetch error: {e}")
                
                live_games = [g for g in games if g['status'] == 'in' and 'live_home_prob' in g]
                upcoming_games = [g for g in games if g['status'] == 'pre' and 'pregame_home_prob' in g]
                finished_games = [g for g in games if g['status'] == 'post']
                
                # Cache Kalshi positions once per cycle (avoid repeated API calls)
                self._cached_kalshi_positions = None
                if self.live_mode and self.kalshi_client:
                    try:
                        self._cached_kalshi_positions = self.kalshi_client.get_positions()
                    except Exception as e:
                        print(f"  ⚠️ Failed to cache Kalshi positions: {e}")
                
                # Cache Polymarket positions once per cycle
                self._cached_poly_positions = None
                if self.live_mode and self.poly_client:
                    try:
                        self._cached_poly_positions = self._get_poly_positions()
                    except Exception as e:
                        print(f"  ⚠️ Failed to cache Poly positions: {e}")
                
                # Periodic reconciliation: correct drift between tracked and actual positions
                if self.live_mode and cycle_number % RECONCILE_EVERY_N_CYCLES == 0:
                    self._reconcile_positions()
                
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
                
                # Get responsive layout params
                L = get_layout()
                W = L['sep_width']
                TW = L['team_width']
                MW = L['matchup_width']
                compact = L['compact']
                
                # Clean up cooldowns for finished games
                active_game_ids = {g['game_id'] for g in live_games + upcoming_games}
                expired = [key for key in self.exit_cooldowns.keys() if key[0] not in active_game_ids]
                for key in expired:
                    del self.exit_cooldowns[key]
                
                # Helper to get bid for a contract type
                def get_contract_bid(odds, contract_type, market_source):
                    """Get the bid price for a specific contract type on the given venue"""
                    if market_source == 'Polymarket':
                        poly_bid_map = {'home_yes': 'poly_home_bid', 'away_yes': 'poly_away_bid'}
                        bid_key = poly_bid_map.get(contract_type)
                        if bid_key and odds.get(bid_key):
                            return odds[bid_key] / 100.0
                        # Fallback: if Poly-only dict, standard keys have Poly prices
                        bid_key = f'{contract_type.split("_")[0]}_yes_bid'
                        return odds.get(bid_key, 0) / 100.0 if odds.get(bid_key) else 0
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
                
                # kalshi_portfolio and poly_portfolio already fetched in parallel above
                
                print(f"\n{'═'*W}")
                # Real ROI = profit / peak capital deployed
                roi_pct = (self.realized_pnl / self.peak_capital_deployed * 100) if self.peak_capital_deployed > 0 else 0
                kalshi_count = len(self.kalshi_odds) // 2
                poly_count = len(self.polymarket_odds) // 2 if self.polymarket_odds else 0
                
                # Compact header with ROI
                time_str = datetime.now().strftime('%H:%M:%S')
                
                if self.live_mode:
                    ws_str = " │ WS" if self.use_ws else ""
                    
                    # Build venue-specific portfolio lines
                    k_line = ""
                    p_line = ""
                    combined_total = 0.0
                    combined_positions = 0
                    
                    if kalshi_portfolio:
                        session_pnl = kalshi_portfolio['session_pnl']
                        pnl_str = f"+${session_pnl:.2f}" if session_pnl >= 0 else f"-${abs(session_pnl):.2f}"
                        if compact:
                            k_line = f"K:${kalshi_portfolio['total']:.2f} ({pnl_str})"
                        else:
                            k_line = f"K: ${kalshi_portfolio['total']:.2f} (${kalshi_portfolio['cash']:.2f} cash + ${kalshi_portfolio['position_value']:.2f} pos×{kalshi_portfolio['position_count']}) {pnl_str}"
                        combined_total += kalshi_portfolio['total']
                        combined_positions += kalshi_portfolio['position_count']
                    
                    if poly_portfolio:
                        if compact:
                            p_line = f"P:${poly_portfolio['total']:.2f}"
                        else:
                            p_line = f"P: ${poly_portfolio['total']:.2f} (${poly_portfolio['cash']:.2f} cash + ${poly_portfolio['position_value']:.2f} pos×{poly_portfolio['position_count']})"
                        combined_total += poly_portfolio['total']
                        combined_positions += poly_portfolio['position_count']
                    
                    # Print header
                    if k_line and p_line:
                        # Both venues active — show combined + per-venue
                        if compact:
                            print(f"  🔴 {time_str} │ ${combined_total:.2f} │ {combined_positions}pos │ K:{kalshi_count} P:{poly_count}")
                            print(f"     {k_line} │ {p_line}")
                        else:
                            print(f"  🔴 LIVE │ {time_str} │ COMBINED: ${combined_total:.2f} ({combined_positions} pos) │ Markets: K:{kalshi_count} P:{poly_count}{ws_str}")
                            print(f"     {k_line}")
                            print(f"     {p_line}")
                    elif k_line:
                        if compact:
                            print(f"  🔴 {time_str} │ {k_line}{ws_str}")
                        else:
                            print(f"  🔴 LIVE │ {time_str} │ {k_line}{ws_str}")
                    elif p_line:
                        if compact:
                            print(f"  🔴 {time_str} │ {p_line}{ws_str}")
                        else:
                            print(f"  🔴 LIVE │ {time_str} │ {p_line}{ws_str}")
                    else:
                        print(f"  🔴 LIVE │ {time_str} │ K:{kalshi_count} P:{poly_count} │ (portfolio fetch failed){ws_str}")
                else:
                    roi_str = f" ({roi_pct:+.1f}% ROI)" if self.trade_count > 0 else ""
                    print(f"  PAPER TRADER │ {time_str} │ K:{kalshi_count} P:{poly_count} │ {len(self.open_positions)} pos │ Net: {self.realized_pnl:+.0f}¢{roi_str} ({self.trade_count} trades) │ GrossUnreal: {unrealized_pnl:+.0f}¢")
                print(f"{'═'*W}")
                
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
                                
                                entry_price = pos['entry_price']
                                
                                pnl_cents = (current_bid - entry_price) * 100
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
                        
                        # Kalshi asks
                        k_home_yes = odds.get('home_yes_ask')
                        k_away_no = odds.get('away_no_ask')
                        k_home_ask = min(filter(None, [k_home_yes, k_away_no]), default=None)
                        
                        k_away_yes = odds.get('away_yes_ask')
                        k_home_no = odds.get('home_no_ask')
                        k_away_ask = min(filter(None, [k_away_yes, k_home_no]), default=None)
                        
                        # Polymarket asks
                        p_home_ask = odds.get('poly_home_ask') if odds.get('has_polymarket') else None
                        p_away_ask = odds.get('poly_away_ask') if odds.get('has_polymarket') else None
                        
                        # Best ask across venues
                        home_asks = [x for x in [k_home_ask, p_home_ask] if x]
                        away_asks = [x for x in [k_away_ask, p_away_ask] if x]
                        home_ask = min(home_asks) if home_asks else None
                        away_ask = min(away_asks) if away_asks else None
                        
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
                    if compact:
                        print(f"  {'MATCHUP':<{MW}} {'SCORE':^7} {'TIME':>6} {'MDL':^5} {'─K─':^8} {'─P─':^8} {'EDGE':>6}")
                        print(f"  {'─'*(MW+44)}")
                    else:
                        print(f"  {'MATCHUP':<40} {'SCORE':^8} {'TIME':>8} {'MODEL':^7} {'─KALSHI─':^10} {'──POLY──':^10} {'EDGE':>7}")
                        print(f"  {'─'*96}")
                    
                    games_with_edge = []
                    for game in live_games:
                        odds = self._get_best_market_odds(game['home_team'], game['away_team'])
                        best_edge = None
                        best_side = None
                        
                        if odds:
                            our_home = game['live_home_prob']
                            
                            # Get best Kalshi ask for HOME (min of home_yes_ask, away_no_ask)
                            k_home_yes = odds.get('home_yes_ask')
                            k_away_no = odds.get('away_no_ask')
                            k_home_ask = min(filter(None, [k_home_yes, k_away_no]), default=None)
                            
                            # Get best Kalshi ask for AWAY (min of away_yes_ask, home_no_ask)
                            k_away_yes = odds.get('away_yes_ask')
                            k_home_no = odds.get('home_no_ask')
                            k_away_ask = min(filter(None, [k_away_yes, k_home_no]), default=None)
                            
                            # Get Polymarket asks
                            p_home_ask = odds.get('poly_home_ask') if odds.get('has_polymarket') else None
                            p_away_ask = odds.get('poly_away_ask') if odds.get('has_polymarket') else None
                            
                            # Best ask across venues for each side
                            home_asks = [x for x in [k_home_ask, p_home_ask] if x]
                            away_asks = [x for x in [k_away_ask, p_away_ask] if x]
                            best_home_ask = min(home_asks) if home_asks else None
                            best_away_ask = min(away_asks) if away_asks else None
                            
                            # Calculate edges from best ask
                            if best_home_ask:
                                home_edge = our_home - (best_home_ask / 100.0)
                                if best_edge is None or home_edge > (best_edge or -999):
                                    best_edge = home_edge
                                    best_side = 'home'
                            if best_away_ask:
                                away_edge = (1 - our_home) - (best_away_ask / 100.0)
                                if best_edge is None or away_edge > (best_edge or -999):
                                    best_edge = away_edge
                                    best_side = 'away'
                            
                            # If no positive edge, show the better negative one
                            if best_edge is None:
                                if best_home_ask and best_away_ask:
                                    home_e = our_home - (best_home_ask / 100.0)
                                    away_e = (1 - our_home) - (best_away_ask / 100.0)
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
                        away = strip_mascot(game['away_team'])[:TW]
                        home = strip_mascot(game['home_team'])[:TW]
                        if compact:
                            matchup = f"{away:<{TW}} @ {home}"
                        else:
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
                        
                        k_str = "   --"
                        p_str = "   --"
                        edge_str = "   --"
                        
                        if odds:
                            # Get Kalshi prices for the edge side (best of YES vs NO)
                            if edge_side == 'home':
                                k_yes_ask = odds.get('home_yes_ask')
                                k_no_ask = odds.get('away_no_ask')
                                k_yes_bid = odds.get('home_yes_bid')
                                k_no_bid = odds.get('away_no_bid')
                                p_ask_raw = odds.get('poly_home_ask') if odds.get('has_polymarket') else None
                                p_bid_raw = odds.get('poly_home_bid') if odds.get('has_polymarket') else None
                            else:
                                k_yes_ask = odds.get('away_yes_ask')
                                k_no_ask = odds.get('home_no_ask')
                                k_yes_bid = odds.get('away_yes_bid')
                                k_no_bid = odds.get('home_no_bid')
                                p_ask_raw = odds.get('poly_away_ask') if odds.get('has_polymarket') else None
                                p_bid_raw = odds.get('poly_away_bid') if odds.get('has_polymarket') else None
                            
                            # Pick best Kalshi price (YES vs NO)
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
                            
                            # Determine best ask across venues and add star
                            k_star = ""
                            p_star = ""
                            if k_ask and p_ask_raw:
                                if k_ask <= p_ask_raw:
                                    k_star = "★"
                                else:
                                    p_star = "★"
                            elif k_ask:
                                k_star = "★"
                            elif p_ask_raw:
                                p_star = "★"
                            
                            # Format Kalshi column
                            if k_ask and k_bid:
                                k_spread = k_ask - k_bid
                                k_str = f"{k_ask:2.0f}¢({k_spread:.0f}){k_star}"
                            elif k_ask:
                                k_str = f"{k_ask:2.0f}¢   {k_star}"
                            
                            # Format Poly column
                            if p_ask_raw and p_bid_raw:
                                p_spread = p_ask_raw - p_bid_raw
                                p_str = f"{p_ask_raw:2.0f}¢({p_spread:.0f}){p_star}"
                            elif p_ask_raw:
                                p_str = f"{p_ask_raw:2.0f}¢   {p_star}"
                            
                            # Edge from best book
                            if edge is not None:
                                edge_str = f"{edge*100:+5.1f}%"
                        
                        if compact:
                            time_short = f"{game['period']}{game['clock']:>5}"
                            print(f"  {matchup:<{MW}} {score:^7} {time_short:>6} {model_str:^5} {k_str:^8} {p_str:^8} {edge_str:>6}")
                        else:
                            print(f"  {matchup:<40} {score:^8} {time_str:>8} {model_str:^7} {k_str:^10} {p_str:^10} {edge_str:>7}")
                        
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
                    if compact:
                        print(f"  {'GAME':<{MW}} {'TY':<4} {'ENT':>4} {'BID':>5} {'SL@':>5} {'GRS':>5}")
                        print(f"  {'-'*(MW+27)}")
                    else:
                        print(f"  {'GAME':<40} {'TYPE':<8} {'ENTRY':>6} {'K BID':>7} {'P BID':>7} {'SELL@':>6} {'GROSS':>6} {'STATUS':<10}")
                        print(f"  {'-'*100}")
                    
                    for p in live_positions:
                        pos = p['pos']
                        current_game = p['game']
                        pnl_cents = p['pnl_cents']
                        
                        away = strip_mascot(current_game['away_team'])[:TW]
                        home = strip_mascot(current_game['home_team'])[:TW]
                        if compact:
                            matchup = f"{away:<{TW}} @ {home}"
                        else:
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
                        abs_margin = abs(current_game.get('home_score', 0) - current_game.get('away_score', 0))
                        min_sell = self.compute_min_sell_bid(time_rem, our_prob, abs_margin, venue=p['market_source'])
                        
                        # Get bids from both venues
                        odds = p['odds']
                        k_bid_cents = 0
                        p_bid_cents = 0
                        
                        if odds:
                            # Kalshi bid for this contract type
                            k_bid_map = {
                                'home_yes': 'home_yes_bid', 'home_no': 'home_no_bid',
                                'away_yes': 'away_yes_bid', 'away_no': 'away_no_bid'
                            }
                            k_bid_key = k_bid_map.get(p['contract_type'], 'home_yes_bid')
                            k_bid_cents = odds.get(k_bid_key, 0) or 0
                            
                            # Polymarket bid (YES contracts only, mapped to our side)
                            if odds.get('has_polymarket'):
                                if side == 'home':
                                    p_bid_cents = odds.get('poly_home_bid', 0) or 0
                                else:
                                    p_bid_cents = odds.get('poly_away_bid', 0) or 0
                        
                        # Star the venue we're actually on
                        k_star = "★" if p['market_source'] == 'Kalshi' else ""
                        p_star = "★" if p['market_source'] == 'Polymarket' else ""
                        
                        k_bid_str = f"{k_bid_cents:4.0f}¢{k_star}" if k_bid_cents > 0 else "   --"
                        p_bid_str = f"{p_bid_cents:4.0f}¢{p_star}" if p_bid_cents > 0 else "   --"
                        
                        entry_display = pos['entry_price'] * 100
                        
                        # Display gross: starred bid minus entry
                        our_bid_cents = k_bid_cents if p['market_source'] == 'Kalshi' else p_bid_cents
                        gross_display = our_bid_cents - entry_display
                        
                        # Status based on display gross
                        if self.STOP_LOSS_CENTS and gross_display <= -self.STOP_LOSS_CENTS:
                            status = "⚠️ STOP"
                        elif gross_display > 5:
                            status = "✅ UP"
                        elif gross_display < -5:
                            status = "❌ DOWN"
                        else:
                            status = "⬜ FLAT"
                        
                        if compact:
                            # Pre-format to fixed widths for alignment
                            ty_str = f"{type_label}{venue}"
                            ent_str = f"{entry_display:.0f}¢"
                            bid_str = f"{our_bid_cents:.0f}¢★" if our_bid_cents > 0 else "--"
                            sl_str = f"{min_sell:.0f}¢"
                            grs_str = f"{gross_display:+.0f}¢"
                            print(f"  {matchup:<{MW}} {ty_str:<4} {ent_str:>4} {bid_str:>5} {sl_str:>5} {grs_str:>5}")
                        else:
                            print(f"  {matchup:<40} {type_label}{venue:<5} {entry_display:5.0f}¢ {k_bid_str:>7} {p_bid_str:>7} {min_sell:5.0f}¢ {gross_display:+5.0f}¢ {status:<10}")
                
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
                
                print(f"\n{'═'*W}")
                # Real ROI = profit / peak capital deployed
                roi_pct = (self.realized_pnl / self.peak_capital_deployed * 100) if self.peak_capital_deployed > 0 else 0
                roi_str = f" ({roi_pct:+.1f}% ROI)" if self.trade_count > 0 else ""
                
                # Smart sleep: only wait the remainder of poll_interval
                elapsed = time.time() - cycle_start
                sleep_time = max(1.0 - elapsed, 0)
                cycle_info = f"cycle: {elapsed:.1f}s"
                
                if self.live_mode and (kalshi_portfolio or poly_portfolio):
                    combined_total = 0.0
                    combined_positions = 0
                    parts = []
                    if kalshi_portfolio:
                        combined_total += kalshi_portfolio['total']
                        combined_positions += kalshi_portfolio['position_count']
                        session_pnl = kalshi_portfolio['session_pnl']
                        pnl_str = f"+${session_pnl:.2f}" if session_pnl >= 0 else f"-${abs(session_pnl):.2f}"
                        parts.append(f"K:${kalshi_portfolio['total']:.2f} ({pnl_str})")
                    if poly_portfolio:
                        combined_total += poly_portfolio['total']
                        combined_positions += poly_portfolio['position_count']
                        parts.append(f"P:${poly_portfolio['total']:.2f}")
                    venue_str = " │ ".join(parts)
                    if compact:
                        print(f"  📊 ${combined_total:.2f} │ {venue_str} │ {combined_positions} pos │ {cycle_info}")
                    else:
                        print(f"  📊 Combined: ${combined_total:.2f} │ {venue_str} │ {combined_positions} positions │ {cycle_info} │ Ctrl+C to stop")
                else:
                    print(f"  📊 Net: {self.realized_pnl:+.0f}¢{roi_str} realized | {len(self.open_positions)} open ({unrealized_pnl:+.0f}¢ gross) | {self.trade_count} trades | {cycle_info} | Ctrl+C to stop")
                
                time.sleep(max(1.5 - (time.time() - cycle_start), 0.1))
                
        except KeyboardInterrupt:
            # Clean up WebSockets
            if self.ws_book:
                self.ws_book.stop()
            if self.poly_ws_book:
                self.poly_ws_book.stop()
            print("\n\nStopped. Use --review to see logged trades.")
    
    def _handle_entry(self, opp: dict):
        """Handle entry into a new position"""
        game_id = opp['game_id']
        side = opp['side']
        entry_price = opp['entry_price']
        status = opp.get('status', 'in')
        market_source = opp.get('market_source', 'Kalshi')
        contract_type = opp.get('contract_type', f'{side}_yes')  # e.g., home_yes, away_no
        
        # === SAFETY: Check for existing position on same EVENT (not just same game_id) ===
        # Prevents doubling up via different contract types (e.g., home_no + away_yes)
        opp_ticker = opp.get('market_ticker')
        if opp_ticker:
            opp_stem = self._ticker_event_stem(opp_ticker)
            if opp_stem:
                for pos_key, pos in self.open_positions.items():
                    pos_ticker = pos.get('market_ticker')
                    if pos_ticker:
                        pos_stem = self._ticker_event_stem(pos_ticker)
                        if pos_stem == opp_stem:
                            existing_ct = pos.get('contract_type', 'unknown')
                            if existing_ct != contract_type:
                                print(f"\n  🛡️ BLOCKED: Already have {existing_ct} on this event, won't add {contract_type}")
                                return
                            # Same contract type = top-up, allowed to continue
        
        # === LIVE TRADING: Place order FIRST ===
        live_fill_count = 0
        live_avg_price = None
        live_order_id = None
        
        if self.live_mode and self.kalshi_client and market_source == 'Kalshi':
            ticker = opp.get('market_ticker')
            if ticker:
                # === HARD CAP: Fresh Kalshi query as SOLE source of truth ===
                contracts_to_order = self.live_contracts
                try:
                    fresh_positions = self.kalshi_client.get_positions()  # Returns Dict[str, Position]
                    opp_stem = self._ticker_event_stem(ticker)
                    
                    # Count ALL contracts across ALL tickers on this event
                    total_event_contracts = 0
                    for k_ticker, k_pos in fresh_positions.items():
                        if k_pos.yes_count != 0:
                            k_stem = self._ticker_event_stem(k_ticker)
                            if k_stem and k_stem == opp_stem:
                                total_event_contracts += abs(k_pos.yes_count)
                    
                    if total_event_contracts >= self.live_contracts:
                        print(f"\n  🛡️ HARD CAP: Already {total_event_contracts} contracts on event (target={self.live_contracts})")
                        # Register in open_positions so we stop checking
                        if game_id not in self.open_positions:
                            self.open_positions[game_id] = {
                                'trade_id': None, 'side': side, 'entry_price': entry_price,
                                'entry_time': None, 'entry_status': 'unknown',
                                'home_team': opp['home_team'], 'away_team': opp['away_team'],
                                'market_ticker': ticker, 'market_source': 'Kalshi',
                                'contract_type': contract_type,
                                'live_fill_count': total_event_contracts,
                                'live_avg_price': None, 'live_order_id': None
                            }
                        return
                    
                    contracts_to_order = min(contracts_to_order, self.live_contracts - total_event_contracts)
                    if contracts_to_order <= 0:
                        return
                    
                    if total_event_contracts > 0:
                        print(f"  📈 TOP-UP: {total_event_contracts}/{self.live_contracts} on Kalshi, ordering {contracts_to_order} more")
                        
                except Exception as e:
                    print(f"  🛡️ Fresh position check failed: {e} — BLOCKING order for safety")
                    return  # If we can't verify, don't risk over-accumulation
                
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
        
        # === LIVE TRADING: Polymarket ===
        if self.live_mode and self.poly_client and market_source == 'Polymarket':
            slug = opp.get('market_ticker')  # For Poly, market_ticker holds the slug
            if slug:
                contracts_to_order = self.live_contracts
                
                # === HARD CAP: Fresh Polymarket query as SOLE source of truth ===
                # Same safety posture as Kalshi: query API before every order
                try:
                    poly_positions = self._get_poly_positions()
                    if poly_positions is None:
                        print(f"\n  🛡️ Poly position check failed (API unavailable) — BLOCKING order for safety")
                        return  # If we can't verify, don't risk over-accumulation
                    
                    # Count existing contracts on this slug
                    existing_poly_count = 0
                    if slug in poly_positions:
                        existing_poly_count = poly_positions[slug]['quantity']
                    
                    if existing_poly_count >= self.live_contracts:
                        print(f"\n  🛡️ HARD CAP: Already {existing_poly_count} contracts on Polymarket (target={self.live_contracts})")
                        # Register in open_positions so we stop checking
                        if game_id not in self.open_positions:
                            self.open_positions[game_id] = {
                                'trade_id': None, 'side': side, 'entry_price': entry_price,
                                'entry_time': None, 'entry_status': 'unknown',
                                'home_team': opp['home_team'], 'away_team': opp['away_team'],
                                'market_ticker': slug, 'market_source': 'Polymarket',
                                'contract_type': contract_type,
                                'live_fill_count': existing_poly_count,
                                'live_avg_price': None, 'live_order_id': None
                            }
                        return
                    
                    contracts_to_order = min(contracts_to_order, self.live_contracts - existing_poly_count)
                    if contracts_to_order <= 0:
                        return
                    
                    if existing_poly_count > 0:
                        print(f"  📈 TOP-UP: {existing_poly_count}/{self.live_contracts} on Polymarket, ordering {contracts_to_order} more")
                    
                except Exception as e:
                    print(f"  🛡️ Fresh Poly position check failed: {e} — BLOCKING order for safety")
                    return  # If we can't verify, don't risk over-accumulation
                
                # Map side to Polymarket intent
                # BBO is for outcome team (away in CBB)
                # home_yes → BUY_SHORT (bet against outcome team = bet home)
                # away_yes → BUY_LONG (bet on outcome team = bet away)
                if 'away' in contract_type:
                    intent = 'ORDER_INTENT_BUY_LONG'
                else:
                    intent = 'ORDER_INTENT_BUY_SHORT'
                
                # Smart price walk-up (keep full float precision from BBO)
                home_price_cents = entry_price * 100
                time_remaining = opp.get('time_remaining_sec', 2400)
                required_edge = self.get_required_edge(entry_price, time_remaining)
                current_edge = opp['our_prob'] - entry_price
                
                buffer = 0
                if current_edge - 0.01 >= required_edge:
                    buffer = 1
                    if current_edge - 0.02 >= required_edge:
                        buffer = 2
                
                if 'away' in contract_type:
                    # BUY_LONG: price = away ask, higher = more aggressive
                    raw_price_cents = home_price_cents
                    price_cents = min(99, raw_price_cents + buffer)
                    buf_str = f" +{buffer}¢ buf" if buffer > 0 else ""
                else:
                    # BUY_SHORT: SDK sends SELL on away token
                    # Must complement price: home 55¢ → sell away at 45¢ floor
                    # Lower floor = more aggressive for sells
                    raw_price_cents = 100 - home_price_cents
                    price_cents = max(1, raw_price_cents - buffer)
                
                price_decimal = price_cents / 100.0
                bet_team = opp['home_team'][:20] if side == 'home' else opp['away_team'][:20]
                if intent == 'ORDER_INTENT_BUY_SHORT':
                    print(f"\n  🟣 POLY BUY SHORT: {bet_team} @ {home_price_cents:.1f}¢ (CLOB: sell away @ {price_cents:.1f}¢{f' -{buffer}¢ buf' if buffer else ''}) x {contracts_to_order}")
                else:
                    print(f"\n  🟣 POLY BUY LONG: {bet_team} @ {price_cents:.1f}¢{f' (+{buffer}¢ buf)' if buffer else ''} x {contracts_to_order}")
                
                try:
                    result = self.poly_client.orders.create({
                        "marketSlug": slug,
                        "intent": intent,
                        "type": "ORDER_TYPE_LIMIT",
                        "price": {"value": f"{price_cents / 100:.4f}", "currency": "USD"},
                        "quantity": contracts_to_order,
                        "tif": "TIME_IN_FORCE_IMMEDIATE_OR_CANCEL",
                    })
                    
                    # SDK returns 'executions' not 'fills'
                    fills = result.get('executions', []) or result.get('fills', [])
                    if fills:
                        print(f"  📋 Execution[0] keys: {list(fills[0].keys()) if isinstance(fills[0], dict) else type(fills[0])}")
                    fill_count = sum(int(float(f.get('quantity', 0))) if isinstance(f.get('quantity', 0), str) else f.get('quantity', 0) for f in fills)
                    poly_pos = None  # Will be populated if needed
                    
                    # Fallback: check positions API if no fills parsed
                    if fill_count == 0:
                        print(f"  📋 Order response keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
                        # Check for alternative fill indicators
                        status_str = result.get('status', '') if isinstance(result, dict) else ''
                        qty_filled = result.get('quantityFilled') or result.get('filledQuantity') or result.get('filled_quantity')
                        if qty_filled:
                            fill_count = int(float(qty_filled))
                            print(f"  📋 Found fill via quantityFilled: {fill_count}")
                        elif status_str and 'FILL' in status_str.upper():
                            # Status indicates fill, verify via positions API
                            print(f"  📋 Order status: {status_str}, checking positions...")
                            fill_count = contracts_to_order  # Assume full fill, verify below
                        
                        # Last resort: check positions API for actual holdings
                        if fill_count == 0:
                            import time as _time
                            _time.sleep(0.5)
                            poly_pos = self._get_poly_positions()
                            if poly_pos and slug in poly_pos:
                                actual_qty = poly_pos[slug]['quantity']
                                if actual_qty > 0:
                                    fill_count = min(actual_qty, contracts_to_order)
                                    print(f"  📋 Detected fill via positions API: {fill_count} contracts on {slug}")
                    
                    if fill_count > 0:
                        # Entry price in home-probability terms from our known limit.
                        # IOC fills at limit or not at all, so this is exact.
                        # Avoids ambiguity in SDK fill price semantics (CLOB vs buyer side).
                        if intent == 'ORDER_INTENT_BUY_SHORT':
                            # We sent SELL away @ price_cents (e.g. 80¢). Home equiv = complement.
                            avg_price_decimal = 1.0 - (price_cents / 100.0)
                        else:
                            # BUY_LONG: CLOB price IS the entry price
                            avg_price_decimal = price_cents / 100.0
                        
                        # Fees on raw CLOB price
                        fee_cents = poly_fee_cents(price_cents / 100.0) * fill_count
                        
                        live_fill_count = fill_count
                        live_avg_price = round(avg_price_decimal * 100, 1)
                        live_order_id = result.get('orderId') or result.get('id') or result.get('order_id')
                        entry_price = avg_price_decimal
                        opp['entry_price'] = entry_price
                        print(f"  ✓ FILLED {fill_count} @ {live_avg_price:.0f}¢ (fees: ~{fee_cents:.2f}¢)")
                        
                        # === POST-TRADE VERIFICATION ===
                        verified_qty = self._verify_poly_position(slug, fill_count, action='buy')
                        if verified_qty is not None and verified_qty == 0:
                            print(f"  ⚠️ ANOMALY: Fill reported but no position found on Polymarket!")
                            # Still log the trade, but flag it
                    else:
                        print(f"  ❌ NO FILL: Order not filled on Polymarket")
                        return
                except Exception as e:
                    print(f"  ❌ POLY ORDER ERROR: {e}")
                    return
        
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
    parser.add_argument('--venue', choices=['kalshi', 'polymarket', 'best'], default='best',
                        help='Trading venue: kalshi (Kalshi only), polymarket (Poly only), best (pick best net EV)')
    args = parser.parse_args()
    
    # Safety confirmation for live mode
    if args.live:
        print("⚠️  WARNING: LIVE TRADING MODE ⚠️")
        venues = {'kalshi': 'Kalshi', 'polymarket': 'Polymarket', 'best': 'Kalshi + Polymarket (best EV)'}
        print(f"This will place REAL orders with REAL money on {venues[args.venue]}!")
        print(f"Position size: {args.contracts} contracts per trade")
        confirm = input("\nType 'live' to confirm: ")
        if confirm != 'live':
            print("Cancelled.")
            sys.exit(0)
    
    trader = PaperTrader(args.db, live_mode=args.live, live_contracts=args.contracts, venue=args.venue)
    
    if args.review:
        trader.review_opportunities()
    elif args.analyze:
        trader.analyze_results()
    else:
        trader.run(poll_interval=args.interval)


if __name__ == "__main__":
    main()