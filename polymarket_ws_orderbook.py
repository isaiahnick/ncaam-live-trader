#!/usr/bin/env python3
"""
polymarket_ws_orderbook.py - Real-time Polymarket US Orderbook via WebSocket

Maintains in-memory BBO state from Polymarket US WebSocket stream.
Mirror architecture of kalshi_ws_orderbook.py for drop-in integration.

ARCHITECTURE:
- Runs in a background thread with asyncio event loop
- Subscribes to market_data_lite channels for active slugs
- Maintains BBO (best bid/ask) state per market
- Provides thread-safe access to current prices
- Tracks update timestamps for freshness gating

KEY DIFFERENCE FROM KALSHI:
- Polymarket has ONE market per game (not separate home/away tickers)
- BBO represents the outcome team (away team in CBB, ordering='away')
- LONG = away team wins, SHORT = home team wins
- get_prices() returns both home and away perspectives (derived from one BBO)

USAGE:
    from polymarket_ws_orderbook import PolymarketWSOrderbook
    
    ws_book = PolymarketWSOrderbook(key_id='...', secret_key='...')
    ws_book.start()
    
    ws_book.subscribe(['aec-cbb-iowa-mary-2026-02-11'])
    
    prices = ws_book.get_prices('aec-cbb-iowa-mary-2026-02-11')
    # Returns: {
    #   'away_bid': 85, 'away_ask': 87,   (cents, outcome team = away)
    #   'home_bid': 13, 'home_ask': 15,   (cents, derived: 100 - away)
    #   'spread': 2, 'last_update': ..., 'age_ms': 42
    # }
    
    ws_book.stop()

DOCS:
- Polymarket US WebSocket: wss://api.polymarket.us/v1/ws/markets
- Auth: Ed25519 HMAC via polymarket_us.auth.create_auth_headers
"""

import asyncio
import json
import threading
import time
from typing import Dict, Optional, List, Set
from dataclasses import dataclass

import websockets

# Use SDK's auth helper for Ed25519 signing
from polymarket_us.auth import create_auth_headers


@dataclass
class BBOState:
    """
    In-memory best-bid/offer state for a single Polymarket market.
    
    Polymarket CBB markets have a single outcome team (away, ordering='away').
    BBO is for that outcome team's LONG side.
    """
    slug: str
    away_bid: Optional[int] = None   # Best bid for outcome (away) team, in cents
    away_ask: Optional[int] = None   # Best ask for outcome (away) team, in cents
    last_trade: Optional[int] = None # Last trade price, in cents
    last_update_ts: float = 0.0      # Unix timestamp of last WS message
    has_data: bool = False
    
    @property
    def home_bid(self) -> Optional[int]:
        """Home bid = 100 - away ask (SHORT side bid)"""
        return (100 - self.away_ask) if self.away_ask is not None else None
    
    @property
    def home_ask(self) -> Optional[int]:
        """Home ask = 100 - away bid (SHORT side ask)"""
        return (100 - self.away_bid) if self.away_bid is not None else None
    
    @property
    def spread(self) -> Optional[int]:
        """Spread in cents"""
        if self.away_bid is not None and self.away_ask is not None:
            return self.away_ask - self.away_bid
        return None


class PolymarketWSOrderbook:
    """
    Maintains real-time BBO state via Polymarket US WebSocket.
    
    Thread-safe: runs asyncio loop in background thread, provides
    synchronized access to BBO state from main thread.
    
    Mirrors KalshiWSOrderbook interface for easy integration.
    """
    
    WS_URL = "wss://api.polymarket.us/v1/ws/markets"
    WS_PATH = "/v1/ws/markets"
    
    def __init__(self, key_id: str, secret_key: str, verbose: bool = False):
        """
        Initialize WebSocket orderbook manager.
        
        Args:
            key_id: Polymarket API key ID (UUID)
            secret_key: Base64-encoded Ed25519 secret key
            verbose: Print debug messages
        """
        self.key_id = key_id
        self.secret_key = secret_key
        self.verbose = verbose
        
        # Thread-safe state
        self._lock = threading.Lock()
        self._markets: Dict[str, BBOState] = {}
        self._subscribed_slugs: Set[str] = set()
        self._pending_subscriptions: Set[str] = set()
        self._request_counter = 0
        
        # Connection state
        self._ws = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False
        self.last_any_message_ts = 0.0
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 30.0
    
    def start(self):
        """Start background thread with WebSocket connection"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        # Wait for connection (up to 5 seconds)
        for _ in range(50):
            if self._connected:
                break
            time.sleep(0.1)
    
    def stop(self):
        """Stop WebSocket and background thread"""
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5.0)
    
    def subscribe(self, slugs: List[str]):
        """
        Subscribe to BBO updates for given market slugs.
        Thread-safe: can be called from main thread.
        """
        with self._lock:
            new_slugs = set(slugs) - self._subscribed_slugs
            self._pending_subscriptions.update(new_slugs)
            
            for slug in new_slugs:
                if slug not in self._markets:
                    self._markets[slug] = BBOState(slug=slug)
        
        # Trigger subscription in WS thread
        if self._loop and self._connected:
            self._loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self._send_subscriptions())
            )
    
    def unsubscribe(self, slugs: List[str]):
        """Unsubscribe from BBO updates"""
        with self._lock:
            for slug in slugs:
                self._subscribed_slugs.discard(slug)
                self._markets.pop(slug, None)
        
        if self._loop and self._connected:
            self._loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self._send_unsubscribe(slugs))
            )
    
    def get_prices(self, slug: str) -> Optional[Dict]:
        """
        Get current best prices for a market slug.
        Thread-safe: can be called from main thread.
        
        Returns dict with:
            away_bid, away_ask (cents, outcome/away team)
            home_bid, home_ask (cents, derived from away)
            spread (away_ask - away_bid in cents)
            last_update (Unix timestamp)
            age_ms (milliseconds since last update)
        """
        with self._lock:
            state = self._markets.get(slug)
            if not state or not state.has_data:
                return None
            
            now = time.time()
            return {
                'away_bid': state.away_bid,
                'away_ask': state.away_ask,
                'home_bid': state.home_bid,
                'home_ask': state.home_ask,
                'spread': state.spread,
                'last_trade': state.last_trade,
                'last_update': state.last_update_ts,
                'age_ms': int((now - state.last_update_ts) * 1000)
            }
    
    def get_update_age_ms(self, slug: str) -> Optional[int]:
        """Get milliseconds since last BBO update for slug"""
        with self._lock:
            state = self._markets.get(slug)
            if not state or state.last_update_ts == 0:
                return None
            return int((time.time() - state.last_update_ts) * 1000)
    
    def is_fresh(self, slug: str, max_age_ms: int = 500) -> bool:
        """Check if BBO data is fresh enough"""
        age = self.get_update_age_ms(slug)
        return age is not None and age <= max_age_ms
    
    def get_all_subscribed(self) -> List[str]:
        """Get list of all subscribed slugs"""
        with self._lock:
            return list(self._subscribed_slugs)
    
    @property
    def connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self._connected
    
    # ===== INTERNAL METHODS =====
    
    def _next_request_id(self) -> str:
        """Generate unique request ID"""
        self._request_counter += 1
        return f"poly-ws-{self._request_counter}-{int(time.time() * 1000)}"
    
    def _run_loop(self):
        """Background thread entry point"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._connection_loop())
        except Exception as e:
            if self.verbose:
                print(f"[POLY-WS] Loop error: {e}")
        finally:
            self._loop.close()
    
    async def _connection_loop(self):
        """Main connection loop with auto-reconnect"""
        while self._running:
            try:
                await self._connect_and_run()
            except Exception as e:
                if self.verbose:
                    print(f"[POLY-WS] Connection error: {e}")
                self._connected = False
            
            if self._running:
                if self.verbose:
                    print(f"[POLY-WS] Reconnecting in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2,
                    self._max_reconnect_delay
                )
    
    async def _connect_and_run(self):
        """Connect to WebSocket and process messages"""
        # Generate auth headers using SDK's Ed25519 signer
        headers = create_auth_headers(
            self.key_id,
            self.secret_key,
            "GET",
            self.WS_PATH
        )
        
        if self.verbose:
            print(f"[POLY-WS] Connecting to {self.WS_URL}...")
        
        async with websockets.connect(
            self.WS_URL,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=10
        ) as ws:
            self._ws = ws
            self._connected = True
            self._reconnect_delay = 1.0  # Reset on successful connect
            
            if self.verbose:
                print("[POLY-WS] Connected!")
            
            # Re-subscribe everything (reconnect scenario)
            with self._lock:
                self._pending_subscriptions.update(self._subscribed_slugs)
            await self._send_subscriptions()
            
            # Process messages
            async for message in ws:
                if not self._running:
                    break
                self._handle_message(message)
    
    async def _send_subscriptions(self):
        """Send subscription requests for pending slugs"""
        with self._lock:
            slugs_to_sub = list(self._pending_subscriptions)
            self._pending_subscriptions.clear()
        
        if not slugs_to_sub or not self._ws:
            return
        
        # Polymarket WS subscribe format
        request_id = self._next_request_id()
        msg = {
            "subscribe": {
                "requestId": request_id,
                "subscriptionType": "SUBSCRIPTION_TYPE_MARKET_DATA_LITE",
                "marketSlugs": slugs_to_sub
            }
        }
        
        if self.verbose:
            print(f"[POLY-WS] Subscribing to {len(slugs_to_sub)} markets")
        
        await self._ws.send(json.dumps(msg))
        
        with self._lock:
            self._subscribed_slugs.update(slugs_to_sub)
    
    async def _send_unsubscribe(self, slugs: List[str]):
        """Send unsubscribe request"""
        if not self._ws or not slugs:
            return
        
        # Find request IDs to unsubscribe — just use a new ID
        request_id = self._next_request_id()
        msg = {
            "unsubscribe": {
                "requestId": request_id
            }
        }
        
        await self._ws.send(json.dumps(msg))
    
    def _handle_message(self, raw: str):
        """Process incoming WebSocket message"""
        self.last_any_message_ts = time.time()
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        
        # BBO update (market_data_lite)
        if "marketDataLite" in msg:
            self._handle_bbo(msg)
        
        # Full orderbook (market_data) — extract BBO from it
        elif "marketData" in msg:
            self._handle_market_data(msg)
        
        # Heartbeat
        elif "heartbeat" in msg:
            pass  # Keepalive, nothing to do
        
        # Error
        elif "error" in msg:
            if self.verbose:
                print(f"[POLY-WS] Error: {msg.get('error')}")
    
    def _handle_bbo(self, msg: dict):
        """Handle market_data_lite (BBO) update
        
        Format:
        {
            "marketDataLite": {
                "marketSlug": "aec-cbb-iowa-mary-2026-02-11",
                "bestBid": {"value": "0.847", "currency": "USD"},
                "bestAsk": {"value": "0.850", "currency": "USD"},
                "lastTradePx": {"value": "0.849", "currency": "USD"}
            },
            "requestId": "...",
            "subscriptionType": "SUBSCRIPTION_TYPE_MARKET_DATA_LITE"
        }
        
        BBO is for the outcome team (away in CBB).
        """
        data = msg.get("marketDataLite", {})
        slug = data.get("marketSlug")
        
        if not slug:
            return
        
        best_bid = data.get("bestBid", {})
        best_ask = data.get("bestAsk", {})
        last_trade = data.get("lastTradePx", {})
        
        bid_val = best_bid.get("value") if best_bid else None
        ask_val = best_ask.get("value") if best_ask else None
        trade_val = last_trade.get("value") if last_trade else None
        
        if bid_val is None and ask_val is None:
            return
        
        with self._lock:
            state = self._markets.get(slug)
            if not state:
                state = BBOState(slug=slug)
                self._markets[slug] = state
            
            if bid_val is not None:
                state.away_bid = round(float(bid_val) * 100)
            if ask_val is not None:
                state.away_ask = round(float(ask_val) * 100)
            if trade_val is not None:
                state.last_trade = round(float(trade_val) * 100)
            
            state.last_update_ts = time.time()
            state.has_data = True
        
        if self.verbose:
            prices = self.get_prices(slug)
            if prices:
                print(f"[POLY-WS] BBO {slug}: Away {prices['away_bid']}/{prices['away_ask']} | Home {prices['home_bid']}/{prices['home_ask']} | age={prices['age_ms']}ms")
    
    def _handle_market_data(self, msg: dict):
        """Handle full market_data update — extract BBO from orderbook
        
        Format:
        {
            "marketData": {
                "marketSlug": "...",
                "bids": [{"px": {"value": "0.85"}, "qty": "10"}, ...],
                "offers": [{"px": {"value": "0.87"}, "qty": "5"}, ...],
                "state": "STATE_OPEN",
                "stats": {...}
            }
        }
        """
        data = msg.get("marketData", {})
        slug = data.get("marketSlug")
        
        if not slug:
            return
        
        bids = data.get("bids", [])
        offers = data.get("offers", [])
        
        # Extract best bid (highest) and best ask (lowest)
        best_bid = None
        best_ask = None
        
        if bids:
            bid_prices = [float(b.get("px", {}).get("value", 0)) for b in bids if b.get("px")]
            if bid_prices:
                best_bid = max(bid_prices)
        
        if offers:
            ask_prices = [float(a.get("px", {}).get("value", 0)) for a in offers if a.get("px")]
            if ask_prices:
                best_ask = min(ask_prices)
        
        if best_bid is None and best_ask is None:
            return
        
        with self._lock:
            state = self._markets.get(slug)
            if not state:
                state = BBOState(slug=slug)
                self._markets[slug] = state
            
            if best_bid is not None:
                state.away_bid = round(best_bid * 100)
            if best_ask is not None:
                state.away_ask = round(best_ask * 100)
            
            state.last_update_ts = time.time()
            state.has_data = True


# ===== STANDALONE TEST =====
if __name__ == "__main__":
    import os
    
    KEY_ID = os.environ.get("POLY_KEY_ID")
    SECRET_KEY = os.environ.get("POLY_SECRET_KEY")
    
    if not KEY_ID or not SECRET_KEY:
        print("Set POLY_KEY_ID and POLY_SECRET_KEY environment variables")
        exit(1)
    
    print("Starting Polymarket WebSocket orderbook manager...")
    
    ws_book = PolymarketWSOrderbook(KEY_ID, SECRET_KEY, verbose=True)
    ws_book.start()
    
    time.sleep(2)
    
    if not ws_book.connected:
        print("Failed to connect!")
        exit(1)
    
    # Subscribe to a test market
    test_slug = "aec-cbb-iowa-mary-2026-02-11"
    print(f"\nSubscribing to {test_slug}")
    ws_book.subscribe([test_slug])
    
    # Monitor for 30 seconds
    print("\nMonitoring for 30 seconds...")
    for i in range(30):
        prices = ws_book.get_prices(test_slug)
        if prices:
            print(f"  {i}: Away {prices['away_bid']}/{prices['away_ask']} | Home {prices['home_bid']}/{prices['home_ask']} | spread={prices['spread']}¢ | age={prices['age_ms']}ms | fresh={ws_book.is_fresh(test_slug, 500)}")
        else:
            print(f"  {i}: No data yet")
        time.sleep(1)
    
    ws_book.stop()
    print("\nDone!")