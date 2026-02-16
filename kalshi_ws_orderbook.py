#!/usr/bin/env python3
"""
kalshi_ws_orderbook.py - Real-time Kalshi Orderbook via WebSocket

Maintains in-memory orderbooks from Kalshi WebSocket stream to eliminate 
the latency gap between ESPN updates and REST polling.

ARCHITECTURE:
- Runs in a background thread with asyncio event loop
- Subscribes to orderbook channels for active tickers
- Maintains snapshot + delta state per market
- Provides thread-safe access to current best bid/ask
- Tracks update timestamps for freshness gating

USAGE:
    from kalshi_ws_orderbook import KalshiWSOrderbook
    
    # Initialize and start
    ws_book = KalshiWSOrderbook(api_key, private_key)
    ws_book.start()  # Starts background thread
    
    # Subscribe to markets you care about
    ws_book.subscribe(['KXNCAAMBGAME-26FEB03-DUKE-UCONNA', 'KXNCAAMBGAME-26FEB03-DUKE-UCONNB'])
    
    # Get current prices (thread-safe)
    prices = ws_book.get_prices('KXNCAAMBGAME-26FEB03-DUKE-UCONNA')
    # Returns: {'yes_bid': 55, 'yes_ask': 57, 'no_bid': 43, 'no_ask': 45, 'last_update': 1706900000.123}
    
    # Check freshness
    age_ms = ws_book.get_update_age_ms('KXNCAAMBGAME-26FEB03-DUKE-UCONNA')
    if age_ms > 500:  # Data more than 500ms stale
        pass  # Skip trade
    
    # Clean shutdown
    ws_book.stop()

DOCS:
- WebSocket Connection: https://docs.kalshi.com/websockets/websocket-connection
- Orderbook Updates: https://docs.kalshi.com/websockets/orderbook-updates
"""

import asyncio
import json
import threading
import time
import base64
from datetime import datetime
from typing import Dict, Optional, List, Set
from dataclasses import dataclass, field
import websockets
import requests
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key


@dataclass
class OrderbookState:
    """
    In-memory orderbook state for a single market.
    
    IMPORTANT: Kalshi only returns BIDS. In binary markets:
    - YES bid at price P implies NO ask at (100-P)
    - NO bid at price P implies YES ask at (100-P)
    
    So to find the best YES ask, we look at the lowest NO bid and compute 100 - price.
    """
    ticker: str
    yes_bids: Dict[int, int] = field(default_factory=dict)  # price -> quantity (BIDS only)
    no_bids: Dict[int, int] = field(default_factory=dict)   # price -> quantity (BIDS only)
    last_update_ts: float = 0.0  # Unix timestamp of last WS message
    last_seq: int = 0  # Sequence number for delta ordering
    has_snapshot: bool = False
    
    def best_yes_bid(self) -> Optional[int]:
        """Highest YES bid price (cents) - what you'd get selling YES"""
        return max(self.yes_bids.keys()) if self.yes_bids else None
    
    def best_yes_ask(self) -> Optional[int]:
        """Lowest YES ask price (cents) - what you'd pay buying YES
        
        Calculated from NO bids: someone bidding NO at X is asking YES at (100-X)
        """
        if not self.no_bids:
            return None
        # The highest NO bid gives the lowest YES ask
        highest_no_bid = max(self.no_bids.keys())
        return 100 - highest_no_bid
    
    def best_no_bid(self) -> Optional[int]:
        """Highest NO bid price (cents) - what you'd get selling NO"""
        return max(self.no_bids.keys()) if self.no_bids else None
    
    def best_no_ask(self) -> Optional[int]:
        """Lowest NO ask price (cents) - what you'd pay buying NO
        
        Calculated from YES bids: someone bidding YES at X is asking NO at (100-X)
        """
        if not self.yes_bids:
            return None
        # The highest YES bid gives the lowest NO ask
        highest_yes_bid = max(self.yes_bids.keys())
        return 100 - highest_yes_bid
    
    def apply_delta(self, side: str, price: int, delta: int):
        """Apply a quantity delta to the orderbook
        
        Args:
            side: 'yes' or 'no' - which side's bids to update
            price: Price level in cents
            delta: Quantity change (positive or negative)
        """
        book = self.yes_bids if side == 'yes' else self.no_bids
        if price in book:
            book[price] += delta
            if book[price] <= 0:
                del book[price]
        elif delta > 0:
            book[price] = delta
    
    def set_level(self, side: str, price: int, quantity: int):
        """Set absolute quantity at a price level (for snapshots)"""
        book = self.yes_bids if side == 'yes' else self.no_bids
        if quantity > 0:
            book[price] = quantity
        elif price in book:
            del book[price]
    
    def clear(self):
        """Clear all levels (before applying new snapshot)"""
        self.yes_bids.clear()
        self.no_bids.clear()
        self.has_snapshot = False


class KalshiWSOrderbook:
    """
    Maintains real-time orderbook state via Kalshi WebSocket.
    
    Thread-safe: runs asyncio loop in background thread, provides 
    synchronized access to orderbook state from main thread.
    """
    
    WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    
    def __init__(self, api_key: str, private_key_path: str, verbose: bool = False):
        """
        Initialize WebSocket orderbook manager.
        
        Args:
            api_key: Kalshi API key ID
            private_key_path: Path to PEM private key file
            verbose: Print debug messages
        """
        self.api_key = api_key
        self.private_key_path = private_key_path
        self.verbose = verbose
        
        # Thread-safe state
        self._lock = threading.Lock()
        self._orderbooks: Dict[str, OrderbookState] = {}
        self._subscribed_tickers: Set[str] = set()
        self._pending_subscriptions: Set[str] = set()
        
        # Connection state
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False
        self.last_any_message_ts = 0.0
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 30.0
        
        # Load private key
        with open(private_key_path, 'rb') as f:
            self._private_key = load_pem_private_key(f.read(), password=None)
    
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
    
    def subscribe(self, tickers: List[str]):
        """
        Subscribe to orderbook updates for given tickers.
        Thread-safe: can be called from main thread.
        """
        with self._lock:
            new_tickers = set(tickers) - self._subscribed_tickers
            self._pending_subscriptions.update(new_tickers)
            
            # Initialize orderbook state for new tickers
            for ticker in new_tickers:
                if ticker not in self._orderbooks:
                    self._orderbooks[ticker] = OrderbookState(ticker=ticker)
        
        # Trigger subscription in WS thread
        if self._loop and self._connected:
            self._loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self._send_subscriptions())
            )
    
    def unsubscribe(self, tickers: List[str]):
        """Unsubscribe from orderbook updates"""
        with self._lock:
            for ticker in tickers:
                self._subscribed_tickers.discard(ticker)
                self._orderbooks.pop(ticker, None)
        
        if self._loop and self._connected:
            self._loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self._send_unsubscribe(tickers))
            )
    
    def get_prices(self, ticker: str) -> Optional[Dict]:
        """
        Get current best prices for a ticker.
        Thread-safe: can be called from main thread.
        
        Returns dict with:
            yes_bid, yes_ask, no_bid, no_ask (all in cents, or None)
            spread (yes_ask - yes_bid in cents)
            last_update (Unix timestamp)
            age_ms (milliseconds since last update)
        """
        with self._lock:
            book = self._orderbooks.get(ticker)
            if not book or not book.has_snapshot:
                return None
            
            now = time.time()
            yes_bid = book.best_yes_bid()
            yes_ask = book.best_yes_ask()
            no_bid = book.best_no_bid()
            no_ask = book.best_no_ask()
            
            spread = None
            if yes_bid is not None and yes_ask is not None:
                spread = yes_ask - yes_bid
            
            return {
                'yes_bid': yes_bid,
                'yes_ask': yes_ask,
                'no_bid': no_bid,
                'no_ask': no_ask,
                'spread': spread,
                'last_update': book.last_update_ts,
                'age_ms': int((now - book.last_update_ts) * 1000)
            }
    
    def get_update_age_ms(self, ticker: str) -> Optional[int]:
        """Get milliseconds since last orderbook update for ticker"""
        with self._lock:
            book = self._orderbooks.get(ticker)
            if not book or book.last_update_ts == 0:
                return None
            return int((time.time() - book.last_update_ts) * 1000)
    
    def is_fresh(self, ticker: str, max_age_ms: int = 500) -> bool:
        """Check if orderbook data is fresh enough"""
        age = self.get_update_age_ms(ticker)
        return age is not None and age <= max_age_ms
    
    def get_all_subscribed(self) -> List[str]:
        """Get list of all subscribed tickers"""
        with self._lock:
            return list(self._subscribed_tickers)
    
    @property
    def connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self._connected
    
    # ===== INTERNAL METHODS =====
    
    def _run_loop(self):
        """Background thread entry point"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._connection_loop())
        except Exception as e:
            if self.verbose:
                print(f"[WS] Loop error: {e}")
        finally:
            self._loop.close()
    
    async def _connection_loop(self):
        """Main connection loop with auto-reconnect"""
        while self._running:
            try:
                await self._connect_and_run()
            except Exception as e:
                if self.verbose:
                    print(f"[WS] Connection error: {e}")
                self._connected = False
            
            if self._running:
                if self.verbose:
                    print(f"[WS] Reconnecting in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, 
                    self._max_reconnect_delay
                )
    
    async def _connect_and_run(self):
        """Connect to WebSocket and process messages"""
        # Generate auth signature using RSA-PSS (per Kalshi docs)
        # Must use time.time() not datetime.utcnow() to match Kalshi's expected format
        timestamp_ms = str(int(time.time() * 1000))
        # Kalshi WS auth: sign "timestamp + GET + /trade-api/ws/v2"
        message = f"{timestamp_ms}GET/trade-api/ws/v2".encode('utf-8')
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )
        sig_b64 = base64.b64encode(signature).decode()
        
        # Connect with auth headers
        headers = {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": sig_b64,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms
        }
        
        if self.verbose:
            print(f"[WS] Connecting to {self.WS_URL}...")
        
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
                print("[WS] Connected!")
            
            # Subscribe to any pending tickers
            await self._send_subscriptions()
            
            # Process messages
            async for message in ws:
                if not self._running:
                    break
                await self._handle_message(message)
    
    async def _send_subscriptions(self):
        """Send subscription requests for pending tickers"""
        with self._lock:
            tickers_to_sub = list(self._pending_subscriptions)
            self._pending_subscriptions.clear()
        
        if not tickers_to_sub or not self._ws:
            return
        
        # Subscribe to orderbook channel
        # Kalshi WS expects: {"cmd": "subscribe", "params": {"channels": ["orderbook_delta"], "market_tickers": [...]}}
        msg = {
            "id": int(time.time() * 1000),
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_tickers": tickers_to_sub
            }
        }
        
        if self.verbose:
            print(f"[WS] Subscribing to {len(tickers_to_sub)} tickers")
        
        await self._ws.send(json.dumps(msg))
        
        with self._lock:
            self._subscribed_tickers.update(tickers_to_sub)
    
    async def _send_unsubscribe(self, tickers: List[str]):
        """Send unsubscribe request"""
        if not self._ws or not tickers:
            return
        
        msg = {
            "id": int(time.time() * 1000),
            "cmd": "unsubscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_tickers": tickers
            }
        }
        
        await self._ws.send(json.dumps(msg))
    
    async def _handle_message(self, raw: str):
        """Process incoming WebSocket message"""
        self.last_any_message_ts = time.time()
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        
        msg_type = msg.get("type")
        
        # Handle orderbook snapshot
        if msg_type == "orderbook_snapshot":
            self._handle_snapshot(msg)
        
        # Handle orderbook delta
        elif msg_type == "orderbook_delta":
            self._handle_delta(msg)
        
        # Handle subscription confirmation
        elif msg_type == "subscribed":
            if self.verbose:
                print(f"[WS] Subscription confirmed: {msg.get('msg', {}).get('channel')}")
        
        # Handle errors
        elif msg_type == "error":
            if self.verbose:
                print(f"[WS] Error: {msg.get('msg')}")
    
    def _handle_snapshot(self, msg: dict):
        """Apply orderbook snapshot
        
        Kalshi snapshot format:
        {
            "type": "orderbook_snapshot",
            "msg": {
                "market_ticker": "...",
                "yes": [[price1, qty1], [price2, qty2], ...],  # BIDS only
                "no": [[price1, qty1], [price2, qty2], ...]    # BIDS only
            }
        }
        """
        data = msg.get("msg", {})
        ticker = data.get("market_ticker")
        
        if not ticker:
            return
        
        with self._lock:
            book = self._orderbooks.get(ticker)
            if not book:
                book = OrderbookState(ticker=ticker)
                self._orderbooks[ticker] = book
            
            # Clear existing state
            book.clear()
            
            # Apply YES bids - format is [[price, quantity], ...]
            for level in data.get("yes", []):
                if isinstance(level, list) and len(level) >= 2:
                    price = level[0]
                    quantity = level[1]
                    if price and quantity > 0:
                        book.yes_bids[price] = quantity
            
            # Apply NO bids - format is [[price, quantity], ...]
            for level in data.get("no", []):
                if isinstance(level, list) and len(level) >= 2:
                    price = level[0]
                    quantity = level[1]
                    if price and quantity > 0:
                        book.no_bids[price] = quantity
            
            book.last_update_ts = time.time()
            book.last_seq = data.get("seq", msg.get("seq", 0))
            book.has_snapshot = True
        
        if self.verbose:
            prices = self.get_prices(ticker)
            if prices:
                print(f"[WS] Snapshot {ticker}: YES {prices['yes_bid']}/{prices['yes_ask']}, NO {prices['no_bid']}/{prices['no_ask']}")
    
    def _handle_delta(self, msg: dict):
        """Apply orderbook delta update
        
        Kalshi delta format:
        {
            "type": "orderbook_delta",
            "seq": 123,
            "msg": {
                "market_ticker": "...",
                "price": 55,
                "delta": -10,
                "side": "yes"  # or "no"
            }
        }
        """
        data = msg.get("msg", {})
        ticker = data.get("market_ticker")
        
        if not ticker:
            return
        
        with self._lock:
            book = self._orderbooks.get(ticker)
            if not book or not book.has_snapshot:
                # Need snapshot first - can't apply deltas without baseline
                return
            
            # Check sequence
            seq = msg.get("seq", data.get("seq", 0))
            if seq and seq <= book.last_seq:
                return  # Old message, skip
            
            # Apply delta
            price = data.get("price")
            delta = data.get("delta", 0)
            side = data.get("side")  # "yes" or "no"
            
            if price and side in ('yes', 'no'):
                book.apply_delta(side, price, delta)
            
            book.last_update_ts = time.time()
            if seq:
                book.last_seq = seq
        
        if self.verbose:
            prices = self.get_prices(ticker)
            if prices:
                print(f"[WS] Delta {ticker}: YES {prices['yes_bid']}/{prices['yes_ask']}, age={prices['age_ms']}ms")


# ===== STANDALONE TEST =====
if __name__ == "__main__":
    import os
    
    # Test with env vars
    API_KEY = os.environ.get("KALSHI_API_KEY")
    PRIVATE_KEY_PATH = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "kalshi_private_key.pem")
    
    if not API_KEY:
        print("Set KALSHI_API_KEY environment variable")
        exit(1)
    
    print("Starting Kalshi WebSocket orderbook manager...")
    
    ws_book = KalshiWSOrderbook(API_KEY, PRIVATE_KEY_PATH, verbose=True)
    ws_book.start()
    
    # Wait for connection
    time.sleep(2)
    
    if not ws_book.connected:
        print("Failed to connect!")
        exit(1)
    
    # Subscribe to a test market (replace with actual ticker)
    test_ticker = "KXNCAAMBGAME-26FEB03-TEST-A"
    print(f"\nSubscribing to {test_ticker}")
    ws_book.subscribe([test_ticker])
    
    # Monitor for 30 seconds
    print("\nMonitoring for 30 seconds...")
    for i in range(30):
        prices = ws_book.get_prices(test_ticker)
        if prices:
            print(f"  {i}: YES {prices['yes_bid']}/{prices['yes_ask']} | age={prices['age_ms']}ms | fresh={ws_book.is_fresh(test_ticker, 500)}")
        else:
            print(f"  {i}: No data yet")
        time.sleep(1)
    
    ws_book.stop()
    print("\nDone!")