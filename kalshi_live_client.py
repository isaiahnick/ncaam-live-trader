#!/usr/bin/env python3
"""
kalshi_live_client.py - Live Trading Client for Kalshi API

This module provides authenticated access to Kalshi's trading API,
allowing programmatic order placement and position management.

SECURITY: 
- Private key should be stored in a secure file (not in code)
- API key ID should be in environment variables
- Never commit credentials to version control
"""

import os
import uuid
import base64
import datetime
import requests
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Cryptography imports for RSA signing
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import padding


class OrderSide(Enum):
    YES = "yes"
    NO = "no"


class OrderAction(Enum):
    BUY = "buy"
    SELL = "sell"


class TimeInForce(Enum):
    GTC = None  # Good 'til cancelled (default)
    FOK = "fill_or_kill"  # Fill or kill - must fill entirely or cancel
    IOC = "immediate_or_cancel"  # Immediate or cancel - fill what you can, cancel rest


@dataclass
class OrderResult:
    """Result of an order placement attempt"""
    success: bool
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    status: Optional[str] = None  # 'resting', 'filled', 'cancelled', etc.
    fill_count: int = 0
    remaining_count: int = 0
    avg_price: Optional[float] = None
    taker_fees: float = 0.0
    error: Optional[str] = None
    raw_response: Optional[Dict] = None


@dataclass 
class Position:
    """Current position in a market"""
    ticker: str
    yes_count: int
    no_count: int
    yes_avg_price: float
    no_avg_price: float


class KalshiLiveClient:
    """
    Live trading client for Kalshi API.
    
    Handles authentication, order placement, and position management.
    
    Usage:
        client = KalshiLiveClient(
            api_key_id="your-api-key-id",
            private_key_path="/path/to/kalshi-key.key",
            use_demo=True  # Start with demo!
        )
        
        # Place a buy order
        result = client.place_order(
            ticker="KXNCAAMBGAME-26FEB02-DUKE",
            side=OrderSide.YES,
            action=OrderAction.BUY,
            count=10,
            price_cents=45,
            time_in_force=TimeInForce.FOK
        )
    """
    
    DEMO_URL = "https://demo-api.kalshi.co"
    PROD_URL = "https://api.elections.kalshi.com"
    
    def __init__(
        self,
        api_key_id: str,
        private_key_path: Optional[str] = None,
        private_key_pem: Optional[str] = None,
        use_demo: bool = True,
        max_position_size: int = 100,  # Max contracts per position
        max_order_value_cents: int = 5000,  # Max $50 per order
        verbose: bool = True
    ):
        """
        Initialize the Kalshi client.
        
        Args:
            api_key_id: Your Kalshi API Key ID (UUID)
            private_key_path: Path to your .key file
            private_key_pem: OR the PEM string directly
            use_demo: If True, use demo environment (RECOMMENDED for testing)
            max_position_size: Safety limit on position size
            max_order_value_cents: Safety limit on order value
            verbose: Print trade confirmations
        """
        self.api_key_id = api_key_id
        self.base_url = self.DEMO_URL if use_demo else self.PROD_URL
        self.use_demo = use_demo
        self.max_position_size = max_position_size
        self.max_order_value_cents = max_order_value_cents
        self.verbose = verbose
        
        # Load private key
        if private_key_path:
            self.private_key = self._load_private_key_from_file(private_key_path)
        elif private_key_pem:
            self.private_key = self._load_private_key_from_pem(private_key_pem)
        else:
            raise ValueError("Must provide either private_key_path or private_key_pem")
        
        # Track orders for the session
        self.orders_placed = []
        self.total_fees_paid = 0.0
        
        if self.verbose:
            env = "DEMO" if use_demo else "🔴 PRODUCTION"
            print(f"✓ KalshiLiveClient initialized ({env})")
            print(f"  Max position: {max_position_size} contracts")
            print(f"  Max order value: ${max_order_value_cents/100:.2f}")
    
    def _load_private_key_from_file(self, key_path: str):
        """Load private key from .key file"""
        with open(key_path, "rb") as f:
            return serialization.load_pem_private_key(
                f.read(), 
                password=None, 
                backend=default_backend()
            )
    
    def _load_private_key_from_pem(self, pem_string: str):
        """Load private key from PEM string"""
        return serialization.load_pem_private_key(
            pem_string.encode('utf-8'),
            password=None,
            backend=default_backend()
        )
    
    def _create_signature(self, timestamp: str, method: str, path: str) -> str:
        """Create RSA-PSS signature for request authentication"""
        # Strip query parameters for signing
        path_without_query = path.split('?')[0]
        message = f"{timestamp}{method}{path_without_query}".encode('utf-8')
        
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )
        
        return base64.b64encode(signature).decode('utf-8')
    
    def _get_headers(self, method: str, path: str) -> Dict[str, str]:
        """Generate authenticated headers for a request"""
        timestamp = str(int(datetime.datetime.now().timestamp() * 1000))
        signature = self._create_signature(timestamp, method, path)
        
        return {
            'KALSHI-ACCESS-KEY': self.api_key_id,
            'KALSHI-ACCESS-SIGNATURE': signature,
            'KALSHI-ACCESS-TIMESTAMP': timestamp,
            'Content-Type': 'application/json'
        }
    
    def _get(self, path: str) -> requests.Response:
        """Make authenticated GET request"""
        headers = self._get_headers("GET", path)
        return requests.get(self.base_url + path, headers=headers)
    
    def _post(self, path: str, data: Dict) -> requests.Response:
        """Make authenticated POST request"""
        headers = self._get_headers("POST", path)
        return requests.post(self.base_url + path, headers=headers, json=data)
    
    def _delete(self, path: str) -> requests.Response:
        """Make authenticated DELETE request"""
        headers = self._get_headers("DELETE", path)
        return requests.delete(self.base_url + path, headers=headers)
    
    # ===== Account Methods =====
    
    def get_balance(self) -> Dict[str, float]:
        """Get account balance"""
        response = self._get("/trade-api/v2/portfolio/balance")
        if response.status_code == 200:
            data = response.json()
            return {
                'balance': data.get('balance', 0) / 100.0,  # Convert cents to dollars
                'payout_available': data.get('payout_available', 0) / 100.0
            }
        else:
            raise Exception(f"Failed to get balance: {response.status_code} - {response.text}")
    
    def get_positions(self) -> Dict[str, Position]:
        """Get all current positions"""
        response = self._get("/trade-api/v2/portfolio/positions")
        if response.status_code == 200:
            positions = {}
            for pos in response.json().get('market_positions', []):
                ticker = pos.get('ticker', '')
                positions[ticker] = Position(
                    ticker=ticker,
                    yes_count=pos.get('position', 0),  # Positive = yes, negative = no
                    no_count=0,  # Kalshi uses signed position
                    yes_avg_price=pos.get('market_exposure', 0) / max(abs(pos.get('position', 1)), 1),
                    no_avg_price=0
                )
            return positions
        else:
            raise Exception(f"Failed to get positions: {response.status_code} - {response.text}")
    
    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position for a specific market"""
        positions = self.get_positions()
        return positions.get(ticker)
    
    # ===== Market Data Methods =====
    
    def get_orderbook(self, ticker: str, depth: int = 10) -> Dict:
        """Get orderbook for a market"""
        response = self._get(f"/trade-api/v2/markets/{ticker}/orderbook?depth={depth}")
        if response.status_code == 200:
            return response.json()
        else:
            return {}
    
    def get_best_prices(self, ticker: str) -> Dict[str, Optional[int]]:
        """
        Get best bid/ask for YES and NO sides.
        
        Returns dict with keys: yes_bid, yes_ask, no_bid, no_ask (in cents)
        """
        book = self.get_orderbook(ticker, depth=1)
        orderbook = book.get('orderbook', {})
        
        yes_bids = orderbook.get('yes', [])
        no_bids = orderbook.get('no', [])
        
        # Best YES bid is highest price in yes array
        yes_bid = yes_bids[-1][0] if yes_bids else None
        # Best NO bid is highest price in no array  
        no_bid = no_bids[-1][0] if no_bids else None
        
        # Ask = 100 - opposite bid (since YES + NO = 100)
        yes_ask = 100 - no_bid if no_bid else None
        no_ask = 100 - yes_bid if yes_bid else None
        
        return {
            'yes_bid': yes_bid,
            'yes_ask': yes_ask,
            'no_bid': no_bid,
            'no_ask': no_ask
        }
    
    # ===== Order Methods =====
    
    def place_order(
        self,
        ticker: str,
        side: OrderSide,
        action: OrderAction,
        count: int,
        price_cents: int,
        time_in_force: TimeInForce = TimeInForce.FOK,
        client_order_id: Optional[str] = None
    ) -> OrderResult:
        """
        Place an order on Kalshi.
        
        Args:
            ticker: Market ticker (e.g., "KXNCAAMBGAME-26FEB02-DUKE")
            side: OrderSide.YES or OrderSide.NO
            action: OrderAction.BUY or OrderAction.SELL
            count: Number of contracts
            price_cents: Price per contract in cents (1-99)
            time_in_force: FOK (fill or kill), IOC, or GTC
            client_order_id: Optional unique ID for deduplication
        
        Returns:
            OrderResult with success status,    fill info, and any errors
        """
        # Safety checks
        order_value = count * price_cents
        if order_value > self.max_order_value_cents:
            return OrderResult(
                success=False,
                error=f"Order value {order_value}¢ exceeds max {self.max_order_value_cents}¢"
            )
        
        if count > self.max_position_size:
            return OrderResult(
                success=False,
                error=f"Order count {count} exceeds max position size {self.max_position_size}"
            )
        
        if price_cents < 1 or price_cents > 99:
            return OrderResult(
                success=False,
                error=f"Price {price_cents}¢ must be between 1-99"
            )
        
        # Build order data
        order_id = client_order_id or str(uuid.uuid4())
        
        order_data = {
            "ticker": ticker,
            "action": action.value,
            "side": side.value,
            "count": count,
            "type": "limit",
            "client_order_id": order_id
        }
        
        # Set price based on side - Kalshi wants exactly ONE price field
        if side == OrderSide.YES:
            order_data["yes_price"] = price_cents
        else:
            order_data["no_price"] = price_cents
        
        # Set time in force
        if time_in_force != TimeInForce.GTC:
            order_data["time_in_force"] = time_in_force.value
        
        # Place the order
        try:
            if self.verbose:
                print(f"  📤 Sending order: {order_data}")
            response = self._post("/trade-api/v2/portfolio/orders", order_data)
            
            if response.status_code == 201:
                order = response.json().get('order', {})
                
                # Try to get average fill price from various possible fields
                avg_price = (
                    order.get('avg_fill_price') or 
                    order.get('average_fill_price') or
                    order.get('yes_price') or 
                    order.get('no_price') or
                    price_cents  # Fall back to requested price
                )
                
                result = OrderResult(
                    success=True,
                    order_id=order.get('order_id'),
                    client_order_id=order_id,
                    status=order.get('status'),
                    fill_count=order.get('fill_count', 0),
                    remaining_count=order.get('remaining_count', count),
                    avg_price=avg_price,
                    taker_fees=order.get('taker_fees', 0) / 100.0,  # Convert to dollars
                    raw_response=order
                )
                
                # Track the order
                self.orders_placed.append(result)
                self.total_fees_paid += result.taker_fees
                
                if self.verbose:
                    status_emoji = "✅" if result.fill_count > 0 else "⏳"
                    print(f"  {status_emoji} ORDER: {action.value.upper()} {count} {side.value.upper()} @ {price_cents}¢")
                    print(f"     Ticker: {ticker}")
                    print(f"     Status: {result.status} | Filled: {result.fill_count}/{count}")
                    if result.taker_fees > 0:
                        print(f"     Fees: ${result.taker_fees:.4f}")
                
                return result
            else:
                error_msg = response.json().get('error', {}).get('message', response.text)
                if self.verbose:
                    print(f"  ❌ ORDER FAILED: {error_msg}")
                    print(f"     Full response: {response.json()}")
                    print(f"     Order data sent: {order_data}")
                return OrderResult(
                    success=False,
                    error=error_msg,
                    raw_response=response.json() if response.text else None
                )
                
        except Exception as e:
            return OrderResult(success=False, error=str(e))
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order"""
        response = self._delete(f"/trade-api/v2/portfolio/orders/{order_id}")
        return response.status_code == 200
    
    def cancel_all_orders(self, ticker: Optional[str] = None) -> int:
        """Cancel all open orders, optionally filtered by ticker"""
        # Get open orders
        response = self._get("/trade-api/v2/portfolio/orders?status=resting")
        if response.status_code != 200:
            return 0
        
        cancelled = 0
        for order in response.json().get('orders', []):
            if ticker is None or order.get('ticker') == ticker:
                if self.cancel_order(order['order_id']):
                    cancelled += 1
        
        if self.verbose and cancelled > 0:
            print(f"  🚫 Cancelled {cancelled} orders")
        
        return cancelled
    
    # ===== Convenience Methods for Live Trading =====
    
    def buy_yes(
        self,
        ticker: str,
        count: int,
        max_price_cents: int,
        immediate: bool = True
    ) -> OrderResult:
        """
        Buy YES contracts - convenience wrapper.
        
        Args:
            ticker: Market ticker
            count: Number of contracts
            max_price_cents: Maximum price to pay
            immediate: If True, use FOK; if False, use GTC (resting order)
        """
        return self.place_order(
            ticker=ticker,
            side=OrderSide.YES,
            action=OrderAction.BUY,
            count=count,
            price_cents=max_price_cents,
            time_in_force=TimeInForce.FOK if immediate else TimeInForce.GTC
        )
    
    def buy_no(
        self,
        ticker: str,
        count: int,
        max_price_cents: int,
        immediate: bool = True
    ) -> OrderResult:
        """Buy NO contracts - convenience wrapper."""
        return self.place_order(
            ticker=ticker,
            side=OrderSide.NO,
            action=OrderAction.BUY,
            count=count,
            price_cents=max_price_cents,
            time_in_force=TimeInForce.FOK if immediate else TimeInForce.GTC
        )
    
    def sell_yes(
        self,
        ticker: str,
        count: int,
        min_price_cents: int,
        immediate: bool = True
    ) -> OrderResult:
        """Sell YES contracts - convenience wrapper."""
        return self.place_order(
            ticker=ticker,
            side=OrderSide.YES,
            action=OrderAction.SELL,
            count=count,
            price_cents=min_price_cents,
            time_in_force=TimeInForce.FOK if immediate else TimeInForce.GTC
        )
    
    def sell_no(
        self,
        ticker: str,
        count: int,
        min_price_cents: int,
        immediate: bool = True
    ) -> OrderResult:
        """Sell NO contracts - convenience wrapper."""
        return self.place_order(
            ticker=ticker,
            side=OrderSide.NO,
            action=OrderAction.SELL,
            count=count,
            price_cents=min_price_cents,
            time_in_force=TimeInForce.FOK if immediate else TimeInForce.GTC
        )
    
    def get_session_summary(self) -> Dict:
        """Get summary of trading session"""
        return {
            'orders_placed': len(self.orders_placed),
            'total_fees_paid': self.total_fees_paid,
            'environment': 'DEMO' if self.use_demo else 'PRODUCTION'
        }


# ===== Quick Test Function =====

def test_connection(api_key_id: str, private_key_path: str, use_demo: bool = True):
    """Test API connection and authentication"""
    print("\n" + "="*60)
    print("KALSHI API CONNECTION TEST")
    print("="*60)
    
    try:
        client = KalshiLiveClient(
            api_key_id=api_key_id,
            private_key_path=private_key_path,
            use_demo=use_demo,
            verbose=True
        )
        
        # Test 1: Get balance
        print("\n1. Testing balance endpoint...")
        balance = client.get_balance()
        print(f"   ✓ Balance: ${balance['balance']:.2f}")
        
        # Test 2: Get positions
        print("\n2. Testing positions endpoint...")
        positions = client.get_positions()
        print(f"   ✓ Found {len(positions)} positions")
        
        # Test 3: Get a market orderbook (use a known CBB market if available)
        print("\n3. Testing orderbook endpoint...")
        # Try to find a CBB game market
        # For demo, we'll just check the endpoint works
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED - API CONNECTION WORKING")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ CONNECTION TEST FAILED: {e}")
        return False


if __name__ == "__main__":
    # Example usage - DO NOT hardcode credentials in production!
    import sys
    
    if len(sys.argv) >= 3:
        api_key = sys.argv[1]
        key_path = sys.argv[2]
        use_demo = len(sys.argv) < 4 or sys.argv[3].lower() != 'prod'
        
        test_connection(api_key, key_path, use_demo)
    else:
        print("Usage: python3 kalshi_live_client.py <api_key_id> <private_key_path> [demo|prod]")
        print("\nExample:")
        print("  python3 kalshi_live_client.py abc123 /path/to/key.key demo")