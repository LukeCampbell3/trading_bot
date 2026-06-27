"""
Test Alpaca API Connection
Verifies that your API credentials work and you can access market data
"""
import sys
from datetime import datetime, timedelta
import pytz

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.client import TradingClient
    from alpaca_config import AlpacaConfig
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Run: pip install -r requirements_alpaca.txt")
    sys.exit(1)

def test_credentials():
    """Test 1: Verify credentials are loaded"""
    print("\n" + "="*60)
    print("TEST 1: Checking API Credentials")
    print("="*60)
    
    try:
        AlpacaConfig.validate()
        print(f"✓ API Key: {AlpacaConfig.API_KEY[:8]}...{AlpacaConfig.API_KEY[-4:]}")
        print(f"✓ API Secret: {'*' * 20}")
        print(f"✓ Paper Trading: {AlpacaConfig.PAPER}")
        return True
    except ValueError as e:
        print(f"❌ {e}")
        return False

def test_trading_client():
    """Test 2: Connect to trading API"""
    print("\n" + "="*60)
    print("TEST 2: Trading Client Connection")
    print("="*60)
    
    try:
        client = TradingClient(
            AlpacaConfig.API_KEY,
            AlpacaConfig.API_SECRET,
            paper=AlpacaConfig.PAPER,
            url_override=AlpacaConfig.BASE_URL
        )
        
        # Get account info
        account = client.get_account()
        
        print(f"✓ Connected to Alpaca Trading API")
        print(f"✓ Account Status: {account.status}")
        print(f"✓ Account Number: {account.account_number}")
        print(f"✓ Equity: ${float(account.equity):,.2f}")
        print(f"✓ Cash: ${float(account.cash):,.2f}")
        print(f"✓ Buying Power: ${float(account.buying_power):,.2f}")
        print(f"✓ Portfolio Value: ${float(account.portfolio_value):,.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Trading client failed: {e}")
        return False

def test_market_data():
    """Test 3: Fetch market data"""
    print("\n" + "="*60)
    print("TEST 3: Market Data Access")
    print("="*60)
    
    try:
        client = StockHistoricalDataClient(
            AlpacaConfig.API_KEY,
            AlpacaConfig.API_SECRET,
            url_override=AlpacaConfig.DATA_URL
        )
        
        # Test with popular stock
        symbol = "AAPL"
        now = datetime.now(pytz.UTC)
        start = now - timedelta(days=1)
        
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start,
            end=now,
            limit=100
        )
        
        bars = client.get_stock_bars(request)
        df = bars.df
        
        if df.empty:
            print(f"⚠ No data returned (market may be closed)")
            return True
        
        print(f"✓ Successfully fetched {len(df)} bars for {symbol}")
        print(f"✓ Date range: {df.index[0][0]} to {df.index[-1][0]}")
        print(f"✓ Latest close: ${df['close'].iloc[-1]:.2f}")
        print(f"✓ Columns: {list(df.columns)}")
        
        return True
    except Exception as e:
        print(f"❌ Market data failed: {e}")
        return False

def test_positions():
    """Test 4: Check positions"""
    print("\n" + "="*60)
    print("TEST 4: Position Management")
    print("="*60)
    
    try:
        client = TradingClient(
            AlpacaConfig.API_KEY,
            AlpacaConfig.API_SECRET,
            paper=AlpacaConfig.PAPER,
            url_override=AlpacaConfig.BASE_URL
        )
        
        positions = client.get_all_positions()
        
        if not positions:
            print("✓ No open positions (clean slate)")
        else:
            print(f"✓ Found {len(positions)} open position(s):")
            for pos in positions:
                print(f"  - {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f}")
                print(f"    Current: ${float(pos.current_price):.2f} | P/L: {float(pos.unrealized_plpc):.2%}")
        
        return True
    except Exception as e:
        print(f"❌ Position check failed: {e}")
        return False

def test_orders():
    """Test 5: Check order history"""
    print("\n" + "="*60)
    print("TEST 5: Order History")
    print("="*60)
    
    try:
        client = TradingClient(
            AlpacaConfig.API_KEY,
            AlpacaConfig.API_SECRET,
            paper=AlpacaConfig.PAPER,
            url_override=AlpacaConfig.BASE_URL
        )
        
        orders = client.get_orders()
        
        if not orders:
            print("✓ No recent orders")
        else:
            print(f"✓ Found {len(orders)} recent order(s):")
            for order in orders[:5]:  # Show last 5
                print(f"  - {order.symbol}: {order.side} {order.qty} @ {order.status}")
        
        return True
    except Exception as e:
        print(f"❌ Order check failed: {e}")
        return False

def test_clock():
    """Test 6: Market clock"""
    print("\n" + "="*60)
    print("TEST 6: Market Clock")
    print("="*60)
    
    try:
        client = TradingClient(
            AlpacaConfig.API_KEY,
            AlpacaConfig.API_SECRET,
            paper=AlpacaConfig.PAPER,
            url_override=AlpacaConfig.BASE_URL
        )
        
        clock = client.get_clock()
        
        print(f"✓ Market is {'OPEN' if clock.is_open else 'CLOSED'}")
        print(f"✓ Current time: {clock.timestamp}")
        print(f"✓ Next open: {clock.next_open}")
        print(f"✓ Next close: {clock.next_close}")
        
        return True
    except Exception as e:
        print(f"❌ Clock check failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("ALPACA API CONNECTION TEST SUITE")
    print("="*60)
    
    tests = [
        ("Credentials", test_credentials),
        ("Trading Client", test_trading_client),
        ("Market Data", test_market_data),
        ("Positions", test_positions),
        ("Orders", test_orders),
        ("Market Clock", test_clock),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your Alpaca connection is working perfectly.")
        return 0
    else:
        print("\n⚠ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
