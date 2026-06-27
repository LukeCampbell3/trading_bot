# Alpaca Paper Trading Setup (Free API)

This setup uses Alpaca's free paper trading API to test your trading strategies with real market data but simulated money.

## Quick Start

### 1. Get Free Alpaca Paper Trading Account

1. Go to https://alpaca.markets/
2. Sign up for a free account
3. Navigate to Paper Trading dashboard: https://app.alpaca.markets/paper/dashboard/overview
4. Generate API keys (Paper Trading keys are free)

### 2. Install Dependencies

```bash
pip install -r requirements_alpaca.txt
```

### 3. Configure API Keys

Create a `.env` file in your project root:

```bash
cp .env.example .env
```

Edit `.env` and add your Alpaca paper trading keys:

```
ALPACA_API_KEY=your_paper_api_key_here
ALPACA_API_SECRET=your_paper_api_secret_here
```

**Important:** Never commit your `.env` file to git!

### 4. Run the Trader

```bash
python alpaca_trader.py
```

## Features

- ✅ Free paper trading with real market data
- ✅ Integrates with your existing ML models
- ✅ Real-time market data streaming
- ✅ Automatic position management
- ✅ Stop loss and take profit
- ✅ Safe credential management

## File Structure

- `alpaca_config.py` - Configuration and credential management
- `alpaca_trader.py` - Main trading bot with ML integration
- `.env` - Your API credentials (create this, never commit)
- `.env.example` - Template for credentials
- `requirements_alpaca.txt` - Python dependencies

## Trading Strategy

The bot uses your trained ML model to generate trading signals:

1. Fetches real-time market data from Alpaca
2. Computes features (z-score, trend, volume, etc.)
3. Runs prediction through your model
4. Places trades based on signal strength
5. Manages risk with stop loss and take profit

## Configuration

Edit `alpaca_trader.py` to customize:

```python
SYMBOL = 'AAPL'  # Stock to trade
MODEL_PATH = 'model/instinct_model.keras'  # Your model
SCALER_PATH = 'model/scaler.pkl'  # Your scaler
```

Trading parameters:
- `buy_threshold` - Minimum signal to enter trade
- `position_size` - Fraction of buying power to use (0.3 = 30%)
- `stop_loss` - Exit if loss exceeds this (-0.02 = -2%)
- `take_profit` - Exit if profit exceeds this (0.03 = 3%)

## Alpaca Free Tier Limits

- ✅ Unlimited paper trading
- ✅ Real-time market data
- ✅ Full API access
- ✅ No credit card required
- ⚠️ Paper trading only (no real money)

## Safety Features

1. **Paper Trading Only** - No real money at risk
2. **Environment Variables** - Credentials never in code
3. **Stop Loss** - Automatic loss protection
4. **Position Sizing** - Limited capital per trade
5. **Market Hours Check** - Only trades during market hours

## Monitoring

The bot prints real-time status:

```
[14:30:15] Price: $150.25 | Signal: 0.0025 | Position: 10 shares | P/L: +2.5%
```

## Troubleshooting

**"API credentials not found"**
- Make sure `.env` file exists
- Check that keys are correct
- Verify you're using paper trading keys

**"Insufficient data"**
- Wait a few minutes for data to accumulate
- Check that market is open
- Verify symbol is valid

**"Order failed"**
- Check buying power
- Verify market hours
- Ensure symbol is tradeable

## Next Steps

1. Test with paper trading first
2. Monitor performance for several days
3. Adjust parameters based on results
4. Consider adding more sophisticated risk management

## Resources

- Alpaca Docs: https://alpaca.markets/docs/
- Paper Trading Dashboard: https://app.alpaca.markets/paper/dashboard/overview
- API Reference: https://alpaca.markets/docs/api-references/trading-api/
