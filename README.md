# AI Trading Bot - Live Trading System

A production-ready trading bot with optimized entry/exit logic, integrated with Webull API for live trading. Designed to achieve $500 daily profit with $10,000 capital (5% daily return).

## Quick Start

### 1. Setup
```bash
# Activate virtual environment
venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Live Trading Bot

**Primary Method - Web Interface (Recommended):**
```bash
python src/web_app.py
```
Then open http://127.0.0.1:5000 in your browser and click "Start Trading" to begin.

**Alternative - Direct Start (Legacy):**
```bash
# Auto-fetch from top gainers
python src/run_live_bot.py auto

# Auto-fetch from swing screener
python src/run_live_bot.py swing

# With specific tickers
python src/run_live_bot.py manual TICKER1 TICKER2 TICKER3
```

### 3. Run Simulations (Testing)

**Weekly simulation:**
```bash
python src/weekly_simulation.py 7
```

**Single ticker simulation:**
```bash
python src/simulate_trading_capital.py TICKER1 TICKER2
```

**Daily simulation:**
```bash
python src/daily_trading_simulator.py 7
```

## Key Features

- **Webull API Integration**: Real-time data fetching from Webull
- **Optimized Entry Logic**: Multi-layered validation (8 critical checks + scoring system)
- **Smart Exit Strategy**: 6 priority levels (setup failure, stop loss, profit target, trailing stop, trend weakness, bearish reversal)
- **Web Dashboard**: Real-time monitoring and control via web interface
- **Extended Trading Hours**: Trades 8:00 AM - 6:00 PM ET (pre-market and after-hours)
- **Auto Stock Discovery**: Automatically fetches top gainers or swing stocks
- **Risk Management**: 3% stop loss, 3% trailing stop, position sizing
- **Daily Profit Target**: $500 per day (5% of $10,000 capital)
- **Database Persistence**: All trades and positions saved to SQLite database

## Configuration

**Optimized Parameters** (from simulation testing):
- Initial Capital: $10,000
- Daily Profit Target: $500
- Position Size: 50% per trade
- Max Positions: 3 concurrent
- Min Confidence: 70%
- Min Entry Price Increase: 5%
- Profit Target: 7%
- Trailing Stop: 3%
- Stop Loss: 3%

**Edit in**: `src/web_app.py` (or `src/run_live_bot.py` for legacy mode)

## Performance

Based on weekly simulation testing:
- **Weekly Return**: +177.56% (7 days)
- **Win Rate**: 50-60%
- **Average P&L per Trade**: $83.31
- **Best Day**: +$11,085.10
- **Target Reached Days**: 3 out of 7 (42.9%)

## Documentation

- **`TRADING_LOGIC_DOCUMENTATION.md`** - Complete trading logic and execution flow
- **`WEBULL_INTEGRATION.md`** - Webull API integration guide
- **`README.md`** - This file

## File Structure

```
src/
├── web_app.py                    # Primary entry point (web interface)
├── run_live_bot.py              # Legacy entry point (direct start)
├── live_trading_bot.py           # Core trading bot
├── trading_web_interface.py       # Web dashboard (Flask)
├── trading_database.py           # SQLite database for trades
├── realtime_trader.py            # Entry/exit logic
├── pattern_detector.py           # Pattern detection
├── webull_data_api.py            # Webull API integration
├── WebullUtil.py                 # Webull utilities
├── api_interface.py              # API abstraction
├── premarket_analyzer.py        # Pre-market analysis
├── utils.py                      # Logger utility
├── simulate_trading_capital.py   # Single ticker simulator
├── weekly_simulation.py          # Weekly simulation
└── daily_trading_simulator.py    # Daily simulation
```

## API Integration

The bot uses **WebullDataAPI** by default for live trading. For testing with CSV files, modify `src/run_live_bot.py`:

```python
# Comment out Webull API
# from webull_data_api import WebullDataAPI
# api = WebullDataAPI()

# Uncomment CSV API
from api_interface import CSVDataAPI
api = CSVDataAPI(data_dir="test_data")
```

## Trading Strategy

### Entry Requirements
- Pattern detected (Strong_Bullish_Setup or Volume_Breakout)
- Confidence ≥ 70%
- 8 critical requirements must pass
- Perfect setup score ≥ 6/8
- Setup confirmed 3+ periods
- Expected gain ≥ 5%
- No false breakout or reverse split

### Exit Conditions (Priority Order)
1. Setup failed after entry (first 5 minutes)
2. Stop loss hit (3% below entry)
3. Profit target reached (7% above entry)
4. Trailing stop hit (3% from peak)
5. Trend weakness detected
6. Bearish reversal pattern

## Important Notes

⚠️ **Paper Trading First**: Always test with paper trading before using real money.

⚠️ **Risk Warning**: Trading involves risk. Past performance does not guarantee future results.

⚠️ **Market Hours**: Bot trades during extended hours (8:00 AM - 6:00 PM ET) by default. Configure in `src/web_app.py`.

## Support

For detailed trading logic, see `TRADING_LOGIC_DOCUMENTATION.md`.
For Webull integration details, see `WEBULL_INTEGRATION.md`.

