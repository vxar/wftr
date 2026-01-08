# Current Trading Interval Configuration

## Current Interval: **15 seconds**

The trading bot currently checks stock movements every **15 seconds**, not 20 seconds.

### Configuration Location

**File**: `src/web/web_app.py`
- **Line 157**: `bot._check_interval = 15`
- **Line 200**: `interval = getattr(bot, '_check_interval', 15)`
- **Line 201**: `logger.info(f"Auto-starting trading bot (interval: {interval} seconds)")`
- **Line 204**: `bot.run_continuous(interval_seconds=interval)`

### Evidence from Logs

From `trading_bot.log`:
```
2026-01-08 08:30:41 - INFO - Auto-starting trading bot (interval: 15 seconds)
2026-01-08 08:30:41 - INFO - Starting trading bot (checking every 15 seconds)
```

### How It Works

1. **Web App Sets Interval**: `src/web/web_app.py` sets `bot._check_interval = 15`
2. **Bot Uses Interval**: `bot.run_continuous(interval_seconds=15)` is called
3. **Trading Cycle**: Every 15 seconds, the bot:
   - Checks all monitored tickers
   - Analyzes for entry/exit signals
   - Executes trades if signals are valid
   - Updates positions

### Default Value

**File**: `src/core/live_trading_bot.py`
- **Line 1724**: `def run_continuous(self, interval_seconds: int = 60):`
- **Default**: 60 seconds (if not specified)
- **Actual**: 15 seconds (set by web app)

### To Change the Interval

To change from 15 seconds to 20 seconds (or any other value):

**Option 1: Modify web_app.py**
```python
# Line 157 in src/web/web_app.py
bot._check_interval = 20  # Change from 15 to 20
```

**Option 2: Modify run_continuous call**
```python
# Line 204 in src/web/web_app.py
bot.run_continuous(interval_seconds=20)  # Change from interval to 20
```

### Impact of Interval

- **15 seconds**: More frequent checks, faster entry/exit, more API calls
- **20 seconds**: Slightly less frequent, fewer API calls, may miss some fast moves
- **60 seconds**: Less frequent, fewer API calls, may miss more opportunities

### Current Status

✅ **Current Interval**: 15 seconds
✅ **Configured In**: `src/web/web_app.py` line 157
✅ **Used In**: `src/core/live_trading_bot.py` `run_continuous()` method
