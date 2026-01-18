# Scheduled Trading Bot

## Overview

The trading bot now includes automatic scheduling functionality that:

- **Automatically starts trading at 4:00 AM ET on weekdays**
- **Automatically stops trading at 8:00 PM ET on weekdays**
- **Enters sleep mode from 8:00 PM to 4:00 AM**
- **Remains in sleep mode on weekends**
- **Handles market volatility pauses automatically**

## Files Created

### Core Components
- `src/core/trading_bot_scheduler.py` - Main scheduler implementation
- `src/core/autonomous_trading_bot.py` - Updated bot with scheduler integration

### Launch Scripts
- `run_scheduled_bot.py` - Main launcher with monitoring
- `run_scheduled_bot.bat` - Windows batch file to start the bot
- `test_scheduler_logic.py` - Test script to verify scheduler logic

## Configuration

The trading window is configured in `src/config/settings.py`:

```python
@dataclass
class TradingWindowConfig:
    """Trading window configuration"""
    start_time: str = "04:00"  # 4:00 AM ET
    end_time: str = "20:00"    # 8:00 PM ET
    timezone: str = "US/Eastern"
```

## How to Use

### Quick Start (Windows)

1. **Double-click the batch file:**
   ```
   run_scheduled_bot.bat
   ```

2. **Or run from command line:**
   ```cmd
   venv\Scripts\python.exe run_scheduled_bot.py
   ```

### Manual Start/Stop

The bot can also be controlled manually:

```python
from src.core.autonomous_trading_bot import AutonomousTradingBot

# Initialize and start
bot = AutonomousTradingBot()
bot.start()  # This automatically starts the scheduler

# Check status
status = bot.get_bot_status()
scheduler_status = bot.get_scheduler_status()

# Stop (this also stops the scheduler)
bot.stop()
```

## Scheduler Behavior

### Weekday Schedule (Monday-Friday)
- **3:59 AM ET**: Bot in sleep mode
- **4:00 AM ET**: Bot automatically starts trading
- **4:00 AM - 8:00 PM ET**: Active trading window
- **8:00 PM ET**: Bot automatically stops trading
- **8:01 PM - 3:59 AM ET**: Sleep mode

### Weekend Schedule (Saturday-Sunday)
- **All day**: Bot remains in sleep mode

### Automatic Features

1. **Market Volatility Management**: The bot automatically pauses during high volatility periods
2. **Graceful Shutdown**: Handles Ctrl+C and system signals properly
3. **Status Monitoring**: Logs status every 30 minutes
4. **Error Recovery**: Automatically retries on errors

## Testing

Test the scheduler logic without running the full bot:

```cmd
venv\Scripts\python.exe test_scheduler_logic.py
```

This will test various time scenarios and verify the scheduling logic is correct.

## Monitoring

When running, the bot provides:

- **Console Logs**: Real-time status updates
- **Log File**: Detailed logs saved to `trading_bot.log`
- **Status Checks**: Automatic status reporting every 30 minutes

### Example Console Output

```
============================================================
TRADING BOT LAUNCHER STARTING
============================================================
Current time: 2026-01-17 18:45:44 ET
Trading window: 04:00 - 20:00 ET
Timezone: US/Eastern
Initializing trading bot...
Starting trading bot with automatic scheduler...
Trading bot started successfully!
The bot will automatically:
  - Start trading at 4:00 AM ET on weekdays
  - Stop trading at 8:00 PM ET on weekdays
  - Remain in sleep mode on weekends
  - Handle market volatility pauses automatically
```

## Configuration Options

You can modify the trading window by editing `src/config/settings.py`:

```python
@dataclass
class TradingWindowConfig:
    start_time: str = "09:30"  # Market open
    end_time: str = "16:00"    # Market close
    timezone: str = "US/Eastern"
```

## Troubleshooting

### Bot Not Starting
1. Ensure virtual environment is activated: `venv\Scripts\activate`
2. Install dependencies: `venv\Scripts\pip install -r requirements.txt`
3. Check logs in `trading_bot.log`

### Scheduler Issues
1. Run test script: `venv\Scripts\python.exe test_scheduler_logic.py`
2. Verify system timezone settings
3. Check trading window configuration

### Dependencies
Make sure all required packages are installed:

```cmd
venv\Scripts\pip install -r requirements.txt
```

Key packages for scheduling:
- `schedule>=1.2.0` - Task scheduling
- `pytz>=2023.3` - Timezone handling

## Integration with Existing Features

The scheduler integrates seamlessly with existing bot features:

- **Position Management**: Active positions are managed during trading hours
- **Risk Management**: All risk controls remain active
- **Market Scanning**: Scanning only occurs during trading hours
- **Dashboard**: Web dashboard shows scheduler status

## Next Steps

1. **Run the bot**: Use `run_scheduled_bot.bat` to start
2. **Monitor logs**: Check `trading_bot.log` for detailed activity
3. **Verify schedule**: Confirm bot starts/stops at expected times
4. **Customize**: Adjust trading window if needed in settings

The scheduled trading bot provides fully automated operation while maintaining all existing trading functionality and risk management features.
