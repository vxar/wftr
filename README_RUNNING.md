# Running the Trading Bot

## Quick Start

### Option 1: Run Web Application (Recommended)
```bash
python run_web_app.py
```
This starts the Flask web interface at http://127.0.0.1:5000

### Option 2: Run Trading Bot Directly
```bash
python run_live_bot.py
```
This runs the trading bot with command-line interface.

## Package Structure

The code is organized into packages:
- `src/core/` - Core trading logic
- `src/analysis/` - Pattern detection and analysis
- `src/data/` - Data access layer
- `src/database/` - Database operations
- `src/web/` - Web interface
- `src/utils/` - Utility functions
- `src/scripts/` - Entry point scripts

## Import Structure

All imports use absolute imports from the `src/` package level:
- `from core.live_trading_bot import LiveTradingBot`
- `from data.webull_data_api import WebullDataAPI`
- `from web.trading_web_interface import set_trading_bot`

The entry point scripts (`run_web_app.py` and `run_live_bot.py`) add `src/` to the Python path, so all imports work correctly.

## Troubleshooting

### Import Errors
If you see import errors, make sure:
1. You're running from the project root directory
2. The `src/` directory is in the Python path (handled automatically by entry scripts)
3. All dependencies are installed: `pip install -r requirements.txt`

### Template Not Found
The templates folder should be at the project root: `templates/dashboard.html`
The web interface automatically finds it.

### Database Errors
The database file `trading_data.db` will be created automatically in the project root.
