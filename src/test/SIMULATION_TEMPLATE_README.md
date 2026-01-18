# Trading Bot Simulation Template

## Overview

The `simulate_ticker_template.py` is a reusable template for backtesting any ticker from a specific detection time. This eliminates the need to create new simulation scripts for each stock.

## Quick Start

1. Open `src/test/simulate_ticker_template.py`
2. Modify the configuration section at the top of the file
3. Run: `python src/test/simulate_ticker_template.py`

Or use the quick runner:
1. Run: `python src/test/quick_simulate.py TICKER DATE TIME [HAD_TRADES]`

## Configuration Parameters

### Required Parameters

```python
TICKER = 'PRFX'  # Stock symbol to simulate
DETECTION_DATE = '2026-01-15'  # Date when ticker was detected (YYYY-MM-DD)
DETECTION_TIME = '16:05'  # Time when ticker was detected (HH:MM format, ET)
```

### Optional Parameters

```python
START_HOUR = 4  # Hour to start data collection from (default: 4 AM)
LIVE_BOT_HAD_TRADES = True  # True if live bot had trades, False otherwise
DETAILED_ANALYSIS_MINUTES = 20  # Number of minutes to analyze in detail after detection

# Trader settings (optional overrides)
MIN_CONFIDENCE = 0.72  # Minimum pattern confidence
PROFIT_TARGET_PCT = 20.0  # Profit target percentage
TRAILING_STOP_PCT = 7.0  # Trailing stop percentage
```

## Usage Examples

### Example 1: PRFX at 4:05 PM (Live bot had losing trade)

```python
TICKER = 'PRFX'
DETECTION_DATE = '2026-01-15'
DETECTION_TIME = '16:05'
LIVE_BOT_HAD_TRADES = True
DETAILED_ANALYSIS_MINUTES = 20
```

### Example 2: VERO at 4:31 PM (Live bot had trades)

```python
TICKER = 'VERO'
DETECTION_DATE = '2026-01-15'
DETECTION_TIME = '16:31'
LIVE_BOT_HAD_TRADES = True
DETAILED_ANALYSIS_MINUTES = 20
```

### Example 3: Stock detected at 4 AM (No live bot trades)

```python
TICKER = 'SPHL'
DETECTION_DATE = '2026-01-15'
DETECTION_TIME = '04:00'
LIVE_BOT_HAD_TRADES = False
DETAILED_ANALYSIS_MINUTES = 30
```

## Output Files

The template generates the following files:

1. **Log File**: `src/test/{TICKER}_SIMULATION_LOG_{DATE}.log`
   - Detailed minute-by-minute analysis
   - Entry/exit signals
   - Rejected entries with reasons

2. **CSV File**: `analysis/BOT_SIMULATION_{TICKER}_{TIME}_{DATE}.csv`
   - Trade results in CSV format
   - Includes entry/exit times, prices, P&L, patterns

3. **Data File**: `test_data/{TICKER}_1min_{DATE}.csv`
   - Downloaded price data (saved for reuse)

## Features

### Automatic Data Management
- Downloads data if not already present
- Reuses existing data files
- Handles timezone conversions automatically

### Detailed Analysis
- First N minutes after detection analyzed in detail
- Minute-by-minute price, volume, and signal logging
- Clear indication of entry/exit points

### Comparison Mode
- `LIVE_BOT_HAD_TRADES = True`: Compares simulation vs live results
- `LIVE_BOT_HAD_TRADES = False`: Analyzes why entry was missed

### Comprehensive Reporting
- Trade summary (win rate, total P&L, average P&L)
- Rejected entries grouped by reason
- Individual trade details

## Workflow

1. **Configure**: Set ticker, date, time, and flags
2. **Run**: Execute the script
3. **Review**: Check log file for detailed analysis
4. **Compare**: Compare CSV results with live bot trades
5. **Analyze**: Use findings to improve bot logic

## Tips

### For Missed Trades Analysis
- Set `LIVE_BOT_HAD_TRADES = False`
- Increase `DETAILED_ANALYSIS_MINUTES` to see more context
- Check rejected entries to understand why entry was missed

### For Trade Comparison
- Set `LIVE_BOT_HAD_TRADES = True`
- Compare entry times and prices
- Analyze why simulation differs from live bot

### For After-Hours Analysis
- Detection times after 4:00 PM are after-hours
- After-hours trades have different characteristics
- Consider tighter stops and higher confidence thresholds

## Troubleshooting

### No Data Available
- Check if ticker symbol is correct
- Verify date format (YYYY-MM-DD)
- Ensure data is available for that date

### No Trades Generated
- Check rejected entries in log
- Verify confidence thresholds
- Review entry validation logic

### Discrepancies with Live Bot
- Compare entry times (simulation may enter earlier/later)
- Check for data quality issues
- Review surge detection vs pattern detection differences

## Advanced Usage

### Batch Processing
Create a script to run multiple simulations:

```python
import subprocess

tickers = [
    ('PRFX', '2026-01-15', '16:05', True),
    ('VERO', '2026-01-15', '16:31', True),
    ('SPHL', '2026-01-15', '04:00', False),
]

for ticker, date, time, had_trades in tickers:
    # Use quick_simulate for each ticker
    subprocess.run(['python', 'src/test/quick_simulate.py', ticker, date, time, str(had_trades)])
```

### Custom Analysis
Modify the `simulate_bot_trading` function to add custom analysis:
- Additional indicators
- Custom entry/exit logic
- Performance metrics

## Notes

- All times are in Eastern Time (ET)
- Data is collected from START_HOUR (default 4 AM) for indicator calculations
- Simulation starts from DETECTION_TIME but uses all earlier data
- Historical data is preserved for accurate indicator calculations
