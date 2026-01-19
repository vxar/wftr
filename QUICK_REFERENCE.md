# Trading Bot Quick Reference Guide

## Key Parameters

### Entry Requirements
- **Minimum Confidence:** 72% (normal), 70% (fast movers), 80% (slow movers)
- **Minimum Price:** $0.50 per share
- **Volume Requirements:**
  - Normal: 500K over 60 min (or 167K over 20 min)
  - Fast movers: Same as normal
  - Slow movers: 200K minimum
- **Expected Gain:** Minimum 5.5%
- **Setup Confirmation:** 4+ out of last 6 periods

### Exit Management
- **Hard Stop Loss:** 15% (always active)
- **Minimum Hold Time:** 20 minutes (normal), 10 minutes (surge/slow mover)
- **Profit Target:** 20% (only after 30+ min AND 20%+ profit)
- **Trailing Stops:** Progressive based on profit level (5-20%)

### Position Sizing
- **Position Size:** 50% of capital per trade
- **Max Positions:** 3 concurrent
- **Max Trades/Day:** 1000

### Trading Window
- **Start:** 4:00 AM ET
- **End:** 8:00 PM ET
- **Sleep:** 8:00 PM - 4:00 AM ET

## Common Workflows

### Starting the Bot
```bash
python run_dashboard_with_bot.py
```
Access dashboard at `http://localhost:5000`

### Monitoring Positions
1. Open web dashboard
2. View "Active Positions" section
3. Check P&L, target, stop loss
4. Use "Update" button to refresh
5. Use "Close" button for manual exit

### Checking Rejected Entries
1. Open web dashboard
2. View "Rejected Entry Signals" section
3. See ticker, price, reason, timestamp
4. Entries persist for the day

### Running Daily Analysis
1. Open web dashboard
2. Navigate to Analytics page
3. Click "Run Analysis" button
4. View performance metrics and recommendations

## Troubleshooting

### Bot Not Starting
- Check if port 5000 is available
- Verify database file exists and is writable
- Check logs for error messages

### No Trades Executing
- Verify trading window is active (4 AM - 8 PM ET)
- Check if max positions reached
- Review rejected entries for reasons
- Verify capital is sufficient ($100+ per trade)

### Positions Not Updating
- Check if bot is running (status indicator)
- Verify data API connection
- Check logs for API errors
- Use "Update" button to force refresh

### Database Errors
- Check database file permissions
- Verify SQLite is installed
- Check for database corruption
- Review logs for specific errors

## Pattern Types

1. **Volume_Breakout_Momentum** - High volume breakout with momentum
2. **RSI_Accumulation_Entry** - RSI 50-65 with accumulation
3. **Golden_Cross_Volume** - MA crossover with volume
4. **Slow_Accumulation** - Gradual accumulation pattern
5. **MACD_Acceleration_Breakout** - MACD acceleration with breakout
6. **Consolidation_Breakout** - Breakout from consolidation

## Exit Reasons

- Stop loss hit
- Profit target reached
- Trailing stop hit
- Strong reversal detected
- Setup failed
- Partial profit taking
- Time-based exit

## Configuration Files

- `src/config/settings.py` - Main configuration
- Environment variables - Override defaults
- `.env` file - Optional configuration file

## Log Files

- `trading_bot.log` - Main log file
- Rotates when >10MB
- Keeps 5 backup files
