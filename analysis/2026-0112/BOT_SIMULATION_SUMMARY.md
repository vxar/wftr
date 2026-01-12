# Bot Simulation Summary - EVTV, UP, BDSX

**Simulation Date**: 2026-01-12  
**Stocks Simulated**: EVTV, UP, BDSX  
**Simulation Period**: From 4 AM ET

---

## Data Files Saved

All 1-minute data has been downloaded and saved to `test_data/` folder:

- **EVTV_1min_20260112.csv** (16.54 KB) - 299 minutes of data
- **UP_1min_20260112.csv** (13.97 KB) - 247 minutes of data
- **BDSX_1min_20260112.csv** (17.85 KB) - 348 minutes of data

---

## Trading Results

### Overall Summary

- **Total Trades**: 5
- **Total P&L**: **+$217.80**
- **Win Rate**: 40.0% (2 wins, 3 losses)
- **Average P&L per Trade**: +$43.56

### By Stock

#### BDSX: ✅ 2 Wins (+$350.00)
1. **Entry**: 09:34:00 @ $7.25 - Strong_Bullish_Setup (85.0%)
   - **Exit**: 09:40:00 @ $7.36 - Trend weakness detected
   - **P&L**: +$110.00 (+1.52%)
   - **Hold Time**: 6 minutes
   - **Max Price**: $7.87 (+8.55%)
   - **Capture Rate**: 17.7%

2. **Entry**: 09:48:00 @ $8.05 - Strong_Bullish_Setup (85.0%)
   - **Exit**: 10:10:00 @ $8.29 - Trend weakness detected
   - **P&L**: +$240.00 (+2.98%)
   - **Hold Time**: 22 minutes
   - **Max Price**: $9.08 (+12.80%)
   - **Capture Rate**: 23.3%

#### EVTV: ❌ 2 Losses (-$103.00)
1. **Entry**: 08:55:00 @ $1.13 - Strong_Bullish_Setup (85.0%)
   - **Exit**: 08:58:00 @ $1.10 - Trend weakness detected
   - **P&L**: -$30.00 (-2.65%)
   - **Hold Time**: 3 minutes
   - **Max Price**: $1.14 (+0.88%)
   - **Issue**: Exited too early, missed potential gains

2. **Entry**: 10:06:00 @ $1.275 - Strong_Bullish_Setup (85.0%)
   - **Exit**: 10:07:00 @ $1.202 - Setup failed - multiple failure signals detected
   - **P&L**: -$73.00 (-5.73%)
   - **Hold Time**: 1 minute
   - **Max Price**: $1.28 (+0.39%)
   - **Issue**: Setup failed immediately after entry

#### UP: ❌ 1 Loss (-$29.20)
1. **Entry**: 10:19:00 @ $0.9636 - Volume_Breakout (80.0%)
   - **Exit**: 10:20:00 @ $0.9344 - Bearish reversal pattern detected
   - **P&L**: -$29.20 (-3.03%)
   - **Hold Time**: 1 minute
   - **Max Price**: $0.9725 (+0.92%)
   - **Issue**: Exited immediately on bearish reversal

---

## Analysis

### Strengths

1. **BDSX Performance**: Both trades were profitable with good capture rates
2. **Entry Timing**: Entries were placed at appropriate times (09:34, 09:48)
3. **Pattern Detection**: Strong_Bullish_Setup pattern worked well for BDSX

### Issues

1. **Early Exits**: 
   - EVTV trades exited too early (3 minutes, 1 minute)
   - UP trade exited immediately (1 minute)
   - "Trend weakness detected" and "Setup failed" are triggering too soon

2. **Low Capture Rates**:
   - BDSX trades captured only 17.7% and 23.3% of max potential
   - Max prices were significantly higher than exit prices

3. **Setup Failure**:
   - EVTV second trade failed immediately after entry
   - May need to relax "Setup failed" detection

### Recommendations

1. **Extend Minimum Hold Time**: 
   - Current: No minimum hold time for trend weakness
   - Proposed: 10-15 minute minimum hold time before trend weakness exit

2. **Relax Setup Failed Detection**:
   - Current: Triggers immediately after entry
   - Proposed: Require 5+ minutes and multiple failure signals

3. **Improve Capture Rate**:
   - Current: Exiting on trend weakness too early
   - Proposed: Use wider trailing stops or partial exits

---

## Files Generated

1. **Data Files** (in `test_data/`):
   - `EVTV_1min_20260112.csv`
   - `UP_1min_20260112.csv`
   - `BDSX_1min_20260112.csv`

2. **Trades CSV**:
   - `analysis/BOT_SIMULATION_TRADES_20260112.csv` - Detailed trade log with P&L

3. **Reports**:
   - `analysis/BOT_SIMULATION_SUMMARY.md` - This summary

---

## Next Steps

1. ✅ **Data Downloaded** - All 1-minute data saved to test_data
2. ✅ **Simulation Complete** - Bot trades simulated and logged
3. ⏳ **Analysis** - Review early exits and low capture rates
4. ⏳ **Optimization** - Adjust exit logic to improve capture rates
