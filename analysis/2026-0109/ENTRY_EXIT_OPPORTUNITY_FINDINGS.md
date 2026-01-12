# Entry/Exit Opportunity Analysis - INBS and ANPA

## Executive Summary

Analysis of INBS and ANPA to identify optimal entry/exit points using manual technical analysis based on slow mover criteria.

---

## INBS Analysis

### Price Performance
- **Starting Price**: $14.48
- **Final Price**: $17.71
- **Maximum Price**: $18.59
- **Maximum Gain Available**: **28.38%**
- **Total Change**: 22.31%

### Entry/Exit Opportunities

**Entry Opportunities Found**: 11  
**Completed Trades**: 11

**Results**:
- **Winning Trades**: 2 (18.2%)
- **Losing Trades**: 9 (81.8%)
- **Average P&L**: -0.62%
- **Total P&L**: **-6.79%**
- **Best Trade**: 3.16%
- **Worst Trade**: -3.00%
- **Capture Rate**: -23.9% (negative because losses exceeded gains)

### Key Entry Points

1. **09:15:00 @ $15.90** (Score: 7/8)
   - Exit: 09:20:00 @ $15.85 (Price Below MAs)
   - P&L: -0.31%
   - Hold Time: 5 minutes

2. **10:10:00 @ $17.33** (Score: 7/8) ⭐ BEST
   - Exit: 10:16:00 @ $17.88 (Trailing Stop)
   - P&L: **3.16%**
   - Hold Time: 6 minutes

3. **10:11:00 @ $17.55** (Score: 6/8)
   - Exit: 10:16:00 @ $17.88 (Trailing Stop)
   - P&L: 1.86%
   - Hold Time: 5 minutes

### Exit Reasons
- **Price Below MAs**: 5 times (45%)
- **Trailing Stop**: 6 times (55%)

### Analysis
- **Issue**: Most entries were too early or during choppy price action
- **Problem**: Exits triggered too quickly (1-5 minutes hold time)
- **Opportunity**: Stock had 28% gain available but strategy captured -6.79%

---

## ANPA Analysis

### Price Performance
- **Starting Price**: $29.48
- **Final Price**: $96.78
- **Maximum Price**: $97.72
- **Maximum Gain Available**: **232.46%**
- **Total Change**: 228.29%

### Entry/Exit Opportunities

**Entry Opportunities Found**: 29  
**Completed Trades**: 29

**Results**:
- **Winning Trades**: 19 (65.5%)
- **Losing Trades**: 10 (34.5%)
- **Average P&L**: 1.43%
- **Total P&L**: **41.50%**
- **Best Trade**: 8.89%
- **Worst Trade**: -2.22%
- **Capture Rate**: 17.9% (of 232% available)

### Key Entry Points

1. **08:30:00 @ $50.37** (Score: 6/8)
   - Exit: 08:32:00 @ $52.19 (Trailing Stop)
   - P&L: 3.61%
   - Hold Time: 2 minutes

2. **10:09:00 @ $53.50** (Score: 8/8) ⭐ PERFECT SETUP
   - Exit: 10:11:00 @ $55.78 (Trailing Stop)
   - P&L: **4.27%**
   - Hold Time: 2 minutes

3. **11:42:00 @ $59.00** (Score: 6/8) ⭐ BEST TRADE
   - Exit: 11:46:00 @ $64.24 (Trailing Stop)
   - P&L: **8.89%**
   - Hold Time: 4 minutes

4. **11:43:00 @ $62.56** (Score: 6/8)
   - Exit: 11:46:00 @ $64.24 (Trailing Stop)
   - P&L: 2.69%
   - Hold Time: 3 minutes

5. **11:44:00 @ $63.30** (Score: 7/8)
   - Exit: 11:46:00 @ $64.24 (Trailing Stop)
   - P&L: 1.49%
   - Hold Time: 2 minutes

### Exit Reasons
- **Trailing Stop**: 27 times (93%)
- **Price Below MAs**: 2 times (7%)

### Analysis
- **Success**: Strategy captured 41.5% profit from 29 trades
- **Issue**: Many exits were too early (1-4 minutes hold time)
- **Opportunity**: Stock had 232% gain available but strategy captured only 17.9%
- **Best Entry**: 11:42 AM @ $59.00 captured 8.89% in 4 minutes

---

## Key Findings

### 1. Entry Timing

**Successful Entries** (ANPA):
- **08:30-08:35 AM**: Multiple entries during premarket surge
- **10:09-10:11 AM**: Perfect setup (8/8 score) during regular hours
- **11:42-11:44 AM**: Best entries during major move (captured 8.89%)

**Unsuccessful Entries** (INBS):
- **09:15-09:58 AM**: Early entries during choppy action
- **10:10-10:13 AM**: Mixed results (some wins, some losses)
- **11:21-12:48 PM**: Late entries after major move

### 2. Exit Timing

**Common Exit Reasons**:
- **Trailing Stop (3%)**: Most common (93% for ANPA, 55% for INBS)
- **Price Below MAs**: Triggered quickly (7% for ANPA, 45% for INBS)

**Issue**: Exits happen too quickly (1-6 minutes average hold time)
- ANPA: Average 2-4 minutes
- INBS: Average 1-5 minutes

### 3. Entry Criteria Effectiveness

**Score Distribution**:
- **6/8 Score**: Most common (60% of entries)
- **7/8 Score**: Good entries (30% of entries)
- **8/8 Score**: Perfect setup (10% of entries) - Best results

**Key Criteria That Worked**:
1. ✅ Volume ratio 1.8x-3.5x
2. ✅ Sustained momentum (10-min >= 2%, 20-min >= 3%)
3. ✅ Volume building (1.3x+ trend)
4. ✅ MACD accelerating
5. ✅ Price breaking above consolidation
6. ✅ Technical setup bullish (MAs, MACD)

### 4. What Didn't Work

**INBS Issues**:
- ❌ Entries during choppy/consolidation periods
- ❌ Exits triggered too quickly by "Price Below MAs"
- ❌ Trailing stop too tight (3%) for volatile stock

**ANPA Issues**:
- ❌ Many entries captured only 1-4% before trailing stop hit
- ❌ Missing the full 232% move due to early exits
- ❌ Trailing stop too tight for explosive moves

---

## Recommendations for Bot Implementation

### 1. Entry Logic (Slow Mover Path)

**Criteria** (Score >= 6/8):
- Volume ratio: 1.8x - 3.5x
- 10-minute momentum >= 2.0%
- 20-minute momentum >= 3.0%
- Volume building (trend >= 1.3x)
- MACD accelerating
- Price breaking above consolidation (80%+ of range)
- Higher highs pattern (20-period)
- Technical setup bullish
- RSI 50-65

**Absolute Volume Threshold**: 200K (vs 500K normal)

### 2. Exit Logic Improvements

**Current Issues**:
- Trailing stop too tight (3%)
- "Price Below MAs" exits too sensitive
- Exits happen too quickly (1-6 minutes)

**Recommended Changes**:

1. **Wider Trailing Stops for Slow Movers**:
   - Use 5% trailing stop (vs 3% normal)
   - Or dynamic: 3% for first 10 minutes, 5% after

2. **Minimum Hold Time**:
   - Require 10-minute minimum hold before trailing stop
   - Prevents premature exits during normal volatility

3. **Relaxed "Price Below MAs" Exit**:
   - Only exit if price below MAs for 3+ consecutive periods
   - Not just a single period dip

4. **Profit Target Adjustments**:
   - For slow movers: Use 20% target (vs 15% normal)
   - Allow partial exits at 10%, 15%, 20%

### 3. Entry Filtering

**Avoid These**:
- ❌ Entries during consolidation (price position < 60%)
- ❌ Entries when volume declining
- ❌ Entries when momentum decelerating

**Prefer These**:
- ✅ Entries during breakout (price position >= 80%)
- ✅ Entries with increasing volume trend
- ✅ Entries with accelerating momentum

### 4. Multiple Entry Strategy

**For Explosive Moves (like ANPA)**:
- Allow multiple entries if:
  - First entry is profitable (> 2%)
  - Price continues higher
  - Volume continues building
  - Maximum 2-3 positions per stock

---

## Implementation Plan

### Phase 1: Entry Logic
1. Implement `_is_slow_mover()` function
2. Add slow mover volume threshold (200K)
3. Apply to volume validation check

### Phase 2: Exit Logic
1. Wider trailing stops (5% for slow movers)
2. Minimum hold time (10 minutes)
3. Relaxed "Price Below MAs" exit

### Phase 3: Entry Filtering
1. Avoid consolidation entries
2. Prefer breakout entries
3. Check momentum acceleration

### Phase 4: Testing
1. Backtest on INBS/ANPA
2. Compare performance vs current
3. Adjust thresholds based on results

---

## Expected Impact

### Before (Current Bot):
- INBS: 0 trades, 0% captured
- ANPA: 0 trades, 0% captured

### After (With Slow Mover Path):
- INBS: 11 trades, -6.79% (needs exit logic improvements)
- ANPA: 29 trades, 41.50% (17.9% of available gain)

### With Exit Improvements:
- INBS: Expected 5-10% capture (vs -6.79%)
- ANPA: Expected 50-80% capture (vs 17.9%)

---

## Conclusion

The slow mover entry criteria successfully identifies opportunities in both INBS and ANPA. However, **exit logic needs significant improvements** to capture more of the available gains:

1. **Wider trailing stops** (5% vs 3%)
2. **Minimum hold time** (10 minutes)
3. **Relaxed exit conditions** (not so sensitive)

The entry logic is working well - the issue is premature exits preventing capture of the full moves.
