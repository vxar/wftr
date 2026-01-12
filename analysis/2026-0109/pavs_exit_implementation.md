# PAVS Exit Criteria Implementation - Priority 1 & 2

## Implementation Summary

Successfully implemented Priority 1 and Priority 2 from the PAVS exit analysis to prevent premature exits during premarket.

## Changes Made

### Priority 1: Minimum Hold Time for Premarket Entries ✅

**Implementation:**
- Added 15-minute minimum hold time for positions entered during premarket (before 9:30 AM ET)
- During this period, trailing stop exits are **disabled**
- Only allows exits for:
  - Hard stop loss
  - Profit target reached
  - Setup failure
  - Trend reversal signals

**Code Location:** `src/core/realtime_trader.py` lines 384-456

**Key Features:**
- Detects if entry was during premarket (hour < 9 or hour == 9 and minute < 30)
- Calculates minutes since entry
- Skips trailing stop check if within 15-minute minimum hold period
- Logs debug message when skipping trailing stop

### Priority 2: Wider Trailing Stops During Premarket ✅

**Implementation:**
- Applies 1.5x multiplier to trailing stop width during premarket
- Caps maximum trailing stop at 6% to prevent excessive risk
- Applies to both:
  - Positions entered during premarket
  - Positions currently being evaluated during premarket hours

**Code Location:** `src/core/realtime_trader.py` lines 477-492

**Key Features:**
- Detects if entry was premarket OR if current time is premarket
- Multiplies normal trailing stop width by 1.5x
- Examples:
  - Normal 2.5% stop → 3.75% during premarket
  - Normal 3.0% stop → 4.5% during premarket
  - Normal 5.0% stop → 6.0% (capped) during premarket
- Logs debug message when wider stop is applied

## Expected Impact

### For PAVS Trade (Example):

**Before:**
- Entry: 09:06:33 @ $2.5400
- Exit: 09:09:42 @ $2.5510 (3 minutes)
- P&L: +0.43% (+$23.89)
- **Missed:** +19.69% (+$1,089)

**After (with new rules):**
- Entry: 09:06:33 @ $2.5400
- **Minimum hold time:** 15 minutes (until 09:21:33)
- **Trailing stop disabled** until 09:21:33
- **Wider trailing stop** after 15 minutes (3.75% instead of 2.5%)
- Would have held through the initial volatility
- Would have captured more of the +19.69% move

## Technical Details

### Timezone Handling
- All times normalized to US/Eastern timezone
- Handles both timezone-aware and timezone-naive datetimes
- Works with pandas Timestamp and Python datetime objects

### Integration with Existing Logic
- Works seamlessly with existing trailing stop logic
- Maintains all existing features:
  - Progressive trailing stops (wider for bigger winners)
  - ATR-based stops (2x ATR)
  - Stop never goes below entry price
  - Stop only moves up, never down

### Logging
- Debug logs when skipping trailing stop during minimum hold time
- Debug logs when applying wider trailing stop for premarket
- Info logs when trailing stop is activated

## Testing Recommendations

1. **Monitor premarket entries** for next few days
2. **Verify minimum hold time** is working (check logs)
3. **Verify wider stops** are being applied (check logs)
4. **Compare results** with previous premarket exits
5. **Adjust parameters** if needed:
   - Minimum hold time (currently 15 minutes)
   - Trailing stop multiplier (currently 1.5x)
   - Maximum cap (currently 6%)

## Configuration

Current settings (can be adjusted if needed):
- `min_hold_time_premarket = 15` minutes
- `trailing_stop_multiplier = 1.5x`
- `max_trailing_stop_pct = 6.0%`

## Files Modified

- `src/core/realtime_trader.py`
  - Added pytz import
  - Added premarket detection logic
  - Added minimum hold time check
  - Added wider trailing stop logic

## Next Steps

1. Monitor live trading to validate improvements
2. Consider implementing Priority 3 (disable trailing stops during premarket) if needed
3. Adjust parameters based on real-world results
4. Document any additional improvements needed
