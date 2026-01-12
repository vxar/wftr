# Slow Mover Entry Logic Implementation Summary

## Overview

Successfully implemented slow mover entry logic as an **alternative path** that only activates when the original entry logic fails to place a trade. The implementation does **NOT disturb the current logic** - it's purely additive.

---

## Changes Made

### 1. ActivePosition Dataclass (`src/core/realtime_trader.py`)

**Added field:**
```python
is_slow_mover_entry: bool = False  # Flag to mark slow mover entries (uses different exit logic)
```

This flag marks positions entered via the slow mover path, enabling different exit logic.

---

### 2. Entry Logic Flow (`analyze_data` method)

**Modified flow:**
```python
# Check for new entry signals (only if no active position)
entry_signal = None
if ticker not in self.active_positions:
    # FIRST: Try original entry logic
    entry_signal = self._check_entry_signal(df, ticker)
    
    # SECOND: If original logic found no entry, try slow mover logic
    if entry_signal is None:
        entry_signal = self._check_slow_mover_entry_signal(df, ticker)
```

**Key Points:**
- Original logic runs **FIRST** (unchanged)
- Slow mover logic **ONLY** runs if original returns `None`
- If original finds a trade, slow mover logic is **never executed**

---

### 3. Slow Mover Entry Check (`_check_slow_mover_entry_signal` method)

**New method that implements slow mover criteria:**

#### Volume Requirements:
- **Absolute Volume**: 200K minimum over 60 minutes (vs 500K normal)
- **Volume Ratio**: 1.8x - 3.5x (moderate-high, not explosive)
- **Volume Building**: Last 5 periods >= 110% of previous 5 periods
- **Volume Acceleration**: Current volume >= 1.3x of 10-period average
- **No Declining Volume**: Volume not declining for 3+ consecutive periods

#### Momentum Requirements:
- **10-minute momentum**: >= 2.0%
- **20-minute momentum**: >= 3.0%
- **Momentum consistency**: 10-min >= 80% of 20-min (not decelerating)

#### Technical Setup:
- **MACD Accelerating**: Histogram increasing over 3+ periods
- **Breakout Confirmation**: Price >= 1.02x of 10-period high (2% breakout)
- **Higher Highs Pattern**: 20-period higher highs confirmed
- **RSI Accumulation Zone**: RSI between 50-65 (not overbought/oversold)
- **Price Above All MAs**: SMA5, SMA10, SMA20
- **MACD Bullish**: MACD above signal line

#### Pattern Quality:
- **Confidence Threshold**: >= 80% (higher bar for slow movers)
- **Pattern Selection**: Best pattern from detected patterns

**Returns:**
- `TradeSignal` with `indicators['is_slow_mover_entry'] = True` if criteria met
- `None` if criteria not met

---

### 4. Position Creation (`enter_position` method)

**Modified to mark slow mover entries:**
```python
# Check if this is a slow mover entry
is_slow_mover_entry = signal.indicators.get('is_slow_mover_entry', False) if signal.indicators else False

position = ActivePosition(
    ...
    is_slow_mover_entry=is_slow_mover_entry
)
```

**Key Points:**
- Reads the slow mover flag from the TradeSignal's indicators dict
- Sets `is_slow_mover_entry` field in the position
- This flag is used by exit logic to determine which exit strategy to use

---

### 5. Exit Logic (`_check_exit_signals` method)

**Modified to use different exit logic for slow movers:**

```python
# SLOW MOVER EXIT LOGIC: Use different logic if this is a slow mover entry
is_slow_mover_entry = position.is_slow_mover_entry

# ... exit conditions check ...

# 3. Trailing stop logic (different for slow movers vs normal)
elif position.max_price_reached > 0 and position.unrealized_pnl_pct >= 3.0:
    if is_slow_mover_entry:
        # SLOW MOVER EXIT LOGIC: Wider trailing stops (5%), minimum hold time 10 minutes
        min_hold_time_slow_mover = 10
        
        # Skip trailing stop if within minimum hold time
        if minutes_since_entry < min_hold_time_slow_mover:
            # Don't exit on trailing stop
        else:
            # Slow movers use wider trailing stop: 5% (fixed)
            trailing_stop_pct = 5.0
            # ... trailing stop logic ...
    else:
        # NORMAL EXIT LOGIC (original logic - unchanged)
        # ... original trailing stop logic ...
```

**Slow Mover Exit Logic:**
- **Minimum Hold Time**: 10 minutes (vs 15 for premarket in normal logic)
- **Trailing Stop**: 5% (fixed, wider than normal)
- **ATR-based Stop**: 2.5x ATR (vs 2x for normal)
- **Profit Target**: 20% (same as normal)

**Normal Exit Logic:**
- **Unchanged** - original logic remains exactly as before
- Only executes when `is_slow_mover_entry == False`

---

### 6. Advanced Indicators (`_calculate_advanced_indicators_for_slow_mover` method)

**New helper method that calculates indicators needed for slow mover detection:**

Calculates:
- `momentum_10`, `momentum_20`
- `volume_ma_10`, `volume_ma_20`
- `high_10`, `low_10`, `high_20`, `low_20`
- `sma_5`, `sma_10`, `sma_20` (if not already calculated)
- `price_above_all_ma`
- `macd`, `macd_signal`, `macd_hist` (if not already calculated)
- `macd_bullish`, `macd_hist_accelerating`
- `higher_high_20`
- `breakout_10`
- `rsi_accumulation`

---

## Key Design Principles

### 1. **Additive, Not Modifying**
- Original logic is **completely unchanged**
- Slow mover logic only runs when original logic fails
- If original finds a trade, slow mover is never executed

### 2. **Clear Separation**
- Slow mover entries are **tagged** with `is_slow_mover_entry = True`
- Exit logic **checks the flag** to determine which strategy to use
- Normal entries use normal exit logic (unchanged)

### 3. **Quality Over Quantity**
- Slow mover logic has **stricter criteria** (80% confidence vs 72% normal)
- Multiple confirmations required (momentum, volume, MACD, breakouts)
- Still requires minimum volume (200K) for liquidity

### 4. **Appropriate Exit Strategy**
- Slow movers use **wider trailing stops** (5% vs 2.5-5% progressive for normal)
- **Shorter minimum hold time** (10 min vs 15 min for premarket)
- Designed to let slow accumulations develop while protecting capital

---

## Testing Checklist

- [ ] Original logic still works unchanged
- [ ] Slow mover logic only activates when original returns None
- [ ] Slow mover entries are correctly tagged
- [ ] Slow mover exits use different logic
- [ ] Normal entries use normal exit logic (unchanged)
- [ ] No performance degradation

---

## Files Modified

1. `src/core/realtime_trader.py`
   - Added `is_slow_mover_entry` field to `ActivePosition`
   - Added `_check_slow_mover_entry_signal` method
   - Added `_calculate_advanced_indicators_for_slow_mover` method
   - Modified `analyze_data` to check slow mover if original returns None
   - Modified `enter_position` to mark slow mover entries
   - Modified `_check_exit_signals` to use slow mover exit logic

---

## Next Steps

1. **Test the implementation** on historical data (ANPA, INBS, etc.)
2. **Verify** that original logic is not impacted
3. **Monitor** slow mover entry/exit performance
4. **Adjust** thresholds if needed based on results

---

## Notes

- The implementation follows the slow mover strategy plan in `analysis/SLOW_MOVER_STRATEGY_PLAN.md`
- Slow mover criteria are based on the comprehensive analysis of stocks like ANPA and INBS
- The exit logic for slow movers uses wider trailing stops to allow slow accumulations to develop
- All changes are backward compatible - existing functionality is preserved
