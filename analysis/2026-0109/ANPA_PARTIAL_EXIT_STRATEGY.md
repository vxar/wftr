# ANPA Partial Exit Strategy - Implementation Plan

## Problem Analysis

Looking at the optimized CSV results and the Friday chart:
- **Stock High**: $108.68
- **Prev Close**: ~$24.20
- **Available Gain**: ~350%
- **Current Capture**: Only 10.9%

### Key Missed Opportunities:

1. **Trade #13** (Entry: $69.00, Exit: $82.00, 18.84% gain)
   - Exited via "Strong Reversal (3 signals)" at $82
   - Stock continued to $108.68
   - **Missed**: $26.68 per share (32.7% more gain)

2. **Trade #14** (Entry: $95.00, Exit: $101.07, 6.39% gain)
   - Exited via "Trailing Stop (7%)" at $101.07
   - Stock continued to $108.68
   - **Missed**: $7.61 per share (7.5% more gain)

3. **Trade #10** (Entry: $54.56, Exit: $65.47, 20% gain)
   - Good exit via profit target
   - But could have scaled out and let remaining position run to $108.68

---

## Solution: Implement Partial Exit Strategy

### Strategy Overview

**Scale-Out Approach**: Lock in profits at multiple levels while letting remaining position capture more gains.

### Implementation Plan

#### 1. **Partial Exit Levels**

**Level 1 (50% position)**: At 20% profit
- Lock in 10% gain on full position
- Remaining 50% continues with wider trailing stop

**Level 2 (25% of original)**: At 40% profit  
- Lock in additional 10% gain on full position
- Remaining 25% continues with very wide trailing stop

**Level 3 (12.5% of original)**: At 80% profit
- Lock in additional 10% gain on full position
- Remaining 12.5% continues with very wide trailing stop (or disabled)

**Remaining (12.5%)**: Let run with:
- Disabled trailing stop for 100%+ profit
- Only exit on hard stop (15%) or strong reversal (6+ signals)
- Capture massive moves (200-400%+)

---

#### 2. **Enhanced Exit Logic**

**For Remaining Position After Partial Exits**:

- **100%+ profit**: Disable trailing stop entirely
- **50%+ profit**: 30% trailing stop (vs 20% current)
- **30%+ profit**: 20% trailing stop (vs 15% current)
- **20%+ profit**: 15% trailing stop (vs 12% current)

**Strong Reversal**:
- **200%+ profit**: 6+ signals required
- **100%+ profit**: 5+ signals required (current)
- **50%+ profit**: 4+ signals required (current)
- **<50% profit**: 3+ signals required (current)

---

#### 3. **Progressive Profit Targets**

Instead of single 20% profit target, use progressive targets:

- **After 30 min AND 30% profit**: Take 25% position (scales out)
- **After 60 min AND 60% profit**: Take 25% position (scales out)
- **After 90 min AND 100% profit**: Take 25% position (scales out)
- **Remaining 25%**: Let run with very wide trailing stop or disabled

---

## Code Implementation

### Modifications Needed in `simulate_trades`:

1. **Track Position Size**:
   ```python
   current_position = {
       ...
       'position_size': 1.0,  # 100% of original position
       'original_entry_price': current_price,
       'partial_exits': []  # Track partial exits
   }
   ```

2. **Partial Exit Logic**:
   ```python
   # Check for partial exits (scale out strategy)
   if current_position['position_size'] >= 0.5 and current_profit_pct >= 20:
       # First partial: 50% at 20% profit
       partial_exit = {
           'time': current_time,
           'price': current_price,
           'size': 0.5,  # 50% of position
           'profit_pct': current_profit_pct,
           'reason': 'Partial Exit 50% at 20% profit'
       }
       current_position['partial_exits'].append(partial_exit)
       current_position['position_size'] = 0.5  # Reduce to 50%
       
   elif current_position['position_size'] >= 0.25 and current_profit_pct >= 40:
       # Second partial: 25% at 40% profit
       partial_exit = {
           'time': current_time,
           'price': current_price,
           'size': 0.25,  # 25% of position
           'profit_pct': current_profit_pct,
           'reason': 'Partial Exit 25% at 40% profit'
       }
       current_position['partial_exits'].append(partial_exit)
       current_position['position_size'] = 0.25  # Reduce to 25%
       
   elif current_position['position_size'] >= 0.125 and current_profit_pct >= 80:
       # Third partial: 12.5% at 80% profit
       partial_exit = {
           'time': current_time,
           'price': current_price,
           'size': 0.125,  # 12.5% of position
           'profit_pct': current_profit_pct,
           'reason': 'Partial Exit 12.5% at 80% profit'
       }
       current_position['partial_exits'].append(partial_exit)
       current_position['position_size'] = 0.125  # Reduce to 12.5%
   ```

3. **Enhanced Trailing Stops** (for remaining position):
   ```python
   # Widen significantly for strong moves
   if current_position['position_size'] <= 0.25:  # After partial exits
       if current_profit_pct >= 100:
           trailing_pct = None  # Disable trailing stop for 100%+ profit
       elif current_profit_pct >= 50:
           trailing_pct = 0.30  # 30% trailing stop for 50%+ profit
       elif current_profit_pct >= 30:
           trailing_pct = 0.20  # 20% trailing stop for 30%+ profit
       else:
           trailing_pct = 0.15  # 15% trailing stop
   ```

4. **Track Total P&L Across All Exits**:
   ```python
   # Calculate total P&L including partial exits
   total_pnl = 0
   for partial in current_position['partial_exits']:
       partial_pnl = ((partial['price'] - entry_price) / entry_price) * 100 * partial['size']
       total_pnl += partial_pnl
   
   # Add remaining position P&L
   remaining_pnl = ((current_price - entry_price) / entry_price) * 100 * current_position['position_size']
   total_pnl += remaining_pnl
   ```

---

## Expected Impact

### Example: Trade #13 with Partial Exits

**Current**:
- Entry: $69.00
- Exit: $82.00 (100% position)
- P&L: 18.84% gain
- Missed: $26.68 per share (32.7% more gain)

**With Partial Exits**:
- Entry: $69.00
- **Partial 1** (50% at 20% = $82.80): Lock in 10% gain
- **Partial 2** (25% at 40% = $96.60): Lock in 10% gain
- **Partial 3** (12.5% at 80% = $124.20): Lock in 10% gain
- **Remaining 12.5%** runs to $108.68: Additional 7.2% gain
- **Total P&L**: ~37.2% gain (vs 18.84% current)

### Overall Expected Performance:
- **Current**: 43.97% total P&L, 10.9% capture rate
- **With Partial Exits**: 150-200%+ total P&L, 40-60% capture rate

---

## Implementation Priority

### Priority 1: ⭐⭐⭐ **Partial Exits at 20%, 40%, 80%**
- **Impact**: Very High
- **Complexity**: High (requires position tracking)
- **Risk**: Low (reduces risk as position size decreases)

### Priority 2: ⭐⭐⭐ **Disable Trailing Stop for 100%+ Profit**
- **Impact**: Very High
- **Complexity**: Low
- **Risk**: High (can give back significant gains)

### Priority 3: ⭐⭐ **Very Wide Trailing Stops After Partial Exits**
- **Impact**: High
- **Complexity**: Low
- **Risk**: Medium

---

## Testing Plan

1. Implement partial exit logic in `simulate_trades`
2. Test on ANPA with Friday's data
3. Compare results: before vs after
4. Check capture rate improvement
5. Verify no increase in losses
6. Test on other stocks to ensure no negative impact
