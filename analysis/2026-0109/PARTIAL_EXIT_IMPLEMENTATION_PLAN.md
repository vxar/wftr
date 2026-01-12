# Partial Exit Implementation Plan for Comprehensive Analysis

## Bot's Current Implementation

The bot (`src/core/realtime_trader.py`) already has partial exit logic:
- **50% at +4% profit** (first partial)
- **25% at +7% profit** (second partial, of remaining position)
- **25% remaining** (let run to target)

## Adaptation for Massive Moves (ANPA-style)

For stocks with massive moves (350%+), we need a different scale-out strategy:

### Proposed Strategy:
- **50% at 20% profit** (lock in 10% gain)
- **25% at 40% profit** (lock in additional 10% gain)
- **12.5% at 80% profit** (lock in additional 10% gain)
- **12.5% remaining** (let run with disabled trailing stop for 100%+ profit)

## Implementation in simulate_trades()

### Step 1: Add Position Size Tracking

Add to `current_position`:
```python
current_position = {
    ...
    'position_size': 1.0,  # 100% of original position
    'partial_exits': []  # Track partial exits
}
```

### Step 2: Implement Partial Exit Logic

Before checking for full exit, check for partial exits:

```python
# Check for partial exits (scale-out strategy for massive moves)
if current_position['position_size'] > 0.5 and current_profit_pct >= 20:
    # First partial: 50% at 20% profit
    partial_exit = {
        'time': current_time,
        'price': current_price,
        'size': 0.5,  # 50% of position
        'profit_pct': current_profit_pct,
        'entry_price': entry_price
    }
    current_position['partial_exits'].append(partial_exit)
    current_position['position_size'] = 0.5  # Reduce to 50%
    
elif current_position['position_size'] > 0.25 and current_profit_pct >= 40:
    # Second partial: 25% at 40% profit
    partial_exit = {
        'time': current_time,
        'price': current_price,
        'size': 0.25,  # 25% of position
        'profit_pct': current_profit_pct,
        'entry_price': entry_price
    }
    current_position['partial_exits'].append(partial_exit)
    current_position['position_size'] = 0.25  # Reduce to 25%
    
elif current_position['position_size'] > 0.125 and current_profit_pct >= 80:
    # Third partial: 12.5% at 80% profit
    partial_exit = {
        'time': current_time,
        'price': current_price,
        'size': 0.125,  # 12.5% of position
        'profit_pct': current_profit_pct,
        'entry_price': entry_price
    }
    current_position['partial_exits'].append(partial_exit)
    current_position['position_size'] = 0.125  # Reduce to 12.5%
```

### Step 3: Adjust Trailing Stop for Remaining Position

After partial exits, use wider trailing stops:

```python
# Adjust trailing stop based on remaining position size and profit
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

### Step 4: Calculate Total P&L Including Partial Exits

When position exits, calculate total P&L:

```python
# Calculate P&L from partial exits
partial_pnl_pct = 0
for partial in current_position['partial_exits']:
    partial_pnl = ((partial['price'] - entry_price) / entry_price) * 100 * partial['size']
    partial_pnl_pct += partial_pnl

# Calculate P&L from remaining position
remaining_pnl_pct = ((exit_price - entry_price) / entry_price) * 100 * current_position['position_size']

# Total P&L
total_pnl_pct = partial_pnl_pct + remaining_pnl_pct
```

## Expected Impact

### Current Performance:
- **Total P&L**: 43.97%
- **Capture Rate**: 10.9%

### Expected Performance (With Partial Exits):
- **Total P&L**: 150-200%+ (3-4x improvement)
- **Capture Rate**: 40-60% (4-6x improvement)
