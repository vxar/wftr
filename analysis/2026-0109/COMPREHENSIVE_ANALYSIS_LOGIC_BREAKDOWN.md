# Comprehensive Stock Analysis Logic Breakdown

## Overview

The `comprehensive_stock_analysis.py` script uses a **simple, focused pattern detection system** with **6 specific patterns** and a **robust exit strategy** that allows trades to run while protecting capital.

---

## Entry Logic

### Pattern Detection Function: `identify_entry_patterns(df, idx)`

**Requirements:**
- Must have at least 30 bars of data (`idx >= 30`)
- Returns a list of pattern dictionaries with: `name`, `score`, `confidence`

### The 6 Patterns Detected:

#### 1. **Volume_Breakout_Momentum** (Score: 8, Confidence: 0.85)
**Criteria:**
- `volume_ratio >= 1.8`
- `momentum_10 >= 2.0%` (10-minute price change)
- `breakout_10 == True` (price >= 1.02x of 10-period high)
- `price_above_all_ma == True` (price above SMA5, SMA10, SMA20)

**This is the PRIMARY pattern** - highest score (8) and appears most in successful trades.

---

#### 2. **Slow_Accumulation** (Score: 7, Confidence: 0.80)
**Criteria:**
- `1.8 <= volume_ratio < 3.5` (moderate volume, not explosive)
- `momentum_10 >= 2.0%`
- `momentum_20 >= 3.0%` (sustained momentum)
- `volume_trend_10 >= 1.3` (volume trending up)
- `macd_hist_accelerating == True`
- `price_position_20 >= 70%` (price in upper 30% of 20-period range)

**Purpose:** Captures slow, steady accumulation moves.

---

#### 3. **MACD_Acceleration_Breakout** (Score: 8, Confidence: 0.82)
**Criteria:**
- `macd_hist_accelerating == True`
- `macd_bullish == True` (MACD above signal)
- `breakout_20 == True` (price >= 1.02x of 20-period high)
- `volume_ratio >= 2.0`
- `momentum_20 >= 3.0%`

**Purpose:** Catches breakouts with strong MACD confirmation.

---

#### 4. **Golden_Cross_Volume** (Score: 7, Confidence: 0.78)
**Criteria:**
- `sma5_above_sma10 == True`
- `sma10_above_sma20 == True`
- `df.iloc[idx-1]['sma10_above_sma20'] == False` **AND** `current['sma10_above_sma20'] == True` (JUST crossed - important!)
- `volume_ratio >= 1.5`
- `momentum_10 >= 1.5%`

**Purpose:** Catches MA crossover breakouts with volume confirmation.

---

#### 5. **Consolidation_Breakout** (Score: 8, Confidence: 0.83)
**Criteria:**
- `in_consolidation == False` (currently NOT consolidating)
- `df.iloc[idx-5:idx]['in_consolidation'].sum() >= 3` (was consolidating in last 5 bars)
- `breakout_10 == True`
- `volume_ratio >= 2.0`
- `price_above_all_ma == True`

**Purpose:** Catches breakouts from consolidation patterns.

---

#### 6. **RSI_Accumulation_Entry** (Score: 7, Confidence: 0.75)
**Criteria:**
- `rsi_accumulation == True` (RSI between 50-65)
- `momentum_10 >= 2.0%`
- `volume_ratio >= 1.8`
- `macd_hist_increasing == True`
- `higher_high_20 == True` (higher highs pattern)

**Purpose:** Enters during RSI accumulation zone with momentum building.

---

### Pattern Selection Logic (in `simulate_trades`):

```python
# Find best pattern
best_pattern = None
for pattern in patterns:
    if pattern['score'] >= min_score:  # min_score = 6
        if best_pattern is None or pattern['score'] > best_pattern['score']:
            best_pattern = pattern
```

**Key Points:**
- ✅ Selects pattern with **HIGHEST SCORE** (not confidence!)
- ✅ Only accepts patterns with `score >= 6`
- ✅ Only **ONE pattern** selected per entry opportunity
- ✅ If multiple patterns have same score, first one wins (but this is rare)

**Entry Decision:**
- If `best_pattern` found → Enter trade
- Entry price: Current close price
- Stop loss: `entry_price * 0.85` (15% stop)
- Target: `entry_price * 1.20` (20% target)

---

## Exit Logic

The exit logic is **sophisticated** and designed to **let winners run** while **protecting capital**.

### Exit Check Order (CRITICAL - order matters!):

#### 1. **Hard Stop Loss (ALWAYS ACTIVE)**
```python
if current_price <= stop_loss:  # stop_loss = entry_price * 0.85
    exit_reason = "Hard Stop Loss (15%)"
    exit_price = stop_loss
```
- **No exceptions** - always active
- **15% loss** from entry price
- Takes priority over all other exit conditions

---

#### 2. **Minimum Hold Time (20 minutes)**
```python
elif hold_time_min < 20:
    exit_reason = None  # No exit allowed (except hard stop)
```
- **BLOCKS all other exits** (except hard stop) for first 20 minutes
- **Key Design Decision:** Prevents premature exits during normal price fluctuations
- Allows trades to "breathe" and develop

---

#### 3. **Dynamic Trailing Stop (Only after 20 minutes)**

**Trailing Stop Calculation:**
```python
if hold_time_min < 30:
    trailing_pct = 0.07  # 7% trailing stop
else:
    trailing_pct = 0.10  # 10% trailing stop after 30 min

# Adjust based on profit level
if current_profit_pct > 10:
    trailing_pct = 0.10  # Widen to 10% if profit > 10%
elif current_profit_pct > 5:
    trailing_pct = 0.07  # Use 7% if profit > 5%

trailing_stop = max_price_during * (1 - trailing_pct)
```

**Logic:**
- Calculated from **maximum price reached** during hold
- **7% trailing stop** for first 30 minutes
- **10% trailing stop** after 30 minutes
- **Adjusts wider** (10%) if profit > 10% (allows winners to run)
- **7% minimum** if profit > 5% (protects smaller gains)

**Key Insight:** Uses `max_price_during` (highest price since entry), not just current price, to give trades room to fluctuate upward.

---

#### 4. **Strong Reversal Signals (Requires 3+ confirmations)**

Only checked if:
- `exit_reason is None` (no other exit triggered)
- `hold_time_min >= 20` (after minimum hold time)

**6 Reversal Signals Detected:**

1. **Price below MAs**: `close < sma_10 AND close < sma_20`
2. **MACD Bearish Crossover**: Previous bar MACD bullish, current bar MACD bearish
3. **MACD Histogram Negative**: `macd_hist < 0`
4. **Volume + Price Decline**: Volume declined 30%+ AND price declined (over 5 bars)
5. **Price 5%+ Below High**: `(max_price_during - current_price) / max_price_during >= 5%`
6. **RSI Drop**: RSI dropped from >70 (overbought) to <50 (bearish) in 3 bars

**Exit Condition:**
```python
if reversal_signals >= 3:  # Need 3+ signals
    exit_reason = f"Strong Reversal ({reversal_signals} signals)"
```

**Key Design:** Requires **multiple confirmations** (3+) to prevent false exits from single bearish signals.

---

#### 5. **Profit Target (Optional)**
```python
if exit_reason is None and hold_time_min >= 30 and current_profit_pct >= 20:
    exit_reason = "Profit Target (20%+)"
    exit_price = target if current_high >= target else current_price
```

**Conditions:**
- Only if no other exit triggered
- Must hold for **30+ minutes**
- Must have **20%+ profit**

**Key Design:** Only takes profit target after sufficient time AND sufficient gain. Allows smaller profits to potentially grow.

---

## Key Design Principles

### Entry Principles:
1. **Simple Pattern Detection**: Only 6 well-defined patterns (not dozens)
2. **Score-Based Selection**: Chooses highest score pattern (prioritizes quality)
3. **Minimum Score Threshold**: Only accepts patterns with score >= 6
4. **Single Pattern Per Entry**: One pattern, one entry (no pattern overlap)

### Exit Principles:
1. **Capital Protection First**: Hard stop always active (15% max loss)
2. **Let Winners Run**: 20-minute minimum hold prevents premature exits
3. **Dynamic Trailing Stops**: Adjusts based on hold time and profit level
4. **Multiple Confirmations**: Strong reversal requires 3+ signals (prevents false exits)
5. **Profit Target Last Resort**: Only after 30+ min and 20%+ gain

### Critical Differences from Other Systems:

1. **20-Minute Minimum Hold**: Most systems exit too quickly (5-10 min). This allows trades to develop.
2. **Trailing Stop from Max Price**: Uses highest price reached, not entry price, giving more room.
3. **Multiple Reversal Signals**: Requires 3+ confirmations, not just 1-2 signals.
4. **Score-Based Selection**: Uses pattern score (quality indicator) not just confidence.
5. **Simple Pattern Set**: Only 6 patterns, each well-defined and tested.

---

## Data Flow

1. **Fetch 1-minute data** from 4 AM onwards
2. **Calculate base indicators** (SMA, RSI, MACD, volume ratios)
3. **Calculate advanced indicators** (momentum, breakouts, price positions, etc.)
4. **Simulate trades**:
   - Loop through each bar starting from index 30
   - Check for entry if no position
   - Check for exit if position exists
5. **Export results** to CSV with detailed trade information

---

## Performance Characteristics

Based on the original analysis results:

- **ANPA**: 11 trades, 63.6% win rate, 35.90% total P&L
- **Volume_Breakout_Momentum**: Best total P&L (26.02% across 10 trades)
- **RSI_Accumulation_Entry**: Best average P&L (4.28% per trade)
- **Overall**: 31 trades across 5 stocks, 58.1% win rate, 50.29% total P&L

---

## Implementation Notes

The original code uses:
- **Simple function-based pattern detection** (`identify_entry_patterns`)
- **Dictionary-based pattern storage** (not class-based PatternSignal objects)
- **Score-based selection** (not confidence-based)
- **Direct DataFrame manipulation** (not complex pattern detector class)

This simplicity is actually a **strength** - it's easier to understand, debug, and verify than complex class hierarchies.
