# Slow Mover Strategy Plan

## Problem Statement

The current bot misses profitable opportunities in "slow movers" - stocks that:
- Have **moderate-high volume ratios** (1.8x-3.5x) but not explosive (4x+)
- Show **steady accumulation** rather than rapid spikes
- Have **lower absolute volume** (200K-400K) but consistent activity
- Build momentum **gradually** over 10-20 minutes rather than 5 minutes

**Examples**: ANPA, INBS - had 3-6x volume ratios but only 200-250K absolute volume, missed because:
1. Absolute volume < 500K threshold
2. May not meet fast mover criteria (momentum might be 2-3% over 10 min, not 3%+ in 5 min)

---

## Current Logic (DO NOT DISTURB)

### Normal Entry Path:
1. **Absolute Volume**: Minimum 500K over 60 minutes (MANDATORY)
2. **Volume Ratio**: Minimum 1.5x (MANDATORY)
3. **Fast Mover Detection**: volume_ratio >= 2.5x AND momentum >= 3.0% in 5 min
   - Gets relaxed thresholds (70% confidence, 5/8 score, etc.)
4. **Pattern Requirements**: Primary patterns or secondary with strong confirmations
5. **Technical Indicators**: MAs, MACD, price action, etc.

### Fast Mover Benefits:
- Lower confidence threshold (70% vs 72%)
- Lower score requirement (5/8 vs 6/8)
- Relaxed volatility check
- Relaxed MACD acceleration

---

## Proposed Solution: Slow Mover Alternative Path

### Strategy: Additive Detection (Runs AFTER Normal Validation Fails)

**Key Principle**: Only attempt slow mover path if normal validation has already failed due to volume thresholds, but other criteria are met.

### Slow Mover Detection Criteria (REFINED):

#### Core Requirements (ALL Must Pass):

1. **Volume Ratio**: 1.8x - 3.5x (moderate-high, not explosive)
   - Too low (< 1.8x): Not enough interest
   - Too high (>= 3.5x): Should use fast mover path

2. **Sustained Momentum** ⭐ CRITICAL:
   - **10-minute momentum >= 2.0%**: Shows sustained move (not just a spike)
   - **20-minute momentum >= 3.0%**: Confirms trend is building
   - **Momentum consistency**: 10-min momentum >= 80% of 20-min momentum (not decelerating)
   - **Why**: Stocks that don't move have low momentum over longer periods

3. **Volume Building Consistently** ⭐ CRITICAL:
   - **Volume trend over 10 periods**: Last 5 periods volume >= 110% of previous 5 periods
   - **Volume acceleration**: Current volume >= 1.3x of 10-period average
   - **No declining volume**: Volume not declining for 3+ consecutive periods
   - **Why**: Stocks that don't move have flat or declining volume

4. **MACD Acceleration Pattern** ⭐ CRITICAL:
   - **MACD accelerating over 3+ periods**: Histogram increasing (hist > hist_1 > hist_2 > hist_3)
   - **MACD momentum**: Current histogram >= 1.5x of 5-period average
   - **Why**: Stocks that don't move have flat or declining MACD

5. **Price Breaking Above Consolidation** ⭐ CRITICAL:
   - **Breakout confirmation**: Price >= 1.02x of 10-period high (2% breakout)
   - **Why**: Stocks that don't move stay in consolidation

6. **Higher Highs Pattern Over Extended Period** ⭐ CRITICAL:
   - **20-period higher highs**: Max of last 10 periods > max of previous 10 periods by 2%+
   - **Consistent uptrend**: 20-period gain >= 3%
   - **Why**: Stocks that don't move have choppy or flat price action

7. **RSI in Optimal Accumulation Zone**:
   - **RSI 50-65**: Not overbought, not oversold
   - **RSI trending up**: Current RSI >= previous RSI
   - **Why**: Overbought RSI may indicate exhaustion

8. **Pattern Quality**:
   - **Primary patterns only**: Volume_Breakout, Golden_Cross, Bullish_Engulfing, etc.
   - **OR secondary with 80%+ confidence**: Higher bar for slow movers
   - **Pattern persistence**: Pattern detected in at least 2 of last 3 periods

9. **Absolute Volume**: Lower threshold for slow movers
   - **200K minimum over 60 minutes** (vs 500K normal)
   - **67K minimum over 20 minutes** (vs 167K normal)
   - Still requires minimum liquidity

10. **Technical Setup**: All other requirements still apply
    - MAs in bullish order
    - MACD bullish
    - Price above all MAs
    - Higher highs, higher lows
    - All scoring requirements (6/8 minimum)

#### Rejection Criteria (Must NOT Have):

1. ❌ Volume declining for 3+ consecutive periods
2. ❌ Price flat (< 1% move in 10 periods)
3. ❌ MACD histogram declining
4. ❌ Price stuck in consolidation (not breaking out)
5. ❌ Lower highs pattern
6. ❌ RSI overbought (>70) or oversold (<45)
7. ❌ No sustained momentum (10-min < 2% or 20-min < 3%)

---

## Implementation Approach

### Phase 1: Detection Function
Create `_is_slow_mover()` function that:
- Checks volume ratio range (1.8x - 3.5x)
- Checks momentum profile (steady over 10-20 min)
- Checks volume consistency
- Returns (is_slow_mover: bool, metrics: dict)

### Phase 2: Alternative Volume Check
In `_validate_entry_signal()`:
1. Run normal volume checks first (existing logic)
2. If volume check fails (absolute volume < 500K):
   - Check if slow mover criteria met
   - If yes, apply slow mover volume threshold (200K)
   - Continue with rest of validation

### Phase 3: Logging & Monitoring
- Log when slow mover path is used
- Track performance separately
- Monitor win rate vs normal/fast movers

---

## Code Structure

```python
def _is_slow_mover(self, df: pd.DataFrame, idx: int) -> Tuple[bool, Dict]:
    """
    Detect slow movers: steady accumulation with moderate-high volume ratio.
    Returns (is_slow_mover, metrics_dict)
    """
    # Check volume ratio range
    # Check momentum profile (10-20 min)
    # Check volume consistency
    # Return result

def _validate_entry_signal(self, signal, df, idx, ...):
    # ... existing validation ...
    
    # Volume check (existing)
    if total_volume_60min < 500000:
        # NEW: Check if slow mover
        is_slow_mover, slow_metrics = self._is_slow_mover(df, idx)
        if is_slow_mover:
            # Apply slow mover threshold (200K)
            if total_volume_60min >= 200000:
                logger.info(f"[{signal.ticker}] SLOW MOVER: Using relaxed volume threshold 200K")
                # Continue validation
            else:
                return False, "Low volume (even for slow mover)"
        else:
            # Normal rejection
            return False, "Low volume stock"
    # ... rest of validation continues unchanged ...
```

---

## Benefits

1. **Non-Disruptive**: Existing logic remains unchanged
2. **Additive**: Only activates when normal path fails
3. **Selective**: Only captures quality slow movers (not all low-volume stocks)
4. **Trackable**: Can monitor performance separately
5. **Configurable**: Thresholds can be adjusted independently

---

## Risks & Mitigations

### Risk 1: Too Many Low-Quality Trades
**Mitigation**: 
- Require strong patterns (primary or 75%+ confidence)
- Require all technical indicators (MAs, MACD, etc.)
- Require 6/8 score (same as normal trades)

### Risk 2: Slippage on Low Volume
**Mitigation**:
- Still require 200K minimum (not too low)
- Require volume consistency (not declining)
- Monitor execution quality

### Risk 3: Missing Fast Movers
**Mitigation**:
- Slow mover only activates if volume < 500K
- Fast movers with high volume still use fast mover path
- Volume ratio range (1.8x-3.5x) excludes explosive moves (4x+)

---

## Testing Plan

1. **Backtest on ANPA/INBS**: Verify slow mover path captures these
2. **Compare Performance**: Slow movers vs normal vs fast movers
3. **Monitor Win Rate**: Ensure quality maintained
4. **Adjust Thresholds**: Fine-tune based on results

---

## Configuration

Add to config (optional):
```python
ENABLE_SLOW_MOVER_PATH = True  # Enable/disable feature
SLOW_MOVER_MIN_VOLUME = 200000  # Minimum absolute volume
SLOW_MOVER_VOL_RATIO_MIN = 1.8  # Minimum volume ratio
SLOW_MOVER_VOL_RATIO_MAX = 3.5  # Maximum volume ratio (above = fast mover)
SLOW_MOVER_MOMENTUM_10MIN_MIN = 2.0  # Minimum 10-min momentum
```

---

## Success Criteria

1. ✅ Captures ANPA/INBS type opportunities
2. ✅ Maintains win rate > 50%
3. ✅ No degradation in normal/fast mover performance
4. ✅ Clear logging for monitoring
5. ✅ Easy to disable if needed
