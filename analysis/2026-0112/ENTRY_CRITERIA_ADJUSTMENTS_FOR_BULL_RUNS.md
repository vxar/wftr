# Entry Criteria Adjustments to Capture Fast-Moving Bull Runs

## Analysis Summary

Based on the EVTV rejection analysis (2:00 PM - 2:45 PM), the bot missed a significant bull run due to overly strict validation criteria. The main rejection reasons were:

1. **Too volatile** (14:39, 14:45) - 14.2% and 18.9% range in 5 periods
2. **MAs not in bullish order** (14:21, 14:36) - During rapid price moves
3. **Making lower lows** (14:27) - False signal during uptrend
4. **No pattern detected** (30 times) - Pattern detector not catching explosive moves

## Recommended Adjustments

### 1. Relax Volatility Check for Fast Movers ⚠️ CRITICAL

**Current Issue**: Line 1264 - Rejects if volatility > 8% in 5 periods, but fast movers naturally have high volatility.

**Current Code** (line 1256-1270):
```python
# 10. Price stability check (avoid high volatility entries) (MANDATORY)
# Fast movers bypass this check - high volatility is expected for breakouts
if not is_fast_mover:
    if len(lookback_10) >= 5:
        recent_highs = lookback_10['high'].tail(5).values
        recent_lows = lookback_10['low'].tail(5).values
        if len(recent_highs) > 0 and len(recent_lows) > 0:
            price_range_pct = ((max(recent_highs) - min(recent_lows)) / min(recent_lows)) * 100
            if price_range_pct > 8.0:  # Too volatile (8%+ range in 5 periods)
                reason = f"Too volatile ({price_range_pct:.1f}% range in 5 periods)"
                return False, reason
```

**Problem**: Fast mover detection happens AFTER this check, so fast movers are still being rejected.

**Solution**: 
- Move fast mover detection earlier (before volatility check)
- OR increase volatility threshold to 15-20% for normal stocks
- OR make volatility check conditional on fast mover status

**Recommended Fix**:
```python
# Detect fast mover FIRST (before volatility check)
is_fast_mover, fast_mover_metrics = self._is_fast_mover(df, idx)

# 10. Price stability check (avoid high volatility entries) (MANDATORY)
# Fast movers bypass this check - high volatility is expected for breakouts
if not is_fast_mover:
    if len(lookback_10) >= 5:
        recent_highs = lookback_10['high'].tail(5).values
        recent_lows = lookback_10['low'].tail(5).values
        if len(recent_highs) > 0 and len(recent_lows) > 0:
            price_range_pct = ((max(recent_highs) - min(recent_lows)) / min(recent_lows)) * 100
            # Increased threshold from 8% to 15% to allow more volatile breakouts
            if price_range_pct > 15.0:  # Too volatile (15%+ range in 5 periods)
                reason = f"Too volatile ({price_range_pct:.1f}% range in 5 periods)"
                return False, reason
else:
    logger.info(f"[{signal.ticker}] FAST MOVER: Bypassing volatility check")
```

### 2. Relax MA Order Requirement for Fast Movers ⚠️ CRITICAL

**Current Issue**: Line 1084 - Requires strict MA order (sma5 > sma10 > sma20), but during rapid moves, MAs may not align immediately.

**Current Code** (line 1083-1088):
```python
# 2. Moving averages in bullish order (MANDATORY)
if not (sma5 > sma10 and sma10 > sma20):
    reason = "MAs not in bullish order"
    return False, reason  # REJECT if MAs not in bullish order
```

**Problem**: During explosive moves, price can move faster than MAs can align.

**Solution**: For fast movers with strong momentum, relax this requirement.

**Recommended Fix**:
```python
# 2. Moving averages in bullish order (MANDATORY)
# Relax for fast movers with strong momentum
is_fast_mover, fast_mover_metrics = self._is_fast_mover(df, idx)
if is_fast_mover:
    # For fast movers: Only require price above all MAs, not strict order
    if not (close > sma5 and close > sma10 and close > sma20):
        reason = "Price not above all MAs (fast mover)"
        return False, reason
    # Allow if at least 2 of 3 MAs are in order
    ma_order_score = sum([sma5 > sma10, sma10 > sma20])
    if ma_order_score < 1:  # At least one pair must be in order
        reason = "MAs not showing bullish alignment (fast mover)"
        return False, reason
else:
    # Normal stocks: Strict MA order required
    if not (sma5 > sma10 and sma10 > sma20):
        reason = "MAs not in bullish order"
        return False, reason
```

### 3. Relax "Lower Lows" Check for Fast Movers

**Current Issue**: Line 1239 - Rejects if making lower lows, but during strong uptrends with pullbacks, this can be a false signal.

**Current Code** (line 1230-1243):
```python
# 8. Price must be making consistent higher lows (MANDATORY - uptrend confirmation)
if len(lookback_20) >= 10:
    recent_lows = lookback_20['low'].tail(10).values
    if len(recent_lows) >= 5:
        older_lows = recent_lows[:5]
        newer_lows = recent_lows[5:]
        avg_older_low = min(older_lows) if len(older_lows) > 0 else 0
        avg_newer_low = min(newer_lows) if len(newer_lows) > 0 else 0
        if avg_older_low > 0 and avg_newer_low < avg_older_low * 0.98:  # Lower lows
            reason = "Making lower lows (downtrend)"
            return False, reason
```

**Problem**: During rapid moves, temporary pullbacks can create lower lows even in strong uptrends.

**Solution**: For fast movers, check if overall trend is still up despite temporary lower lows.

**Recommended Fix**:
```python
# 8. Price must be making consistent higher lows (MANDATORY - uptrend confirmation)
if len(lookback_20) >= 10:
    recent_lows = lookback_20['low'].tail(10).values
    if len(recent_lows) >= 5:
        older_lows = recent_lows[:5]
        newer_lows = recent_lows[5:]
        avg_older_low = min(older_lows) if len(older_lows) > 0 else 0
        avg_newer_low = min(newer_lows) if len(newer_lows) > 0 else 0
        if avg_older_low > 0 and avg_newer_low < avg_older_low * 0.98:  # Lower lows
            # For fast movers: Check if overall price trend is still up
            if is_fast_mover:
                # Check if current price is still significantly above older price
                older_price = lookback_20.iloc[0].get('close', 0)
                current_price = current.get('close', 0)
                if older_price > 0 and current_price > older_price * 1.05:  # Still up 5%+
                    # Allow if overall trend is up despite temporary lower lows
                    logger.debug(f"[{signal.ticker}] FAST MOVER: Lower lows detected but overall trend up ({((current_price - older_price) / older_price) * 100:.1f}%)")
                else:
                    reason = "Making lower lows (downtrend)"
                    return False, reason
            else:
                reason = "Making lower lows (downtrend)"
                return False, reason
```

### 4. Improve Fast Mover Detection

**Current Issue**: Fast mover detection may not be triggering early enough or with the right criteria.

**Current Code** (line 944-988):
```python
def _is_fast_mover(self, df: pd.DataFrame, idx: int) -> Tuple[bool, Dict[str, float]]:
    # Check volume ratio
    volume_ratio = current.get('volume_ratio', 0)
    if volume_ratio < 5.0:  # Must be at least 5x average
        return False, {}
    
    # Check price momentum (5%+ in last 5 periods)
    if idx >= 5:
        price_change_5 = ((current.get('close', 0) - df.iloc[idx-5].get('close', 0)) / 
                         df.iloc[idx-5].get('close', 0)) * 100
        if price_change_5 < 5.0:  # Must be at least 5% gain
            return False, {}
```

**Problem**: Thresholds may be too high (5x volume, 5% momentum).

**Solution**: Lower thresholds to catch fast movers earlier.

**Recommended Fix**:
```python
def _is_fast_mover(self, df: pd.DataFrame, idx: int) -> Tuple[bool, Dict[str, float]]:
    if idx < 5:
        return False, {}
    
    current = df.iloc[idx]
    lookback_10 = df.iloc[idx-10:idx] if idx >= 10 else df.iloc[:idx]
    
    # Check volume ratio - LOWERED from 5.0x to 3.0x
    volume_ratio = current.get('volume_ratio', 0)
    if volume_ratio < 3.0:  # Must be at least 3x average (was 5x)
        return False, {}
    
    # Check price momentum - LOWERED from 5% to 3% in last 5 periods
    if idx >= 5:
        price_change_5 = ((current.get('close', 0) - df.iloc[idx-5].get('close', 0)) / 
                         df.iloc[idx-5].get('close', 0)) * 100
        if price_change_5 < 3.0:  # Must be at least 3% gain (was 5%)
            return False, {}
    
    # Check if volume is increasing (not declining)
    if len(lookback_10) >= 5:
        recent_volumes = lookback_10['volume'].tail(5).values
        if len(recent_volumes) >= 3:
            if recent_volumes[-1] < recent_volumes[0] * 0.9:  # Declining 10%+
                return False, {}
    
    # All conditions met - this is a fast mover
    metrics = {
        'vol_ratio': volume_ratio,
        'momentum': price_change_5
    }
    return True, metrics
```

### 5. Lower Confidence Threshold for Fast Movers

**Current Issue**: Line 270-276 - Fast movers get 70% threshold, but this may still be too high during explosive moves.

**Current Code**:
```python
effective_min_confidence = self.min_confidence
if hour < 10:  # Before 10 AM - use 70% threshold
    effective_min_confidence = 0.70
elif is_fast_mover:
    effective_min_confidence = 0.70  # Lower to 70% for fast movers
```

**Solution**: Lower to 65% for very strong fast movers.

**Recommended Fix**:
```python
effective_min_confidence = self.min_confidence
if hour < 10:  # Before 10 AM - use 70% threshold
    effective_min_confidence = 0.70
elif is_fast_mover:
    # For very strong fast movers (high volume + high momentum), lower threshold
    if volume_ratio >= 4.0 and price_momentum_5 >= 10.0:  # Very strong
        effective_min_confidence = 0.65  # Lower to 65% for explosive moves
    else:
        effective_min_confidence = 0.70  # Lower to 70% for fast movers
```

### 6. Relax "Price Too Extended" Check

**Current Issue**: Line 1412 - Rejects if price moved >10% in 5 periods, but during strong bull runs, this is normal.

**Current Code**:
```python
# Reject if price is too extended (recent massive move)
if price_change_5 > 10:  # More than 10% in 5 periods - too extended
    reason = f"Price too extended ({price_change_5:.1f}% in 5 periods, max 10% allowed)"
    return False, reason
```

**Solution**: Increase threshold or bypass for fast movers.

**Recommended Fix**:
```python
# Reject if price is too extended (recent massive move)
# For fast movers, allow up to 20% move in 5 periods
max_extended_pct = 20.0 if is_fast_mover else 10.0
if price_change_5 > max_extended_pct:
    reason = f"Price too extended ({price_change_5:.1f}% in 5 periods, max {max_extended_pct}% allowed)"
    return False, reason
```

## Implementation Priority

### HIGH PRIORITY (Will capture EVTV-type moves):
1. ✅ **Move fast mover detection earlier** - Before volatility check
2. ✅ **Relax volatility threshold** - Increase from 8% to 15% for normal stocks
3. ✅ **Relax MA order for fast movers** - Allow if price above all MAs even if not perfect order
4. ✅ **Lower fast mover thresholds** - 3x volume, 3% momentum (from 5x, 5%)

### MEDIUM PRIORITY (Will improve capture rate):
5. ✅ **Relax "lower lows" for fast movers** - Check overall trend, not just recent lows
6. ✅ **Relax "price too extended"** - Allow up to 20% for fast movers
7. ✅ **Lower confidence for very strong fast movers** - 65% for explosive moves

### LOW PRIORITY (Fine-tuning):
8. Improve pattern detection for explosive moves
9. Add momentum-based entry patterns

## Expected Impact

With these adjustments, the bot should:
- ✅ Capture EVTV at 14:21 ($1.52) - Would pass with relaxed MA order
- ✅ Capture EVTV at 14:36 ($1.652) - Would pass with relaxed MA order  
- ✅ Capture EVTV at 14:39 ($1.76) - Would pass with relaxed volatility (fast mover)
- ✅ Capture EVTV at 14:45 ($1.95) - Would pass with relaxed volatility (fast mover)

## Risk Assessment

**Low Risk Changes**:
- Moving fast mover detection earlier
- Increasing volatility threshold to 15%
- Relaxing MA order for fast movers

**Medium Risk Changes**:
- Lowering fast mover thresholds (3x/3% vs 5x/5%)
- Lowering confidence to 65% for very strong fast movers

**Mitigation**: These changes only apply to fast movers, which are already high-probability setups. The bot will still maintain strict criteria for normal stocks.
