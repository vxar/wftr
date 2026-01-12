# Slow Mover Setup Analysis - What Leads to Steady Bull Runs

## Problem Statement

We need to identify the **setup characteristics** that distinguish:
- ✅ **Successful slow movers** (steady accumulation leading to bull runs)
- ❌ **Stocks that don't move** (low volume, no momentum, flat price action)
- ❌ **False signals** (brief spikes that don't sustain)

---

## Current Validation Requirements (All Must Pass)

### Mandatory Checks:
1. **Price above all MAs** (SMA5, SMA10, SMA20)
2. **MAs in bullish order** (SMA5 > SMA10 > SMA20)
3. **Volume ratio >= 1.5x**
4. **Absolute volume >= 500K** (60 min) or 167K (20 min)
5. **Average volume >= 50K/min** (or 30K for fast movers)
6. **MACD bullish** (MACD > Signal)
7. **MACD histogram positive AND accelerating** (3%+ increase)
8. **Price making higher highs** (not rejected from highs)
9. **No price weakness** (< 3 declining periods)
10. **Longer-term uptrend** (2%+ over 15 periods)
11. **Higher lows** (uptrend confirmation)
12. **Upward momentum** (recent closes increasing)
13. **Volume not declining** (last 5 periods)
14. **Not at peak without momentum** (1%+ momentum if at high)
15. **Price not too extended** (< 10% in 5 periods)
16. **Perfect setup score >= 6/8** (or 5/8 for fast movers)

---

## Key Differentiators for Successful Slow Movers

### 1. **Sustained Momentum Over Longer Periods** ⭐ CRITICAL

**Observation**: Successful slow movers show momentum building over 10-20 minutes, not just 5 minutes.

**Current Check**: Only checks 5-minute momentum (price_change_5)

**What We Need**:
- ✅ **10-minute momentum >= 2.0%**: Shows sustained move (not just a spike)
- ✅ **20-minute momentum >= 3.0%**: Confirms trend is building
- ✅ **Momentum consistency**: 10-min momentum should be >= 80% of 20-min momentum (not decelerating)

**Why This Matters**: Stocks that don't move will have low momentum over longer periods. Slow movers build momentum gradually.

---

### 2. **Volume Building Consistently** ⭐ CRITICAL

**Observation**: Successful slow movers show volume increasing over 10-20 periods, not just current period.

**Current Check**: Only checks last 5 periods for volume decline

**What We Need**:
- ✅ **Volume trend over 10 periods**: Last 5 periods volume >= 110% of previous 5 periods
- ✅ **Volume consistency**: No sharp drops (each period >= 80% of previous)
- ✅ **Volume acceleration**: Current volume >= 1.3x of 10-period average

**Why This Matters**: Stocks that don't move have flat or declining volume. Slow movers show consistent accumulation.

---

### 3. **MACD Acceleration Pattern** ⭐ CRITICAL

**Observation**: Successful slow movers show MACD histogram accelerating over multiple periods.

**Current Check**: Only checks current vs previous period (3% increase)

**What We Need**:
- ✅ **MACD acceleration trend**: Histogram increasing over last 3 periods
- ✅ **MACD momentum**: Current histogram >= 1.5x of 5-period average
- ✅ **MACD line above signal**: Already checked, but confirm it's widening

**Why This Matters**: Stocks that don't move have flat or declining MACD. Slow movers show building momentum.

---

### 4. **Price Breaking Above Consolidation** ⭐ CRITICAL

**Observation**: Successful slow movers break above recent consolidation, not just bouncing within range.

**Current Check**: Checks if price is at/near recent high (scoring only)

**What We Need**:
- ✅ **Breakout confirmation**: Price >= 1.02x of 10-period high (2% breakout)
- ✅ **Consolidation pattern**: Price was in range (high-low < 3%) for 5+ periods before breakout
- ✅ **Breakout volume**: Volume during breakout >= 1.5x of consolidation period average

**Why This Matters**: Stocks that don't move stay in consolidation. Slow movers break out with volume.

---

### 5. **Higher Highs Pattern Over Extended Period** ⭐ CRITICAL

**Observation**: Successful slow movers show consistent higher highs over 20+ periods.

**Current Check**: Checks 15-period trend (2%+ gain)

**What We Need**:
- ✅ **20-period higher highs**: Max of last 10 periods > max of previous 10 periods by 2%+
- ✅ **Consistent uptrend**: No more than 2 periods where price dropped > 2% from previous high
- ✅ **Trend strength**: 20-period gain >= 3% (stronger than 2% minimum)

**Why This Matters**: Stocks that don't move have choppy or flat price action. Slow movers show clear uptrend.

---

### 6. **RSI in Optimal Accumulation Zone**

**Observation**: Successful slow movers often have RSI in 50-65 range (accumulation, not overbought).

**Current Check**: RSI 45-70 is optimal (scoring only)

**What We Need**:
- ✅ **RSI in accumulation zone**: 50-65 (not overbought, not oversold)
- ✅ **RSI trending up**: Current RSI >= previous RSI (building momentum)

**Why This Matters**: Overbought RSI (>70) may indicate exhaustion. Slow movers build from accumulation zone.

---

### 7. **Pattern Quality - Primary Patterns Only**

**Observation**: Successful slow movers typically show primary patterns (Volume_Breakout, Golden_Cross).

**Current Check**: Accepts secondary patterns with strong confirmations

**What We Need for Slow Movers**:
- ✅ **Primary patterns only**: Volume_Breakout, Golden_Cross, Bullish_Engulfing, etc.
- ✅ **OR secondary with 80%+ confidence**: Higher bar for slow movers
- ✅ **Pattern confirmation**: Pattern detected in last 3 periods (not just current)

**Why This Matters**: Secondary patterns in slow movers may be false signals. Primary patterns are more reliable.

---

## Refined Slow Mover Criteria

### Detection Function: `_is_slow_mover()`

```python
def _is_slow_mover(self, df: pd.DataFrame, idx: int) -> Tuple[bool, Dict]:
    """
    Detect slow movers: steady accumulation with sustained momentum.
    Returns (is_slow_mover, metrics_dict)
    """
    if idx < 20:  # Need at least 20 periods
        return False, {}
    
    current = df.iloc[idx]
    
    # 1. Volume ratio in moderate-high range (1.8x - 3.5x)
    volume_ratio = current.get('volume_ratio', 0)
    if volume_ratio < 1.8 or volume_ratio >= 3.5:
        return False, {}
    
    # 2. Sustained momentum over longer periods
    price_10min_ago = df.iloc[idx-10].get('close', 0) if idx >= 10 else 0
    price_20min_ago = df.iloc[idx-20].get('close', 0) if idx >= 20 else 0
    current_price = current.get('close', 0)
    
    momentum_10 = ((current_price - price_10min_ago) / price_10min_ago) * 100 if price_10min_ago > 0 else 0
    momentum_20 = ((current_price - price_20min_ago) / price_20min_ago) * 100 if price_20min_ago > 0 else 0
    
    # Must have sustained momentum
    if momentum_10 < 2.0 or momentum_20 < 3.0:
        return False, {}
    
    # Momentum should be consistent (not decelerating)
    if momentum_10 < momentum_20 * 0.8:
        return False, {}
    
    # 3. Volume building consistently over 10 periods
    if idx >= 10:
        recent_volumes = df.iloc[idx-10:idx+1]['volume'].values
        if len(recent_volumes) >= 10:
            last_5_avg = recent_volumes[-5:].mean()
            prev_5_avg = recent_volumes[-10:-5].mean() if len(recent_volumes) >= 10 else 0
            
            if prev_5_avg > 0 and last_5_avg < prev_5_avg * 1.1:  # Not building
                return False, {}
            
            # Current volume should be above average
            current_vol = current.get('volume', 0)
            avg_vol_10 = recent_volumes.mean()
            if current_vol < avg_vol_10 * 1.3:  # Not accelerating
                return False, {}
    
    # 4. MACD acceleration over multiple periods
    macd_hist = current.get('macd_hist', 0)
    if macd_hist <= 0:
        return False, {}
    
    if idx >= 3:
        hist_3 = df.iloc[idx-3].get('macd_hist', 0)
        hist_2 = df.iloc[idx-2].get('macd_hist', 0)
        hist_1 = df.iloc[idx-1].get('macd_hist', 0)
        
        # Should be accelerating (increasing)
        if not (macd_hist > hist_1 > hist_2 > hist_3):
            return False, {}
    
    # 5. Price breaking above consolidation
    if idx >= 10:
        recent_highs = df.iloc[idx-10:idx]['high'].values
        max_recent_high = max(recent_highs) if len(recent_highs) > 0 else 0
        
        if max_recent_high > 0:
            # Should be breaking out (2%+ above recent high)
            if current_price < max_recent_high * 1.02:
                return False, {}
    
    # 6. Higher highs pattern over 20 periods
    if idx >= 20:
        older_highs = df.iloc[idx-20:idx-10]['high'].values
        newer_highs = df.iloc[idx-10:idx+1]['high'].values
        
        if len(older_highs) > 0 and len(newer_highs) > 0:
            max_older = max(older_highs)
            max_newer = max(newer_highs)
            
            if max_newer < max_older * 1.02:  # Not making higher highs
                return False, {}
    
    # 7. RSI in accumulation zone (50-65)
    rsi = current.get('rsi', 50)
    if rsi < 50 or rsi > 65:
        return False, {}
    
    # All checks passed - this is a slow mover
    metrics = {
        'vol_ratio': volume_ratio,
        'momentum_10': momentum_10,
        'momentum_20': momentum_20,
        'rsi': rsi
    }
    return True, metrics
```

---

## Additional Filters to Avoid Stocks That Don't Move

### 1. **Minimum Price Movement**
- Require price has moved >= 1% in last 10 periods (not flat)
- Require price has moved >= 2% in last 20 periods (sustained move)

### 2. **Volume Activity**
- Require at least 3 periods with volume >= 1.5x average in last 10 periods
- Reject if volume has been declining for 3+ consecutive periods

### 3. **Price Range Expansion**
- Require current price range (high-low) >= 1.5x of 10-period average range
- Shows increasing volatility/activity

### 4. **Pattern Persistence**
- Require pattern detected in at least 2 of last 3 periods (not just appeared)
- Confirms pattern is developing, not just a flash

---

## Summary: What Makes a Successful Slow Mover

### ✅ **Must Have** (All Required):
1. Volume ratio 1.8x - 3.5x (moderate-high)
2. 10-minute momentum >= 2.0%
3. 20-minute momentum >= 3.0%
4. Momentum consistent (not decelerating)
5. Volume building over 10 periods
6. MACD accelerating over 3+ periods
7. Price breaking above consolidation (2%+)
8. Higher highs over 20 periods
9. RSI in 50-65 range
10. Primary pattern OR secondary with 80%+ confidence

### ❌ **Must NOT Have** (Reject If):
1. Volume declining for 3+ periods
2. Price flat (< 1% move in 10 periods)
3. MACD histogram declining
4. Price stuck in consolidation
5. Lower highs pattern
6. RSI overbought (>70) or oversold (<45)

---

## Implementation Priority

1. **Phase 1**: Add `_is_slow_mover()` function with all checks
2. **Phase 2**: Integrate into volume validation (apply 200K threshold if slow mover)
3. **Phase 3**: Add additional filters (price movement, volume activity, etc.)
4. **Phase 4**: Test on ANPA/INBS and similar stocks
5. **Phase 5**: Monitor and adjust thresholds

---

## Expected Impact

### Before:
- ANPA: 0 trades (volume < 500K)
- INBS: 0 trades (volume < 500K)

### After (with slow mover path):
- ANPA: Should capture if meets all slow mover criteria
- INBS: Should capture if meets all slow mover criteria
- Other slow movers: Will capture quality setups only

### Risk Mitigation:
- Still requires all technical indicators (MAs, MACD, etc.)
- Still requires 6/8 perfect setup score
- Only activates for stocks with sustained momentum
- Rejects stocks that don't move (flat price, declining volume)
