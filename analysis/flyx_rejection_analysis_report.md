# FLYX Rejection Analysis (15:42 - 15:49)

## Overview
This report analyzes why FLYX trade was rejected between 15:42 to 15:49. The bot appeared to miss a trading opportunity during this time window.

## Analysis Method
1. Download latest FLYX data from Webull API
2. Run the trading bot's entry validation logic minute-by-minute for the 15:42-15:49 window
3. Capture all rejection reasons logged by the `RealtimeTrader`
4. Analyze indicators at each rejection point

## Rejection Criteria (Current Bot Settings)

The bot uses strict entry criteria:

### 1. **Minimum Price**: $0.50
- Rejects stocks below $0.50

### 2. **Minimum Volume**: 500K shares daily
- Checks total volume over 60 minutes (or extrapolates from 20 minutes)
- Rejects if total volume < 500K shares

### 3. **Minimum Confidence**: 72%
- Pattern confidence must be >= 72%

### 4. **Pattern Requirements**
- Only accepts: `Strong_Bullish_Setup`, `Bullish_Reversal`, `Momentum_Breakout`
- Rejects all other patterns

### 5. **Price Above All Moving Averages**
- Price must be above SMA 5, SMA 10, and SMA 20
- Rejects if price is below any MA

### 6. **Moving Averages in Bullish Order**
- Must have: SMA 5 > SMA 10 > SMA 20
- Rejects if MAs are not in bullish order

### 7. **Volume Ratio**: >= 1.5x
- Volume must be at least 1.5x the average volume
- Rejects if volume ratio < 1.5x

### 8. **Minimum Entry Price Increase**: 5.5%
- Expected gain (target - entry) / entry must be >= 5.5%
- Rejects if expected gain < 5.5%

### 9. **Setup Confirmation**
- Setup must be confirmed for multiple periods (not just appeared)
- Rejects if setup is not sustainable

### 10. **False Breakout Detection**
- Checks for false breakouts (price spikes that reverse)
- Rejects if false breakout detected

### 11. **Reverse Split Detection**
- Checks for reverse splits
- Rejects if reverse split detected

### 12. **MACD Requirements**
- MACD must be above signal line
- MACD acceleration must be positive (for fast movers: >= 2%)
- Rejects if MACD conditions not met

### 13. **RSI Requirements**
- RSI should be in bullish range (typically 50-80)
- Rejects if RSI indicates overbought or oversold conditions

## How to Run Analysis

To analyze the rejection window, run:

```bash
python analysis/simulate_flyx_full_day.py --rejection
```

This will:
1. Download latest FLYX data
2. Analyze the 15:42-15:49 window
3. Log all rejection reasons
4. Save a detailed report to `analysis/flyx_rejection_analysis.txt`

## Expected Output

The analysis will show:
- **Entry Signals Found**: Number of valid entry signals in the window
- **Rejection Events**: Number of times entry was rejected
- **Most Common Rejection Reasons**: Summary of why entries were rejected
- **Detailed Rejection Analysis**: For each rejection, shows:
  - Timestamp and price
  - All rejection reasons
  - Indicator values at rejection time (price, volume, RSI, MACD, MAs, etc.)

## Common Rejection Reasons

Based on the bot's validation logic, common reasons for rejection include:

1. **Low Volume**: Volume ratio < 1.5x or total volume < 500K shares
2. **Price Below MAs**: Price not above all moving averages
3. **MAs Not Bullish**: Moving averages not in bullish order (SMA 5 > SMA 10 > SMA 20)
4. **Low Confidence**: Pattern confidence < 72%
5. **Insufficient Gain**: Expected gain < 5.5%
6. **False Breakout**: Detected false breakout pattern
7. **Setup Not Confirmed**: Setup not confirmed for multiple periods
8. **MACD Issues**: MACD below signal or negative acceleration
9. **RSI Issues**: RSI in overbought/oversold range

## Next Steps

1. Run the analysis script to get actual rejection reasons for the 15:42-15:49 window
2. Review the detailed rejection log to understand why entries were rejected
3. If the rejection was too strict, consider adjusting:
   - Minimum confidence threshold
   - Volume ratio requirement
   - Moving average requirements
   - Expected gain threshold

## Notes

- The bot checks on the **5th second of every minute** (not every 15 seconds)
- This means it checks at 15:42:05, 15:43:05, 15:44:05, etc.
- The analysis will show what the bot would have seen at each check time
