# Improved Exit Strategies for Trending Stocks

## Research-Based Exit Strategies

### 1. Dynamic Trailing Stops
- **Initial**: 3% trailing stop for first 10 minutes
- **After 10 min**: 5% trailing stop
- **After 20 min**: 7% trailing stop
- **After 30 min**: 10% trailing stop
- **Rationale**: Give strong moves room to breathe, wider stops as profit builds

### 2. Minimum Hold Time
- **Requirement**: 15-20 minutes minimum before allowing trailing stops
- **Rationale**: Prevents premature exits during normal volatility

### 3. Hard Stop Loss Only
- **Hard Stop**: 15% from entry (absolute, never moves)
- **Rationale**: Only exit on significant reversal, not small pullbacks

### 4. Strong Reversal Signals (Multiple Confirmations Required)
- **Require 3+ of these**:
  - Price below SMA10 AND SMA20
  - MACD bearish crossover AND histogram negative
  - Volume declining 30%+ AND price declining
  - Price making lower lows (5%+ below recent high)
  - RSI dropping below 50 from overbought

### 5. Profit-Based Exit Rules
- **0-5% profit**: Only exit on hard stop or strong reversal
- **5-10% profit**: Allow 7% trailing stop
- **10%+ profit**: Allow 10% trailing stop
- **Rationale**: Protect profits as they grow, but give room for continuation

### 6. Trend Strength Indicators
- **Stay in if**:
  - Price above all MAs
  - MACD still bullish
  - Volume above average
  - Making higher highs
- **Exit only if**: Multiple of these fail

### 7. Time-Based Exits
- **End of Day**: Exit all positions
- **After 4 hours**: Consider partial exit if profit > 10%
- **Rationale**: Lock in profits, avoid overnight risk

---

## Implementation Strategy

### Conservative Exit Logic:
1. **Hard Stop Loss**: 15% from entry (always active)
2. **Minimum Hold Time**: 20 minutes before any exit (except hard stop)
3. **Dynamic Trailing Stop**: 
   - 0-10 min: No trailing stop
   - 10-20 min: 7% trailing stop
   - 20+ min: 10% trailing stop
4. **Strong Reversal Only**: Require 3+ reversal signals
5. **End of Day**: Exit all positions

### Exit Signals (Require Multiple):
- Price below SMA10 AND SMA20: 1 signal
- MACD bearish crossover: 1 signal
- MACD histogram negative: 1 signal
- Volume declining 30%+ AND price declining: 1 signal
- Price 5%+ below recent high: 1 signal
- RSI < 50 from > 70: 1 signal

**Exit if**: 3+ signals OR hard stop hit
