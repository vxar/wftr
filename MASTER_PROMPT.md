# Master Prompt: Trading Bot Development

## Project Overview
Create a professional, autonomous trading bot system that identifies bullish stock patterns, executes trades, and manages positions with sophisticated entry/exit logic. The system should include a modern web dashboard for monitoring and control.

## Core Requirements

### 1. Trading Bot Functionality

#### Stock Discovery
- **Primary Source**: Monitor only "Top Gainers" list (remove all other ticker sources)
- **No Hardcoded Tickers**: All ticker symbols must be dynamically fetched, no hardcoded values
- **Monitoring**: Track top gainers throughout trading day

#### Entry Logic
- **Pattern Detection**: Identify bullish patterns including:
  - Volume_Breakout_Momentum (preferred pattern)
  - RSI_Accumulation_Entry (preferred pattern)
  - Golden_Cross_Volume
  - Slow_Accumulation
  - Strong_Bullish_Setup
  - MACD_Bullish_Cross
  - Consolidation_Breakout
- **Confidence Threshold**: 
  - Normal stocks: 72% minimum confidence
  - Fast movers (volume ratio >= 2.5x AND 5-min momentum >= 3%): 70% minimum confidence
  - Slow movers: 80% minimum confidence (alternative path when original logic fails)
- **Volume Requirements**:
  - Normal stocks: 500K minimum over 60 minutes (or 167K over 20 minutes)
  - Fast movers: Same as normal (500K/60min or 167K/20min)
  - Slow movers: 200K minimum (alternative path)
  - For Volume_Breakout_Momentum and RSI_Accumulation_Entry patterns: 200K absolute volume threshold
- **Price Filter**: Minimum $0.50 per share
- **Setup Confirmation**: Setup must be confirmed for 4+ out of last 6 periods
- **False Breakout Detection**: Filter out false breakouts (price spike + reversal)
- **Reverse Split Detection**: Filter out reverse split patterns
- **Validation Checks**: Comprehensive validation including price above MAs, MACD bullish, volume above average, positive momentum

#### Exit Logic
- **Hard Stop Loss**: 15% (always active)
- **Minimum Hold Time**: 20 minutes (prevents premature exits)
- **Dynamic Trailing Stops**:
  - 50%+ profit: 20% trailing stop
  - 30%+ profit: 15% trailing stop
  - 20%+ profit: 12% trailing stop
  - 10%+ profit: 10% trailing stop
  - 5%+ profit: 7% trailing stop
  - <5% profit: 5% trailing stop
- **Strong Reversal Detection**:
  - 50%+ profit: 5+ reversal signals required
  - 20%+ profit: 4+ reversal signals required
  - <20% profit: 3+ reversal signals required
- **Profit Target**: Only triggers after 30+ minutes AND profit >= 20%
- **Partial Exits**:
  - 50% of position at +20% profit
  - 25% of position at +40% profit
  - 12.5% of position at +80% profit
  - Adjust trailing stops and reversal requirements for remaining position
  - Disable profit target if partial exits taken
- **Slow Mover Exit Logic** (for slow mover entries):
  - Minimum hold time: 10 minutes
  - Trailing stop: 5% (fixed, wider than normal)
  - ATR-based trailing stop: `max_price - (atr * 2.5)`
  - Ensure trailing stop never goes below entry price

#### Slow Mover Strategy (Alternative Path)
- **Activation**: Only when original entry logic fails to place a trade
- **Criteria**:
  - Volume ratio: 1.8x - 3.5x (moderate-high, not explosive)
  - Absolute volume: 200K minimum (vs 500K normal)
  - Sustained momentum: 10-min >= 2.0%, 20-min >= 3.0%
  - MACD accelerating, breakouts, higher highs, RSI 50-65
  - Confidence threshold: 80%
- **Exit**: Uses separate slow mover exit logic (see above)

#### Risk Management
- **Position Sizing**: Configurable percentage per trade
- **Max Positions**: Configurable limit
- **Daily Trade Limit**: Configurable max trades per day
- **Consecutive Loss Limit**: DISABLED (testing mode - all trades allowed)
- **Capital Management**: Track portfolio value, daily profit, total return

### 2. Data Persistence

#### Database Requirements
- **SQLite Database**: Store all trading data
- **Tables**:
  - `trades`: Completed trades with entry/exit details
  - `positions`: Active positions
  - `rejected_entries`: Rejected entry signals (persist for the day)
- **Rejected Entries**: Must persist in database, not just memory, so they remain visible even if web app stops
- **Position Restoration**: Restore active positions from database on bot startup

### 3. Web Dashboard

#### UI/UX Requirements
- **Modern Professional Design**: 
  - Clean, professional color scheme (inspired by Bloomberg Terminal, TradingView)
  - Primary: #0066cc (Professional Blue)
  - Success: #00c853 (Modern Green)
  - Danger: #d32f2f (Alert Red)
  - Warning: #ff9800 (Orange)
- **Layout**:
  - Top statistics bar with key metrics (Portfolio Value, Total Return, Daily Profit, Positions, Total Trades, Monitored)
  - Collapsible sections for better space utilization
  - Two-column layout (main content + side panel)
- **Statistics Display**:
  - Portfolio Value
  - Total Return (percentage)
  - Daily Profit (dollars)
  - Active Positions / Max Positions
  - Total Trades
  - Monitored Tickers
- **Sections**:
  - Rejected Entry Signals (collapsible)
  - Active Positions (collapsible, with update/close buttons)
  - Completed Trades (collapsible, table format)
  - Trading Statistics (collapsible)
  - Monitoring List - Top Gainers (side panel, collapsible)
- **Controls**:
  - Start/Stop buttons (in top bar)
  - Bot status indicator (colored circle with pulse animation)
  - Auto-refresh checkbox (in top bar)
  - Remove refresh button (consolidate into auto-refresh)
- **Auto-Refresh**:
  - Refresh every 30 seconds
  - Label: "Auto-refresh (30s)"
  - Toggle on/off via checkbox

#### Data Display
- **Top Gainers**:
  - Display current price from minute data
  - Calculate change percentage:
    - Premarket/Regular hours: Use previous day's close
    - After-hours: Use today's close
  - Sort by calculated change percentage
  - Show status badges (MONITORING, ACTIVE, REJECTED)
  - Display rejection reasons if applicable
- **Rejected Entries**:
  - Show ticker, price, reason, timestamp
  - Persist in database for the day
  - Clear when position is successfully entered
- **Active Positions**:
  - Ticker, Entry Price, Current Price, Shares
  - P&L (dollars and percentage)
  - Target Price and Stop Loss (editable)
  - Update and Close buttons
- **Completed Trades**:
  - Full trade history with entry/exit details
  - P&L, pattern, exit reason, status (WIN/LOSS)

### 4. Technical Requirements

#### Code Quality
- **No Hardcoded Values**: All ticker symbols must be dynamic
- **Error Handling**: Robust error handling throughout
- **Logging**: Comprehensive logging (ASCII-safe, no emoji characters for Windows compatibility)
- **Type Hints**: Use type hints where appropriate
- **Documentation**: Clear code comments and docstrings

#### Performance
- **Efficient Data Fetching**: Optimize API calls
- **Real-time Processing**: Process 1-minute data efficiently
- **Database Optimization**: Efficient queries and indexing

#### Compatibility
- **Windows Support**: Ensure all code works on Windows
- **Encoding**: Use ASCII-safe characters in logging (no Unicode emoji)
- **Timezone Handling**: Proper timezone conversion (US/Eastern)

### 5. Analysis and Testing Tools

#### Analysis Scripts
- **Comprehensive Stock Analysis**: Analyze multiple stocks, identify patterns, simulate trades
- **Entry/Exit Opportunity Analysis**: Check possible entry and exit points
- **Missed Trade Analysis**: Investigate why trades were missed
- **CSV Export**: Export detailed trade logs for visual validation

#### Testing Requirements
- **Simulation Mode**: Ability to run simulations on historical data
- **Trade Validation**: Visual validation using CSV exports
- **Pattern Performance**: Track pattern performance (count, win rate, P&L)

### 6. Key Features Implemented

#### Pattern Detection
- Advanced indicator calculation (momentum, volume trends, price position, MA relationships, MACD, RSI zones)
- Multiple bullish pattern detection
- Pattern scoring and confidence calculation

#### Entry Validation
- False breakout detection
- Reverse split detection
- Setup confirmation (multiple periods)
- Price validation
- Expected gain validation

#### Exit Management
- Multiple exit strategies (stop loss, trailing stop, profit target, strong reversal)
- Partial profit taking
- Dynamic trailing stops based on profit levels
- Minimum hold time enforcement

#### Data Management
- Real-time data fetching (1-minute intervals)
- Indicator calculation
- Historical data analysis
- Database persistence

### 7. Specific Implementation Details

#### Change Percentage Calculation
- Use minute-level data for accurate calculations
- Premarket/Regular hours: Calculate from previous day's close
- After-hours: Calculate from today's close
- Always use current price from minute data for display

#### Rejected Entries Tracking
- Store in database with ticker, price, reason, timestamp, date
- Display in web dashboard
- Clear when position successfully entered
- Persist for the day

#### Bot Synchronization
- Backend refreshes at 5th second of every minute
- Webapp refreshes every 30 seconds (simple interval, no complex synchronization needed)

#### Color Theme
- Modern, professional appearance
- Gradient buttons and cards
- Enhanced shadows and elevation
- Smooth transitions
- Professional typography

### 8. Error Handling

#### Common Issues to Handle
- Python execution blocked (Windows App Store Python)
- Unicode encoding errors (use ASCII-safe characters)
- Database connection errors
- API rate limits
- Missing data
- Corrupted database rows

### 9. Future Enhancements (Not Yet Implemented)

#### Potential Improvements
- Pre-market trading support
- After-hours trading support
- More sophisticated pattern detection
- Machine learning integration
- Advanced risk management
- Portfolio optimization
- Multi-timeframe analysis

## Development Guidelines

1. **Incremental Development**: Implement features incrementally, test thoroughly
2. **Code Reusability**: Create reusable components and functions
3. **Error Recovery**: Implement robust error handling and recovery
4. **User Feedback**: Provide clear feedback in UI and logs
5. **Performance**: Optimize for real-time processing
6. **Maintainability**: Write clean, well-documented code
7. **Testing**: Test with real data and simulations
8. **Documentation**: Document all major features and decisions

## Success Criteria

- Bot successfully identifies bullish patterns
- Trades are executed with proper risk management
- Exit logic captures profits while protecting capital
- Web dashboard provides clear visibility into bot operations
- System handles errors gracefully
- All data persists correctly
- Performance is acceptable for real-time trading
