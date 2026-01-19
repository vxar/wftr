# Trading Bot API Reference

## Core Classes

### LiveTradingBot

Main trading bot orchestrator.

**Location:** `src/core/live_trading_bot.py`

**Key Methods:**
- `run_single_cycle()` - Execute one trading cycle
- `_process_ticker(ticker)` - Process ticker for entry/exit signals
- `_execute_entry(signal)` - Execute entry trade
- `_execute_exit(signal)` - Execute exit trade
- `get_portfolio_value()` - Calculate total portfolio value

**Configuration:**
```python
initial_capital: float = 10000.0
position_size_pct: float = 0.50
max_positions: int = 3
max_trades_per_day: int = 1000
```

### RealtimeTrader

Generates entry and exit signals.

**Location:** `src/core/realtime_trader.py`

**Key Methods:**
- `analyze_data(df, ticker)` - Analyze data for signals
- `_check_entry_signal(df, ticker)` - Check for entry opportunities
- `_check_exit_signals(df, ticker)` - Check exit conditions
- `enter_position(signal)` - Create active position
- `exit_position(signal)` - Close position

**Parameters:**
```python
min_confidence: float = 0.72
min_entry_price_increase: float = 5.5
surge_detection_enabled: bool = True
```

### PatternDetector

Detects bullish and bearish patterns.

**Location:** `src/analysis/pattern_detector.py`

**Key Methods:**
- `calculate_indicators(df)` - Calculate technical indicators
- `detect_patterns(df, ticker, date)` - Detect patterns
- `_detect_bullish_patterns(...)` - Detect bullish patterns
- `_is_false_breakout(...)` - Check for false breakouts
- `_is_reverse_split(...)` - Check for reverse splits

**Supported Patterns:**
- Volume_Breakout_Momentum
- RSI_Accumulation_Entry
- Golden_Cross_Volume
- Slow_Accumulation
- MACD_Acceleration_Breakout
- Consolidation_Breakout

### TradingDatabase

Handles data persistence.

**Location:** `src/database/trading_database.py`

**Key Methods:**
- `add_trade(trade)` - Record completed trade
- `add_position(position)` - Create/update position
- `close_position(ticker)` - Close position
- `get_active_positions()` - Get all active positions
- `get_statistics()` - Calculate statistics
- `get_current_capital_from_db(initial_capital)` - Calculate capital

**Database Tables:**
- `trades` - Completed trades
- `positions` - Active positions
- `rejected_entries` - Rejected entry signals

### EnhancedGainerScanner

Discovers top gainers.

**Location:** `src/scanning/enhanced_gainer_scanner.py`

**Key Methods:**
- `fetch_and_analyze_gainers(page_size)` - Fetch top gainers
- `_calculate_change_pct(gainer, rank_type)` - Calculate change %
- `get_current_rank_type()` - Get market session type

**Market Sessions:**
- `preMarket` - 4:00 AM - 9:30 AM ET
- `1d` - 9:30 AM - 4:00 PM ET
- `afterMarket` - 4:00 PM - 8:00 PM ET

### EnhancedDashboard

Web interface for monitoring.

**Location:** `src/web/enhanced_dashboard.py`

**Key Methods:**
- `run()` - Start web server
- `_get_bot_status()` - Get bot status
- `_get_positions_data()` - Get positions
- `_get_trades_data(limit)` - Get trades
- `_get_performance_data()` - Get performance metrics

**API Endpoints:**
- `GET /api/status` - Bot status
- `GET /api/positions` - Active positions
- `GET /api/trades` - Trade history
- `POST /api/start` - Start bot
- `POST /api/stop` - Stop bot

## Data Structures

### TradeSignal

Entry or exit signal.

```python
@dataclass
class TradeSignal:
    signal_type: str  # 'entry' or 'exit'
    ticker: str
    timestamp: datetime
    price: float
    pattern_name: str
    confidence: float
    reason: str
    target_price: Optional[float]
    stop_loss: Optional[float]
    indicators: Dict
```

### ActivePosition

Active trading position.

```python
@dataclass
class ActivePosition:
    ticker: str
    entry_time: datetime
    entry_price: float
    entry_pattern: str
    entry_confidence: float
    target_price: float
    stop_loss: float
    current_price: float
    shares: float
    max_price_reached: float
    unrealized_pnl_pct: float
    trailing_stop_price: Optional[float]
    is_slow_mover_entry: bool
    is_surge_entry: bool
```

### PatternSignal

Pattern detection signal.

```python
@dataclass
class PatternSignal:
    ticker: str
    date: str
    pattern_type: str
    pattern_name: str
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    timestamp: str
    indicators: Dict
```

## Return Value Formats

### Bot Status
```python
{
    'running': bool,
    'current_capital': float,
    'portfolio_value': float,
    'total_return': float,
    'daily_profit': float,
    'active_positions': int,
    'total_trades': int
}
```

### Positions Data
```python
[
    {
        'ticker': str,
        'entry_price': float,
        'current_price': float,
        'shares': float,
        'pnl_pct': float,
        'pnl_dollars': float,
        'target_price': float,
        'stop_loss': float
    }
]
```

### Trades Data
```python
[
    {
        'ticker': str,
        'entry_time': str,
        'exit_time': str,
        'entry_price': float,
        'exit_price': float,
        'pnl_pct': float,
        'pnl_dollars': float,
        'entry_pattern': str,
        'exit_reason': str
    }
]
```

## Error Handling

All methods include error handling:
- Database errors: Logged, fallback to in-memory
- API errors: Retry with exponential backoff
- Data errors: Skip ticker, continue processing
- Validation errors: Reject entry, log reason

## Performance Considerations

- **Data Caching:** 30-minute refresh interval
- **Incremental Updates:** 5-minute chunks for existing tickers
- **Database:** WAL mode for better concurrency
- **API Rate Limiting:** 0.1s delay between requests
