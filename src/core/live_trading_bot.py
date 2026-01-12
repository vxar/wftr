"""
Live Trading Bot - Production Ready
Identifies bullish trends, places trades, and exits on trend reversal or indicator signals
Designed to grow capital from $10,000 to $100,000+
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import logging
from core.realtime_trader import RealtimeTrader, TradeSignal, ActivePosition
from data.api_interface import DataAPI
from analysis.premarket_analyzer import PreMarketAnalyzer
from database.trading_database import TradingDatabase, TradeRecord, PositionRecord
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a completed trade"""
    ticker: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    shares: float
    entry_value: float
    exit_value: float
    pnl_pct: float
    pnl_dollars: float
    entry_pattern: str
    exit_reason: str
    confidence: float


class LiveTradingBot:
    """
    Production-ready live trading bot
    Monitors stocks, identifies bullish trends, places trades, and exits on reversals
    """
    
    def __init__(self,
                 data_api: DataAPI,
                 initial_capital: float = 10000.0,
                 target_capital: float = 100000.0,
                 min_confidence: float = 0.72,  # BALANCED: 72% - high-quality trades with reasonable opportunities
                 min_entry_price_increase: float = 5.5,  # BALANCED: 5.5% - good quality setups
                 trailing_stop_pct: float = 2.5,  # REFINED: 2.5% - tighter stops, cut losses faster
                 profit_target_pct: float = 8.0,  # REFINED: 8% - realistic profit target
                 position_size_pct: float = 0.50,  # REFINED: 50% - more conservative sizing
                 max_positions: int = 3,
                 max_loss_per_trade_pct: float = 2.5,  # REFINED: 2.5% - cut losses faster
                 daily_profit_target_min: float = 500.0,
                 daily_profit_target_max: float = 500.0,
                 max_trades_per_day: int = 8,  # NEW: Limit trades per day for quality
                 max_daily_loss: float = -300.0,  # NEW: Stop trading if daily loss exceeds this
                 consecutive_loss_limit: int = 3,  # NEW: Pause after N consecutive losses
                 trading_start_time: str = "04:00",
                 trading_end_time: str = "20:00"):  # Trading window: 4:00 AM - 8:00 PM ET
        """
        Args:
            data_api: DataAPI instance for fetching live data
            initial_capital: Starting capital in USD
            target_capital: Target capital to reach
            min_confidence: Minimum pattern confidence (0-1)
            min_entry_price_increase: Minimum expected price increase (%)
            trailing_stop_pct: Trailing stop loss percentage
            profit_target_pct: Profit target percentage
            position_size_pct: Percentage of capital to use per trade
            max_positions: Maximum concurrent positions
            max_loss_per_trade_pct: Maximum loss per trade before forced exit (%)
            daily_profit_target_min: Minimum daily profit target in USD
            daily_profit_target_max: Maximum daily profit target in USD
            trading_start_time: Trading window start time (HH:MM format, ET)
            trading_end_time: Trading window end time (HH:MM format, ET)
        """
        self.data_api = data_api
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.target_capital = target_capital
        self.trader = RealtimeTrader(
            min_confidence=min_confidence,
            min_entry_price_increase=min_entry_price_increase,
            trailing_stop_pct=trailing_stop_pct,
            profit_target_pct=profit_target_pct,
            data_api=data_api  # Pass data_api for multi-timeframe analysis
        )
        self.premarket_analyzer = PreMarketAnalyzer(
            min_confidence=min_confidence,
            min_entry_price_increase=min_entry_price_increase
        )
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.max_loss_per_trade_pct = max_loss_per_trade_pct
        
        # Daily profit targets
        self.daily_profit_target_min = daily_profit_target_min
        self.daily_profit_target_max = daily_profit_target_max
        self.daily_profit = 0.0
        self.daily_start_capital = initial_capital
        self.current_date = None
        
        # Trading window
        self.trading_start_time = trading_start_time
        self.trading_end_time = trading_end_time
        self.et_timezone = pytz.timezone('America/New_York')
        
        self.tickers: List[str] = []
        self.trade_history: List[Trade] = []  # Keep for backward compatibility
        self.running = False
        self.premarket_signals: Dict[str, Dict] = {}  # Store premarket analysis
        self.monitoring_status: Dict[str, Dict] = {}  # Track monitoring status for each ticker
        self.last_stock_discovery: Optional[datetime] = None  # Track when we last refreshed stock list
        self.stock_discovery_interval_minutes = 1  # Refresh stock list every 1 minute (keep list fresh)
        self.top_gainers: List[str] = []  # Track top gainers
        self.top_gainers_data: List[Dict] = []  # Store full gainer data with change % for sorting
        self.rejected_entries: List[Dict] = []  # Track rejected entry signals for display (keep last 50)
        
        # Re-entry tracking: track when tickers were exited to allow re-entry after cooldown
        self.ticker_exit_times: Dict[str, datetime] = {}  # Track exit time per ticker
        self.re_entry_cooldown_minutes = 10  # Wait 10 minutes before allowing re-entry
        
        # Autonomous trading safety limits
        self.max_trades_per_day = max_trades_per_day
        self.max_daily_loss = max_daily_loss  # Kept for logging but not enforced
        self.consecutive_loss_limit = consecutive_loss_limit
        self.daily_trade_count = 0
        self.consecutive_losses = 0
        self.trading_paused = False  # Auto-pause flag
        self.pause_reason = ""  # Reason for pause
        
        # Performance tracking for auto-adjustment
        self.recent_trades: List[Dict] = []  # Last 20 trades for performance analysis
        self.win_rate_window = 20  # Analyze last N trades
        
        # Initialize database
        self.db = TradingDatabase()
        logger.info("Database initialized for trade persistence")
        
        # Load trade history from database
        self._load_trade_history_from_db()
        
        # Clean up any stale positions (positions with completed trades)
        try:
            cleaned = self.db.cleanup_orphaned_positions()
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} stale position(s) on startup")
        except Exception as e:
            logger.warning(f"Could not cleanup stale positions: {e}")
        
        # Restore active positions from database (if any)
        self._restore_active_positions_from_db()
        
        logger.info(f"Trading Bot Initialized:")
        logger.info(f"  Initial Capital: ${initial_capital:,.2f}")
        logger.info(f"  Target Capital: ${target_capital:,.2f}")
        logger.info(f"  Daily Profit Target: ${daily_profit_target_min:,.2f} - ${daily_profit_target_max:,.2f}")
        logger.info(f"  Trading Window: {trading_start_time} - {trading_end_time} ET")
        logger.info(f"  Position Size: {position_size_pct*100:.0f}% per trade")
        logger.info(f"  Max Positions: {max_positions}")
        logger.info(f"  Min Confidence: {min_confidence*100:.0f}%")
        logger.info(f"  Min Expected Gain: {min_entry_price_increase:.1f}%")
    
    def _load_trade_history_from_db(self):
        """Load trade history from database"""
        try:
            db_trades = self.db.get_all_trades()
            logger.info(f"Loaded {len(db_trades)} trades from database")
        except Exception as e:
            logger.warning(f"Could not load trades from database: {e}")
    
    def _restore_active_positions_from_db(self):
        """Restore active positions from database after restart"""
        try:
            db_positions = self.db.get_active_positions()
            if db_positions:
                logger.info(f"Found {len(db_positions)} active positions in database, restoring to memory...")
                
                for db_pos in db_positions:
                    try:
                        ticker = db_pos.get('ticker')
                        if not ticker:
                            continue
                        
                        # Skip if already in memory
                        if ticker in self.trader.active_positions:
                            continue
                        
                        # Parse entry_time
                        entry_time_str = db_pos.get('entry_time')
                        if isinstance(entry_time_str, str):
                            try:
                                entry_time = pd.to_datetime(entry_time_str)
                            except:
                                entry_time = datetime.now()
                        else:
                            entry_time = datetime.now()
                        
                        # Get current price for the position
                        try:
                            current_price = self.data_api.get_current_price(ticker)
                            if current_price is None or current_price <= 0:
                                # Try to get from 1-minute data
                                df = self.data_api.get_1min_data(ticker, minutes=1)
                                if len(df) > 0:
                                    current_price = df.iloc[-1]['close']
                                else:
                                    current_price = db_pos.get('entry_price', 0)
                        except:
                            current_price = db_pos.get('entry_price', 0)
                        
                        # Create ActivePosition object
                        position = ActivePosition(
                            ticker=ticker,
                            entry_time=entry_time,
                            entry_price=float(db_pos.get('entry_price', 0)),
                            entry_pattern=db_pos.get('entry_pattern', 'Unknown'),
                            entry_confidence=float(db_pos.get('confidence', 0)),
                            target_price=float(db_pos.get('target_price', 0)) if db_pos.get('target_price') else None,
                            stop_loss=float(db_pos.get('stop_loss', 0)) if db_pos.get('stop_loss') else None,
                            current_price=current_price,
                            shares=float(db_pos.get('shares', 0)),
                            entry_value=float(db_pos.get('entry_value', 0)),
                            partial_profit_taken=bool(db_pos.get('partial_profit_taken', False)),
                            partial_profit_taken_second=bool(db_pos.get('partial_profit_taken_second', False)),
                            original_shares=float(db_pos.get('shares', 0))  # Assume no partial exit yet
                        )
                        
                        # Calculate unrealized P&L
                        if position.entry_price > 0:
                            position.unrealized_pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                        else:
                            position.unrealized_pnl_pct = 0.0
                        
                        # Set max_price_reached to current price (conservative)
                        position.max_price_reached = current_price
                        
                        # Add to trader's active positions
                        self.trader.active_positions[ticker] = position
                        
                        logger.info(f"Restored position: {ticker} - {int(position.shares)} shares @ ${position.entry_price:.4f} (Entry Value: ${position.entry_value:,.2f})")
                        
                    except Exception as e:
                        logger.warning(f"Could not restore position {db_pos.get('ticker', 'UNKNOWN')}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        continue
                
                restored_count = len([t for t in self.trader.active_positions.keys() if t in [p.get('ticker') for p in db_positions]])
                logger.info(f"Successfully restored {restored_count} active positions to memory")
                
                # Update current capital from database to reflect actual cash available
                # (Capital tied up in positions is already accounted for in the database)
                try:
                    current_cash = self.db.get_current_capital_from_db(self.initial_capital)
                    if current_cash is not None:
                        self.current_capital = current_cash
                        logger.info(f"Updated current capital from database: ${self.current_capital:,.2f}")
                except Exception as e:
                    logger.warning(f"Could not update capital from database: {e}")
                
        except Exception as e:
            logger.warning(f"Could not restore positions from database: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def add_ticker(self, ticker: str):
        """Add a ticker to monitor"""
        if ticker not in self.tickers:
            self.tickers.append(ticker)
            logger.info(f"Added {ticker} to monitoring list")
    
    def remove_ticker(self, ticker: str):
        """Remove a ticker from monitoring"""
        if ticker in self.tickers:
            self.tickers.remove(ticker)
            logger.info(f"Removed {ticker} from monitoring list")
    
    def update_tickers_from_gainers(self, max_tickers: int = 20):
        """
        Update ticker list from top gainers (if using WebullDataAPI)
        
        Args:
            max_tickers: Maximum number of tickers to fetch
        """
        if hasattr(self.data_api, 'get_stock_list_from_gainers'):
            try:
                # Get full gainer data (not just symbols) to access change percentage
                gainer_data = self.data_api.get_top_gainers(page_size=max_tickers)
                # Sort by change percentage (descending - highest first)
                gainer_data_sorted = sorted(
                    gainer_data, 
                    key=lambda x: x.get('change_ratio', 0) or x.get('changeRatio', 0) or 0, 
                    reverse=True
                )
                # Update top gainers list
                new_tickers = [g.get('symbol', '') for g in gainer_data_sorted if g.get('symbol')]
                self.top_gainers = new_tickers[:max_tickers]
                # Store full gainer data with change % for API endpoint
                self.top_gainers_data = gainer_data_sorted[:max_tickers]
                
                # Add new tickers that aren't already being monitored
                added = 0
                for ticker in new_tickers:
                    if ticker not in self.tickers:
                        self.add_ticker(ticker)
                        added += 1
                if added > 0:
                    logger.info(f"Updated ticker list: Added {added} new tickers from top gainers")
            except Exception as e:
                logger.error(f"Error updating tickers from gainers: {e}")
    
    
    def _get_current_et_time(self) -> datetime:
        """Get current time in Eastern Time"""
        return datetime.now(self.et_timezone)
    
    def _is_sleep_time(self) -> bool:
        """Check if current time is within sleep period (8 PM to 4 AM ET)"""
        current_time = self._get_current_et_time()
        current_hour = current_time.hour
        current_minute = current_time.minute
        current_time_value = current_hour * 60 + current_minute
        
        # Sleep period: 8:00 PM (1200) to 4:00 AM (240) next day
        # This means: time >= 1200 (8 PM) OR time < 240 (4 AM)
        return current_time_value >= 1200 or current_time_value < 240
    
    def _is_trading_window(self) -> bool:
        """Check if current time is within trading window (4 AM to 8 PM, excludes sleep time)"""
        # First check if we're in sleep time
        if self._is_sleep_time():
            return False
        
        current_time = self._get_current_et_time()
        current_hour = current_time.hour
        current_minute = current_time.minute
        current_time_value = current_hour * 60 + current_minute
        
        start_hour, start_minute = map(int, self.trading_start_time.split(':'))
        end_hour, end_minute = map(int, self.trading_end_time.split(':'))
        start_time_value = start_hour * 60 + start_minute
        end_time_value = end_hour * 60 + end_minute
        
        # Trading window includes premarket if start_time is before 9:30 AM
        # But only if we're not in sleep time (already checked above)
        return start_time_value <= current_time_value <= end_time_value
    
    def _is_premarket(self) -> bool:
        """Check if current time is premarket (7:00 AM - 9:30 AM ET)"""
        current_time = self._get_current_et_time()
        current_hour = current_time.hour
        current_minute = current_time.minute
        current_time_value = current_hour * 60 + current_minute
        
        # Premarket: 7:00 AM (420) to 9:30 AM (570)
        return 420 <= current_time_value < 570
    
    def _is_after_trading_window(self) -> bool:
        """Check if current time is after trading window (after 4:00 PM ET)"""
        current_time = self._get_current_et_time()
        current_hour = current_time.hour
        current_minute = current_time.minute
        current_time_value = current_hour * 60 + current_minute
        
        end_hour, end_minute = map(int, self.trading_end_time.split(':'))
        end_time_value = end_hour * 60 + end_minute
        
        return current_time_value > end_time_value
    
    def _cleanup_positions(self):
        """
        Clean up stale positions from database
        Should be called periodically to ensure positions table only contains active positions
        """
        try:
            # Remove positions that have completed trades
            cleaned_orphaned = self.db.cleanup_orphaned_positions()
            
            # Remove any inactive positions (shouldn't exist if close_position deletes them, but just in case)
            cleaned_inactive = self.db.cleanup_inactive_positions()
            
            total_cleaned = cleaned_orphaned + cleaned_inactive
            if total_cleaned > 0:
                logger.info(f"Position cleanup: Removed {total_cleaned} stale position(s)")
            
            return total_cleaned
        except Exception as e:
            logger.warning(f"Error during position cleanup: {e}")
            return 0
    
    def _reset_daily_tracking(self):
        """Reset daily profit tracking for new trading day"""
        current_time = self._get_current_et_time()
        current_date = current_time.date()
        
        if self.current_date != current_date:
            self.current_date = current_date
            
            # Get daily start capital and profit from database
            try:
                date_str = current_date.strftime('%Y-%m-%d')
                daily_data = self.db.get_daily_profit_from_db(self.initial_capital, date_str)
                self.daily_start_capital = daily_data['daily_start_capital']
                self.daily_profit = daily_data['daily_profit']
            except Exception as e:
                logger.warning(f"Error getting daily data from database: {e}, using portfolio value")
                self.daily_start_capital = self.get_portfolio_value()
                self.daily_profit = 0.0
            
            # Reset daily counters for new day
            self.daily_trade_count = 0
            self.consecutive_losses = 0
            self.trading_paused = False
            self.pause_reason = ""
            
            logger.info(f"New Trading Day: {current_date}")
            logger.info(f"   Starting Capital: ${self.daily_start_capital:,.2f}")
            logger.info(f"   Daily Limits: Max Trades={self.max_trades_per_day}, Max Loss=DISABLED (unlimited)")
            logger.info(f"   Consecutive Loss Limit: DISABLED (testing mode - all trades allowed)")
            logger.info(f"   Daily Profit Target: ${self.daily_profit_target_min:,.2f} - ${self.daily_profit_target_max:,.2f}")
            logger.info(f"   Current Daily Profit: ${self.daily_profit:+,.2f}")
    
    def _update_daily_profit(self):
        """Update daily profit tracking from database"""
        try:
            from datetime import datetime
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # Get daily profit from database
            daily_data = self.db.get_daily_profit_from_db(self.initial_capital, current_date)
            
            self.daily_profit = daily_data['daily_profit']
            self.daily_start_capital = daily_data['daily_start_capital']
            
            # Also update current_capital to match database
            portfolio_value = self.get_portfolio_value()
            # Current capital should be: daily_start_capital (from DB) + daily_profit (from DB)
            # But we also need to account for active positions, so use get_portfolio_value which handles that
        except Exception as e:
            logger.error(f"Error updating daily profit from database: {e}")
            # Fallback to in-memory calculation
            portfolio_value = self.get_portfolio_value()
            self.daily_profit = portfolio_value - self.daily_start_capital
    
    def get_current_positions(self) -> Dict[str, ActivePosition]:
        """Get all current active positions"""
        return self.trader.active_positions.copy()
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value (cash + positions) from database"""
        try:
            # Get current cash capital from database (recalculated for accuracy)
            current_cash = self.db.get_current_capital_from_db(self.initial_capital)
            
            # Update in-memory current_capital to match database
            self.current_capital = current_cash
            
            # Start with cash
            total_value = current_cash
            
            # Add value of all open positions (from memory and database)
            # Use 1-minute data for accurate current prices
            # Create a copy to avoid "dictionary changed size during iteration" error
            active_positions_copy = dict(self.trader.active_positions)  # Create a snapshot
            # Add positions from memory
            for position in active_positions_copy.values():
                try:
                    # Get current price from latest 1-minute data
                    df = self.data_api.get_1min_data(position.ticker, minutes=1)
                    if len(df) > 0:
                        current_price = df.iloc[-1]['close']
                    else:
                        current_price = self.data_api.get_current_price(position.ticker)
                    
                    shares = position.shares if hasattr(position, 'shares') else 0
                    position_value = shares * current_price
                    total_value += position_value
                except Exception as e:
                    logger.warning(f"Could not get current price for {position.ticker}: {e}")
            
            # Add positions from database (that might not be in memory after restart)
            if hasattr(self, 'db'):
                db_positions = self.db.get_active_positions()
                memory_tickers = set(self.trader.active_positions.keys())
                # Create a copy of db_positions list to avoid iteration issues
                db_positions_copy = list(db_positions)
                for db_pos in db_positions_copy:
                    if db_pos['ticker'] not in memory_tickers:
                        try:
                            # Get current price from latest 1-minute data
                            df = self.data_api.get_1min_data(db_pos['ticker'], minutes=1)
                            if len(df) > 0:
                                current_price = df.iloc[-1]['close']
                            else:
                                current_price = self.data_api.get_current_price(db_pos['ticker'])
                            
                            shares = float(db_pos.get('shares', 0))
                            position_value = shares * current_price
                            total_value += position_value
                        except Exception as e:
                            logger.warning(f"Could not get current price for {db_pos['ticker']}: {e}")
            
            return total_value
        except Exception as e:
            logger.error(f"Error calculating portfolio value from database: {e}")
            # Fallback to in-memory calculation
            total_value = self.current_capital
            for position in self.trader.active_positions.values():
                try:
                    current_price = self.data_api.get_current_price(position.ticker)
                    position_value = position.shares * current_price if hasattr(position, 'shares') else 0
                    total_value += position_value
                except Exception as e:
                    logger.warning(f"Could not get current price for {position.ticker}: {e}")
            return total_value
    
    def _process_ticker(self, ticker: str) -> Tuple[Optional[TradeSignal], List[TradeSignal]]:
        """
        Process a single ticker: check for entry/exit signals
        
        Returns:
            Tuple of (entry_signal, exit_signals)
        """
        # Initialize variables to avoid scope errors
        entry_signal = None
        exit_signals = []
        
        try:
            # Initialize monitoring status for this ticker
            if ticker not in self.monitoring_status:
                self.monitoring_status[ticker] = {
                    'status': 'monitoring',
                    'last_check': None,
                    'rejection_reasons': [],
                    'has_position': False,
                    'current_price': None,
                    'is_fast_mover': False,
                    'fast_mover_vol_ratio': 0.0,
                    'fast_mover_momentum': 0.0
                }
            
            # Update last check time
            self.monitoring_status[ticker]['last_check'] = self._get_current_et_time()
            self.monitoring_status[ticker]['has_position'] = ticker in self.trader.active_positions
            
            # Fetch latest data (800 minutes = ~13 hours of 1-min data)
            df = self.data_api.get_1min_data(ticker, minutes=800)
            logger.debug(f"[{ticker}] Fetched {len(df)} minutes of data")
            
            if len(df) < 50:
                reason = f"Insufficient data: only {len(df)} minutes"
                logger.warning(f"{reason} for {ticker}")
                self.monitoring_status[ticker]['rejection_reasons'] = [reason]
                self.monitoring_status[ticker]['status'] = 'rejected'
                return None, []
            
            # Get current price
            try:
                current_price = self.data_api.get_current_price(ticker)
                self.monitoring_status[ticker]['current_price'] = current_price
            except:
                current_price = df.iloc[-1]['close'] if len(df) > 0 else 0
                self.monitoring_status[ticker]['current_price'] = current_price
            
            # During premarket, analyze and execute trades
            if self._is_premarket():
                # Analyze premarket data for entry signals
                premarket_trades = self.premarket_analyzer.analyze_premarket(df, ticker)
                if premarket_trades:
                    # Store the best premarket signal
                    best_signal = max(premarket_trades, key=lambda x: x.get('confidence', 0))
                    self.premarket_signals[ticker] = best_signal
                    logger.info(f"PREMARKET analysis for {ticker}: {best_signal.get('pattern')} "
                              f"(Confidence: {best_signal.get('confidence', 0)*100:.1f}%)")
                    
                    # Create entry signal for premarket trading
                    entry_signal = TradeSignal(
                        signal_type='entry',
                        ticker=ticker,
                        timestamp=self._get_current_et_time(),
                        price=best_signal.get('entry_price', current_price),
                        pattern_name=best_signal.get('pattern'),
                        confidence=best_signal.get('confidence', 0),
                        reason=f"Premarket signal: {best_signal.get('pattern')}",
                        target_price=best_signal.get('target_price'),
                        stop_loss=best_signal.get('stop_loss')
                    )
                    return entry_signal, []
                
                # Also check for regular entry signals during premarket
                # Pass current_price to ensure positions show current premarket price
                entry_signal, exit_signals = self.trader.analyze_data(df, ticker, current_price=current_price)
                return entry_signal, exit_signals
            
            # During trading window, analyze for entry/exit signals
            if self._is_trading_window():
                # Analyze for entry/exit signals
                # Pass current_price to ensure positions show current price
                entry_signal, exit_signals = self.trader.analyze_data(df, ticker, current_price=current_price)
                
            # Update monitoring status based on analysis (silently, no logging)
            if entry_signal:
                self.monitoring_status[ticker]['status'] = 'entry_signal'
                self.monitoring_status[ticker]['rejection_reasons'] = []
            elif ticker in self.trader.last_rejection_reasons:
                self.monitoring_status[ticker]['status'] = 'rejected'
                self.monitoring_status[ticker]['rejection_reasons'] = self.trader.last_rejection_reasons.get(ticker, [])
            else:
                self.monitoring_status[ticker]['status'] = 'no_signal'
                self.monitoring_status[ticker]['rejection_reasons'] = ['No pattern detected']
            
            # Add fast mover information if detected
            if ticker in self.trader.last_fast_mover_status:
                fast_mover_info = self.trader.last_fast_mover_status[ticker]
                self.monitoring_status[ticker]['is_fast_mover'] = True
                self.monitoring_status[ticker]['fast_mover_vol_ratio'] = fast_mover_info.get('vol_ratio', 0)
                self.monitoring_status[ticker]['fast_mover_momentum'] = fast_mover_info.get('momentum', 0)
            else:
                self.monitoring_status[ticker]['is_fast_mover'] = False
                
                # If we have a premarket signal and no current position, consider using it
                if not entry_signal and ticker in self.premarket_signals and ticker not in self.trader.active_positions:
                    premarket_signal = self.premarket_signals[ticker]
                    # Validate the premarket signal is still valid
                    if self._validate_premarket_signal_still_valid(df, premarket_signal):
                        entry_signal = TradeSignal(
                            signal_type='entry',
                            ticker=ticker,
                            timestamp=self._get_current_et_time(),
                            price=premarket_signal.get('entry_price', df.iloc[-1]['close']),
                            pattern_name=premarket_signal.get('pattern'),
                            confidence=premarket_signal.get('confidence', 0),
                            reason=f"Premarket signal confirmed: {premarket_signal.get('pattern')}",
                            target_price=premarket_signal.get('target_price'),
                            stop_loss=premarket_signal.get('stop_loss')
                        )
                
                return entry_signal, exit_signals
            
            # After trading window, only check exits for positions that no longer meet criteria
            if self._is_after_trading_window():
                # Check if we should hold or exit positions
                exit_signals = self._check_hold_or_exit_after_window(df, ticker)
                return None, exit_signals
            
            return None, []
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            return None, []
    
    def _validate_premarket_signal_still_valid(self, df: pd.DataFrame, premarket_signal: Dict) -> bool:
        """Validate that premarket signal is still valid at market open"""
        if len(df) < 5:
            return False
        
        current = df.iloc[-1]
        current_price = current['close']
        entry_price = premarket_signal.get('entry_price', current_price)
        
        # Price should be close to premarket entry price (within 2%)
        if abs(current_price - entry_price) / entry_price > 0.02:
            return False
        
        return True
    
    def _check_hold_or_exit_after_window(self, df: pd.DataFrame, ticker: str) -> List[TradeSignal]:
        """
        After 4:00 PM (market close), check if positions should be held (if setup still valid and in uptrend)
        or exited (if setup failed or trend reversed)
        """
        exit_signals = []
        
        if ticker not in self.trader.active_positions:
            return exit_signals
        
        position = self.trader.active_positions[ticker]
        
        # Calculate indicators
        df_with_indicators = self.trader.pattern_detector.calculate_indicators(df)
        
        if len(df_with_indicators) < 1:
            return exit_signals
        
        current = df_with_indicators.iloc[-1]
        current_price = current['close']
        
        # Check if setup is still valid and stock is in uptrend
        setup_still_valid = self._is_setup_still_valid(df_with_indicators, position)
        in_uptrend = self._is_in_uptrend(df_with_indicators)
        
        # Hold if setup is valid AND in uptrend
        if setup_still_valid and in_uptrend:
            logger.info(f"HOLDING {ticker} after market close - Setup valid and in uptrend")
            return exit_signals
        
        # Otherwise, exit (setup failed or trend reversed)
        exit_reason = "Setup no longer valid or trend reversed after trading window"
        if not setup_still_valid:
            exit_reason = "Setup conditions no longer met after trading window"
        elif not in_uptrend:
            exit_reason = "Uptrend reversed after trading window"
        
        exit_signal = TradeSignal(
            signal_type='exit',
            ticker=ticker,
            timestamp=self._get_current_et_time(),
            price=current_price,
            reason=exit_reason,
            confidence=1.0
        )
        exit_signals.append(exit_signal)
        
        return exit_signals
    
    def _is_setup_still_valid(self, df: pd.DataFrame, position: ActivePosition) -> bool:
        """Check if the entry setup conditions are still valid"""
        if len(df) < 5:
            return False
        
        current = df.iloc[-1]
        
        # Check key setup conditions
        conditions_met = 0
        
        # 1. Price above key MAs
        if (current.get('close', 0) > current.get('sma_5', 0) and
            current.get('close', 0) > current.get('sma_10', 0)):
            conditions_met += 1
        
        # 2. MACD still bullish
        if current.get('macd', 0) > current.get('macd_signal', 0):
            conditions_met += 1
        
        # 3. Volume still decent
        if current.get('volume_ratio', 0) > 1.0:
            conditions_met += 1
        
        # 4. Price not significantly below entry
        if current.get('close', 0) > position.entry_price * 0.95:  # Within 5% of entry
            conditions_met += 1
        
        # Need at least 3 out of 4 conditions
        return conditions_met >= 3
    
    def _is_in_uptrend(self, df: pd.DataFrame) -> bool:
        """Check if stock is still in uptrend"""
        if len(df) < 10:
            return False
        
        current = df.iloc[-1]
        recent = df.iloc[-10:]
        
        # Check multiple uptrend indicators
        uptrend_signals = 0
        
        # 1. Price above MAs in bullish order
        if (current.get('sma_5', 0) > current.get('sma_10', 0) and
            current.get('sma_10', 0) > current.get('sma_20', 0)):
            uptrend_signals += 1
        
        # 2. Price making higher highs
        recent_highs = recent['high'].values
        if len(recent_highs) >= 5:
            first_half_max = max(recent_highs[:len(recent_highs)//2])
            second_half_max = max(recent_highs[len(recent_highs)//2:])
            if second_half_max > first_half_max * 0.98:  # Higher or similar highs
                uptrend_signals += 1
        
        # 3. MACD bullish
        if current.get('macd', 0) > current.get('macd_signal', 0):
            uptrend_signals += 1
        
        # 4. Price momentum positive
        if len(recent) >= 5:
            price_change = ((current.get('close', 0) - recent.iloc[0].get('close', 0)) / 
                          recent.iloc[0].get('close', 0)) * 100
            if price_change > -1.0:  # Not down more than 1%
                uptrend_signals += 1
        
        # Need at least 3 out of 4 signals
        return uptrend_signals >= 3
    
    def _execute_entry(self, signal: TradeSignal) -> bool:
        """
        Execute an entry trade with autonomous safety checks
        
        Returns:
            True if trade was executed, False otherwise
        """
        try:
            # === AUTONOMOUS SAFETY CHECKS ===
            
            # 1. Check if trading is paused
            if self.trading_paused:
                reason = f"Trading paused: {self.pause_reason}"
                logger.warning(f"[REJECTED] Entry: {signal.ticker} @ ${signal.price:.4f} - {reason}")
                self._add_rejected_entry(signal.ticker, signal.price, reason)
                return False
            
            # 2. Check daily trade limit
            if self.daily_trade_count >= self.max_trades_per_day:
                reason = f"Daily trade limit reached ({self.daily_trade_count}/{self.max_trades_per_day})"
                logger.warning(f"[REJECTED] Entry: {signal.ticker} @ ${signal.price:.4f} - {reason}")
                self._add_rejected_entry(signal.ticker, signal.price, reason)
                return False
            
            # 3. Daily loss limit REMOVED - allow trading regardless of daily loss
            # This prevents missing opportunities like JTAI where a big move happens after initial losses
            
            # 4. Consecutive loss limit REMOVED - allow trading regardless of consecutive losses
            # This is disabled for testing mode to allow all possible trades for fine-tuning
            
            # Check if we have enough capital
            position_value = self.current_capital * self.position_size_pct
            
            # Reject stocks below $0.50 minimum price
            if signal.price < 0.50:
                reason = "Price below minimum $0.50"
                logger.warning(f"[REJECTED] Entry: {signal.ticker} @ ${signal.price:.4f} - {reason}")
                self._add_rejected_entry(signal.ticker, signal.price, reason)
                return False
            
            if position_value < 100:  # Minimum $100 to trade
                reason = f"Insufficient capital (${self.current_capital:.2f} < $100)"
                logger.warning(f"[REJECTED] Entry: {signal.ticker} @ ${signal.price:.4f} - {reason}")
                self._add_rejected_entry(signal.ticker, signal.price, reason)
                return False
            
            # Calculate shares (round to whole number)
            shares = round(position_value / signal.price)
            
            # Check if we already have a position in this ticker
            if signal.ticker in self.trader.active_positions:
                reason = f"Already have position in {signal.ticker}"
                logger.warning(f"[REJECTED] Entry: {signal.ticker} @ ${signal.price:.4f} - {reason}")
                self._add_rejected_entry(signal.ticker, signal.price, reason)
                return False
            
            # Check database for existing position
            if hasattr(self, 'db'):
                db_positions = self.db.get_active_positions()
                if any(p['ticker'] == signal.ticker for p in db_positions):
                    reason = "Already have position in database"
                    logger.warning(f"[REJECTED] Entry: {signal.ticker} @ ${signal.price:.4f} - {reason}")
                    self._add_rejected_entry(signal.ticker, signal.price, reason)
                    return False
            
            # RE-ENTRY LOGIC: Check if we recently exited this ticker and if cooldown has passed
            if signal.ticker in self.ticker_exit_times:
                exit_time = self.ticker_exit_times[signal.ticker]
                time_since_exit = (datetime.now() - exit_time).total_seconds() / 60  # minutes
                
                if time_since_exit < self.re_entry_cooldown_minutes:
                    reason = f"Re-entry cooldown active: {time_since_exit:.1f} min < {self.re_entry_cooldown_minutes} min"
                    logger.warning(f"[REJECTED] Entry: {signal.ticker} @ ${signal.price:.4f} - {reason}")
                    self._add_rejected_entry(signal.ticker, signal.price, reason)
                    return False
                else:
                    # Cooldown passed, allow re-entry
                    logger.info(f"[OK] Re-entry cooldown passed for {signal.ticker} ({time_since_exit:.1f} min). Allowing re-entry after previous exit.")
                    # Remove from exit tracking to allow entry
                    del self.ticker_exit_times[signal.ticker]
            
            # Check if we're at max positions (include both memory and database positions)
            memory_position_count = len(self.trader.active_positions)
            db_position_count = 0
            if hasattr(self, 'db'):
                db_positions = self.db.get_active_positions()
                memory_tickers = set(self.trader.active_positions.keys())
                # Count database positions that aren't in memory
                db_position_count = sum(1 for p in db_positions if p['ticker'] not in memory_tickers)
            
            total_positions = memory_position_count + db_position_count
            if total_positions >= self.max_positions:
                reason = f"At max positions ({total_positions}/{self.max_positions})"
                logger.warning(f"[REJECTED] Entry: {signal.ticker} @ ${signal.price:.4f} - {reason}")
                self._add_rejected_entry(signal.ticker, signal.price, reason)
                return False
            
            # Deduct capital
            self.current_capital -= position_value
            
            # Enter position (pass DataFrame for ATR calculation)
            # Get recent data for ATR calculation
            try:
                df_for_atr = self.data_api.get_1min_data(signal.ticker, minutes=100)
            except:
                df_for_atr = None
            
            position = self.trader.enter_position(signal, df=df_for_atr)
            position.shares = shares
            position.original_shares = shares  # Store original shares
            position.entry_value = position_value
            
            # Clear any rejected entries for this ticker since we successfully entered
            self.rejected_entries = [r for r in self.rejected_entries if r['ticker'] != signal.ticker]
            # Also clear from database
            try:
                self.db.clear_rejected_entries_for_ticker(signal.ticker)
            except Exception as e:
                logger.error(f"Error clearing rejected entries from database: {e}")
            
            # Validate ticker before saving position
            ticker = str(signal.ticker).strip() if signal.ticker else None
            if not ticker:
                logger.error(f"Cannot save position: ticker is null or empty for signal {signal}")
                return False
            
            # Save position to database
            try:
                position_record = PositionRecord(
                    ticker=ticker,
                    entry_time=signal.timestamp if hasattr(signal.timestamp, 'isoformat') else datetime.now(),
                    entry_price=signal.price,
                    shares=shares,
                    entry_value=position_value,
                    entry_pattern=signal.pattern_name if hasattr(signal, 'pattern_name') else 'Unknown',
                    confidence=signal.confidence if hasattr(signal, 'confidence') else 0.0,
                    target_price=signal.target_price if hasattr(signal, 'target_price') else None,
                    stop_loss=signal.stop_loss if hasattr(signal, 'stop_loss') else None,
                    is_active=True
                )
                self.db.add_position(position_record)
            except Exception as e:
                logger.error(f"Error saving position to database: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # Clear exit tracking for this ticker since we're entering a new position
            if signal.ticker in self.ticker_exit_times:
                del self.ticker_exit_times[signal.ticker]
                logger.debug(f"Cleared exit tracking for {signal.ticker} (new position entered)")
            
            logger.info(f"ENTRY: {signal.ticker} @ ${signal.price:.4f}")
            logger.info(f"   Pattern: {signal.pattern_name} (Confidence: {signal.confidence*100:.1f}%)")
            logger.info(f"   Shares: {shares}")
            logger.info(f"   Position Value: ${position_value:,.2f}")
            logger.info(f"   Target: ${signal.target_price:.4f}")
            logger.info(f"   Stop Loss: ${signal.stop_loss:.4f}")
            logger.info(f"   Capital Remaining: ${self.current_capital:,.2f}")
            logger.info(f"   Reason: {signal.reason}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing entry for {signal.ticker}: {e}")
            return False
    
    def _execute_exit(self, signal: TradeSignal) -> Optional[Trade]:
        """
        Execute an exit trade
        
        Returns:
            Trade object if successful, None otherwise
        """
        try:
            position = self.trader.exit_position(signal)
            
            if not position:
                return None
            
            # Calculate P&L
            pnl_pct = ((signal.price - position.entry_price) / position.entry_price) * 100
            
            # Get position value
            position_value = position.shares * signal.price if hasattr(position, 'shares') else 0
            pnl_dollars = position_value - position.entry_value if hasattr(position, 'entry_value') else 0
            
            # Update capital
            self.current_capital += position_value
            
            # Close position in database
            try:
                self.db.close_position(signal.ticker)
            except Exception as e:
                logger.error(f"Error closing position in database: {e}")
            
            # Validate ticker before creating trade record
            ticker = str(signal.ticker).strip() if signal.ticker else None
            if not ticker:
                logger.error(f"Cannot create trade record: ticker is null or empty for signal {signal}")
                return None
            
            # Create trade record
            trade = Trade(
                ticker=ticker,
                entry_time=position.entry_time,
                exit_time=signal.timestamp,
                entry_price=position.entry_price,
                exit_price=signal.price,
                shares=position.shares if hasattr(position, 'shares') else 0,
                entry_value=position.entry_value if hasattr(position, 'entry_value') else 0,
                exit_value=position_value,
                pnl_pct=pnl_pct,
                pnl_dollars=pnl_dollars,
                entry_pattern=position.entry_pattern if hasattr(position, 'entry_pattern') else 'Unknown',
                exit_reason=signal.reason if hasattr(signal, 'reason') else 'Unknown',
                confidence=position.entry_confidence if hasattr(position, 'entry_confidence') else 0.0
            )
            
            self.trade_history.append(trade)
            
            # Save to database (only if ticker is valid)
            if trade.ticker and str(trade.ticker).strip():
                try:
                    trade_record = TradeRecord(
                        ticker=str(trade.ticker).strip(),
                        entry_time=trade.entry_time,
                        exit_time=trade.exit_time,
                        entry_price=trade.entry_price,
                        exit_price=trade.exit_price,
                        shares=trade.shares,
                        entry_value=trade.entry_value,
                        exit_value=trade.exit_value,
                        pnl_pct=trade.pnl_pct,
                        pnl_dollars=trade.pnl_dollars,
                        entry_pattern=trade.entry_pattern if trade.entry_pattern else 'Unknown',
                        exit_reason=trade.exit_reason if trade.exit_reason else 'Unknown',
                        confidence=trade.confidence
                    )
                    self.db.add_trade(trade_record)
                except Exception as e:
                    logger.error(f"Error saving trade to database: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            else:
                logger.error(f"Cannot save trade to database: invalid ticker '{trade.ticker}'")
            
            # Track trade for performance analysis
            self.daily_trade_count += 1
            trade_result = {
                'ticker': signal.ticker,
                'pnl_dollars': pnl_dollars,
                'pnl_pct': pnl_pct,
                'timestamp': signal.timestamp,
                'is_win': pnl_dollars > 0
            }
            self.recent_trades.append(trade_result)
            
            # Keep only last N trades for analysis
            if len(self.recent_trades) > self.win_rate_window:
                self.recent_trades.pop(0)
            
            # Update consecutive losses counter (for logging/informational purposes only)
            if pnl_dollars < 0:
                self.consecutive_losses += 1
                logger.debug(f"Consecutive losses: {self.consecutive_losses} (tracking only, not blocking trades)")
            else:
                self.consecutive_losses = 0  # Reset on win
            
            # Track exit time for re-entry logic
            self.ticker_exit_times[signal.ticker] = signal.timestamp
            logger.debug(f"Tracked exit time for {signal.ticker}: {signal.timestamp} (re-entry allowed after {self.re_entry_cooldown_minutes} min cooldown)")
            
            # Log exit
            status = "WIN" if pnl_dollars > 0 else "LOSS"
            logger.info(f"{status} EXIT: {signal.ticker} @ ${signal.price:.4f}")
            logger.info(f"   Entry: ${position.entry_price:.4f} @ {position.entry_time}")
            logger.info(f"   Exit: ${signal.price:.4f} @ {signal.timestamp}")
            logger.info(f"   Shares: {int(trade.shares)}")
            logger.info(f"   Entry Value: ${trade.entry_value:,.2f}")
            logger.info(f"   Exit Value: ${trade.exit_value:,.2f}")
            logger.info(f"   P&L: {pnl_pct:+.2f}% (${pnl_dollars:+,.2f})")
            logger.info(f"   Capital: ${self.current_capital:,.2f}")
            logger.info(f"   Reason: {signal.reason}")
            
            return trade
            
        except Exception as e:
            logger.error(f"Error executing exit for {signal.ticker}: {e}")
            return None
    
    def _execute_partial_exit(self, signal: TradeSignal) -> bool:
        """
        Execute progressive partial profit taking:
        - First exit: 50% at +4% profit
        - Second exit: 25% at +7% profit (of remaining position)
        - Hold remaining 25% to target
        
        Args:
            signal: Partial exit signal
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if signal.ticker not in self.trader.active_positions:
                return False
            
            position = self.trader.active_positions[signal.ticker]
            
            # Determine which partial exit this is
            if not position.partial_profit_taken:
                # First partial exit: 50% at +4%
                shares_to_sell = round(position.shares * 0.5)
                exit_pct = 50
            elif hasattr(position, 'partial_profit_taken_second') and not position.partial_profit_taken_second:
                # Second partial exit: 25% at +7% (of remaining position)
                shares_to_sell = round(position.shares * 0.5)  # 50% of remaining = 25% of original
                exit_pct = 25
            else:
                # Already taken both partial exits
                return False
            
            if shares_to_sell < 1:
                return False
            
            # Calculate P&L for partial exit
            partial_entry_value = (shares_to_sell / position.shares) * position.entry_value
            partial_exit_value = shares_to_sell * signal.price
            partial_pnl = partial_exit_value - partial_entry_value
            partial_pnl_pct = ((signal.price - position.entry_price) / position.entry_price) * 100
            
            # Update capital
            self.current_capital += partial_exit_value
            
            # Update position
            position.shares -= shares_to_sell
            position.entry_value -= partial_entry_value
            
            # Mark which partial exit was taken
            if not position.partial_profit_taken:
                position.partial_profit_taken = True
                # Move stop loss to breakeven after first partial exit
                position.stop_loss = position.entry_price
                logger.info(f"[{signal.ticker}] Stop moved to breakeven after first partial profit taking")
            else:
                position.partial_profit_taken_second = True
                # Tighten stop loss after second partial exit (trailing stop will handle it)
                logger.info(f"[{signal.ticker}] Second partial profit taken, trailing stop active")
            
            # Update position in database
            try:
                self.db.update_position(
                    signal.ticker,
                    target_price=position.target_price,
                    stop_loss=position.stop_loss,
                    shares=position.shares,
                    entry_value=position.entry_value
                )
            except Exception as e:
                logger.error(f"Error updating position in database: {e}")
            
            # Create partial trade record (only if ticker is valid)
            ticker = str(signal.ticker).strip() if signal.ticker else None
            if not ticker:
                logger.error(f"Cannot create partial trade record: ticker is null or empty for signal {signal}")
                return False
            
            try:
                from database.trading_database import TradeRecord
                from datetime import datetime
                
                partial_trade = TradeRecord(
                    ticker=ticker,
                    entry_time=position.entry_time,
                    exit_time=signal.timestamp if hasattr(signal.timestamp, 'isoformat') else datetime.now(),
                    entry_price=position.entry_price,
                    exit_price=signal.price,
                    shares=shares_to_sell,
                    entry_value=partial_entry_value,
                    exit_value=partial_exit_value,
                    pnl_pct=partial_pnl_pct,
                    pnl_dollars=partial_pnl,
                    entry_pattern=position.entry_pattern if hasattr(position, 'entry_pattern') else 'Unknown',
                    exit_reason=f"Partial profit taking ({exit_pct}%) at +{partial_pnl_pct:.2f}%",
                    confidence=position.entry_confidence if hasattr(position, 'entry_confidence') else 0.0
                )
                self.db.add_trade(partial_trade)
            except Exception as e:
                logger.error(f"Error saving partial trade to database: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # Log partial exit
            logger.info(f"PARTIAL EXIT: {signal.ticker} @ ${signal.price:.4f}")
            logger.info(f"   Sold: {shares_to_sell} shares ({exit_pct}% of original position)")
            logger.info(f"   P&L: {partial_pnl_pct:+.2f}% (${partial_pnl:+,.2f})")
            logger.info(f"   Remaining: {position.shares} shares")
            logger.info(f"   Stop Loss: Moved to breakeven @ ${position.entry_price:.4f}")
            logger.info(f"   Capital: ${self.current_capital:,.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing partial exit for {signal.ticker}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _check_forced_exits(self) -> List[TradeSignal]:
        """
        Check for positions that need forced exit (max loss exceeded)
        
        Returns:
            List of exit signals for forced exits
        """
        exit_signals = []
        
        # Create a copy to avoid "dictionary changed size during iteration" error
        active_positions_copy = dict(self.trader.active_positions)
        for ticker, position in active_positions_copy.items():
            try:
                current_price = self.data_api.get_current_price(ticker)
                loss_pct = ((position.entry_price - current_price) / position.entry_price) * 100
                
                # Force exit if loss exceeds maximum
                if loss_pct > self.max_loss_per_trade_pct:
                    exit_signal = TradeSignal(
                        signal_type='exit',
                        ticker=ticker,
                        timestamp=datetime.now(),
                        price=current_price,
                        reason=f"Max loss exceeded ({loss_pct:.2f}% > {self.max_loss_per_trade_pct}%)",
                        confidence=1.0
                    )
                    exit_signals.append(exit_signal)
                    logger.warning(f"FORCED EXIT: {ticker} - Max loss exceeded ({loss_pct:.2f}%)")
                    
            except Exception as e:
                logger.error(f"Error checking forced exit for {ticker}: {e}")
        
        return exit_signals
    
    def _find_underperforming_stocks(self) -> List[Tuple[str, ActivePosition, float]]:
        """
        Find stocks with low returns that could be replaced
        
        Returns:
            List of tuples (ticker, position, return_pct) sorted by worst performers first
        """
        underperformers = []
        
        # Create a copy to avoid "dictionary changed size during iteration" error
        active_positions_copy = dict(self.trader.active_positions)
        for ticker, position in active_positions_copy.items():
            try:
                current_price = self.data_api.get_current_price(ticker)
                return_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                
                # Consider stocks with low or negative returns as candidates for replacement
                # Only during trading window (9:30-11:00)
                if self._is_trading_window() and return_pct < 2.0:  # Less than 2% return
                    underperformers.append((ticker, position, return_pct))
                    
            except Exception as e:
                logger.error(f"Error checking performance for {ticker}: {e}")
        
        # Sort by worst performers first
        underperformers.sort(key=lambda x: x[2])
        
        return underperformers
    
    def _find_better_opportunities(self, exclude_tickers: List[str]) -> List[TradeSignal]:
        """
        Find better entry opportunities from available tickers
        
        Args:
            exclude_tickers: List of tickers to exclude (already have positions)
            
        Returns:
            List of entry signals sorted by confidence/expected gain
        """
        opportunities = []
        
        for ticker in self.tickers:
            if ticker in exclude_tickers:
                continue
            
            if ticker in self.trader.active_positions:
                continue
            
            try:
                entry_signal, _ = self._process_ticker(ticker)
                if entry_signal:
                    opportunities.append(entry_signal)
            except Exception as e:
                logger.error(f"Error checking opportunity for {ticker}: {e}")
        
        # Sort by confidence and expected gain
        opportunities.sort(key=lambda x: (
            x.confidence,
            ((x.target_price - x.price) / x.price) * 100 if x.target_price else 0
        ), reverse=True)
        
        return opportunities
    
    def _replace_underperforming_stocks(self):
        """
        Replace low-performing stocks with better opportunities during trading window
        """
        if not self._is_trading_window():
            return
        
        if len(self.trader.active_positions) >= self.max_positions:
            # Find underperformers
            underperformers = self._find_underperforming_stocks()
            
            if not underperformers:
                return
            
            # Find better opportunities
            exclude_tickers = list(self.trader.active_positions.keys())
            opportunities = self._find_better_opportunities(exclude_tickers)
            
            if not opportunities:
                return
            
            # Replace worst performer with best opportunity
            worst_ticker, worst_position, worst_return = underperformers[0]
            best_opportunity = opportunities[0]
            
            # Check if replacement makes sense
            best_expected_gain = ((best_opportunity.target_price - best_opportunity.price) / 
                                 best_opportunity.price) * 100 if best_opportunity.target_price else 0
            
            # Only replace if new opportunity is significantly better
            if best_expected_gain > abs(worst_return) + 3.0:  # At least 3% better
                logger.info(f"🔄 REPLACING: {worst_ticker} (Return: {worst_return:.2f}%) "
                          f"with {best_opportunity.ticker} (Expected: {best_expected_gain:.2f}%)")
                
                # Exit underperformer
                current_price = self.data_api.get_current_price(worst_ticker)
                exit_signal = TradeSignal(
                    signal_type='exit',
                    ticker=worst_ticker,
                    timestamp=self._get_current_et_time(),
                    price=current_price,
                    reason=f"Replaced with better opportunity ({best_opportunity.ticker})",
                    confidence=1.0
                )
                self._execute_exit(exit_signal)
                
                # Enter new position
                self._execute_entry(best_opportunity)
    
    def run_single_cycle(self):
        """Run a single trading cycle (check all tickers once)"""
        # Don't run during sleep time (8 PM to 4 AM)
        if self._is_sleep_time():
            logger.info("Sleep time (8 PM - 4 AM) - Skipping trading cycle")
            return
        
        # Reset daily tracking if new day
        self._reset_daily_tracking()
        self._update_daily_profit()
        
        current_time = self._get_current_et_time()
        
        # Periodically refresh stock list from top gainers
        should_refresh = False
        if self.last_stock_discovery is None:
            # First run - refresh immediately
            should_refresh = True
        else:
            # Check if enough time has passed (refresh every 1 minute to keep list fresh)
            time_since_refresh = (current_time - self.last_stock_discovery).total_seconds() / 60
            if time_since_refresh >= self.stock_discovery_interval_minutes:
                should_refresh = True
        
        if should_refresh:
            try:
                logger.info("Refreshing stock list from top gainers...")
                self.update_tickers_from_gainers(max_tickers=20)
                self.last_stock_discovery = current_time
                logger.info(f"Stock list refreshed. Now monitoring {len(self.tickers)} tickers from top gainers")
            except Exception as e:
                logger.warning(f"Error refreshing stock list from top gainers: {e}")
        
        if not self.tickers:
            logger.warning("No tickers to monitor")
            return
        
        logger.info(f"\n{'='*80}")
        logger.info(f"TRADING CYCLE - {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.info(f"{'='*80}")
        
        # Check if market is premarket, trading window, or after hours
        if self._is_premarket():
            logger.info("PREMARKET: Trading enabled - Analyzing and executing trades")
            # Process premarket trading (analyze and execute)
            for ticker in self.tickers:
                try:
                    entry_signal, exit_signals = self._process_ticker(ticker)
                    
                    # Process exit signals first (to free up capital)
                    for exit_signal in exit_signals:
                        if exit_signal.signal_type == 'partial_exit':
                            self._execute_partial_exit(exit_signal)
                        else:
                            self._execute_exit(exit_signal)
                    
                    # Process entry signal (if we have capital and room)
                    if entry_signal and self.current_capital > 100:
                        # Check if we're at max positions
                        if len(self.trader.active_positions) < self.max_positions:
                            self._execute_entry(entry_signal)
                except Exception as e:
                    logger.error(f"Error in premarket trading for {ticker}: {e}")
            
            # Print portfolio status after premarket cycle
            portfolio_value = self.get_portfolio_value()
            logger.info(f"PREMARKET Portfolio Value: ${portfolio_value:,.2f}")
            return
        
        # Check for forced exits first
        forced_exits = self._check_forced_exits()
        for exit_signal in forced_exits:
            self._execute_exit(exit_signal)
        
        # During trading window, check for stock replacements
        if self._is_trading_window():
            self._replace_underperforming_stocks()
        
        # Process each ticker
        logger.info(f"Processing {len(self.tickers)} tickers for entry/exit signals...")
        for ticker in self.tickers:
            try:
                logger.debug(f"Checking {ticker} for trading opportunities...")
                entry_signal, exit_signals = self._process_ticker(ticker)
                
                # Process exit signals first (to free up capital)
                for exit_signal in exit_signals:
                    if exit_signal.signal_type == 'partial_exit':
                        self._execute_partial_exit(exit_signal)
                    else:
                        self._execute_exit(exit_signal)
                
                # Process entry signal (if we have capital and room)
                if entry_signal:
                    logger.info(f"[ENTRY SIGNAL] Found: {entry_signal.ticker} @ ${entry_signal.price:.4f} - {entry_signal.pattern_name} ({entry_signal.confidence*100:.1f}%)")
                    if self.current_capital <= 100:
                        logger.warning(f"[REJECTED] Entry: {entry_signal.ticker} - Insufficient capital (${self.current_capital:.2f} < $100)")
                    elif len(self.trader.active_positions) >= self.max_positions:
                        logger.warning(f"[REJECTED] Entry: {entry_signal.ticker} - At max positions ({len(self.trader.active_positions)}/{self.max_positions})")
                    else:
                        self._execute_entry(entry_signal)
                
            except Exception as e:
                logger.error(f"Error in trading cycle for {ticker}: {e}")
        
        # After trading window, only manage existing positions
        if self._is_after_trading_window():
            logger.info("After trading window - Managing existing positions only")
            for ticker in list(self.trader.active_positions.keys()):
                try:
                    _, exit_signals = self._process_ticker(ticker)
                    for exit_signal in exit_signals:
                        if exit_signal.signal_type == 'partial_exit':
                            self._execute_partial_exit(exit_signal)
                        else:
                            self._execute_exit(exit_signal)
                except Exception as e:
                    logger.error(f"Error managing position for {ticker}: {e}")
        
        # Print portfolio status
        portfolio_value = self.get_portfolio_value()
        total_return = ((portfolio_value - self.initial_capital) / self.initial_capital) * 100
        
        logger.info(f"\nPORTFOLIO STATUS:")
        logger.info(f"   Cash: ${self.current_capital:,.2f}")
        logger.info(f"   Portfolio Value: ${portfolio_value:,.2f}")
        logger.info(f"   Total Return: {total_return:+.2f}%")
        logger.info(f"   Daily Profit: ${self.daily_profit:+,.2f}")
        logger.info(f"   Daily Target: ${self.daily_profit_target_min:,.2f} - ${self.daily_profit_target_max:,.2f}")
        # Count active positions (both in-memory and database-only)
        memory_position_count = len(self.trader.active_positions)
        db_position_count = 0
        if hasattr(self, 'db'):
            db_positions = self.db.get_active_positions()
            memory_tickers = set(self.trader.active_positions.keys())
            db_position_count = sum(1 for p in db_positions if p.get('ticker') not in memory_tickers)
        total_active_positions = memory_position_count + db_position_count
        logger.info(f"   Active Positions: {total_active_positions} (Memory: {memory_position_count}, DB-only: {db_position_count})")
        logger.info(f"   Total Trades: {len(self.trade_history)}")
        
        if self.trade_history:
            winning_trades = [t for t in self.trade_history if t.pnl_dollars > 0]
            win_rate = len(winning_trades) / len(self.trade_history) * 100
            total_pnl = sum(t.pnl_dollars for t in self.trade_history)
            logger.info(f"   Win Rate: {win_rate:.1f}%")
            logger.info(f"   Total P&L: ${total_pnl:+,.2f}")
        
        # Check if daily profit target reached
        if self.daily_profit >= self.daily_profit_target_min:
            if self.daily_profit >= self.daily_profit_target_max:
                logger.info(f"Daily profit target MAX reached: ${self.daily_profit:,.2f}")
            else:
                logger.info(f"Daily profit target MIN reached: ${self.daily_profit:,.2f}")
        
        logger.info(f"{'='*80}\n")
        
        # End of day analysis (after market close, around 4:00 PM ET)
        if current_time.hour >= 16:  # After 4 PM ET
            self._end_of_day_analysis()
    
    def _end_of_day_analysis(self):
        """Perform end-of-day analysis and identify areas for improvement"""
        current_time = self._get_current_et_time()
        
        # Only run once per day
        if not hasattr(self, '_eod_analysis_done') or self._eod_analysis_done != current_time.date():
            self._eod_analysis_done = current_time.date()
        else:
            return  # Already done today
        
        logger.info(f"\n{'='*80}")
        logger.info(f"END OF DAY ANALYSIS - {current_time.strftime('%Y-%m-%d')}")
        logger.info(f"{'='*80}")
        
        portfolio_value = self.get_portfolio_value()
        daily_return = ((portfolio_value - self.daily_start_capital) / self.daily_start_capital) * 100
        
        # Daily performance summary
        logger.info(f"\nDAILY PERFORMANCE:")
        logger.info(f"   Starting Capital: ${self.daily_start_capital:,.2f}")
        logger.info(f"   Ending Capital: ${portfolio_value:,.2f}")
        logger.info(f"   Daily Profit/Loss: ${self.daily_profit:+,.2f}")
        logger.info(f"   Daily Return: {daily_return:+.2f}%")
        logger.info(f"   Target Range: ${self.daily_profit_target_min:,.2f} - ${self.daily_profit_target_max:,.2f}")
        
        # Trade analysis
        today_trades = [t for t in self.trade_history 
                       if t.entry_time.date() == current_time.date() or 
                          t.exit_time.date() == current_time.date()]
        
        # Initialize variables to avoid scope errors
        winning_trades = []
        losing_trades = []
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        
        if today_trades:
            winning_trades = [t for t in today_trades if t.pnl_dollars > 0]
            losing_trades = [t for t in today_trades if t.pnl_dollars <= 0]
            
            win_rate = len(winning_trades) / len(today_trades) * 100 if today_trades else 0
            avg_win = sum(t.pnl_dollars for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t.pnl_dollars for t in losing_trades) / len(losing_trades) if losing_trades else 0
            
            logger.info(f"\nTRADE ANALYSIS:")
            logger.info(f"   Total Trades Today: {len(today_trades)}")
            logger.info(f"   Winning Trades: {len(winning_trades)} ({win_rate:.1f}%)")
            logger.info(f"   Losing Trades: {len(losing_trades)} ({100-win_rate:.1f}%)")
            logger.info(f"   Average Win: ${avg_win:+,.2f}")
            logger.info(f"   Average Loss: ${avg_loss:+,.2f}")
            
            if winning_trades:
                best_trade = max(winning_trades, key=lambda x: x.pnl_dollars)
                logger.info(f"   Best Trade: {best_trade.ticker} - {best_trade.pnl_pct:+.2f}% (${best_trade.pnl_dollars:+,.2f})")
            
            if losing_trades:
                worst_trade = min(losing_trades, key=lambda x: x.pnl_dollars)
                logger.info(f"   Worst Trade: {worst_trade.ticker} - {worst_trade.pnl_pct:+.2f}% (${worst_trade.pnl_dollars:+,.2f})")
        
        # Active positions analysis
        if self.trader.active_positions:
            logger.info(f"\nACTIVE POSITIONS:")
            # Create a copy to avoid "dictionary changed size during iteration" error
            active_positions_copy = dict(self.trader.active_positions)
            for ticker, position in active_positions_copy.items():
                try:
                    current_price = self.data_api.get_current_price(ticker)
                    unrealized_pnl = ((current_price - position.entry_price) / position.entry_price) * 100
                    logger.info(f"   {ticker}: {unrealized_pnl:+.2f}% "
                              f"(Entry: ${position.entry_price:.4f}, Current: ${current_price:.4f})")
                except Exception as e:
                    logger.warning(f"   {ticker}: Could not get current price - {e}")
        
        # Improvement recommendations
        logger.info(f"\nIMPROVEMENT RECOMMENDATIONS:")
        
        recommendations = []
        
        # Check if daily target was met
        if self.daily_profit < self.daily_profit_target_min:
            recommendations.append(f"Daily profit target not met (${self.daily_profit:,.2f} < ${self.daily_profit_target_min:,.2f})")
        
        # Clean up stale positions at end of day
        try:
            self._cleanup_positions()
        except Exception as e:
            logger.warning(f"Error cleaning up positions during end-of-day analysis: {e}")
            recommendations.append("   - Consider: More aggressive entry criteria, larger position sizes, or better stock selection")
        
        # Check win rate
        if today_trades:
            if win_rate < 50:
                recommendations.append(f"WARNING: Low win rate ({win_rate:.1f}%)")
                recommendations.append("   - Consider: Stricter entry criteria, better pattern validation, or improved exit timing")
            
            # Check risk/reward
            if losing_trades and winning_trades:
                risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
                if risk_reward_ratio < 1.5:
                    recommendations.append(f"WARNING: Poor risk/reward ratio ({risk_reward_ratio:.2f})")
                    recommendations.append("   - Consider: Tighter stop losses, better profit targets, or earlier exits on winners")
        
        # Check if too many trades
        if len(today_trades) > 10:
            recommendations.append(f"WARNING: High number of trades ({len(today_trades)})")
            recommendations.append("   - Consider: More selective entry criteria to reduce overtrading")
        
        # Check if too few trades
        if len(today_trades) < 2 and self._is_trading_window():
            recommendations.append(f"WARNING: Low number of trades ({len(today_trades)})")
            recommendations.append("   - Consider: Relaxing entry criteria slightly or monitoring more stocks")
        
        # Check for large losses
        if losing_trades:
            max_loss = min(t.pnl_dollars for t in losing_trades)
            if abs(max_loss) > 200:
                recommendations.append(f"WARNING: Large loss detected (${max_loss:,.2f})")
                recommendations.append("   - Consider: Tighter stop losses or faster exit on setup failure")
        
        if not recommendations:
            recommendations.append("No major issues identified - strategy performing well")
        
        for rec in recommendations:
            logger.info(f"   {rec}")
        
        # Pattern performance analysis
        if today_trades:
            pattern_performance = {}
            for trade in today_trades:
                pattern = trade.entry_pattern
                if pattern not in pattern_performance:
                    pattern_performance[pattern] = {'wins': 0, 'losses': 0, 'total_pnl': 0}
                if trade.pnl_dollars > 0:
                    pattern_performance[pattern]['wins'] += 1
                else:
                    pattern_performance[pattern]['losses'] += 1
                pattern_performance[pattern]['total_pnl'] += trade.pnl_dollars
            
            logger.info(f"\nPATTERN PERFORMANCE:")
            for pattern, stats in pattern_performance.items():
                total_trades = stats['wins'] + stats['losses']
                win_rate = (stats['wins'] / total_trades * 100) if total_trades > 0 else 0
                logger.info(f"   {pattern}: {stats['wins']}W/{stats['losses']}L "
                          f"({win_rate:.1f}% win rate, ${stats['total_pnl']:+,.2f} P&L)")
        
        logger.info(f"\n{'='*80}\n")
    
    def run_continuous(self, interval_seconds: int = 60, check_on_second: int = 5):
        """
        Run trading bot continuously
        
        Args:
            interval_seconds: DEPRECATED - kept for compatibility, not used
            check_on_second: Second of each minute to run check (default: 5)
                            Bot will check on the Nth second of every minute
                            This prevents premature exits from intra-minute price volatility
        """
        self.running = True
        logger.info(f"Starting trading bot (checking on {check_on_second}th second of every minute)")
        logger.info(f"Monitoring tickers: {', '.join(self.tickers)}")
        
        cycle_count = 0
        
        def get_next_check_time():
            """Calculate next time to run check (on Nth second of next minute)"""
            current_time = datetime.now()
            # Get next minute with the specified second
            next_check = current_time.replace(second=check_on_second, microsecond=0)
            if next_check <= current_time:
                # If we've passed the check time this minute, go to next minute
                next_check += timedelta(minutes=1)
            return next_check
        
        try:
            while self.running:
                # Check if we're in sleep time (8 PM to 4 AM)
                if self._is_sleep_time():
                    current_time = self._get_current_et_time()
                    logger.info(f"Sleep time (8 PM - 4 AM): Current time is {current_time.strftime('%H:%M:%S')} ET")
                    
                    # Calculate seconds until 4 AM
                    if current_time.hour >= 20:  # After 8 PM, sleep until 4 AM next day
                        # Calculate time until 4 AM next day
                        next_4am = (current_time + timedelta(days=1)).replace(hour=4, minute=0, second=0, microsecond=0)
                    else:  # Before 4 AM, sleep until 4 AM today
                        next_4am = current_time.replace(hour=4, minute=0, second=0, microsecond=0)
                        if next_4am <= current_time:
                            # If 4 AM already passed today, sleep until 4 AM tomorrow
                            next_4am = (current_time + timedelta(days=1)).replace(hour=4, minute=0, second=0, microsecond=0)
                    
                    sleep_seconds = (next_4am - current_time).total_seconds()
                    logger.info(f"Sleeping until 4:00 AM ET ({sleep_seconds/3600:.1f} hours)")
                    
                    # Sleep in chunks to allow for graceful shutdown
                    sleep_chunk = 300  # Check every 5 minutes
                    slept = 0
                    while slept < sleep_seconds and self.running:
                        time.sleep(min(sleep_chunk, sleep_seconds - slept))
                        slept += sleep_chunk
                    
                    if not self.running:
                        break
                    
                    logger.info("Wake time reached (4:00 AM ET) - Resuming trading")
                    continue
                
                # Calculate next check time (5th second of next minute)
                next_check = get_next_check_time()
                current_time = datetime.now()
                wait_seconds = (next_check - current_time).total_seconds()
                
                # If we're very close to or past the check time, run immediately
                if wait_seconds < 0.5:
                    wait_seconds = 0
                elif wait_seconds > 60:
                    # If wait is more than 60 seconds, something's wrong, wait 1 second and recalculate
                    time.sleep(1)
                    continue
                
                # Wait until the check time
                if wait_seconds > 0 and self.running:
                    time.sleep(wait_seconds)
                
                # Run trading cycle
                if self.running:
                    cycle_count += 1
                    logger.info(f"\nCycle #{cycle_count} - {datetime.now().strftime('%H:%M:%S')}")
                    
                    self.run_single_cycle()
                    
                    # Check if we've reached target
                    portfolio_value = self.get_portfolio_value()
                    if portfolio_value >= self.target_capital:
                        logger.info(f"TARGET REACHED! Portfolio value: ${portfolio_value:,.2f}")
                        break
                    
        except KeyboardInterrupt:
            logger.info("\nTrading bot stopped by user")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False
        logger.info("Trading bot stopped")
        
        # Print final summary
        portfolio_value = self.get_portfolio_value()
        total_return = ((portfolio_value - self.initial_capital) / self.initial_capital) * 100
        
        logger.info(f"\n{'='*80}")
        logger.info(f"FINAL SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Final Portfolio Value: ${portfolio_value:,.2f}")
        logger.info(f"Total Return: {total_return:+.2f}%")
        logger.info(f"Profit/Loss: ${portfolio_value - self.initial_capital:+,.2f}")
        logger.info(f"Total Trades: {len(self.trade_history)}")
        
        if self.trade_history:
            winning_trades = [t for t in self.trade_history if t.pnl_dollars > 0]
            losing_trades = [t for t in self.trade_history if t.pnl_dollars <= 0]
            
            win_rate = len(winning_trades) / len(self.trade_history) * 100
            total_pnl = sum(t.pnl_dollars for t in self.trade_history)
            avg_pnl = total_pnl / len(self.trade_history)
            
            logger.info(f"Winning Trades: {len(winning_trades)} ({win_rate:.1f}%)")
            logger.info(f"Losing Trades: {len(losing_trades)} ({100-win_rate:.1f}%)")
            logger.info(f"Total P&L: ${total_pnl:+,.2f}")
            logger.info(f"Average P&L per Trade: ${avg_pnl:+,.2f}")
            
            if winning_trades:
                best_trade = max(self.trade_history, key=lambda x: x.pnl_dollars)
                logger.info(f"\nBest Trade:")
                logger.info(f"  {best_trade.ticker}: {best_trade.pnl_pct:+.2f}% (${best_trade.pnl_dollars:+,.2f})")
            
            if losing_trades:
                worst_trade = min(self.trade_history, key=lambda x: x.pnl_dollars)
                logger.info(f"\nWorst Trade:")
                logger.info(f"  {worst_trade.ticker}: {worst_trade.pnl_pct:+.2f}% (${worst_trade.pnl_dollars:+,.2f})")
        
        logger.info(f"{'='*80}\n")
    
    def _add_rejected_entry(self, ticker: str, price: float, reason: str):
        """
        Add a rejected entry to the tracking list and database for UI display
        
        Args:
            ticker: Stock ticker symbol
            price: Entry price that was rejected
            reason: Reason for rejection
        """
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        
        rejected_entry = {
            'ticker': ticker,
            'price': price,
            'reason': reason,
            'timestamp': now
        }
        
        # Add to in-memory list (for quick access)
        self.rejected_entries.append(rejected_entry)
        # Keep only last 50 entries in memory
        if len(self.rejected_entries) > 50:
            self.rejected_entries = self.rejected_entries[-50:]
        
        # Save to database for persistence
        try:
            self.db.add_rejected_entry(ticker, price, reason, now)
        except Exception as e:
            logger.error(f"Error saving rejected entry to database: {e}")


def main():
    """Main entry point for live trading bot"""
    import sys
    from api_interface import CSVDataAPI  # For testing, replace with your live API
    
    # Configuration
    INITIAL_CAPITAL = 10000.0
    TARGET_CAPITAL = 100000.0
    
    # Tickers to monitor (replace with your preferred stocks)
    if len(sys.argv) > 1:
        tickers = sys.argv[1:]
    else:
        # No hardcoded tickers - bot will fetch from top gainers automatically
        tickers = []
        print("No tickers provided. Bot will fetch tickers from top gainers automatically.")
        print("Usage: python live_trading_bot.py TICKER1 TICKER2 ...")
    
    # Initialize data API (replace CSVDataAPI with your live API)
    # For production, use: from your_api_module import YourLiveAPI
    # api = YourLiveAPI(api_key="your_key")
    api = CSVDataAPI(data_dir="test_data")  # For testing only
    
    # Create and configure bot
    bot = LiveTradingBot(
        data_api=api,
        initial_capital=INITIAL_CAPITAL,
        target_capital=TARGET_CAPITAL,
        min_confidence=0.70,  # 70% confidence - balanced quality and opportunities (OPTIMIZED)
        min_entry_price_increase=5.0,  # 5% expected gain - capture more opportunities (OPTIMIZED)
        trailing_stop_pct=3.0,  # 3% trailing stop - balanced, let winners run (OPTIMIZED)
        profit_target_pct=7.0,  # 7% profit target - capture more gains (OPTIMIZED)
        position_size_pct=0.50,  # 50% of capital per trade - maximize returns (OPTIMIZED)
        max_positions=3,  # Up to 3 positions at once
        max_loss_per_trade_pct=3.0  # Max 3% loss per trade
    )
    
    # Add tickers
    for ticker in tickers:
        bot.add_ticker(ticker)
    
    # Run bot
    try:
        # For live trading, use: bot.run_continuous(interval_seconds=60)
        # For testing, run a single cycle
        print("\nRunning single test cycle...")
        bot.run_single_cycle()
        
        # Uncomment for continuous live trading:
        # bot.run_continuous(interval_seconds=60)
        
    except KeyboardInterrupt:
        print("\nStopped by user")
        bot.stop()


if __name__ == "__main__":
    main()

