"""
Autonomous Trading Bot - Clean Version
Clean version without learning system and other complex features
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import pytz
from pathlib import Path
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from dataclasses import dataclass, asdict

# Import core components
from ..scanning.enhanced_gainer_scanner import EnhancedGainerScanner
from ..analysis.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from ..analysis.manipulation_detector import ManipulationDetector
from .intelligent_position_manager import IntelligentPositionManager, PositionType
from ..risk.volatility_manager import VolatilityManager
from .trading_bot_scheduler import TradingBotScheduler
from .realtime_trader import RealtimeTrader
from ..data.webull_data_api import WebullDataAPI
from ..config.settings import settings
from ..database.trading_database import TradingDatabase, PositionRecord, TradeRecord
from ..utils.trade_processing import process_exit_to_trade_data, process_entry_signal

logger = logging.getLogger(__name__)

class AutonomousTradingBot:
    """
    Clean autonomous trading bot with core functionality only
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the autonomous trading bot
        
        Args:
            config: Optional configuration overrides
        """
        self.et_timezone = pytz.timezone('America/New_York')
        self.running = False
        
        # Default configuration
        base_config = {
            'initial_capital': 10000.0,
            'target_capital': 25000.0,
            'max_positions': 3,
            'position_size_pct': 0.33,
            'risk_per_trade': 0.02,
            
            # Scanning parameters
            'scanner_max_tickers': 30,
            'scanner_update_interval': 60,  # seconds
            
            # Dashboard
            'dashboard_enabled': True,
            'dashboard_port': 5000,
            
            # Data retention
            'data_retention_days': 90
        }
        
        if config:
            base_config.update(config)
        
        self.config = base_config
        
        # Initialize components
        try:
            # Database
            db_path = Path("trading_data.db")
            self.db = TradingDatabase(str(db_path))
            
            # Data API
            self.data_api = WebullDataAPI()
            
            # Manipulation Detector
            self.manipulation_detector = ManipulationDetector()
            
            # Enhanced Scanner
            self.scanner = EnhancedGainerScanner(
                min_volume=50000,
                min_price=0.50,
                max_price=1000.0,
                max_manipulation_score=0.9,  # Increased from 0.7 to 0.9 (less strict)
                min_quality_score=0.3       # Decreased from 0.6 to 0.3 (more permissive)
            )
            
            # Multi-timeframe Analyzer
            self.multi_timeframe_analyzer = MultiTimeframeAnalyzer(self.data_api)
            
            # RealtimeTrader for pattern detection and entry/exit analysis
            self.realtime_trader = RealtimeTrader(
                min_confidence=0.72,
                min_entry_price_increase=5.5,
                trailing_stop_pct=2.5,
                profit_target_pct=8.0,
                data_api=self.data_api,
                rejection_callback=self._log_rejected_trade_callback
            )
            
            # Position Manager
            self.position_manager = IntelligentPositionManager(
                max_positions=self.config['max_positions'],
                position_size_pct=self.config['position_size_pct'],
                risk_per_trade=self.config['risk_per_trade']
            )
            
            # Volatility Manager
            self.volatility_manager = VolatilityManager()
            
            # Performance tracking
            self.performance_metrics = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0
            }
            
            # Daily tracking
            self.current_date = datetime.now(self.et_timezone).date()
            self.daily_start_capital = self.config['initial_capital']
            self.daily_profit = 0.0
            
            # Capital tracking for dashboard
            self.initial_capital = self.config['initial_capital']
            self.current_capital = self.config['initial_capital']
            
            # Monitored tickers storage (for dashboard)
            self.monitored_tickers = []
            self.last_ticker_update = None
            
            # Initialize scheduler
            self.scheduler = TradingBotScheduler(self, settings.trading_window)
            
            # Restore active positions from database on startup
            self._restore_active_positions_from_db()
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def start(self):
        """Start autonomous trading bot"""
        try:
            logger.info("Bot start() method called")
            if self.running:
                logger.warning("Bot is already running")
                return
            
            logger.info("Setting running=True")
            self.running = True
            
            # Start scheduler first
            logger.info("Starting scheduler...")
            self.scheduler.start_scheduler()
            logger.info("Scheduler started successfully")
            
            # Run trading loop in separate thread
            logger.info("Starting trading loop thread...")
            trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
            trading_thread.start()
            logger.info("Trading loop thread started")
            
        except Exception as e:
            logger.error(f"Error starting trading loop: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def stop(self):
        """Stop autonomous trading bot"""
        try:
            # Don't stop if not running (prevents thread join issues)
            if not self.running:
                return
                
            self.running = False
            
            # Stop scheduler
            if hasattr(self, 'scheduler'):
                self.scheduler.stop_scheduler()
            
            logger.info("Autonomous Trading Bot stopped")
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
    
    
    def _trading_loop(self):
        """Main trading loop"""
        try:
            logger.info("Trading loop started - beginning main execution")
            while self.running:
                try:
                    logger.debug("Trading loop iteration - checking for opportunities")
                    # Get current time
                    current_time = datetime.now(self.et_timezone)
                    
                    # Reset daily tracking if new day
                    self._check_daily_reset(current_time)
                    
                    # Market condition monitoring (no pause logic)
                    try:
                        logger.debug("Checking market conditions...")
                        market_condition = self.volatility_manager.check_market_conditions({})
                        logger.debug(f"Market condition: {market_condition}")
                    except Exception as e:
                        logger.error(f"Error checking market conditions: {e}")
                        time.sleep(5)
                        continue
                    
                    # Step 1: Scan for opportunities (simplified - no filtering)
                    try:
                        logger.debug("Fetching gainers...")
                        top_gainers = self.scanner.fetch_and_analyze_gainers(self.config['scanner_max_tickers'])
                        logger.debug(f"Found {len(top_gainers) if top_gainers else 0} gainers")
                        
                        # Update monitored tickers list for dashboard
                        self._update_monitored_tickers(top_gainers, current_time)
                        
                    except Exception as e:
                        logger.error(f"Error fetching gainers: {e}")
                        time.sleep(self.config['scanner_update_interval'])
                        continue
                    
                    if not top_gainers:
                        logger.warning("No gainers found - check scanner logs for details")
                        time.sleep(self.config['scanner_update_interval'])
                        continue
                    
                    # Step 2: Analyze each opportunity
                    for gainer in top_gainers:
                        try:
                            if not self.running:  # Check if bot was stopped
                                break
                            self._analyze_and_process_gainer(gainer, current_time)
                        except Exception as e:
                            logger.error(f"Error processing gainer {gainer.symbol}: {e}")
                            continue
                    
                    # Step 3: Update existing positions
                    self._update_positions()
                    
                    # Step 4: Sleep until next cycle
                    time.sleep(self.config['scanner_update_interval'])
                    
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}")
                    time.sleep(30)  # Wait before retrying
        except Exception as e:
            logger.error(f"CRITICAL ERROR in trading loop thread: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _analyze_and_process_gainer(self, gainer, current_time):
        """Analyze and potentially process a gainer using RealtimeTrader"""
        ticker = gainer.symbol
        
        try:
            logger.info(f"{'='*80}")
            logger.info(f"ANALYZING TICKER: {ticker}")
            logger.info(f"{'='*80}")
            
            # Get 1-minute data up to current minute from Webull
            try:
                df = self.data_api.get_1min_data(ticker, minutes=200)
                if df is None or df.empty:
                    logger.warning(f"[{ticker}] No 1-minute data available - skipping analysis")
                    return
                
                logger.info(f"[{ticker}] Fetched {len(df)} bars of 1-minute data")
                logger.info(f"[{ticker}] Data range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
                
                # Filter out incomplete/future minutes
                # Only use complete minutes (exclude future minutes, but include current minute)
                # At 17:21:20, we should use the 17:21:00 bar as it's the last complete minute
                et_tz = pytz.timezone('America/New_York')
                now_et = datetime.now(et_tz)
                # Get the current minute (rounded down) - this is the last complete minute bar
                # At 17:21:20, this gives us 17:21:00 which is the bar we want to use
                last_complete_minute = now_et.replace(second=0, microsecond=0)
                
                # Ensure timestamp is datetime and in ET timezone
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Convert timestamps to ET if they're not already
                if df['timestamp'].dt.tz is None:
                    df['timestamp'] = df['timestamp'].dt.tz_localize('America/New_York')
                elif str(df['timestamp'].dt.tz) != 'America/New_York':
                    df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')
                
                # Filter to only include complete minutes (up to last_complete_minute)
                df = df[df['timestamp'] <= last_complete_minute].copy()
                
                if df.empty:
                    logger.warning(f"[{ticker}] No complete minute data available after filtering - skipping analysis")
                    return
                
                # Sort by timestamp to ensure correct order
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                logger.info(f"[{ticker}] After filtering incomplete minutes: {len(df)} bars, range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
                
            except Exception as e:
                logger.error(f"[{ticker}] Error fetching 1-minute data: {e}")
                return
            
            # Get current price - try quote API first, fallback to last close from 1-min data
            # Note: Quote API may fail for halted/delisted stocks, after-hours, or API issues
            # Using last close from 1-min data is acceptable for analysis
            current_price = None
            last_close = df['close'].iloc[-1]
            last_timestamp = df['timestamp'].iloc[-1]
            
            try:
                current_price = self.data_api.get_current_price(ticker)
                if current_price is None:
                    current_price = last_close
                    logger.debug(f"[{ticker}] Quote API returned None - using last close ${last_close:.4f} from {last_timestamp}")
            except Exception as e:
                # Quote API failed - this is common for:
                # - Halted or delisted stocks
                # - After-hours trading
                # - Stocks with limited liquidity
                # - API rate limits or temporary issues
                current_price = last_close
                error_msg = str(e)
                if "No quote data available" in error_msg:
                    logger.debug(f"[{ticker}] No quote data available (may be halted/after-hours) - using last close ${last_close:.4f} from {last_timestamp}")
                else:
                    logger.debug(f"[{ticker}] Quote API error: {error_msg} - using last close ${last_close:.4f} from {last_timestamp}")
            
            logger.info(f"[{ticker}] Current price: ${current_price:.4f} (from {'quote API' if current_price != last_close else 'last 1-min close'})")
            
            # Calculate and log key indicators before analysis
            try:
                df_with_indicators = self.realtime_trader.pattern_detector.calculate_indicators(df)
                if len(df_with_indicators) > 0:
                    current = df_with_indicators.iloc[-1]
                    
                    # Log key indicators
                    logger.info(f"[{ticker}] KEY INDICATORS:")
                    logger.info(f"  Price: ${current.get('close', 0):.4f}")
                    logger.info(f"  Volume: {current.get('volume', 0):,.0f}")
                    logger.info(f"  Volume Ratio: {current.get('volume_ratio', 0):.2f}x")
                    
                    # RSI
                    rsi = current.get('rsi', None)
                    if pd.notna(rsi):
                        logger.info(f"  RSI: {rsi:.1f}")
                    
                    # MACD
                    macd = current.get('macd', None)
                    macd_signal = current.get('macd_signal', None)
                    macd_hist = current.get('macd_hist', None)
                    if pd.notna(macd) and pd.notna(macd_signal):
                        macd_status = "BULLISH" if macd > macd_signal else "BEARISH"
                        logger.info(f"  MACD: {macd:.4f} | Signal: {macd_signal:.4f} | Hist: {macd_hist:.4f} ({macd_status})")
                    
                    # Moving Averages
                    sma_5 = current.get('sma_5', None)
                    sma_20 = current.get('sma_20', None)
                    if pd.notna(sma_5) and pd.notna(sma_20):
                        price_vs_sma5 = "ABOVE" if current_price > sma_5 else "BELOW"
                        price_vs_sma20 = "ABOVE" if current_price > sma_20 else "BELOW"
                        logger.info(f"  SMA5: ${sma_5:.4f} (Price {price_vs_sma5}) | SMA20: ${sma_20:.4f} (Price {price_vs_sma20})")
                    
                    # Momentum
                    momentum_5 = current.get('momentum_5', None)
                    momentum_10 = current.get('momentum_10', None)
                    if pd.notna(momentum_5):
                        logger.info(f"  5-min Momentum: {momentum_5:.2f}%")
                    if pd.notna(momentum_10):
                        logger.info(f"  10-min Momentum: {momentum_10:.2f}%")
                    
                    # Price change
                    if len(df) >= 2:
                        price_change_1m = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
                        logger.info(f"  1-min Change: {price_change_1m:.2f}%")
                    
            except Exception as e:
                logger.warning(f"[{ticker}] Error calculating indicators: {e}")
            
            # Analyze using RealtimeTrader (this performs full pattern detection and analysis)
            logger.info(f"[{ticker}] Running pattern detection and analysis...")
            entry_signal, exit_signals = self.realtime_trader.analyze_data(df, ticker, current_price=current_price)
            
            # Log analysis results
            if entry_signal:
                logger.info(f"[{ticker}] ENTRY SIGNAL DETECTED:")
                logger.info(f"  Pattern: {entry_signal.pattern_name}")
                logger.info(f"  Confidence: {entry_signal.confidence*100:.1f}%")
                logger.info(f"  Price: ${entry_signal.price:.4f}")
                logger.info(f"  Target: ${entry_signal.target_price:.4f} ({((entry_signal.target_price/entry_signal.price - 1)*100):.1f}%)")
                logger.info(f"  Stop Loss: ${entry_signal.stop_loss:.4f} ({((entry_signal.stop_loss/entry_signal.price - 1)*100):.1f}%)")
                logger.info(f"  Reason: {entry_signal.reason}")
                
                # Use shared entry processing logic
                entry_config = {
                    'max_positions': self.config['max_positions'],
                    'position_size_pct': self.config['position_size_pct'],
                    'initial_capital': self.config['initial_capital'],
                    'min_capital': 100.0
                }
                
                # Add volume ratio to entry signal if available
                if 'volume_ratio' in df.columns and len(df) > 0:
                    if not hasattr(entry_signal, 'indicators'):
                        entry_signal.indicators = {}
                    entry_signal.indicators['volume_ratio'] = df['volume_ratio'].iloc[-1] if 'volume_ratio' in df.columns else 1.0
                
                entry_result = process_entry_signal(
                    entry_signal,
                    self.position_manager,
                    entry_config,
                    current_capital=self.current_capital,
                    timestamp=datetime.now(self.et_timezone)
                )
                
                if entry_result and entry_result.get('success'):
                    shares = entry_result['shares']
                    position_value = entry_result['position_value']
                    
                    logger.info(f"[{ticker}] POSITION ENTERED: {shares:.2f} shares @ ${entry_signal.price:.4f} (Value: ${position_value:.2f})")
                    self.current_capital -= position_value
                    
                    # Save position to database
                    try:
                        position_record = PositionRecord(
                            ticker=ticker,
                            entry_time=entry_result['timestamp'],
                            entry_price=entry_signal.price,
                            shares=shares,
                            entry_value=position_value,
                            entry_pattern=entry_result['entry_pattern'],
                            confidence=entry_result['confidence'],
                            target_price=entry_result.get('target_price'),
                            stop_loss=entry_result.get('stop_loss'),
                            is_active=True
                        )
                        self.db.add_position(position_record)
                        logger.debug(f"[{ticker}] Position saved to database")
                    except Exception as e:
                        logger.error(f"[{ticker}] Error saving position to database: {e}")
                elif entry_result:
                    logger.warning(f"[{ticker}] Entry rejected: {entry_result.get('reason', 'Unknown reason')}")
                else:
                    logger.warning(f"[{ticker}] Entry processing returned None")
            else:
                # Log detailed rejection information
                logger.info(f"[{ticker}] NO ENTRY SIGNAL DETECTED")
                
                # Check if patterns were detected but rejected
                if ticker in self.realtime_trader.last_rejection_reasons:
                    reasons = self.realtime_trader.last_rejection_reasons[ticker]
                    if reasons:
                        logger.info(f"[{ticker}] REJECTION REASONS:")
                        for reason in reasons:
                            logger.info(f"  - {reason}")
                    else:
                        logger.info(f"[{ticker}] Patterns detected but all rejected (check confidence thresholds)")
                else:
                    logger.info(f"[{ticker}] No patterns detected or insufficient data for pattern detection")
                
                # Log fast mover status if available
                if ticker in self.realtime_trader.last_fast_mover_status:
                    fm_status = self.realtime_trader.last_fast_mover_status[ticker]
                    logger.info(f"[{ticker}] Fast Mover Status: Vol Ratio={fm_status.get('volume_ratio', 0):.2f}x, Momentum={fm_status.get('momentum', 0):.2f}%")
            
            # Process exit signals if any
            if exit_signals:
                for exit_signal in exit_signals:
                    logger.info(f"[{ticker}] EXIT SIGNAL: {exit_signal.signal_type} @ ${exit_signal.price:.4f} - {exit_signal.reason}")
                    # Exit logic would be handled by position manager
                    # For now, just log it
            
            logger.info(f"[{ticker}] Analysis complete")
            logger.info(f"{'='*80}")
            
        except Exception as e:
            logger.error(f"[{ticker}] Error analyzing gainer: {e}")
            import traceback
            logger.error(f"[{ticker}] Traceback: {traceback.format_exc()}")
    
    def _update_positions(self):
        """Update all active positions using both RealtimeTrader and PositionManager"""
        try:
            # Get current market data for all positions
            position_summary = self.position_manager.get_position_summary()
            active_positions = position_summary['active_positions']
            market_data = {}
            
            logger.info(f"Updating positions: {len(active_positions)} active position(s)")
            
            # Analyze each active position with RealtimeTrader for exit signals
            for ticker in active_positions.keys():
                logger.info(f"[{ticker}] Processing position for exit analysis")
                current_price = None
                df = None
                
                try:
                    # Get 1-minute data for exit analysis
                    df = self.data_api.get_1min_data(ticker, minutes=200)
                    if df is not None and not df.empty:
                        # Get current price - try quote API, fallback to DataFrame close
                        # "No quote data available" is not an error - it's common for halted stocks or after-hours
                        try:
                            current_price = self.data_api.get_current_price(ticker)
                        except Exception as e:
                            # Quote API failed - use DataFrame price (this is expected for some stocks)
                            error_msg = str(e)
                            if "No quote data available" not in error_msg:
                                logger.debug(f"[{ticker}] Quote API error (non-critical): {e}")
                        
                        if current_price is None:
                            current_price = df['close'].iloc[-1]
                            logger.debug(f"[{ticker}] Using DataFrame close price ${current_price:.4f} (quote API unavailable)")
                        
                        # Analyze for exit signals using RealtimeTrader
                        entry_signal, exit_signals = self.realtime_trader.analyze_data(df, ticker, current_price=current_price)
                        
                        # Log exit signals if any
                        if exit_signals:
                            for exit_signal in exit_signals:
                                logger.info(f"[{ticker}] EXIT SIGNAL from RealtimeTrader: {exit_signal.signal_type} @ ${exit_signal.price:.4f} - {exit_signal.reason}")
                        
                        # Create market data from cached DataFrame (same data used for exit analysis)
                        try:
                            # Calculate indicators from DataFrame using PatternDetector
                            from src.analysis.pattern_detector import PatternDetector
                            detector = PatternDetector()
                            df_with_indicators = detector.calculate_indicators(df)
                            
                            if len(df_with_indicators) > 0:
                                latest = df_with_indicators.iloc[-1]
                                
                                # Extract values safely
                                volume_ratio = float(latest.get('volume_ratio', 1.0)) if pd.notna(latest.get('volume_ratio')) else 1.0
                                rsi = float(latest.get('rsi', 50.0)) if pd.notna(latest.get('rsi')) else 50.0
                                macd_hist = float(latest.get('macd_hist', 0.0)) if pd.notna(latest.get('macd_hist')) else 0.0
                                volatility = float(latest.get('volatility', 0.0)) if pd.notna(latest.get('volatility')) else 0.0
                                
                                # Calculate price change
                                if len(df) > 1:
                                    price_change_pct = ((current_price - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100)
                                else:
                                    price_change_pct = 0.0
                                
                                data = {
                                    'price': current_price,
                                    'volume_ratio': volume_ratio,
                                    'rsi': rsi,
                                    'macd_hist': macd_hist,
                                    'volatility_score': volatility,
                                    'price_change_pct': price_change_pct
                                }
                                
                                market_data[ticker] = data
                                logger.debug(f"[{ticker}] Market data from DataFrame: price=${current_price:.4f}, RSI={rsi:.1f}, VolRatio={volume_ratio:.2f}x")
                            else:
                                # Minimal fallback if indicators calculation failed
                                data = {
                                    'price': current_price,
                                    'volume_ratio': 1.0,
                                    'rsi': 50.0,
                                    'macd_hist': 0.0,
                                    'volatility_score': 0.0,
                                    'price_change_pct': 0.0
                                }
                                market_data[ticker] = data
                                logger.warning(f"[{ticker}] Using minimal market data (indicator calc failed): price=${current_price:.4f}")
                        except Exception as e:
                            logger.warning(f"[{ticker}] Error creating market data from DataFrame: {e}, using minimal fallback")
                            # Minimal fallback
                            data = {
                                'price': current_price,
                                'volume_ratio': 1.0,
                                'rsi': 50.0,
                                'macd_hist': 0.0,
                                'volatility_score': 0.0,
                                'price_change_pct': 0.0
                            }
                            market_data[ticker] = data
                    else:
                        logger.warning(f"[{ticker}] No 1-minute data available for position analysis")
                        # Try to get current price from position manager as fallback
                        if ticker in self.position_manager.active_positions:
                            current_price = self.position_manager.active_positions[ticker].current_price
                            data = {
                                'price': current_price,
                                'volume_ratio': 1.0,
                                'rsi': 50.0,
                                'macd_hist': 0.0,
                                'volatility_score': 0.0,
                                'price_change_pct': 0.0
                            }
                            market_data[ticker] = data
                            logger.warning(f"[{ticker}] Using position manager price as fallback: ${current_price:.4f}")
                        else:
                            logger.error(f"[{ticker}] Cannot determine current price for exit check - skipping")
                        
                except Exception as e:
                    error_msg = str(e)
                    # Don't log "No quote data available" as an error - it's expected for some stocks
                    if "No quote data available" in error_msg:
                        logger.debug(f"[{ticker}] No quote data available (using DataFrame price): {e}")
                    else:
                        logger.error(f"Error analyzing position {ticker} for exits: {e}")
                    
                    # Still try to get market data
                    try:
                        data = self._get_current_market_data(ticker)
                        if data:
                            market_data[ticker] = data
                    except Exception:
                        pass  # Ignore errors in fallback
            
            # Update positions using position manager
            logger.info(f"Calling position_manager.update_positions() with market_data for {len(market_data)} ticker(s)")
            exits = self.position_manager.update_positions(market_data)
            
            # Log exit detection status
            if exits:
                logger.info(f"Exit signals detected: {len(exits)} position(s) to exit")
                for exit_decision in exits:
                    ticker = exit_decision.get('position_id') or exit_decision.get('ticker', 'UNKNOWN')
                    exit_reason = exit_decision.get('exit_reason', 'Unknown')
                    exit_price = exit_decision.get('exit_price') or exit_decision.get('price', 0)
                    logger.info(f"  - {ticker}: {exit_reason} @ ${exit_price:.4f}")
            elif len(market_data) > 0:
                logger.info(f"No exit signals detected for {len(market_data)} active position(s)")
            else:
                logger.warning(f"No market data available for position updates - exit checks skipped")
            
            # Update positions in database (for active positions)
            # Note: We update positions periodically, but not on every cycle to avoid excessive DB writes
            # Positions are already saved on entry, and will be updated when they change significantly
            
            # Process exits
            for exit_decision in exits:
                self._process_position_exit(exit_decision)
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def _process_position_exit(self, exit_decision):
        """Process a position exit using shared trade processing logic"""
        try:
            # Use shared trade processing function
            trade_data = process_exit_to_trade_data(
                exit_decision,
                self.position_manager,
                datetime.now(self.et_timezone)
            )
            
            if trade_data is None:
                logger.error(f"Failed to process exit: {exit_decision}")
                return
            
            ticker = trade_data['ticker']
            is_partial_exit = trade_data['is_partial_exit']
            
            # Save completed trade to database
            try:
                trade_record = TradeRecord(
                    ticker=trade_data['ticker'],
                    entry_time=trade_data['entry_time'],
                    exit_time=trade_data['exit_time'],
                    entry_price=trade_data['entry_price'],
                    exit_price=trade_data['exit_price'],
                    shares=trade_data['shares'],
                    entry_value=trade_data['entry_value'],
                    exit_value=trade_data['exit_value'],
                    pnl_pct=trade_data['pnl_pct'],
                    pnl_dollars=trade_data['pnl'],
                    entry_pattern=trade_data['entry_pattern'],
                    exit_reason=trade_data['exit_reason'],
                    confidence=trade_data['confidence']
                )
                self.db.add_trade(trade_record)
                logger.debug(f"[{ticker}] Trade saved to database")
                
                # Only close position in database if it's a complete exit (not partial)
                if not is_partial_exit:
                    self.db.close_position(ticker)
                    logger.debug(f"[{ticker}] Position closed in database (complete exit)")
                else:
                    logger.debug(f"[{ticker}] Position remains active (partial exit: {trade_data['shares']:.2f} shares sold)")
            except Exception as e:
                logger.error(f"[{ticker}] Error saving trade to database: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # Update performance metrics
            self.performance_metrics['total_trades'] += 1
            if trade_data['pnl'] > 0:
                self.performance_metrics['winning_trades'] += 1
                self.performance_metrics['total_pnl'] += trade_data['pnl']
            else:
                self.performance_metrics['losing_trades'] += 1
            
            # Return capital
            self.current_capital += trade_data['exit_value']
            
            # Calculate hold time
            hold_time = trade_data['exit_time'] - trade_data['entry_time']
            hold_minutes = hold_time.total_seconds() / 60
            
            # Log detailed exit analysis
            status = "WIN" if trade_data['pnl'] > 0 else "LOSS"
            exit_type = "PARTIAL" if is_partial_exit else "COMPLETE"
            logger.info(f"\n{'='*80}")
            logger.info(f"{status} {exit_type} EXIT: {ticker} @ ${trade_data['exit_price']:.4f}")
            logger.info(f"   Entry: ${trade_data['entry_price']:.4f} @ {trade_data['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"   Exit: ${trade_data['exit_price']:.4f} @ {trade_data['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"   Shares: {int(trade_data['shares'])} {'(partial)' if is_partial_exit else '(full)'}")
            logger.info(f"   Entry Value: ${trade_data['entry_value']:,.2f}")
            logger.info(f"   Exit Value: ${trade_data['exit_value']:,.2f}")
            logger.info(f"   P&L: {trade_data['pnl_pct']:+.2f}% (${trade_data['pnl']:+,.2f})")
            logger.info(f"   Hold Time: {hold_minutes:.1f} minutes ({hold_time})")
            logger.info(f"   Capital: ${self.current_capital:,.2f}")
            logger.info(f"   Pattern: {trade_data['entry_pattern']}")
            logger.info(f"   Confidence: {trade_data['confidence']:.2%}")
            logger.info(f"   Exit Reason: {trade_data['exit_reason']}")
            logger.info(f"{'='*80}\n")
            
        except Exception as e:
            logger.error(f"Error processing position exit: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _get_current_market_data(self, ticker: str) -> Optional[Dict]:
        """Get current market data for a ticker"""
        try:
            # Get current price
            current_price = self.data_api.get_current_price(ticker)
            if current_price is None:
                return None
            
            # Get recent data for calculations
            recent_data = self.data_api.get_1min_data(ticker, minutes=20)
            if recent_data is None or recent_data.empty:
                return None
            
            # Calculate metrics
            volume_ratio = recent_data['volume'].iloc[-1] / recent_data['volume'].iloc[:-1].mean()
            price_change = (current_price - recent_data['close'].iloc[-2]) / recent_data['close'].iloc[-2] * 100
            
            # Simple volatility calculation
            price_changes = recent_data['close'].pct_change().dropna()
            volatility_score = price_changes.std() if len(price_changes) > 1 else 0
            
            return {
                'ticker': ticker,
                'price': current_price,
                'volume': recent_data['volume'].iloc[-1],
                'volume_ratio': volume_ratio,
                'price_change_pct': price_change,
                'volatility_score': volatility_score,
                'timestamp': datetime.now(self.et_timezone)
            }
            
        except Exception as e:
            # Only log error if it's not a common "no quote data" issue
            if "No quote data available" not in str(e):
                logger.error(f"Error getting market data for {ticker}: {e}")
            else:
                logger.debug(f"No quote data available for {ticker} - skipping")
            return None
    
    def _log_rejected_trade(self, ticker: str, reason: str, details: str):
        """Log a rejected trade"""
        try:
            # Log rejection reason for audit trail
            logger.warning(f"REJECTED TRADE: {ticker} - Reason: {reason} - Details: {details}")
            
        except Exception as e:
            logger.error(f"Error logging rejected trade: {e}")
    
    def _log_rejected_trade_callback(self, ticker: str, price: float, reason: str):
        """Callback for RealtimeTrader to log rejected trades"""
        try:
            logger.warning(f"REJECTED TRADE: {ticker} @ ${price:.4f} - Reason: {reason}")
        except Exception as e:
            logger.error(f"Error in rejection callback: {e}")
    
    def _update_monitored_tickers(self, gainers, current_time):
        """Update the monitored tickers list for dashboard display"""
        try:
            self.monitored_tickers = []
            self.last_ticker_update = current_time
            
            for i, gainer in enumerate(gainers[:30]):  # Limit to 30
                # Get additional analysis data for this ticker
                ticker_data = {
                    'symbol': gainer.symbol,
                    'price': gainer.price,
                    'change_pct': gainer.change_pct,
                    'volume': gainer.volume,
                    'surge_score': getattr(gainer, 'surge_score', 0.0),
                    'quality_score': getattr(gainer, 'quality_score', 0.0),
                    'rank': i + 1,
                    
                    # Additional analysis details (will be populated as analysis happens)
                    'manipulation_score': 0.0,
                    'volatility': 0.0,
                    'price_trend': 'neutral',
                    'volume_ratio': getattr(gainer, 'volume_ratio', 1.0),
                    'momentum': 0.0,
                    'rsi': 50.0,
                    'macd_signal': 'neutral',
                    'support_level': 0.0,
                    'resistance_level': 0.0,
                    'analysis_confidence': 0.5,
                    'entry_signal': 'hold',
                    'risk_level': 'medium',
                    'recommendation': 'HOLD'
                }
                
                self.monitored_tickers.append(ticker_data)
                
            logger.debug(f"Updated monitored tickers list with {len(self.monitored_tickers)} tickers")
            
        except Exception as e:
            logger.error(f"Error updating monitored tickers: {e}")
    
    def _check_daily_reset(self, current_time):
        """Check if daily tracking should be reset"""
        try:
            if self.current_date != current_time.date():
                self.current_date = current_time.date()
                self.daily_start_capital = self.config['initial_capital']
                self.daily_profit = 0.0
                
        except Exception as e:
            logger.error(f"Error in daily reset: {e}")
    
    def get_bot_status(self) -> Dict:
        """Get comprehensive bot status"""
        try:
            position_summary = self.position_manager.get_position_summary()
            scheduler_status = self.scheduler.get_scheduler_status() if hasattr(self, 'scheduler') else {}
            
            return {
                'running': self.running,
                'current_capital': self.config['initial_capital'],
                'daily_profit': self.daily_profit,
                'performance_metrics': self.performance_metrics,
                'positions': position_summary,
                'scheduler': scheduler_status
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_scheduler_status(self) -> Dict:
        """Get scheduler status"""
        if hasattr(self, 'scheduler'):
            return self.scheduler.get_scheduler_status()
        return {'error': 'Scheduler not initialized'}
    
    def force_scheduler_check(self) -> Dict:
        """Force a scheduler check"""
        if hasattr(self, 'scheduler'):
            return self.scheduler.force_check()
        return {'error': 'Scheduler not initialized'}
    
    def _restore_active_positions_from_db(self):
        """Restore active positions from database after restart"""
        try:
            db_positions = self.db.get_active_positions()
            if not db_positions:
                logger.info("No active positions found in database")
                return
            
            logger.info(f"Found {len(db_positions)} active positions in database, restoring to position manager...")
            
            for db_pos in db_positions:
                try:
                    ticker = db_pos.get('ticker')
                    if not ticker:
                        continue
                    
                    # Skip if already in position manager
                    position_summary = self.position_manager.get_position_summary()
                    if ticker in position_summary['active_positions']:
                        logger.debug(f"[{ticker}] Position already in memory, skipping restore")
                        continue
                    
                    # Parse entry_time
                    entry_time_str = db_pos.get('entry_time')
                    if isinstance(entry_time_str, str):
                        try:
                            entry_time = pd.to_datetime(entry_time_str)
                            if entry_time.tzinfo is None:
                                entry_time = self.et_timezone.localize(entry_time)
                        except:
                            entry_time = datetime.now(self.et_timezone)
                    else:
                        entry_time = datetime.now(self.et_timezone)
                    
                    # Get position details
                    entry_price = db_pos.get('entry_price', 0)
                    shares = db_pos.get('shares', 0)
                    entry_pattern = db_pos.get('entry_pattern', 'Unknown')
                    confidence = db_pos.get('confidence', 0.0)
                    
                    # Get current price
                    try:
                        current_price = self.data_api.get_current_price(ticker)
                        if current_price is None or current_price <= 0:
                            # Try to get from 1-minute data
                            df = self.data_api.get_1min_data(ticker, minutes=1)
                            if df is not None and len(df) > 0:
                                current_price = df.iloc[-1]['close']
                            else:
                                current_price = entry_price
                    except:
                        current_price = entry_price
                    
                    # Determine position type from pattern
                    position_type = PositionType.SWING  # Default
                    if 'SURGE' in entry_pattern:
                        position_type = PositionType.SURGE
                    elif 'SLOW' in entry_pattern or 'ACCUMULATION' in entry_pattern:
                        position_type = PositionType.SLOW_MOVER
                    elif 'BREAKOUT' in entry_pattern:
                        position_type = PositionType.BREAKOUT
                    
                    # Restore position in position manager
                    success = self.position_manager.enter_position(
                        ticker=ticker,
                        entry_price=entry_price,
                        shares=shares,
                        position_type=position_type,
                        entry_pattern=entry_pattern,
                        entry_confidence=confidence,
                        multi_timeframe_confidence=confidence
                    )
                    
                    if success:
                        logger.info(f"[{ticker}] Position restored from database: {shares:.2f} shares @ ${entry_price:.4f}")
                        # Update current price in restored position
                        if hasattr(self.position_manager, 'active_positions') and ticker in self.position_manager.active_positions:
                            self.position_manager.active_positions[ticker].current_price = current_price
                    else:
                        logger.warning(f"[{ticker}] Failed to restore position from database")
                        
                except Exception as e:
                    logger.error(f"Error restoring position {db_pos.get('ticker', 'UNKNOWN')}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
            
            logger.info(f"Position restoration complete")
            
        except Exception as e:
            logger.error(f"Error restoring active positions from database: {e}")
            import traceback
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Bot is typically started via web dashboard or scheduler
    pass
