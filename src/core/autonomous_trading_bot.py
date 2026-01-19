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
from .intelligent_position_manager import IntelligentPositionManager
from ..risk.volatility_manager import VolatilityManager
from .trading_bot_scheduler import TradingBotScheduler
from ..data.webull_data_api import WebullDataAPI
from ..config.settings import settings

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
        self.paused = False
        
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
            
            # Initialize scheduler
            self.scheduler = TradingBotScheduler(self, settings.trading_window)
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def start(self):
        """Start autonomous trading bot"""
        try:
            if self.running:
                logger.warning("Bot is already running")
                return
            
            self.running = True
            self.paused = False
            
            logger.info("Starting Autonomous Trading Bot...")
            
            # Start scheduler first
            self.scheduler.start_scheduler()
            
            # Run trading loop in separate thread
            trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
            trading_thread.start()
            
        except Exception as e:
            logger.error(f"Error starting trading loop: {e}")
            raise
    
    def stop(self):
        """Stop autonomous trading bot"""
        try:
            self.running = False
            self.paused = False
            
            # Stop scheduler
            if hasattr(self, 'scheduler'):
                self.scheduler.stop_scheduler()
            
            logger.info("Autonomous Trading Bot stopped")
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
    
    def pause_trading(self):
        """Pause trading activities"""
        try:
            self.paused = True
            logger.info("Trading paused")
        except Exception as e:
            logger.error(f"Error pausing trading: {e}")
    
    def resume_trading(self):
        """Resume trading activities"""
        try:
            self.paused = False
            logger.info("Trading resumed")
        except Exception as e:
            logger.error(f"Error resuming trading: {e}")
    
    def _trading_loop(self):
        """Main trading loop"""
        while self.running:
            try:
                # Check if bot is paused
                if self.paused:
                    time.sleep(10)
                    continue
                
                current_time = datetime.now(self.et_timezone)
                
                # Reset daily tracking if new day
                self._check_daily_reset(current_time)
                
                # Check market conditions
                market_condition = self.volatility_manager.check_market_conditions({})
                
                # Check if trading should be paused due to volatility
                should_trade, reason = self.volatility_manager.should_trade()
                if not should_trade:
                    logger.info(f"Trading paused: {reason}")
                    time.sleep(60)  # Wait 1 minute before retrying
                    continue
                
                # Step 1: Scan for opportunities (simplified - no filtering)
                logger.info("Starting scanner to fetch gainers...")
                top_gainers = self.scanner.fetch_and_analyze_gainers(self.config['scanner_max_tickers'])
                logger.info(f"Scanner returned {len(top_gainers)} gainers")
                
                if not top_gainers:
                    logger.warning("No gainers found - check scanner logs for details")
                    time.sleep(self.config['scanner_update_interval'])
                    continue
                
                # Display all found gainers in dashboard (no filtering)
                logger.info(f"Found {len(top_gainers)} gainers to monitor")
                for gainer in top_gainers[:5]:  # Show first 5 in logs
                    logger.info(f"  Monitoring: {gainer.symbol} - {gainer.change_pct:.2f}% (${gainer.price:.2f})")
                
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
    
    def _analyze_and_process_gainer(self, gainer, current_time):
        """Analyze and potentially process a gainer"""
        ticker = gainer.symbol
        
        # Debug logging to check what we're actually passing
        logger.info(f"Processing gainer: {gainer}")
        logger.info(f"Extracted ticker: {ticker} (type: {type(ticker)})")
        
        try:
            # Get current market data
            current_data = self._get_current_market_data(ticker)
            if not current_data:
                return
            
            # Multi-timeframe analysis
            mt_analysis = self.multi_timeframe_analyzer.analyze_multi_timeframe(ticker)
            if not mt_analysis:
                return
            
            # Manipulation detection
            historical_data = self.data_api.get_1min_data(ticker, minutes=200)
            manipulation_signals = self.manipulation_detector.analyze_for_manipulation(
                ticker, current_data, historical_data, {}
            )
            
            # Check for manipulation
            high_risk_manipulation = [s for s in manipulation_signals if s.severity in ['high', 'critical']]
            if high_risk_manipulation:
                logger.warning(f"Skipping {ticker} due to manipulation detection")
                self._log_rejected_trade(ticker, "manipulation_detected", 
                                       f"Manipulation signals: {[s.manipulation_type.value for s in high_risk_manipulation]}")
                return
            
            # Prepare signal data for evaluation
            signal_data = {
                'entry_price': current_data['price'],
                'signal_strength': mt_analysis.confidence,
                'pattern_name': 'multi_timeframe_signal',
                'volume_ratio': current_data.get('volume_ratio', 1),
                'volatility_score': current_data.get('volatility_score', 0),
                'multi_timeframe_confidence': mt_analysis.confidence,
                'market_condition': self.volatility_manager.current_condition.value
            }
            
            # Evaluate entry
            should_enter, position_type, reason = self.position_manager.evaluate_entry_signal(
                ticker, current_data['price'], mt_analysis.confidence, 
                asdict(mt_analysis), current_data, signal_data
            )
            
            if should_enter:
                # Calculate position size
                position_value = self.config['initial_capital'] * self.config['position_size_pct']
                shares = position_value / current_data['price']
                
                # Enter position
                success = self.position_manager.enter_position(
                    ticker=ticker,
                    entry_price=current_data['price'],
                    shares=shares,
                    position_type=position_type,
                    entry_pattern='multi_timeframe_signal',
                    entry_confidence=mt_analysis.confidence,
                    multi_timeframe_confidence=mt_analysis.confidence
                )
                
                if success:
                    logger.info(f"Entered position: {ticker} - {shares:.2f} shares @ ${current_data['price']:.4f}")
                else:
                    logger.warning(f"Failed to enter position for {ticker}")
                    self._log_rejected_trade(ticker, "insufficient_confidence", reason)
            
        except Exception as e:
            logger.error(f"Error analyzing gainer {ticker}: {e}")
    
    def _update_positions(self):
        """Update all active positions"""
        try:
            # Get current market data for all positions
            position_summary = self.position_manager.get_position_summary()
            market_data = {}
            
            for ticker in position_summary['active_positions'].keys():
                data = self._get_current_market_data(ticker)
                if data:
                    market_data[ticker] = data
            
            # Update positions
            exits = self.position_manager.update_positions(market_data)
            
            # Process exits
            for exit_decision in exits:
                self._process_position_exit(exit_decision)
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def _process_position_exit(self, exit_decision):
        """Process a position exit"""
        try:
            ticker = exit_decision['ticker']
            exit_price = exit_decision['price']
            exit_reason = exit_decision['reason']
            
            # Calculate P&L
            position_summary = self.position_manager.get_position_summary()
            if ticker in position_summary['active_positions']:
                position_info = position_summary['active_positions'][ticker]
                pnl = position_info['unrealized_pnl_dollars']
                pnl_pct = position_info['unrealized_pnl_pct']
                
                # Update performance metrics
                self.performance_metrics['total_trades'] += 1
                if pnl > 0:
                    self.performance_metrics['winning_trades'] += 1
                    self.performance_metrics['total_pnl'] += pnl
                else:
                    self.performance_metrics['losing_trades'] += 1
                
                logger.info(f"Exited position: {ticker} - P&L: ${pnl:.2f} ({pnl_pct:.2f}%) - {exit_reason}")
            
        except Exception as e:
            logger.error(f"Error processing position exit: {e}")
    
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
            logger.error(f"Error getting market data for {ticker}: {e}")
            return None
    
    def _log_rejected_trade(self, ticker: str, reason: str, details: str):
        """Log a rejected trade"""
        try:
            # This is simplified - would need more detailed signal data
            signal_data = {
                'entry_price': 0,
                'signal_strength': 0.5,
                'pattern_name': 'unknown',
                'volume_ratio': 1.0,
                'volatility_score': 0.5,
                'multi_timeframe_confidence': 0.5,
                'market_condition': 'normal'
            }
            
            thresholds = {
                'confidence': 0.7,
                'volume': 1.5,
                'volatility': 0.7
            }
            
            # Log rejection
            logger.info(f"Rejected trade: {ticker} - {reason}")
            
        except Exception as e:
            logger.error(f"Error logging rejected trade: {e}")
    
    def _check_daily_reset(self, current_time):
        """Check if daily tracking should be reset"""
        try:
            if self.current_date != current_time.date():
                self.current_date = current_time.date()
                self.daily_start_capital = self.config['initial_capital']
                self.daily_profit = 0.0
                logger.info(f"New trading day: {self.current_date}")
                
        except Exception as e:
            logger.error(f"Error in daily reset: {e}")
    
    def get_bot_status(self) -> Dict:
        """Get comprehensive bot status"""
        try:
            position_summary = self.position_manager.get_position_summary()
            scheduler_status = self.scheduler.get_scheduler_status() if hasattr(self, 'scheduler') else {}
            
            return {
                'running': self.running,
                'paused': self.paused,
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

if __name__ == "__main__":
    main()
