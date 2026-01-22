"""
Simple Autonomous Trading Bot
Minimal version without complex dependencies
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import pytz
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)

class SimpleAutonomousBot:
    """
    Simple autonomous trading bot with basic functionality
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the simple autonomous trading bot
        
        Args:
            config: Optional configuration overrides
        """
        self.et_timezone = pytz.timezone('America/New_York')
        self.running = False
        
        # Default configuration
        base_config = {
            'initial_capital': 10000.0,
            'max_positions': 3,
            'position_size_pct': 0.33,
            'risk_per_trade': 0.02,
            'scanner_update_interval': 60,  # seconds
            'dashboard_enabled': True,
            'dashboard_port': 5000
        }
        
        if config:
            base_config.update(config)
        
        self.config = base_config
        
        # Initialize basic components
        self.active_positions = {}
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0
        }
        
        # Dashboard thread
        self.dashboard_thread = None
        
        logger.info("Simple Autonomous Bot initialized successfully")
    
    def start(self):
        """Start autonomous trading bot"""
        try:
            if self.running:
                logger.warning("Bot is already running")
                return
            
            self.running = True
            
            logger.info("Starting Simple Autonomous Trading Bot...")
            
            # Start dashboard if enabled
            if self.config.get('dashboard_enabled', False):
                self._start_dashboard()
            
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
            logger.info("Simple Autonomous Trading Bot stopped")
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
    
    
    def _start_dashboard(self):
        """Start the dashboard in a separate thread"""
        try:
            from simple_dashboard import set_bot_instance, run_dashboard
            
            # Set bot instance for dashboard
            set_bot_instance(self)
            
            # Start dashboard in separate thread
            self.dashboard_thread = threading.Thread(
                target=run_dashboard,
                args=(self.config['dashboard_port'],),
                daemon=True
            )
            self.dashboard_thread.start()
            
            logger.info(f"Dashboard started on port {self.config['dashboard_port']}")
            
        except ImportError as e:
            logger.warning(f"Could not start dashboard: {e}")
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
    
    def _trading_loop(self):
        """Main trading loop"""
        try:
            while self.running:
                # Simulate trading cycle
                self._simulate_trading_cycle()
                
                # Sleep until next cycle
                time.sleep(self.config['scanner_update_interval'])
                
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    def _simulate_trading_cycle(self):
        """Simulate a trading cycle"""
        try:
            current_time = datetime.now(self.et_timezone)
            
            # Simulate scanning for opportunities
            opportunities = self._scan_for_opportunities()
            
            if not opportunities:
                logger.info("No trading opportunities found")
                return
            
            # Process each opportunity
            for opportunity in opportunities:
                try:
                    self._process_opportunity(opportunity, current_time)
                except Exception as e:
                    logger.error(f"Error processing opportunity {opportunity['ticker']}: {e}")
                    continue
            
            # Update positions
            self._update_positions()
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    def _scan_for_opportunities(self) -> List[Dict]:
        """Simulate scanning for trading opportunities"""
        # Simulate finding some tickers
        opportunities = []
        
        # Add some simulated opportunities
        tickers = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOG']
        for ticker in tickers:
            if np.random.random() > 0.7:  # 30% chance of opportunity
                opportunities.append({
                    'ticker': ticker,
                    'price': np.random.uniform(50, 500),
                    'volume_ratio': np.random.uniform(1, 10),
                    'signal_strength': np.random.uniform(0.5, 0.9)
                })
        
        return opportunities
    
    def _process_opportunity(self, opportunity: Dict, current_time: datetime):
        """Process a trading opportunity"""
        ticker = opportunity['ticker']
        
        try:
            # Check if already in position
            if ticker in self.active_positions:
                return
            
            # Check max positions
            if len(self.active_positions) >= self.config['max_positions']:
                return
            
            # Simple entry logic
            if opportunity['signal_strength'] > 0.6:
                # Calculate position size
                position_value = self.config['initial_capital'] * self.config['position_size_pct']
                shares = position_value / opportunity['price']
                
                # Enter position
                self.active_positions[ticker] = {
                    'entry_price': opportunity['price'],
                    'shares': shares,
                    'entry_time': current_time,
                    'stop_loss': opportunity['price'] * 0.95,  # 5% stop loss
                    'take_profit': opportunity['price'] * 1.08,  # 8% take profit
                    'current_price': opportunity['price']
                }
                
                logger.info(f"Entered position: {ticker} - {shares:.2f} shares @ ${opportunity['price']:.2f}")
            
        except Exception as e:
            logger.error(f"Error processing opportunity {ticker}: {e}")
    
    def _update_positions(self):
        """Update all active positions"""
        try:
            positions_to_remove = []
            
            for ticker, position in self.active_positions.items():
                # Simulate price movement
                price_change = np.random.uniform(-0.02, 0.02)  # ±2% price change
                position['current_price'] = position['current_price'] * (1 + price_change)
                
                # Check exit conditions
                if position['current_price'] <= position['stop_loss']:
                    # Stop loss hit
                    self._close_position(ticker, position['stop_loss'], 'stop_loss')
                    positions_to_remove.append(ticker)
                elif position['current_price'] >= position['take_profit']:
                    # Take profit hit
                    self._close_position(ticker, position['take_profit'], 'take_profit')
                    positions_to_remove.append(ticker)
            
            # Remove closed positions
            for ticker in positions_to_remove:
                del self.active_positions[ticker]
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def _close_position(self, ticker: str, exit_price: float, exit_reason: str):
        """Close a position"""
        try:
            if ticker not in self.active_positions:
                return
            
            position = self.active_positions[ticker]
            
            # Calculate P&L
            pnl = (exit_price - position['entry_price']) * position['shares']
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
            
            # Update performance metrics
            self.performance_metrics['total_trades'] += 1
            if pnl > 0:
                self.performance_metrics['winning_trades'] += 1
                self.performance_metrics['total_pnl'] += pnl
            else:
                self.performance_metrics['losing_trades'] += 1
                self.performance_metrics['total_pnl'] += pnl
            
            logger.info(f"Closed position: {ticker} - P&L: ${pnl:.2f} ({pnl_pct:.2f}%) - {exit_reason}")
            
        except Exception as e:
            logger.error(f"Error closing position {ticker}: {e}")
    
    def get_bot_status(self) -> Dict:
        """Get bot status"""
        try:
            return {
                'running': self.running,
                'config': self.config,
                'performance_metrics': self.performance_metrics,
                'active_positions': len(self.active_positions),
                'last_update': datetime.now(self.et_timezone).isoformat()
            }
        except Exception as e:
            return {'error': str(e)}

if __name__ == "__main__":
    # Test the bot
    bot = SimpleAutonomousBot()
    print("Bot initialized:", bot.get_bot_status())
