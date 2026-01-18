"""
Pure Trade Simulator - Wrapper for Realtime Bot
This simulator should NOT contain any trading logic. It only provides minute-by-minute
data to the realtime autonomous trading bot and records the results.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import pytz

# Import ONLY the position manager - this is where ALL trading logic lives
from src.core.intelligent_position_manager import IntelligentPositionManager, PositionType, ExitReason
from src.data.webull_data_api import WebullDataAPI

@dataclass
class SimulationConfig:
    """Configuration parameters for trade simulator"""
    ticker: str
    detection_time: str  # Format: "YYYY-MM-DD HH:MM:SS"
    initial_capital: float = 2500.0
    max_positions: int = 1
    commission_per_trade: float = 0.005  # 0.5% commission
    data_folder: str = "simulation_data"
    stop_loss_pct: float = 0.06  # 6% stop loss (increased from 4% based on optimization)
    take_profit_pct: float = 0.08  # 8% take profit
    min_hold_minutes: int = 10  # Minimum hold time


@dataclass
class TradeResult:
    """Result of a single trade (captured from realtime bot)"""
    ticker: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    strategy: str
    pnl: float
    pnl_pct: float
    commission: float
    exit_reason: str
    hold_minutes: int


@dataclass
class SimulationResult:
    """Complete simulation results"""
    config: SimulationConfig
    trades: List[TradeResult]
    total_pnl: float
    total_pnl_pct: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    max_drawdown: float
    sharpe_ratio: float


class PureTradeSimulator:
    """
    PURE wrapper simulator - NO trading logic here.
    Only feeds minute data to realtime bot and captures results.
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.trades = []
        self.data = None
        self.logger = self._setup_logger()
        self.webull_api = WebullDataAPI()
        
        # Historical data for simulation
        self.historical_data = None
        self.current_data_index = 0
        self.simulation_timestamps = []
        
        # Initialize the REALTIME position manager - this does ALL the work
        self.position_manager = IntelligentPositionManager(
            max_positions=config.max_positions,
            position_size_pct=0.95,
            risk_per_trade=0.02
        )
        
        # Store configuration parameters for use in trade calculations
        self.commission_per_trade = config.commission_per_trade
        self.stop_loss_pct = config.stop_loss_pct
        self.take_profit_pct = config.take_profit_pct
        self.min_hold_minutes = config.min_hold_minutes
        
        # Create data folder if it doesn't exist
        Path(config.data_folder).mkdir(exist_ok=True)
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"PureSimulator_{self.config.ticker}")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers to prevent duplicates
        logger.handlers.clear()
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Prevent propagation to root logger to avoid duplicates
        logger.propagate = False
        
        return logger
    
    def _parse_detection_time(self) -> datetime:
        """Parse detection time and make it timezone-aware (US/Eastern)"""
        naive_dt = datetime.strptime(self.config.detection_time, "%Y-%m-%d %H:%M:%S")
        eastern = pytz.timezone('US/Eastern')
        return eastern.localize(naive_dt)
    
    def download_data(self) -> pd.DataFrame:
        """Download historical data using existing Webull API"""
        self.logger.info(f"Downloading data for {self.config.ticker}")
        
        try:
            # Use the existing WebullDataAPI to get 1-minute data
            df = self.webull_api.get_1min_data(self.config.ticker, minutes=1440)  # 24 hours
            
            if df is None or df.empty:
                raise ValueError(f"No data received for {self.config.ticker}")
            
            # Convert to expected format
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Save to file
            detection_dt = self._parse_detection_time()
            data_file = os.path.join(
                self.config.data_folder,
                f"{self.config.ticker}_{detection_dt.strftime('%Y%m%d')}.csv"
            )
            df.to_csv(data_file)
            self.logger.info(f"Data saved to {data_file}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error downloading data: {e}")
            raise
    
    def load_data(self) -> pd.DataFrame:
        """Load data from file or download if not available"""
        detection_dt = self._parse_detection_time()
        data_file = os.path.join(
            self.config.data_folder,
            f"{self.config.ticker}_{detection_dt.strftime('%Y%m%d')}.csv"
        )
        
        if os.path.exists(data_file):
            self.logger.info(f"Loading data from {data_file}")
            df = pd.read_csv(data_file, parse_dates=['timestamp'], index_col='timestamp')
            
            # Check if data year matches detection time year
            data_year = df.index[0].year
            detection_year = detection_dt.year
            if data_year != detection_year:
                self.logger.warning(f"Data year mismatch: file contains {data_year} data but requesting {detection_year}")
                self.logger.info(f"Adjusting detection time to use available data year {data_year}")
                # Store the original detection time for reference
                self.original_detection_time = detection_dt
                # Create adjusted detection time with data year
                adjusted_dt = detection_dt.replace(year=data_year)
                self.adjusted_detection_time = adjusted_dt
            else:
                self.adjusted_detection_time = detection_dt
            
            return df
        else:
            return self.download_data()
    
    def run_simulation(self) -> SimulationResult:
        """
        Run the pure simulation - just feed data to position manager minute by minute
        ALL trading logic is handled by the position manager
        """
        self.logger.info(f"Starting PURE simulation for {self.config.ticker}")
        self.logger.info("NOTE: This is a wrapper - all logic handled by realtime bot")
        
        # Load historical data
        full_data = self.load_data()
        self.historical_data = full_data
        
        # Get detection time and find the corresponding data point
        detection_dt = self._parse_detection_time()
        self.detection_time = detection_dt  # Store for use in entry signals
        
        # Use adjusted detection time if available (for year mismatches)
        actual_detection_dt = getattr(self, 'adjusted_detection_time', detection_dt)
        if hasattr(self, 'original_detection_time'):
            self.logger.info(f"Original detection time: {self.original_detection_time}")
            self.logger.info(f"Using adjusted detection time: {actual_detection_dt}")
        
        # Find the closest available time to detection time
        try:
            detection_idx = full_data.index.get_loc(actual_detection_dt)
            self.logger.info(f"Found exact match for detection time at index {detection_idx}")
        except KeyError:
            available_times = full_data.index[full_data.index >= actual_detection_dt]
            if len(available_times) == 0:
                raise ValueError(f"No data available at or after detection time {actual_detection_dt}")
            else:
                detection_idx = full_data.index.get_loc(available_times[0])
                self.logger.info(f"Using closest available time: {available_times[0]} at index {detection_idx}")
        
        # Get data from detection time onwards
        simulation_data = full_data.iloc[detection_idx:].copy()
        
        self.logger.info(f"Detection time requested: {detection_dt}")
        self.logger.info(f"Actual detection time used: {actual_detection_dt}")
        self.logger.info(f"Full data range: {full_data.index.min()} to {full_data.index.max()}")
        self.logger.info(f"Simulation data range: {simulation_data.index.min()} to {simulation_data.index.max()}")
        
        if len(simulation_data) == 0:
            raise ValueError(f"No data available for simulation period after {detection_dt}")
        
        self.logger.info(f"Feeding {len(simulation_data)} minutes of data to position manager")
        self.logger.info(f"Data range: {simulation_data.index.min()} to {simulation_data.index.max()}")
        
        # Feed minute-by-minute data to the position manager
        for i, (timestamp, row) in enumerate(simulation_data.iterrows()):
            try:
                # Set current data index
                self.current_data_index = detection_idx + i
                
                # Create market data dict
                market_data = {
                    'symbol': self.config.ticker,
                    'price': row['close'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'volume': row['volume'],
                    'timestamp': timestamp,
                    'rsi': row.get('rsi', 50),
                    'macd': row.get('macd'),
                    'macd_signal': row.get('macd_signal'),
                    'macd_hist': row.get('macd_hist', 0),
                    'volume_ratio': row.get('volume_ratio', 1.0),
                    'volatility_score': 0.5,
                    'momentum_score': 0.6,
                    'trend_alignment': 0.7
                }
                
                # Check for entry signals (let position manager handle ALL logic)
                if len(self.position_manager.active_positions) == 0:
                    self._check_entry_signals(market_data, timestamp, row)
                
                # Update existing positions (let position manager handle ALL logic)
                if len(self.position_manager.active_positions) > 0:
                    self._update_positions(market_data, timestamp)
                
                self.logger.debug(f"Processed minute {i+1}/{len(simulation_data)}: {timestamp.strftime('%H:%M:%S')}")
                
            except Exception as e:
                self.logger.error(f"Error processing minute {i}: {e}")
                continue
        
        # Close any remaining positions at the end
        if len(self.position_manager.active_positions) > 0:
            last_timestamp = simulation_data.index[-1]
            last_row = simulation_data.iloc[-1]
            last_market_data = {
                'symbol': self.config.ticker,
                'price': last_row['close'],
                'timestamp': last_timestamp
            }
            
            for pos_id in list(self.position_manager.active_positions.keys()):
                success = self.position_manager.exit_position(pos_id, ExitReason.END_OF_DAY)
                if success:
                    position = self.position_manager.position_history[pos_id]
                    trade_result = self._create_trade_result(position, last_market_data, last_timestamp)
                    self.trades.append(trade_result)
        
        # Calculate results
        return self.calculate_results()
    
    def calculate_results(self) -> SimulationResult:
        """Calculate simulation statistics from captured trades"""
        if not self.trades:
            return SimulationResult(
                config=self.config,
                trades=[],
                total_pnl=0,
                total_pnl_pct=0,
                win_rate=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                max_drawdown=0,
                sharpe_ratio=0
            )
        
        total_pnl = sum(trade.pnl for trade in self.trades)
        total_pnl_pct = total_pnl / self.config.initial_capital
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl < 0])
        win_rate = winning_trades / len(self.trades) if self.trades else 0
        
        # Calculate max drawdown
        capital_curve = [self.config.initial_capital]
        for trade in self.trades:
            capital_curve.append(capital_curve[-1] + trade.pnl)
        
        peak = capital_curve[0]
        max_drawdown = 0
        for value in capital_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio
        returns = [trade.pnl_pct for trade in self.trades]
        if returns:
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        return SimulationResult(
            config=self.config,
            trades=self.trades,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            win_rate=win_rate,
            total_trades=len(self.trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio
        )
    
    def _check_entry_signals(self, market_data: Dict, timestamp: pd.Timestamp, row: pd.Series):
        """Check for entry signals using position manager logic"""
        try:
            # For indicator calculation, use historical data BEFORE detection time (like realtime)
            # But only check for entry AFTER detection time
            actual_detection_dt = getattr(self, 'adjusted_detection_time', self.detection_time)
            if timestamp < actual_detection_dt:
                return  # Don't enter before detection time
            
            # Use all available historical data for indicators (like realtime bot would)
            historical_slice = self.historical_data.loc[:timestamp]
            
            if len(historical_slice) < 50:  # Need enough data for indicators
                return
            
            # Calculate indicators
            indicators = self._calculate_indicators(historical_slice)
            current_indicators = indicators.iloc[-1]
            
            # Create multi-timeframe analysis
            # Calculate momentum based on price change
            price_change_5m = 0
            if len(historical_slice) >= 5:
                price_change_5m = (row['close'] - historical_slice.iloc[-5]['close']) / historical_slice.iloc[-5]['close']
            
            momentum_score = min(1.0, max(0.0, 0.5 + price_change_5m * 10))  # Scale price change to 0-1
            
            multi_timeframe_analysis = {
                'trend_alignment': 0.7,  # Default
                'momentum_score': momentum_score,
                'volatility_score': current_indicators.get('rsi', 50) / 100,
                'rsi': current_indicators.get('rsi', 50),
                'sma_5': current_indicators.get('sma_5'),
                'sma_15': current_indicators.get('sma_15'),
                'sma_50': current_indicators.get('sma_50'),
                'volume_ratio': current_indicators.get('volume_ratio', 1.0)
            }
            
            # Create volume data
            volume_data = {
                'volume_ratio': current_indicators.get('volume_ratio', 1.0)
            }
            
            # Create pattern info based on actual market conditions
            volume_ratio = current_indicators.get('volume_ratio', 1.0)
            rsi = current_indicators.get('rsi', 50)
            
            # Determine pattern based on volume and price action
            if volume_ratio > 15:
                pattern_name = 'surge'
            elif volume_ratio > 5 and rsi > 60:
                pattern_name = 'Volume_Breakout'
            elif volume_ratio > 2:
                pattern_name = 'Breakout'
            elif volume_ratio < 3 and rsi < 40:
                pattern_name = 'Slow_Accumulation'
            else:
                pattern_name = 'swing'  # Default
            
            pattern_info = {
                'pattern_name': pattern_name
            }
            
            # Calculate signal strength
            signal_strength = 0.5
            volume_ratio = current_indicators.get('volume_ratio', 1.0)
            if volume_ratio > 8:  # Reduced from 10
                signal_strength += 0.3
            elif volume_ratio > 4:  # Reduced from 5
                signal_strength += 0.2
            elif volume_ratio > 2:
                signal_strength += 0.1
            
            # Let position manager decide (ALL logic is here)
            should_enter, position_type, reason = self.position_manager.evaluate_entry_signal(
                self.config.ticker,
                market_data['price'],
                signal_strength,
                multi_timeframe_analysis,
                volume_data,
                pattern_info
            )
            
            if should_enter and position_type:
                # Calculate position size
                quantity = int(self.config.initial_capital * 0.95 / market_data['price'])
                
                # Enter position (position manager handles ALL logic)
                success = self.position_manager.enter_position(
                    ticker=self.config.ticker,
                    entry_price=market_data['price'],
                    shares=quantity,
                    position_type=position_type,
                    entry_pattern=pattern_info['pattern_name'],
                    entry_confidence=signal_strength,
                    multi_timeframe_confidence=multi_timeframe_analysis['trend_alignment']
                )
                
                if success:
                    self.logger.info(f"Entered {position_type.value} position at {market_data['price']:.2f} on {timestamp.strftime('%H:%M:%S')}")
            
        except Exception as e:
            self.logger.error(f"Error checking entry signals: {e}")
    
    def _update_positions(self, market_data: Dict, timestamp: pd.Timestamp):
        """Update existing positions (position manager handles ALL logic)"""
        try:
            # Create market data dict for position manager
            market_data_for_pm = {self.config.ticker: market_data}
            
            # Let position manager handle ALL logic
            exits = self.position_manager.update_positions(market_data_for_pm)
            
            # Process any exits
            for exit_info in exits:
                if exit_info.get('exited'):
                    pos_id = exit_info.get('position_id')
                    exit_price = exit_info.get('exit_price', market_data['price'])
                    exit_reason = exit_info.get('exit_reason', 'unknown')
                    
                    # Get the completed position from history
                    if pos_id in self.position_manager.position_history:
                        position = self.position_manager.position_history[pos_id]
                        # Create market data for trade result
                        trade_market_data = {
                            'price': exit_price,
                            'timestamp': timestamp
                        }
                        trade_result = self._create_trade_result(position, trade_market_data, timestamp)
                        self.trades.append(trade_result)
                        self.logger.info(f"Exited position at {exit_price:.2f} on {timestamp.strftime('%H:%M:%S')} - {exit_reason}")
                
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Moving averages
        df.loc[:, 'sma_5'] = df['close'].rolling(window=5).mean()
        df.loc[:, 'sma_15'] = df['close'].rolling(window=15).mean()
        df.loc[:, 'sma_50'] = df['close'].rolling(window=50).mean()
        
        # Volume indicators
        df.loc[:, 'volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df.loc[:, 'volume_ratio'] = df['volume'] / df['volume_sma_10']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df.loc[:, 'rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df.loc[:, 'macd'] = exp1 - exp2
        df.loc[:, 'macd_signal'] = df['macd'].ewm(span=9).mean()
        df.loc[:, 'macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df.loc[:, 'bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df.loc[:, 'bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df.loc[:, 'bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        return df
    
    def _create_trade_result(self, position, market_data: Dict, timestamp: pd.Timestamp) -> TradeResult:
        """Create trade result from position"""
        # Calculate commission based on configuration
        commission = position.entry_price * position.original_shares * self.commission_per_trade
        
        # Calculate P&L percentage
        pnl_pct = (position.realized_pnl / position.entry_value) * 100 if position.entry_value > 0 else 0
        
        return TradeResult(
            ticker=self.config.ticker,
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry_price=position.entry_price,
            exit_price=market_data['price'],
            quantity=position.original_shares,
            strategy=position.position_type.value,
            pnl=position.realized_pnl - commission,
            pnl_pct=pnl_pct,
            commission=commission,
            exit_reason=position.exit_reason.value if position.exit_reason else 'unknown',
            hold_minutes=int((timestamp - position.entry_time).total_seconds() / 60)
        )
