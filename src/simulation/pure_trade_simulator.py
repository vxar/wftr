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

# Import the position manager and RealtimeTrader - use same logic as realtime bot
from src.core.intelligent_position_manager import IntelligentPositionManager, PositionType, ExitReason
from src.core.realtime_trader import RealtimeTrader
from src.data.webull_data_api import WebullDataAPI
from src.utils.trade_processing import process_exit_to_trade_data, process_entry_signal

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
        
        # Initialize RealtimeTrader - use same entry logic as realtime bot
        self.realtime_trader = RealtimeTrader(
            min_confidence=0.72,
            min_entry_price_increase=5.5,
            trailing_stop_pct=2.5,
            profit_target_pct=8.0,
            data_api=self.webull_api,
            rejection_callback=None  # No callback needed for simulator
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
            
            # Ensure timestamps are timezone-aware (US/Eastern) if they're not already
            if df.index.tz is None:
                # Assume loaded timestamps are in US/Eastern
                eastern = pytz.timezone('US/Eastern')
                df.index = df.index.tz_localize(eastern)
            elif str(df.index.tz) != 'US/Eastern':
                # Convert to US/Eastern if different timezone
                df.index = df.index.tz_convert('US/Eastern')
            
            # Don't adjust detection time - use it as provided
            # We'll find the closest match in the data during simulation
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
        
        # Ensure timestamps are timezone-aware (US/Eastern)
        if full_data.index.tz is None:
            # If no timezone, assume US/Eastern (like realtime bot)
            eastern = pytz.timezone('US/Eastern')
            full_data.index = full_data.index.tz_localize(eastern)
        elif str(full_data.index.tz) != 'US/Eastern':
            # Convert to US/Eastern if different timezone
            full_data.index = full_data.index.tz_convert('US/Eastern')
        
        self.historical_data = full_data
        
        # Get detection time (already in ET from _parse_detection_time)
        detection_dt = self._parse_detection_time()
        self.detection_time = detection_dt  # Store for use in entry signals
        
        # Ensure detection time is in same timezone as data (should already be ET)
        data_tz = full_data.index.tz
        if data_tz is not None:
            if detection_dt.tzinfo is None:
                detection_dt = data_tz.localize(detection_dt.replace(tzinfo=None))
            elif detection_dt.tzinfo != data_tz:
                detection_dt = detection_dt.astimezone(data_tz)
            # Update stored detection time
            self.detection_time = detection_dt
        
        # Find the closest available time to detection time in the data
        # Use the detection time as provided, find closest match
        try:
            detection_idx = full_data.index.get_loc(detection_dt)
            actual_detection_dt = detection_dt  # Exact match found
            self.logger.info(f"Found exact match for detection time at index {detection_idx}")
        except KeyError:
            # Find closest available time at or after detection time
            available_times = full_data.index[full_data.index >= detection_dt]
            if len(available_times) == 0:
                # Try to find closest time before detection time if no data after
                available_times = full_data.index[full_data.index <= detection_dt]
                if len(available_times) == 0:
                    raise ValueError(f"No data available near detection time {detection_dt}")
                else:
                    detection_idx = full_data.index.get_loc(available_times[-1])
                    actual_detection_dt = available_times[-1]
                    self.logger.warning(f"No data at or after detection time, using closest before: {actual_detection_dt} at index {detection_idx}")
            else:
                detection_idx = full_data.index.get_loc(available_times[0])
                actual_detection_dt = available_times[0]
                self.logger.info(f"Using closest available time after detection: {actual_detection_dt} at index {detection_idx} (requested: {detection_dt})")
        
        # Store actual detection time used (for entry signal checks)
        self.actual_detection_dt = actual_detection_dt
        
        # Split data: historical (before detection) and simulation (from detection onwards)
        # Historical data is used for analysis (like realtime bot would have accumulated)
        historical_data_before_detection = full_data.iloc[:detection_idx].copy()
        simulation_data = full_data.iloc[detection_idx:].copy()
        
        self.logger.info(f"Detection time requested: {detection_dt}")
        self.logger.info(f"Actual detection time used (closest match in data): {actual_detection_dt}")
        self.logger.info(f"Full data range: {full_data.index.min()} to {full_data.index.max()}")
        self.logger.info(f"Historical data (before detection): {len(historical_data_before_detection)} bars, range: {historical_data_before_detection.index.min() if len(historical_data_before_detection) > 0 else 'N/A'} to {historical_data_before_detection.index.max() if len(historical_data_before_detection) > 0 else 'N/A'}")
        self.logger.info(f"Simulation data (from detection): {len(simulation_data)} bars, range: {simulation_data.index.min()} to {simulation_data.index.max()}")
        
        if len(simulation_data) == 0:
            raise ValueError(f"No data available for simulation period after {detection_dt}")
        
        # Store historical data for use in entry signal analysis
        self.historical_data_before_detection = historical_data_before_detection
        
        self.logger.info(f"Starting simulation from detection time - will process {len(simulation_data)} minutes minute-by-minute")
        
        # Feed minute-by-minute data to the position manager (starting from detection time)
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
                    'volume_ratio': self._calculate_volume_ratio(row, simulation_data, i),  # Add volume ratio
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
            
            for pos_id in list(self.position_manager.active_positions.keys()):
                success = self.position_manager.exit_position(pos_id, ExitReason.END_OF_DAY)
                if success:
                    # Create exit_info for end-of-day exit
                    exit_info = {
                        'position_id': pos_id,
                        'exit_price': last_row['close'],
                        'exit_reason': 'end_of_day',
                        'exited': True
                    }
                    
                    # Use shared trade processing function
                    trade_data = process_exit_to_trade_data(
                        exit_info,
                        self.position_manager,
                        last_timestamp
                    )
                    
                    if trade_data:
                        trade_result = self._create_trade_result_from_data(trade_data)
                        self.trades.append(trade_result)

        # Calculate results
        return self.calculate_results()

    def _calculate_volume_ratio(self, current_row: pd.Series, df: pd.DataFrame, current_idx: int) -> float:
        """Calculate volume ratio for current row"""
        try:
            # Get baseline volume from previous bars
            lookback = min(20, current_idx)  # Use up to 20 bars for baseline
            if current_idx >= lookback:
                baseline_df = df.iloc[current_idx-lookback:current_idx]
                baseline_volume = baseline_df['volume'].mean()
                current_volume = current_row.get('volume', 0)
                return current_volume / baseline_volume if baseline_volume > 0 else 1.0
        except Exception:
            return 1.0

    def _calculate_sma(self, data: pd.DataFrame, current_idx: int, period: int) -> float:
        """Calculate SMA for given period"""
        if current_idx >= period:
            return data.iloc[current_idx-period:current_idx]['close'].mean()
        return data['close'].iloc[current_idx]  # Fallback to current price

    def _calculate_multi_timeframe_analysis(self, row: pd.Series, df: pd.DataFrame, current_idx: int) -> Dict:
        """Calculate multi-timeframe analysis for entry evaluation"""
        try:
            # Calculate SMAs for trend confirmation
            sma_5 = self._calculate_sma(df, current_idx, 5)
            sma_15 = self._calculate_sma(df, current_idx, 15)
            sma_50 = self._calculate_sma(df, current_idx, 50)
            
            # Calculate momentum score (price change from previous)
            momentum_score = 0.0
            if current_idx > 0:
                prev_price = df.iloc[current_idx-1]['close']
                current_price = row['close']
                momentum_score = ((current_price - prev_price) / prev_price) * 100 if prev_price > 0 else 0.0
            
            return {
                'momentum_score': momentum_score,
                'trend_alignment': 0.7 if (row['close'] > sma_5 and sma_5 > sma_15 and sma_15 > sma_50) else 0.5,
                'sma_5': sma_5,
                'sma_15': sma_15,
                'sma_50': sma_50,
                'rsi': row.get('rsi', 50),
                'volatility_score': 0.3
            }
        except Exception:
            return {
                'momentum_score': 0.0,
                'trend_alignment': 0.5,
                'sma_5': row['close'],
                'sma_15': row['close'],
                'sma_50': row['close'],
                'rsi': 50,
                'volatility_score': 0.3
            }

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
        """Check for entry signals using RealtimeTrader (same logic as realtime bot)"""
        try:
            # For indicator calculation, use historical data BEFORE detection time (like realtime)
            # But only check for entry AFTER detection time
            actual_detection_dt = getattr(self, 'actual_detection_dt', self.detection_time)
            if timestamp < actual_detection_dt:
                return  # Don't enter before detection time
            
            # Build the data for analysis: historical data BEFORE detection + data up to current timestamp
            # This mimics how realtime bot would have historical data accumulated before detection time
            historical_before = getattr(self, 'historical_data_before_detection', pd.DataFrame())
            
            # Ensure timezone consistency before slicing
            # Both timestamps must be in the same timezone as the historical_data index
            historical_tz = self.historical_data.index.tz
            
            # Normalize actual_detection_dt to match historical_data timezone
            if historical_tz is not None:
                if actual_detection_dt.tzinfo is None:
                    actual_detection_dt = historical_tz.localize(actual_detection_dt.replace(tzinfo=None))
                else:
                    # Convert to historical timezone
                    actual_detection_dt = actual_detection_dt.astimezone(historical_tz)
            elif actual_detection_dt.tzinfo is not None:
                # Historical data has no timezone, but detection time does - remove timezone
                actual_detection_dt = actual_detection_dt.replace(tzinfo=None)
            
            # Normalize timestamp to match historical_data timezone
            # Timestamp comes from simulation_data index, which should already match, but normalize to be safe
            if historical_tz is not None:
                if timestamp.tzinfo is None:
                    timestamp = historical_tz.localize(timestamp.replace(tzinfo=None))
                else:
                    # Convert to historical timezone
                    timestamp = timestamp.astimezone(historical_tz)
            elif timestamp.tzinfo is not None:
                # Historical data has no timezone, but timestamp does - remove timezone
                timestamp = timestamp.replace(tzinfo=None)
            
            # Get data from detection time up to current timestamp (inclusive)
            # Use try/except to handle any remaining timezone issues gracefully
            try:
                data_from_detection = self.historical_data.loc[actual_detection_dt:timestamp].copy()
            except ValueError as e:
                if "UTC offset" in str(e):
                    # Last resort: convert both to naive timestamps if timezone mismatch persists
                    self.logger.warning(f"Timezone mismatch in slice, converting to naive: {e}")
                    actual_detection_dt_naive = actual_detection_dt.replace(tzinfo=None) if actual_detection_dt.tzinfo else actual_detection_dt
                    timestamp_naive = timestamp.replace(tzinfo=None) if timestamp.tzinfo else timestamp
                    historical_data_naive = self.historical_data.copy()
                    historical_data_naive.index = historical_data_naive.index.tz_localize(None) if historical_data_naive.index.tz else historical_data_naive.index
                    data_from_detection = historical_data_naive.loc[actual_detection_dt_naive:timestamp_naive].copy()
                else:
                    raise
            
            # Combine: historical before detection + data from detection to current
            if len(historical_before) > 0 and len(data_from_detection) > 0:
                # Combine the two dataframes and remove any duplicates (in case detection time overlaps)
                historical_slice = pd.concat([historical_before, data_from_detection])
                # Remove duplicates based on index (timestamp)
                historical_slice = historical_slice[~historical_slice.index.duplicated(keep='first')]
                # Sort by timestamp to ensure correct order
                historical_slice = historical_slice.sort_index()
            elif len(historical_before) > 0:
                # Only historical data (shouldn't happen, but handle it)
                historical_slice = historical_before.copy()
            elif len(data_from_detection) > 0:
                # Only data from detection (no historical - should have at least some)
                historical_slice = data_from_detection.copy()
            else:
                self.logger.warning(f"No data available for analysis at {timestamp}")
                return
            
            # Need at least 4 bars for surge detection, 50 for full pattern detection
            if len(historical_slice) < 4:
                self.logger.debug(f"Insufficient data for analysis at {timestamp}: {len(historical_slice)} bars (need at least 4)")
                return
            
            # Log data composition for debugging (only at detection time to avoid spam)
            if timestamp == actual_detection_dt:
                self.logger.info(f"Data composition at detection time: {len(historical_before)} historical bars + {len(data_from_detection)} bars from detection = {len(historical_slice)} total bars")
            
            # Convert to DataFrame format expected by RealtimeTrader
            # RealtimeTrader expects: timestamp, open, high, low, close, volume
            # The historical_slice has timestamp as index, so reset it to make it a column
            df_for_analysis = historical_slice.reset_index()
            
            # Rename index column to 'timestamp' if it exists
            if df_for_analysis.index.name == 'timestamp' or 'timestamp' not in df_for_analysis.columns:
                # If timestamp is the index, it should already be in the columns after reset_index
                # But check if we need to rename it
                if df_for_analysis.index.name == 'timestamp':
                    df_for_analysis = df_for_analysis.reset_index()
            
            # Ensure we have a timestamp column
            if 'timestamp' not in df_for_analysis.columns:
                # Use the index if timestamp column doesn't exist
                df_for_analysis['timestamp'] = df_for_analysis.index
            
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df_for_analysis['timestamp']):
                df_for_analysis['timestamp'] = pd.to_datetime(df_for_analysis['timestamp'])
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df_for_analysis.columns]
            if missing_cols:
                self.logger.warning(f"Missing required columns: {missing_cols}")
                return
            
            # Use RealtimeTrader to analyze (same logic as realtime bot)
            entry_signal, exit_signals = self.realtime_trader.analyze_data(
                df_for_analysis, 
                self.config.ticker, 
                current_price=market_data['price']
            )
            
            # If entry signal detected, use shared entry processing logic (same as realtime bot)
            if entry_signal:
                # Add volume ratio to entry signal if available
                if 'volume_ratio' in market_data:
                    if not hasattr(entry_signal, 'indicators'):
                        entry_signal.indicators = {}
                    entry_signal.indicators['volume_ratio'] = market_data['volume_ratio']
                
                # Use shared entry processing function (same as bot)
                entry_config = {
                    'max_positions': self.config.max_positions,
                    'position_size_pct': 0.95,  # Simulator uses 95% of capital (same as before, but now uses shared logic)
                    'initial_capital': self.config.initial_capital,
                    'min_capital': 100.0
                }
                
                entry_result = process_entry_signal(
                    entry_signal,
                    self.position_manager,
                    entry_config,
                    current_capital=self.config.initial_capital,  # Simulator uses initial capital (no capital tracking)
                    timestamp=timestamp
                )
                
                if entry_result and entry_result.get('success'):
                    position_type = entry_result.get('position_type')
                    position_type_str = position_type.value if hasattr(position_type, 'value') else str(position_type)
                    self.logger.info(f"Entered {position_type_str} position at {entry_signal.price:.2f} on {timestamp.strftime('%H:%M:%S')} (Pattern: {entry_result['entry_pattern']}, Confidence: {entry_result['confidence']*100:.1f}%)")
                elif entry_result:
                    self.logger.info(f"Entry signal rejected: {entry_result.get('reason', 'Unknown reason')}")
                else:
                    self.logger.warning(f"Entry processing returned None")
            
        except Exception as e:
            self.logger.error(f"Error checking entry signals: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _update_positions(self, market_data: Dict, timestamp: pd.Timestamp):
        """Update existing positions (position manager handles ALL logic)"""
        try:
            # Create market data dict for position manager
            market_data_for_pm = {self.config.ticker: market_data}
            
            # Let position manager handle ALL logic
            exits = self.position_manager.update_positions(market_data_for_pm)
            
            # Process any exits using shared trade processing logic
            for exit_info in exits:
                if exit_info.get('exited'):
                    # Use shared trade processing function (same as bot)
                    trade_data = process_exit_to_trade_data(
                        exit_info,
                        self.position_manager,
                        timestamp
                    )
                    
                    if trade_data:
                        # Convert to TradeResult format for simulator
                        trade_result = self._create_trade_result_from_data(trade_data)
                        self.trades.append(trade_result)
                        
                        exit_type = "PARTIAL" if trade_data['is_partial_exit'] else "COMPLETE"
                        self.logger.info(f"{exit_type} exit at {trade_data['exit_price']:.2f} on {timestamp.strftime('%H:%M:%S')} - {trade_data['exit_reason']}")
                
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
    
    def _create_trade_result_from_data(self, trade_data: Dict) -> TradeResult:
        """Create TradeResult from standardized trade data (shared with bot)"""
        # Calculate commission based on configuration
        commission = trade_data['entry_price'] * trade_data['shares'] * self.commission_per_trade
        
        # Get position type from position manager
        strategy = 'unknown'
        if hasattr(self.position_manager, 'active_positions') and trade_data['ticker'] in self.position_manager.active_positions:
            strategy = self.position_manager.active_positions[trade_data['ticker']].position_type.value
        elif hasattr(self.position_manager, 'position_history') and trade_data['ticker'] in self.position_manager.position_history:
            strategy = self.position_manager.position_history[trade_data['ticker']].position_type.value
        
        return TradeResult(
            ticker=trade_data['ticker'],
            entry_time=trade_data['entry_time'],
            exit_time=trade_data['exit_time'],
            entry_price=trade_data['entry_price'],
            exit_price=trade_data['exit_price'],
            quantity=int(trade_data['shares']),  # Use actual shares (partial or full)
            strategy=strategy,
            pnl=trade_data['pnl'] - commission,
            pnl_pct=trade_data['pnl_pct'],
            commission=commission,
            exit_reason=trade_data['exit_reason'],
            hold_minutes=int((trade_data['exit_time'] - trade_data['entry_time']).total_seconds() / 60)
        )
