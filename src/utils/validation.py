"""
Input validation utilities for the trading bot
Provides comprehensive validation for trading data and parameters
"""
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime, timedelta
import re
from ..exceptions import DataValidationError, ConfigurationError

logger = logging.getLogger(__name__)


class DataFrameValidator:
    """Validator for pandas DataFrames"""
    
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame, min_rows: int = 1) -> None:
        """
        Validate OHLCV DataFrame format and content
        
        Args:
            df: DataFrame to validate
            min_rows: Minimum number of rows required
            
        Raises:
            DataValidationError: If DataFrame is invalid
        """
        if df is None or df.empty:
            raise DataValidationError("DataFrame is None or empty")
        
        if len(df) < min_rows:
            raise DataValidationError(f"DataFrame has {len(df)} rows, minimum {min_rows} required")
        
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")
        
        # Validate data types
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except Exception as e:
                raise DataValidationError(f"Invalid timestamp format: {e}")
        
        # Validate price data
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise DataValidationError(f"Column {col} must be numeric")
            
            if (df[col] <= 0).any():
                raise DataValidationError(f"Column {col} contains non-positive values")
        
        # Validate volume
        if not pd.api.types.is_numeric_dtype(df['volume']):
            raise DataValidationError("Volume column must be numeric")
        
        if (df['volume'] < 0).any():
            raise DataValidationError("Volume column contains negative values")
        
        # Validate OHLC relationships
        invalid_high = df['high'] < df[['open', 'low', 'close']].max(axis=1)
        invalid_low = df['low'] > df[['open', 'high', 'close']].min(axis=1)
        
        if invalid_high.any():
            raise DataValidationError("High prices are lower than other OHLC values")
        
        if invalid_low.any():
            raise DataValidationError("Low prices are higher than other OHLC values")
    
    @staticmethod
    def validate_timestamps(df: pd.DataFrame, max_gap_minutes: int = 60) -> None:
        """
        Validate timestamp continuity and gaps
        
        Args:
            df: DataFrame with timestamp column
            max_gap_minutes: Maximum allowed gap between consecutive timestamps
            
        Raises:
            DataValidationError: If timestamps are invalid
        """
        if 'timestamp' not in df.columns:
            raise DataValidationError("DataFrame missing timestamp column")
        
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff().dropna()
        
        # Check for large gaps
        max_gap = timedelta(minutes=max_gap_minutes)
        large_gaps = time_diffs > max_gap
        
        if large_gaps.any():
            gap_info = time_diffs[large_gaps].describe()
            logger.warning(f"Found {large_gaps.sum()} large time gaps: {gap_info}")
    
    @staticmethod
    def validate_price_ranges(df: pd.DataFrame, max_price_change_pct: float = 50.0) -> None:
        """
        Validate price ranges for anomalies
        
        Args:
            df: DataFrame with price data
            max_price_change_pct: Maximum allowed percentage change between periods
            
        Raises:
            DataValidationError: If price anomalies detected
        """
        price_cols = ['open', 'high', 'low', 'close']
        
        for col in price_cols:
            price_changes = df[col].pct_change().abs()
            extreme_changes = price_changes > (max_price_change_pct / 100)
            
            if extreme_changes.any():
                max_change = price_changes.max()
                logger.warning(f"Extreme price changes detected in {col}: {max_change:.2%}")


class TradingParameterValidator:
    """Validator for trading parameters"""
    
    @staticmethod
    def validate_confidence(confidence: float) -> None:
        """Validate confidence value"""
        if not isinstance(confidence, (int, float)):
            raise DataValidationError("Confidence must be numeric")
        
        if not 0 <= confidence <= 1:
            raise DataValidationError("Confidence must be between 0 and 1")
    
    @staticmethod
    def validate_percentage(value: float, name: str, min_val: float = 0, max_val: float = 100) -> None:
        """Validate percentage values"""
        if not isinstance(value, (int, float)):
            raise DataValidationError(f"{name} must be numeric")
        
        if not min_val <= value <= max_val:
            raise DataValidationError(f"{name} must be between {min_val}% and {max_val}%")
    
    @staticmethod
    def validate_price(price: float, name: str = "price", min_price: float = 0.01) -> None:
        """Validate price values"""
        if not isinstance(price, (int, float)):
            raise DataValidationError(f"{name} must be numeric")
        
        if price < min_price:
            raise DataValidationError(f"{name} must be at least ${min_price}")
    
    @staticmethod
    def validate_volume(volume: Union[int, float], name: str = "volume") -> None:
        """Validate volume values"""
        if not isinstance(volume, (int, float)):
            raise DataValidationError(f"{name} must be numeric")
        
        if volume < 0:
            raise DataValidationError(f"{name} cannot be negative")
    
    @staticmethod
    def validate_ticker(ticker: str) -> None:
        """Validate ticker symbol"""
        if not isinstance(ticker, str):
            raise DataValidationError("Ticker must be a string")
        
        if not ticker:
            raise DataValidationError("Ticker cannot be empty")
        
        # Basic ticker format validation (1-5 letters, optionally with numbers)
        if not re.match(r'^[A-Za-z]{1,5}[0-9]*$', ticker):
            raise DataValidationError(f"Invalid ticker format: {ticker}")
    
    @staticmethod
    def validate_time_window(start_time: str, end_time: str) -> None:
        """Validate trading time window"""
        time_pattern = r'^([01]?[0-9]|2[0-3]):[0-5][0-9]$'
        
        if not re.match(time_pattern, start_time):
            raise DataValidationError(f"Invalid start time format: {start_time}")
        
        if not re.match(time_pattern, end_time):
            raise DataValidationError(f"Invalid end time format: {end_time}")
        
        # Convert to minutes for comparison
        start_minutes = int(start_time.split(':')[0]) * 60 + int(start_time.split(':')[1])
        end_minutes = int(end_time.split(':')[0]) * 60 + int(end_time.split(':')[1])
        
        if start_minutes >= end_minutes:
            raise DataValidationError("Start time must be before end time")


class ConfigurationValidator:
    """Validator for configuration objects"""
    
    @staticmethod
    def validate_trading_config(config) -> None:
        """Validate trading configuration"""
        TradingParameterValidator.validate_confidence(config.min_confidence)
        TradingParameterValidator.validate_percentage(config.min_entry_price_increase, "min_entry_price_increase")
        TradingParameterValidator.validate_percentage(config.trailing_stop_pct, "trailing_stop_pct")
        TradingParameterValidator.validate_percentage(config.profit_target_pct, "profit_target_pct")
        TradingParameterValidator.validate_percentage(config.position_size_pct, "position_size_pct")
        TradingParameterValidator.validate_price(config.min_price_filter, "min_price_filter")
        
        if config.max_positions <= 0:
            raise ConfigurationError("max_positions must be positive")
        
        if config.max_trades_per_day <= 0:
            raise ConfigurationError("max_trades_per_day must be positive")
    
    @staticmethod
    def validate_capital_config(config) -> None:
        """Validate capital configuration"""
        if config.initial_capital <= 0:
            raise ConfigurationError("initial_capital must be positive")
        
        if config.target_capital <= config.initial_capital:
            raise ConfigurationError("target_capital must be greater than initial_capital")
        
        if config.daily_profit_target_min < 0:
            raise ConfigurationError("daily_profit_target_min cannot be negative")
        
        if config.daily_profit_target_max < config.daily_profit_target_min:
            raise ConfigurationError("daily_profit_target_max must be >= daily_profit_target_min")


def validate_trade_signal(signal) -> None:
    """
    Validate a trade signal object
    
    Args:
        signal: TradeSignal object to validate
        
    Raises:
        DataValidationError: If signal is invalid
    """
    if not signal:
        raise DataValidationError("Trade signal is None")
    
    TradingParameterValidator.validate_ticker(signal.ticker)
    TradingParameterValidator.validate_price(signal.price, "signal_price")
    TradingParameterValidator.validate_confidence(signal.confidence)
    
    if signal.target_price is not None:
        TradingParameterValidator.validate_price(signal.target_price, "target_price")
    
    if signal.stop_loss is not None:
        TradingParameterValidator.validate_price(signal.stop_loss, "stop_loss")
    
    if signal.target_price and signal.stop_loss:
        if signal.target_price <= signal.stop_loss:
            raise DataValidationError("Target price must be greater than stop loss")


def validate_position(position) -> None:
    """
    Validate an active position object
    
    Args:
        position: ActivePosition object to validate
        
    Raises:
        DataValidationError: If position is invalid
    """
    if not position:
        raise DataValidationError("Position is None")
    
    TradingParameterValidator.validate_ticker(position.ticker)
    TradingParameterValidator.validate_price(position.entry_price, "entry_price")
    TradingParameterValidator.validate_price(position.current_price, "current_price")
    TradingParameterValidator.validate_price(position.target_price, "target_price")
    TradingParameterValidator.validate_price(position.stop_loss, "stop_loss")
    TradingParameterValidator.validate_volume(position.shares, "shares")
    
    if position.shares <= 0:
        raise DataValidationError("Position shares must be positive")
    
    if position.target_price <= position.stop_loss:
        raise DataValidationError("Target price must be greater than stop loss")
