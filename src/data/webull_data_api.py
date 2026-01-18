"""
Webull Data API Implementation
Integrates WebullUtil with the DataAPI interface for live trading
"""
import pandas as pd
from typing import Optional, List, Dict
from datetime import datetime
from .api_interface import DataAPI
from ..utils.cache import cached, get_cache_key
from ..utils.validation import DataFrameValidator, TradingParameterValidator
from ..exceptions import DataAPIError, InsufficientDataError, RateLimitError, NetworkError
try:
    from .WebullUtil import (
        find_tickerid_for_symbol,
        get_stock_quote as webull_get_stock_quote,
        fetch_top_gainers,
        fetch_swing_stocks,
        fetch_data_array,
        get_rank_type,
        calculate_relative_volume
    )
except ImportError as e:
    raise ImportError(
        "WebullUtil module not found. Please ensure WebullUtil.py is in the src directory. "
        f"Original error: {e}"
    )

import logging

logger = logging.getLogger(__name__)


class WebullDataAPI(DataAPI):
    """
    Webull API implementation for live trading
    Uses WebullUtil methods to fetch real-time stock data
    """
    
    def __init__(self):
        """Initialize Webull Data API"""
        self._ticker_id_cache = {}  # Cache ticker IDs to reduce API calls
        logger.info("WebullDataAPI initialized")
    
    def _get_ticker_id(self, ticker: str) -> Optional[int]:
        """Get ticker ID for a symbol, with caching and validation"""
        TradingParameterValidator.validate_ticker(ticker)
        
        if ticker in self._ticker_id_cache:
            return self._ticker_id_cache[ticker]
        
        try:
            ticker_id = find_tickerid_for_symbol(ticker)
            if ticker_id:
                self._ticker_id_cache[ticker] = ticker_id
            return ticker_id
        except Exception as e:
            logger.error(f"Error getting ticker ID for {ticker}: {e}")
            raise DataAPIError(f"Failed to get ticker ID for {ticker}: {e}")
    
    @cached(ttl_seconds=300, max_size=100)  # 5-minute cache for market data
    def get_1min_data(self, ticker: str, minutes: int = 800) -> pd.DataFrame:
        """
        Fetch 1-minute data for a ticker with caching and validation
        
        Args:
            ticker: Stock ticker symbol
            minutes: Number of minutes of historical data to fetch (default: 800, max: 1200)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            
        Raises:
            DataAPIError: If API call fails
            InsufficientDataError: If not enough data is returned
        """
        try:
            TradingParameterValidator.validate_ticker(ticker)
            
            ticker_id = self._get_ticker_id(ticker)
            if not ticker_id:
                raise DataAPIError(f"Could not find ticker ID for {ticker}")
            
            # Limit to max 1200
            count = min(minutes, 1200)
            
            # Fetch 1-minute data
            df = fetch_data_array(ticker_id=ticker_id, symbol=ticker, timeframe='m1', count=count)
            
            if df is None or df.empty:
                raise InsufficientDataError(f"No data returned for {ticker}")
            
            # Reset index to make timestamp a column
            df = df.reset_index()
            
            # Validate the DataFrame
            try:
                DataFrameValidator.validate_ohlcv(df, min_rows=1)
            except Exception as e:
                raise DataAPIError(f"Invalid data format for {ticker}: {e}")
            
            # Select and return only required columns (avoid copying)
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # Check if all required columns exist
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise DataAPIError(f"Missing required columns for {ticker}: {missing_columns}")
            
            # Create a view instead of copying when possible
            df_result = df[required_columns]
            
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df_result['timestamp']):
                df_result = df_result.copy()  # Need to copy if converting timestamp
                df_result['timestamp'] = pd.to_datetime(df_result['timestamp'])
            
            # Sort by timestamp and remove duplicates (in-place operations to avoid copying)
            df_result = df_result.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
            
            # Return last N minutes if we have more data than requested
            if len(df_result) > minutes:
                df_result = df_result.iloc[-minutes:].reset_index(drop=True)
            else:
                df_result = df_result.reset_index(drop=True)
            
            logger.debug(f"Successfully fetched {len(df_result)} rows of data for {ticker}")
            return df_result
            
        except (DataAPIError, InsufficientDataError):
            raise  # Re-raise our custom exceptions
        except Exception as e:
            logger.error(f"Unexpected error fetching data for {ticker}: {e}")
            raise DataAPIError(f"Unexpected error fetching data for {ticker}: {e}")
    
    def get_current_price(self, ticker: str) -> float:
        """
        Get current price for a ticker with validation
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Current price
            
        Raises:
            DataAPIError: If API call fails
        """
        try:
            TradingParameterValidator.validate_ticker(ticker)
            
            # Try to get current price from API
            ticker_id = self._get_ticker_id(ticker)
            if not ticker_id:
                raise DataAPIError(f"Could not find ticker ID for {ticker}")
            
            quote = webull_get_stock_quote(ticker_id)
            if quote is None or 'price' not in quote:
                raise DataAPIError(f"No quote data available for {ticker}")
            
            price = float(quote['price'])
            TradingParameterValidator.validate_price(price, f"current_price for {ticker}")
            
            return price
            
        except (DataAPIError, ValueError):
            raise  # Re-raise our custom exceptions
        except Exception as e:
            logger.error(f"Unexpected error getting current price for {ticker}: {e}")
            raise DataAPIError(f"Unexpected error getting current price for {ticker}: {e}")
    
    def get_stock_list_from_gainers(self, count: int = 30) -> List[str]:
        """
        Get list of top gainers with validation
        
        Args:
            count: Number of gainers to fetch
            
        Returns:
            List of ticker symbols
            
        Raises:
            DataAPIError: If API call fails
        """
        try:
            if not isinstance(count, int) or count <= 0:
                raise ValueError("Count must be a positive integer")
            
            gainers = fetch_top_gainers(count=count)
            
            if not gainers:
                logger.warning("No gainers returned from API")
                return []
            
            # Validate tickers and filter out invalid ones
            valid_tickers = []
            for ticker in gainers:
                try:
                    TradingParameterValidator.validate_ticker(ticker)
                    valid_tickers.append(ticker)
                except Exception:
                    logger.warning(f"Skipping invalid ticker: {ticker}")
            
            logger.info(f"Retrieved {len(valid_tickers)} valid tickers from {len(gainers)} gainers")
            return valid_tickers
            
        except Exception as e:
            logger.error(f"Error fetching top gainers: {e}")
            raise DataAPIError(f"Error fetching top gainers: {e}")
    
    def clear_cache(self):
        """Clear all internal caches"""
        self._ticker_id_cache.clear()
        logger.info("WebullDataAPI internal cache cleared")
