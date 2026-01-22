"""
Webull Data API Implementation
Integrates WebullUtil with the DataAPI interface for live trading
"""
import pandas as pd
from typing import Optional, List, Dict
from datetime import datetime
import pytz
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
        self._data_cache = {}  # Cache for 1-minute data: {ticker: DataFrame}
        self._cache_timestamps = {}  # Track when cache was last updated: {ticker: datetime}
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
    
    def get_1min_data(self, ticker: str, minutes: int = 800) -> pd.DataFrame:
        """
        Fetch 1-minute data for a ticker with incremental cache updates
        
        The cache is updated incrementally - only new minute data is fetched and appended
        to the cached data, ensuring the cache always has the latest data.
        
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
            
            # Check if we have cached data
            cached_df = self._data_cache.get(ticker)
            cache_time = self._cache_timestamps.get(ticker)
            
            et_tz = pytz.timezone('America/New_York')
            now_et = datetime.now(et_tz)
            
            # If cache exists and is recent (< 2 minutes old), try incremental update
            if cached_df is not None and cache_time is not None:
                cache_age_seconds = (now_et - cache_time).total_seconds()
                
                if cache_age_seconds < 120:  # Cache is less than 2 minutes old
                    try:
                        # Fetch only the last few minutes to get new data
                        # Fetch 5 minutes to ensure we get any new data
                        df_new = fetch_data_array(ticker_id=ticker_id, symbol=ticker, timeframe='m1', count=5)
                        
                        if df_new is not None and not df_new.empty:
                            # Reset index to make timestamp a column
                            df_new = df_new.reset_index()
                            
                            # Ensure timestamp is datetime
                            if not pd.api.types.is_datetime64_any_dtype(df_new['timestamp']):
                                df_new['timestamp'] = pd.to_datetime(df_new['timestamp'])
                            
                            # Convert timestamps to ET if needed
                            if df_new['timestamp'].dt.tz is None:
                                df_new['timestamp'] = df_new['timestamp'].dt.tz_localize('America/New_York')
                            elif str(df_new['timestamp'].dt.tz) != 'America/New_York':
                                df_new['timestamp'] = df_new['timestamp'].dt.tz_convert('America/New_York')
                            
                            # Ensure cached data timestamps are in ET
                            if not pd.api.types.is_datetime64_any_dtype(cached_df['timestamp']):
                                cached_df = cached_df.copy()
                                cached_df['timestamp'] = pd.to_datetime(cached_df['timestamp'])
                            
                            if cached_df['timestamp'].dt.tz is None:
                                cached_df['timestamp'] = cached_df['timestamp'].dt.tz_localize('America/New_York')
                            elif str(cached_df['timestamp'].dt.tz) != 'America/New_York':
                                cached_df['timestamp'] = cached_df['timestamp'].dt.tz_convert('America/New_York')
                            
                            # Get the latest timestamp from cache
                            latest_cached_time = cached_df['timestamp'].max()
                            
                            # Filter new data to only records newer than cache
                            df_new_records = df_new[df_new['timestamp'] > latest_cached_time].copy()
                            
                            if len(df_new_records) > 0:
                                # Select required columns
                                required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                                df_new_records = df_new_records[required_columns]
                                
                                # Merge with cached data
                                df_merged = pd.concat([cached_df, df_new_records], ignore_index=True)
                                df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)
                                # Remove duplicates (keep last if any)
                                df_merged = df_merged.drop_duplicates(subset=['timestamp'], keep='last').sort_values('timestamp').reset_index(drop=True)
                                
                                # Update cache
                                self._data_cache[ticker] = df_merged.copy()
                                self._cache_timestamps[ticker] = now_et
                                
                                logger.debug(f"[{ticker}] Appended {len(df_new_records)} new minutes to cache (total: {len(df_merged)})")
                                
                                # Return last N minutes if we have more data than requested
                                if len(df_merged) > minutes:
                                    df_result = df_merged.iloc[-minutes:].reset_index(drop=True)
                                else:
                                    df_result = df_merged.reset_index(drop=True)
                                
                                return df_result
                            else:
                                # No new data, return cached data (trimmed to requested minutes)
                                logger.debug(f"[{ticker}] No new data, using cached data ({len(cached_df)} minutes)")
                                if len(cached_df) > minutes:
                                    return cached_df.iloc[-minutes:].reset_index(drop=True)
                                else:
                                    return cached_df.reset_index(drop=True)
                    except Exception as e:
                        # If incremental update fails, fall through to full fetch
                        logger.debug(f"[{ticker}] Incremental cache update failed: {e}, fetching full data")
            
            # No cache or cache is stale - fetch full dataset
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
            
            # Update cache with full dataset
            self._data_cache[ticker] = df_result.copy()
            self._cache_timestamps[ticker] = now_et
            
            logger.debug(f"Successfully fetched {len(df_result)} rows of data for {ticker} (cache updated)")
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
    
    def get_top_gainers(self, page_size: int = 30) -> List[Dict]:
        """
        Get full top gainers data with change percentages calculated correctly for market session
        
        Args:
            page_size: Number of gainers to fetch
            
        Returns:
            List of gainer dictionaries with symbol, price, change_ratio, etc.
            Change ratio is calculated correctly based on market session:
            - After-hours: (current_price - today_close) / today_close * 100
            - Regular hours: (current_price - prev_close) / prev_close * 100
            - Pre-market: (current_price - prev_close) / prev_close * 100
            
        Raises:
            DataAPIError: If API call fails
        """
        try:
            if not isinstance(page_size, int) or page_size <= 0:
                raise ValueError("page_size must be a positive integer")
            
            # Get current market session rank type (preMarket, 1d, afterMarket)
            rank_type = get_rank_type()
            logger.debug(f"Fetching top gainers with rank_type: {rank_type}, page_size: {page_size}")
            
            # Fetch top gainers with the correct rank type
            raw_gainers = fetch_top_gainers(rankType=rank_type, pageSize=page_size)
            
            if not raw_gainers:
                logger.warning(f"No gainers returned from API for rank type: {rank_type}")
                return []
            
            # Extract the actual data from the response
            gainers_list = raw_gainers
            if isinstance(raw_gainers, dict) and 'data' in raw_gainers:
                gainers_list = raw_gainers['data']
            elif not isinstance(raw_gainers, list):
                logger.error(f"Unexpected response format: {type(raw_gainers)}")
                return []
            
            if not gainers_list:
                logger.warning(f"No gainers data found in response for rank type: {rank_type}")
                return []
            
            # Process and return gainer data with correct change percentage calculation
            processed_gainers = []
            for gainer in gainers_list:
                if not isinstance(gainer, dict):
                    continue
                
                # Extract symbol (handle nested structure)
                if 'ticker' in gainer and isinstance(gainer['ticker'], dict):
                    ticker_data = gainer['ticker']
                    symbol = (ticker_data.get('symbol') or 
                             ticker_data.get('ticker') or 
                             ticker_data.get('code') or 
                             ticker_data.get('disSymbol') or '')
                else:
                    symbol = (gainer.get('symbol') or 
                             gainer.get('ticker') or 
                             gainer.get('code') or 
                             gainer.get('disSymbol') or '')
                
                if not symbol or not isinstance(symbol, str):
                    continue
                
                # Extract current price
                if 'values' in gainer and isinstance(gainer['values'], dict):
                    current_price = (gainer['values'].get('price') or 
                                   gainer['values'].get('close') or 
                                   gainer['values'].get('last') or 0)
                else:
                    current_price = (gainer.get('price') or 
                                    gainer.get('close') or 
                                    gainer.get('last') or 
                                    gainer.get('pPrice') or 0)
                
                try:
                    current_price = float(current_price) if current_price else 0.0
                except (ValueError, TypeError):
                    current_price = 0.0
                
                if current_price == 0:
                    continue
                
                # Calculate change percentage based on market session
                change_ratio = 0.0
                change = 0.0
                
                if rank_type == 'afterMarket':
                    # After-hours: Change from today's regular close at 4:00 PM (not previous day)
                    # First try to get close from API response
                    regular_close = (gainer.get('close') or 
                                   gainer.get('regularClose') or 
                                   gainer.get('dayClose') or 0)
                    
                    # If close is in nested structure, check there too
                    if regular_close == 0 and 'values' in gainer and isinstance(gainer['values'], dict):
                        regular_close = (gainer['values'].get('close') or 
                                       gainer['values'].get('regularClose') or 0)
                    
                    try:
                        regular_close = float(regular_close) if regular_close else 0.0
                    except (ValueError, TypeError):
                        regular_close = 0.0
                    
                    # If close not in API response, try quote API first, then minute data
                    if regular_close == 0:
                        try:
                            # First try quote API - it might have the regular close
                            ticker_id = self._get_ticker_id(symbol)
                            if ticker_id:
                                quote = webull_get_stock_quote(ticker_id)
                                if quote:
                                    # Try to get close from quote (might be regular session close)
                                    quote_close = (quote.get('close') or 
                                                 quote.get('regularClose') or 
                                                 quote.get('dayClose') or 0)
                                    if quote_close:
                                        try:
                                            regular_close = float(quote_close)
                                            logger.debug(f"Got regular close {regular_close} for {symbol} from quote API")
                                        except (ValueError, TypeError):
                                            pass
                                    
                                    # If still no close, try to get from preClose and calculate
                                    # But for after-hours, we need today's close, not previous day
                                    if regular_close == 0:
                                        # Fetch today's minute data to get the 4:00 PM close
                                        df = fetch_data_array(ticker_id=ticker_id, symbol=symbol, timeframe='m1', count=500)
                                        if df is not None and not df.empty:
                                            # Filter for today's data
                                            et_tz = pytz.timezone('America/New_York')
                                            today = datetime.now(et_tz).date()
                                            
                                            # Reset index if timestamp is the index
                                            if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
                                                df = df.reset_index()
                                            
                                            # Ensure timestamp column exists and is datetime
                                            if 'timestamp' in df.columns:
                                                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                                                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                                                
                                                # Filter for today's data
                                                df['date'] = df['timestamp'].dt.date
                                                today_df = df[df['date'] == today]
                                                
                                                if not today_df.empty:
                                                    # Find the last bar at or before 4:00 PM (16:00)
                                                    # Get bars at or before 4:00 PM (regular session ends at 4:00 PM)
                                                    regular_hours_df = today_df[today_df['timestamp'].dt.hour < 16]
                                                    
                                                    if not regular_hours_df.empty:
                                                        # Get the last close price from regular hours (should be at 4:00 PM)
                                                        regular_close = float(regular_hours_df.iloc[-1]['close'])
                                                        logger.debug(f"Fetched regular close {regular_close} for {symbol} from minute data (4 PM close)")
                                                    else:
                                                        # If no regular hours data yet, try to get the last bar at exactly 16:00
                                                        at_4pm = today_df[(today_df['timestamp'].dt.hour == 16) & (today_df['timestamp'].dt.minute == 0)]
                                                        if not at_4pm.empty:
                                                            regular_close = float(at_4pm.iloc[-1]['close'])
                                                            logger.debug(f"Fetched regular close {regular_close} for {symbol} from minute data (exactly 4 PM)")
                        except Exception as e:
                            logger.debug(f"Could not fetch regular close for {symbol}: {e}")
                            regular_close = 0.0
                    
                    if regular_close > 0:
                        change_ratio = ((current_price - regular_close) / regular_close) * 100
                        change = current_price - regular_close
                        logger.debug(f"After-hours change for {symbol}: {change_ratio:.2f}% (current: {current_price}, close: {regular_close})")
                    else:
                        # Last fallback: try to use preClose if close not available
                        prev_close = (gainer.get('preClose') or 
                                    gainer.get('previousClose') or 
                                    gainer.get('prevClose') or 0)
                        try:
                            prev_close = float(prev_close) if prev_close else 0.0
                        except (ValueError, TypeError):
                            prev_close = 0.0
                        
                        if prev_close > 0:
                            change_ratio = ((current_price - prev_close) / prev_close) * 100
                            change = current_price - prev_close
                            logger.warning(f"Using prev_close for {symbol} after-hours calculation (regular close not available)")
                        else:
                            logger.warning(f"Could not determine regular close for {symbol} - change pct will be 0")
                
                elif rank_type == 'preMarket':
                    # Pre-market: Change from previous close
                    prev_close = (gainer.get('preClose') or 
                                gainer.get('previousClose') or 
                                gainer.get('prevClose') or 0)
                    
                    if prev_close == 0 and 'values' in gainer and isinstance(gainer['values'], dict):
                        prev_close = (gainer['values'].get('preClose') or 
                                    gainer['values'].get('previousClose') or 0)
                    
                    try:
                        prev_close = float(prev_close) if prev_close else 0.0
                    except (ValueError, TypeError):
                        prev_close = 0.0
                    
                    if prev_close > 0:
                        change_ratio = ((current_price - prev_close) / prev_close) * 100
                        change = current_price - prev_close
                
                else:  # '1d' - regular hours
                    # Regular hours: Change from previous close
                    prev_close = (gainer.get('preClose') or 
                                gainer.get('previousClose') or 
                                gainer.get('prevClose') or 0)
                    
                    if prev_close == 0 and 'values' in gainer and isinstance(gainer['values'], dict):
                        prev_close = (gainer['values'].get('preClose') or 
                                    gainer['values'].get('previousClose') or 0)
                    
                    try:
                        prev_close = float(prev_close) if prev_close else 0.0
                    except (ValueError, TypeError):
                        prev_close = 0.0
                    
                    if prev_close > 0:
                        change_ratio = ((current_price - prev_close) / prev_close) * 100
                        change = current_price - prev_close
                    else:
                        # Fallback: use API's change_ratio if available
                        change_ratio = (gainer.get('change_ratio') or 
                                       gainer.get('changeRatio') or 
                                       gainer.get('pctChg') or 0)
                        change = (gainer.get('change') or 
                                gainer.get('chg') or 0)
                
                # Extract volume
                volume = gainer.get('volume') or gainer.get('vol') or 0
                try:
                    volume = int(volume) if volume else 0
                except (ValueError, TypeError):
                    volume = 0
                
                # Build gainer dict in expected format
                gainer_dict = {
                    'symbol': symbol,
                    'price': current_price,
                    'change': float(change) if change else 0.0,
                    'change_ratio': float(change_ratio) if change_ratio else 0.0,
                    'changeRatio': float(change_ratio) if change_ratio else 0.0,  # Support both field names
                    'volume': volume,
                    'rank_type': rank_type
                }
                
                # Add any other useful fields
                if 'marketValue' in gainer:
                    gainer_dict['market_cap'] = gainer.get('marketValue')
                if 'avgVol10D' in gainer:
                    gainer_dict['avg_volume'] = gainer.get('avgVol10D')
                
                processed_gainers.append(gainer_dict)
            
            logger.info(f"Retrieved {len(processed_gainers)} gainers for rank type: {rank_type}")
            return processed_gainers
            
        except Exception as e:
            logger.error(f"Error fetching top gainers: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise DataAPIError(f"Error fetching top gainers: {e}")
    
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
            
            # Use get_top_gainers to get full data, then extract symbols
            gainers = self.get_top_gainers(page_size=count)
            
            if not gainers:
                logger.warning("No gainers returned from API")
                return []
            
            # Extract and validate ticker symbols
            valid_tickers = []
            for gainer in gainers:
                symbol = gainer.get('symbol', '')
                if symbol:
                    try:
                        TradingParameterValidator.validate_ticker(symbol)
                        valid_tickers.append(symbol)
                    except Exception:
                        logger.warning(f"Skipping invalid ticker: {symbol}")
            
            logger.info(f"Retrieved {len(valid_tickers)} valid tickers from {len(gainers)} gainers")
            return valid_tickers
            
        except Exception as e:
            logger.error(f"Error fetching top gainers: {e}")
            raise DataAPIError(f"Error fetching top gainers: {e}")
    
    def clear_cache(self):
        """Clear all internal caches"""
        self._ticker_id_cache.clear()
        self._data_cache.clear()
        self._cache_timestamps.clear()
        logger.info("WebullDataAPI internal cache cleared")
