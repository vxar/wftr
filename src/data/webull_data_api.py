"""
Webull Data API Implementation
Integrates WebullUtil with the DataAPI interface for live trading
"""
import pandas as pd
from typing import Optional, List, Dict
from datetime import datetime
from data.api_interface import DataAPI
try:
    from data.WebullUtil import (
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
        """Get ticker ID for a symbol, with caching"""
        if ticker in self._ticker_id_cache:
            return self._ticker_id_cache[ticker]
        
        try:
            ticker_id = find_tickerid_for_symbol(ticker)
            if ticker_id:
                self._ticker_id_cache[ticker] = ticker_id
            return ticker_id
        except Exception as e:
            logger.error(f"Error getting ticker ID for {ticker}: {e}")
            return None
    
    def get_1min_data(self, ticker: str, minutes: int = 800) -> pd.DataFrame:
        """
        Fetch 1-minute data for a ticker
        
        Args:
            ticker: Stock ticker symbol
            minutes: Number of minutes of historical data to fetch (default: 800, max: 1200)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ticker_id = self._get_ticker_id(ticker)
            if not ticker_id:
                raise ValueError(f"Could not find ticker ID for {ticker}")
            
            # Limit to max 1200
            count = min(minutes, 1200)
            
            # Fetch 1-minute data
            df = fetch_data_array(ticker_id=ticker_id, symbol=ticker, timeframe='m1', count=count)
            
            if df is None or df.empty:
                raise ValueError(f"No data returned for {ticker}")
            
            # Reset index to make timestamp a column
            df = df.reset_index()
            
            # Ensure we have the required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Got: {df.columns.tolist()}")
            
            # Select and return only required columns
            df_result = df[required_columns].copy()
            
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df_result['timestamp']):
                df_result['timestamp'] = pd.to_datetime(df_result['timestamp'])
            
            # Sort by timestamp
            df_result = df_result.sort_values('timestamp').reset_index(drop=True)
            
            # Remove duplicates
            df_result = df_result.drop_duplicates(subset=['timestamp'], keep='last')
            df_result = df_result.sort_values('timestamp').reset_index(drop=True)
            
            logger.debug(f"Fetched {len(df_result)} minutes of data for {ticker}")
            return df_result
            
        except Exception as e:
            logger.error(f"Error fetching 1-minute data for {ticker}: {e}")
            raise
    
    def get_5min_data(self, ticker: str, periods: int = 240) -> pd.DataFrame:
        """
        Fetch 5-minute data for a ticker
        
        Args:
            ticker: Stock ticker symbol
            periods: Number of 5-minute periods to fetch (default: 240 = 20 hours, max: 1200)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ticker_id = self._get_ticker_id(ticker)
            if not ticker_id:
                raise ValueError(f"Could not find ticker ID for {ticker}")
            
            # Limit to max 1200
            count = min(periods, 1200)
            
            # Fetch 5-minute data
            df = fetch_data_array(ticker_id=ticker_id, symbol=ticker, timeframe='m5', count=count)
            
            if df is None or df.empty:
                raise ValueError(f"No data returned for {ticker}")
            
            # Reset index to make timestamp a column
            df = df.reset_index()
            
            # Ensure we have the required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Got: {df.columns.tolist()}")
            
            # Select and return only required columns
            df_result = df[required_columns].copy()
            
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df_result['timestamp']):
                df_result['timestamp'] = pd.to_datetime(df_result['timestamp'])
            
            # Sort by timestamp
            df_result = df_result.sort_values('timestamp').reset_index(drop=True)
            
            # Remove duplicates
            df_result = df_result.drop_duplicates(subset=['timestamp'], keep='last')
            df_result = df_result.sort_values('timestamp').reset_index(drop=True)
            
            logger.debug(f"Fetched {len(df_result)} 5-minute periods of data for {ticker}")
            return df_result
            
        except Exception as e:
            logger.error(f"Error fetching 5-minute data for {ticker}: {e}")
            raise
    
    def get_current_price(self, ticker: str) -> float:
        """
        Get current price for a ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Current price as float
        """
        try:
            ticker_id = self._get_ticker_id(ticker)
            if not ticker_id:
                raise ValueError(f"Could not find ticker ID for {ticker}")
            
            quote = webull_get_stock_quote(ticker_id)
            if quote is None:
                raise ValueError(f"No quote data returned for {ticker}")
            
            # Try to get current price from quote
            # Webull quote may have different field names, try common ones
            price = None
            for field in ['close', 'lastPrice', 'price', 'c']:
                if field in quote:
                    price = float(quote[field])
                    break
            
            if price is None:
                raise ValueError(f"Could not extract price from quote for {ticker}")
            
            return price
            
        except Exception as e:
            logger.error(f"Error getting current price for {ticker}: {e}")
            # Fallback: try to get from latest 1-minute data
            try:
                df = self.get_1min_data(ticker, minutes=1)
                return float(df.iloc[-1]['close'])
            except:
                raise ValueError(f"Could not get current price for {ticker}: {e}")
    
    def get_top_gainers(self, page_size: int = 50) -> List[Dict]:
        """
        Fetch top gainers list
        
        Args:
            page_size: Number of stocks to fetch (default: 50)
            
        Returns:
            List of dictionaries with stock information
        """
        try:
            rank_type = get_rank_type()
            json_data = fetch_top_gainers(rank_type, page_size)
            
            if json_data is None or 'data' not in json_data:
                logger.warning("No top gainers data returned")
                return []
            
            stocks = []
            # Handle different response structures
            data_items = json_data.get('data', [])
            if not data_items and isinstance(json_data, list):
                data_items = json_data
            
            for item in data_items:
                # Webull API returns nested structure: item['ticker']['symbol']
                ticker_info = item.get('ticker', {})
                if isinstance(ticker_info, dict):
                    symbol = ticker_info.get('symbol') or ticker_info.get('disSymbol', '')
                    ticker_id = ticker_info.get('tickerId', '')
                    name = ticker_info.get('name', '')
                else:
                    # Fallback: try direct fields
                    symbol = item.get('symbol') or item.get('tickerSymbol') or item.get('disSymbol', '')
                    ticker_id = item.get('tickerId') or item.get('tickerid', '')
                    name = item.get('name') or item.get('disName', '')
                
                if not symbol:
                    continue
                    
                # Get change ratio (try multiple field names)
                change_ratio = item.get('changeRatio') or item.get('change_ratio') or item.get('changeRatio', 0)
                if change_ratio == 0:
                    # Try calculating from change and price
                    change = item.get('change', 0)
                    price = item.get('close') or item.get('lastPrice') or item.get('price', 0)
                    if price and price > 0:
                        change_ratio = (change / price) * 100
                
                stock_info = {
                    'symbol': symbol,
                    'ticker_id': ticker_id,
                    'name': name,
                    'change': item.get('change', 0),
                    'change_ratio': change_ratio,
                    'changeRatio': change_ratio,  # Also include as changeRatio for compatibility
                    'price': item.get('close') or item.get('lastPrice') or item.get('price', 0),
                    'volume': item.get('volume', 0)
                }
                stocks.append(stock_info)
            
            logger.info(f"Fetched {len(stocks)} top gainers")
            return stocks
            
        except Exception as e:
            logger.error(f"Error fetching top gainers: {e}")
            return []
    
    def get_swing_stocks(self, min_price: float = 10.0, max_price: float = 250.0, 
                        rsi_min: int = 40, rsi_max: int = 60) -> List[Dict]:
        """
        Fetch swing stocks using screener
        
        Args:
            min_price: Minimum price filter
            max_price: Maximum price filter
            rsi_min: Minimum RSI filter
            rsi_max: Maximum RSI filter
            
        Returns:
            List of dictionaries with stock information
        """
        try:
            json_data = fetch_swing_stocks(min_price, max_price, rsi_min, rsi_max)
            
            if json_data is None or 'data' not in json_data:
                logger.warning("No swing stocks data returned")
                return []
            
            stocks = []
            for item in json_data.get('data', []):
                stock_info = {
                    'symbol': item.get('symbol', ''),
                    'ticker_id': item.get('tickerId', ''),
                    'name': item.get('name', ''),
                    'price': item.get('close', 0),
                    'rsi': item.get('rsi', 0),
                    'volume': item.get('volume', 0)
                }
                stocks.append(stock_info)
            
            logger.info(f"Fetched {len(stocks)} swing stocks")
            return stocks
            
        except Exception as e:
            logger.error(f"Error fetching swing stocks: {e}")
            return []
    
    def get_relative_volume(self, ticker: str, window: int = 14) -> Optional[float]:
        """
        Calculate relative volume for a ticker
        
        Args:
            ticker: Stock ticker symbol
            window: Rolling window for average volume (default: 14)
            
        Returns:
            Relative volume ratio (current volume / average volume) or None
        """
        try:
            ticker_id = self._get_ticker_id(ticker)
            if not ticker_id:
                return None
            
            rel_vol = calculate_relative_volume(ticker_id=ticker_id, symbol=ticker, window=window)
            return rel_vol
            
        except Exception as e:
            logger.error(f"Error calculating relative volume for {ticker}: {e}")
            return None
    
    def get_stock_list_from_gainers(self, count: int = 20) -> List[str]:
        """
        Get list of ticker symbols from top gainers
        
        Args:
            count: Number of stocks to fetch (default: 20)
            
        Returns:
            List of ticker symbols
        """
        gainers = self.get_top_gainers(page_size=count)
        symbols = [stock['symbol'] for stock in gainers if stock.get('symbol')]
        logger.info(f"Extracted {len(symbols)} symbols from {len(gainers)} gainers")
        return symbols
    
    def get_stock_list_from_swing_screener(self, count: int = 20, 
                                          min_price: float = 5.0, 
                                          max_price: float = 100.0) -> List[str]:
        """
        Get list of ticker symbols from swing stock screener
        
        Args:
            count: Maximum number of stocks to return (default: 20)
            min_price: Minimum price filter
            max_price: Maximum price filter
            
        Returns:
            List of ticker symbols
        """
        swing_stocks = self.get_swing_stocks(min_price=min_price, max_price=max_price)
        symbols = [stock['symbol'] for stock in swing_stocks[:count] if stock.get('symbol')]
        return symbols

