"""
Stock Discovery Module
Identifies trading opportunities from multiple sources:
- Webull top gainers
- News-driven stocks
- Unusual volume
- Most active stocks
- Breakout candidates
"""
import logging
from typing import List, Dict, Set, Optional
from datetime import datetime, timedelta
import requests
import json

logger = logging.getLogger(__name__)


class StockDiscovery:
    """Discovers trading opportunities from multiple sources"""
    
    def __init__(self, webull_api):
        """
        Initialize stock discovery
        
        Args:
            webull_api: WebullDataAPI instance
        """
        self.api = webull_api
        self._ticker_id_cache = {}
    
    def _get_ticker_id(self, symbol: str) -> Optional[int]:
        """Get ticker ID for a symbol, with caching"""
        if symbol in self._ticker_id_cache:
            return self._ticker_id_cache[symbol]
        
        try:
            from data.WebullUtil import find_tickerid_for_symbol
            ticker_id = find_tickerid_for_symbol(symbol)
            if ticker_id:
                self._ticker_id_cache[symbol] = ticker_id
            return ticker_id
        except Exception as e:
            logger.error(f"Error getting ticker ID for {symbol}: {e}")
            return None
    
    def get_stocks_from_news(self, max_stocks: int = 20) -> List[str]:
        """
        Get stocks mentioned in recent news that might have trading opportunities
        
        Args:
            max_stocks: Maximum number of stocks to return
            
        Returns:
            List of ticker symbols
        """
        stocks = []
        try:
            # Webull news API endpoint
            url = "https://quotes-gw.webullfintech.com/api/information/news/list"
            headers = {
                "device-type": "Web",
                "did": "xtw0doz2stnl2xghaa0hnba6h7kkslni",
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 Edg/128.0.0.0",
                "ver": "4.9.5"
            }
            
            # Get recent news (last 24 hours)
            params = {
                "pageIndex": 1,
                "pageSize": 50,
                "regionId": 6  # US market
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract tickers from news
            seen_tickers = set()
            if 'data' in data and 'items' in data['data']:
                for item in data['data']['items']:
                    # Check if news item has ticker information
                    if 'ticker' in item:
                        ticker_info = item['ticker']
                        if isinstance(ticker_info, dict):
                            symbol = ticker_info.get('symbol') or ticker_info.get('disSymbol', '')
                        else:
                            symbol = str(ticker_info)
                        
                        if symbol and symbol not in seen_tickers:
                            seen_tickers.add(symbol)
                            stocks.append(symbol)
                            
                            if len(stocks) >= max_stocks:
                                break
            
            logger.info(f"Found {len(stocks)} stocks from news: {stocks[:10]}")
            
        except Exception as e:
            logger.warning(f"Error fetching stocks from news: {e}")
        
        return stocks[:max_stocks]
    
    def get_most_active_stocks(self, max_stocks: int = 20) -> List[str]:
        """
        Get most actively traded stocks (by volume)
        
        Args:
            max_stocks: Maximum number of stocks to return
            
        Returns:
            List of ticker symbols
        """
        stocks = []
        try:
            # Webull most active API
            url = "https://quotes-gw.webullfintech.com/api/wlas/ranking/mostActive"
            headers = {
                "device-type": "Web",
                "did": "xtw0doz2stnl2xghaa0hnba6h7kkslni",
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 Edg/128.0.0.0",
                "ver": "4.9.5"
            }
            
            params = {
                "regionId": 6,
                "pageIndex": 1,
                "pageSize": max_stocks
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data:
                for item in data['data']:
                    ticker_info = item.get('ticker', {})
                    if isinstance(ticker_info, dict):
                        symbol = ticker_info.get('symbol') or ticker_info.get('disSymbol', '')
                    else:
                        symbol = str(ticker_info)
                    
                    if symbol:
                        stocks.append(symbol)
            
            logger.info(f"Found {len(stocks)} most active stocks: {stocks[:10]}")
            
        except Exception as e:
            logger.warning(f"Error fetching most active stocks: {e}")
        
        return stocks[:max_stocks]
    
    def get_unusual_volume_stocks(self, max_stocks: int = 20) -> List[str]:
        """
        Get stocks with unusual volume (volume spike)
        
        Args:
            max_stocks: Maximum number of stocks to return
            
        Returns:
            List of ticker symbols
        """
        stocks = []
        try:
            # Get top gainers and check for unusual volume
            gainers = self.api.get_top_gainers(page_size=50)
            
            for stock in gainers:
                symbol = stock.get('symbol', '')
                volume = stock.get('volume', 0)
                change_ratio = stock.get('change_ratio', 0) or stock.get('changeRatio', 0)
                
                # Filter for stocks with high volume and positive movement
                if symbol and volume > 1000000 and change_ratio > 0:
                    stocks.append(symbol)
                    
                    if len(stocks) >= max_stocks:
                        break
            
            logger.info(f"Found {len(stocks)} unusual volume stocks: {stocks[:10]}")
            
        except Exception as e:
            logger.warning(f"Error fetching unusual volume stocks: {e}")
        
        return stocks[:max_stocks]
    
    def get_breakout_candidates(self, max_stocks: int = 20) -> List[str]:
        """
        Get stocks that are breaking out (price and volume)
        
        Args:
            max_stocks: Maximum number of stocks to return
            
        Returns:
            List of ticker symbols
        """
        stocks = []
        try:
            # Get top gainers with high volume
            gainers = self.api.get_top_gainers(page_size=50)
            
            for stock in gainers:
                symbol = stock.get('symbol', '')
                change_ratio = stock.get('change_ratio', 0) or stock.get('changeRatio', 0)
                volume = stock.get('volume', 0)
                price = stock.get('price', 0)
                
                # Filter for breakout candidates:
                # - Price increase > 3%
                # - Volume > 500K
                # - Price between $0.50 and $100
                if (symbol and 
                    change_ratio > 3.0 and 
                    volume > 500000 and
                    0.50 <= price <= 100.0):
                    stocks.append(symbol)
                    
                    if len(stocks) >= max_stocks:
                        break
            
            logger.info(f"Found {len(stocks)} breakout candidates: {stocks[:10]}")
            
        except Exception as e:
            logger.warning(f"Error fetching breakout candidates: {e}")
        
        return stocks[:max_stocks]
    
    def get_top_losers_reversal_candidates(self, max_stocks: int = 10) -> List[str]:
        """
        Get top losers that might reverse (oversold bounce candidates)
        
        Args:
            max_stocks: Maximum number of stocks to return
            
        Returns:
            List of ticker symbols
        """
        stocks = []
        try:
            # Webull top losers API
            url = "https://quotes-gw.webullfintech.com/api/wlas/ranking/topLosers"
            headers = {
                "device-type": "Web",
                "did": "xtw0doz2stnl2xghaa0hnba6h7kkslni",
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 Edg/128.0.0.0",
                "ver": "4.9.5"
            }
            
            params = {
                "regionId": 6,
                "rankType": "1d",
                "pageIndex": 1,
                "pageSize": max_stocks * 2  # Get more to filter
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data:
                for item in data['data']:
                    ticker_info = item.get('ticker', {})
                    if isinstance(ticker_info, dict):
                        symbol = ticker_info.get('symbol') or ticker_info.get('disSymbol', '')
                    else:
                        symbol = str(ticker_info)
                    
                    change_ratio = item.get('changeRatio', 0) or item.get('change_ratio', 0)
                    volume = item.get('volume', 0)
                    price = item.get('close', 0) or item.get('price', 0)
                    
                    # Filter for potential reversal candidates:
                    # - Down 5-15% (not too extreme)
                    # - High volume (institutional interest)
                    # - Price > $0.50
                    if (symbol and 
                        -15.0 <= change_ratio <= -5.0 and
                        volume > 500000 and
                        price > 0.50):
                        stocks.append(symbol)
                        
                        if len(stocks) >= max_stocks:
                            break
            
            logger.info(f"Found {len(stocks)} reversal candidates: {stocks[:10]}")
            
        except Exception as e:
            logger.warning(f"Error fetching reversal candidates: {e}")
        
        return stocks[:max_stocks]
    
    def discover_stocks(self, 
                       include_gainers: bool = True,
                       include_news: bool = True,
                       include_most_active: bool = True,
                       include_unusual_volume: bool = True,
                       include_breakouts: bool = True,
                       include_reversals: bool = False,
                       max_total: int = 30) -> List[str]:
        """
        Discover stocks from multiple sources and combine them
        
        Args:
            include_gainers: Include top gainers
            include_news: Include news-driven stocks
            include_most_active: Include most active stocks
            include_unusual_volume: Include unusual volume stocks
            include_breakouts: Include breakout candidates
            include_reversals: Include reversal candidates (default: False, more risky)
            max_total: Maximum total number of unique stocks to return
            
        Returns:
            List of unique ticker symbols
        """
        all_stocks = set()
        
        # Collect from all sources
        if include_gainers:
            try:
                gainers = self.api.get_stock_list_from_gainers(count=20)
                all_stocks.update(gainers)
                logger.info(f"Added {len(gainers)} stocks from top gainers")
            except Exception as e:
                logger.warning(f"Error getting gainers: {e}")
        
        if include_news:
            try:
                news_stocks = self.get_stocks_from_news(max_stocks=15)
                all_stocks.update(news_stocks)
                logger.info(f"Added {len(news_stocks)} stocks from news")
            except Exception as e:
                logger.warning(f"Error getting news stocks: {e}")
        
        if include_most_active:
            try:
                active_stocks = self.get_most_active_stocks(max_stocks=15)
                all_stocks.update(active_stocks)
                logger.info(f"Added {len(active_stocks)} stocks from most active")
            except Exception as e:
                logger.warning(f"Error getting most active stocks: {e}")
        
        if include_unusual_volume:
            try:
                volume_stocks = self.get_unusual_volume_stocks(max_stocks=15)
                all_stocks.update(volume_stocks)
                logger.info(f"Added {len(volume_stocks)} stocks from unusual volume")
            except Exception as e:
                logger.warning(f"Error getting unusual volume stocks: {e}")
        
        if include_breakouts:
            try:
                breakout_stocks = self.get_breakout_candidates(max_stocks=15)
                all_stocks.update(breakout_stocks)
                logger.info(f"Added {len(breakout_stocks)} stocks from breakouts")
            except Exception as e:
                logger.warning(f"Error getting breakout stocks: {e}")
        
        if include_reversals:
            try:
                reversal_stocks = self.get_top_losers_reversal_candidates(max_stocks=10)
                all_stocks.update(reversal_stocks)
                logger.info(f"Added {len(reversal_stocks)} stocks from reversals")
            except Exception as e:
                logger.warning(f"Error getting reversal stocks: {e}")
        
        # Convert to list and limit
        result = list(all_stocks)[:max_total]
        logger.info(f"Total unique stocks discovered: {len(result)}")
        
        return result

