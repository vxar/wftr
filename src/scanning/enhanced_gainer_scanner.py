"""
Enhanced Top Gainer Scanner
Supports premarket, regular hours, and after-hours scanning with intelligent filtering
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pytz
from dataclasses import dataclass
import logging
from ..data.WebullUtil import fetch_top_gainers, get_rank_type, fetch_data_array, find_tickerid_for_symbol, get_stock_quote

logger = logging.getLogger(__name__)

@dataclass
class GainerData:
    """Enhanced gainer data with analysis metrics"""
    symbol: str
    price: float
    change_pct: float
    volume: int
    avg_volume: int
    volume_ratio: float
    market_cap: float
    sector: str
    rank_type: str  # 'preMarket', '1d', 'afterMarket'
    surge_score: float = 0.0
    manipulation_score: float = 0.0
    quality_score: float = 0.0

class EnhancedGainerScanner:
    """
    Enhanced scanner that filters top gainers for quality and manipulation detection
    """
    
    def __init__(self, 
                 min_volume: int = 50000,
                 min_price: float = 0.50,
                 max_price: float = 1000.0,
                 max_manipulation_score: float = 0.7,
                 min_quality_score: float = 0.6):
        """
        Args:
            min_volume: Minimum volume threshold
            min_price: Minimum stock price
            max_price: Maximum stock price
            max_manipulation_score: Maximum manipulation score (0-1, lower is better)
            min_quality_score: Minimum quality score (0-1, higher is better)
        """
        self.min_volume = min_volume
        self.min_price = min_price
        self.max_price = max_price
        self.max_manipulation_score = max_manipulation_score
        self.min_quality_score = min_quality_score
        self.et_timezone = pytz.timezone('America/New_York')
        
        # Historical data for comparison
        self.historical_gainers = {}  # Track gainers over time
        self.blacklist = set()  # Suspicious tickers
        
    def get_current_rank_type(self) -> str:
        """Get current market session type"""
        return get_rank_type()
    
    def fetch_and_analyze_gainers(self, page_size: int = 30) -> List[GainerData]:
        """
        Fetch top gainers and return them directly without filtering
        
        Args:
            page_size: Number of gainers to fetch
            
        Returns:
            List of gainer data (no filtering applied)
        """
        try:
            rank_type = self.get_current_rank_type()
            raw_gainers = fetch_top_gainers(rankType=rank_type, pageSize=page_size)
            
            if not raw_gainers:
                logger.warning(f"No gainers returned for rank type: {rank_type}")
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
            
            logger.info(f"=== RAW TOP GAINERS ({len(gainers_list)} total) ===")
            simple_gainers = []
            for i, gainer in enumerate(gainers_list):
                if isinstance(gainer, dict):
                    # Debug: Log available fields for first few items
                    if i < 3:
                        logger.info(f"  Debug gainer {i+1} fields: {list(gainer.keys())}")
                    
                    # Get symbol with multiple possible field names
                    # Handle new API structure where symbol is nested in 'ticker' object
                    if 'ticker' in gainer and isinstance(gainer['ticker'], dict):
                        ticker_data = gainer['ticker']
                        symbol = (ticker_data.get('symbol') or 
                                 ticker_data.get('ticker') or 
                                 ticker_data.get('code') or 
                                 ticker_data.get('disSymbol') or f'TICKER_{i+1}')
                    else:
                        # Fallback to old structure
                        symbol = (gainer.get('symbol') or 
                                 gainer.get('ticker') or 
                                 gainer.get('code') or 
                                 gainer.get('disSymbol') or f'TICKER_{i+1}')
                    
                    # Ensure symbol is a string, not a dictionary
                    if not isinstance(symbol, str):
                        symbol = f'TICKER_{i+1}'
                    
                    # Get price with multiple possible field names
                    # Handle new API structure where price data might be in 'values' or 'ticker' object
                    if 'values' in gainer and isinstance(gainer['values'], dict):
                        values_data = gainer['values']
                        price = (values_data.get('price') or 
                                 values_data.get('currentPrice') or 
                                 values_data.get('last') or 
                                 values_data.get('close') or 0)
                    elif 'ticker' in gainer and isinstance(gainer['ticker'], dict):
                        ticker_data = gainer['ticker']
                        price = (ticker_data.get('price') or 
                                 ticker_data.get('currentPrice') or 
                                 ticker_data.get('last') or 
                                 ticker_data.get('close') or 0)
                    else:
                        # Fallback to old structure
                        price = (gainer.get('price') or 
                                 gainer.get('currentPrice') or 
                                 gainer.get('last') or 
                                 gainer.get('close') or 0)
                    
                    # Calculate change percentage
                    change_pct = self._calculate_change_pct(gainer, rank_type)
                    
                    logger.info(f"  {i+1}. {symbol}: {price} ({change_pct:.2f}%)")
                    
                    # Create simple GainerData object
                    # Extract data from new API structure
                    if 'ticker' in gainer and isinstance(gainer['ticker'], dict):
                        ticker_data = gainer['ticker']
                        simple_gainer = GainerData(
                            symbol=symbol,
                            price=float(price) if price else 0,
                            change_pct=change_pct,
                            volume=int(ticker_data.get('volume', 0)),
                            avg_volume=int(ticker_data.get('avgVol10D', 0)),
                            volume_ratio=1.0,
                            market_cap=float(ticker_data.get('marketValue', 0)),
                            sector=ticker_data.get('sector', ''),
                            rank_type=rank_type,
                            surge_score=0.5,
                            manipulation_score=0.3,
                            quality_score=0.7
                        )
                    else:
                        # Fallback to old structure
                        simple_gainer = GainerData(
                            symbol=symbol,
                            price=float(price) if price else 0,
                            change_pct=change_pct,
                            volume=int(gainer.get('volume', 0)),
                            avg_volume=int(gainer.get('avgVol10D', 0)),
                            volume_ratio=1.0,
                            market_cap=float(gainer.get('marketValue', 0)),
                            sector=gainer.get('sector', ''),
                            rank_type=rank_type,
                            surge_score=0.5,
                            manipulation_score=0.3,
                            quality_score=0.7
                        )
                    
                    # Basic filtering to skip problematic tickers
                    if (simple_gainer.price < 0.01 or  # Too low price (likely delisted)
                        simple_gainer.volume < 1000 or   # Very low volume
                        not simple_gainer.symbol or     # No symbol
                        len(simple_gainer.symbol) < 1 or  # Empty symbol
                        simple_gainer.symbol.startswith('TICKER_')):  # Placeholder symbol
                        logger.debug(f"Skipping problematic ticker: {simple_gainer.symbol} (price: ${simple_gainer.price}, volume: {simple_gainer.volume})")
                        continue
                    
                    simple_gainers.append(simple_gainer)
                else:
                    # Safely log non-dict data without showing full content
                    gainer_str = str(gainer)
                    if len(gainer_str) > 100:
                        gainer_str = gainer_str[:100] + "..."
                    logger.info(f"  {i+1}. Non-dict data: {type(gainer).__name__} object")
            
            logger.info("=== END GAINER LIST ===")
            return simple_gainers
            
        except Exception as e:
            logger.error(f"Error fetching gainers: {e}")
            return []
    
    def _calculate_change_pct(self, gainer: Dict, rank_type: str) -> float:
        """
        Calculate percentage change based on market session
        
        Args:
            gainer: Raw gainer data
            rank_type: Current market session ('preMarket', '1d', 'afterMarket')
            
        Returns:
            Percentage change
        """
        try:
            # Handle new API structure where data is in 'ticker' object
            if 'ticker' in gainer and isinstance(gainer['ticker'], dict):
                ticker_data = gainer['ticker']
                # Try multiple possible field names for current price
                # First check 'values' dict (new API structure)
                current_price = 0.0
                if 'values' in gainer and isinstance(gainer['values'], dict):
                    values_data = gainer['values']
                    current_price = float((values_data.get('price') or 
                                         values_data.get('currentPrice') or 
                                         values_data.get('last') or 
                                         values_data.get('close') or 0))
                
                # Fallback to ticker_data
                if current_price == 0:
                    current_price = float((ticker_data.get('price') or 
                                         ticker_data.get('currentPrice') or 
                                         ticker_data.get('last') or 
                                         ticker_data.get('close') or 0))
                
                if current_price == 0:
                    return 0.0
                
                if rank_type == 'preMarket':
                    # Pre-market: Change from previous close
                    prev_close = float((ticker_data.get('preClose') or 
                                      ticker_data.get('previousClose') or 
                                      ticker_data.get('prevClose') or 0))
                    if prev_close == 0:
                        return 0.0
                    change_pct = ((current_price - prev_close) / prev_close) * 100
                    
                elif rank_type == 'afterMarket':
                    # After-hours: Change from today's regular close at 4:00 PM
                    # First check if 'values' dict exists (new API structure)
                    regular_close = 0.0
                    if 'values' in gainer and isinstance(gainer['values'], dict):
                        values_data = gainer['values']
                        regular_close = float((values_data.get('close') or 
                                             values_data.get('regularClose') or 
                                             values_data.get('dayClose') or 0))
                    
                    # Also check ticker_data
                    if regular_close == 0:
                        regular_close = float((ticker_data.get('close') or 
                                             ticker_data.get('regularClose') or 
                                             ticker_data.get('dayClose') or 0))
                    
                    # If still not found, fetch from minute data
                    if regular_close == 0:
                        # Extract symbol from ticker_data
                        symbol = (ticker_data.get('symbol') or 
                                 ticker_data.get('ticker') or 
                                 ticker_data.get('code') or 
                                 ticker_data.get('disSymbol') or '')
                        if symbol:
                            regular_close = self._get_today_regular_close(symbol=symbol, current_price=current_price, gainer=gainer)
                    
                    if regular_close == 0:
                        return 0.0
                    change_pct = ((current_price - regular_close) / regular_close) * 100
                    
                else:  # '1d' - regular hours
                    # Regular hours: Change from previous close
                    prev_close = float((ticker_data.get('preClose') or 
                                      ticker_data.get('previousClose') or 
                                      ticker_data.get('prevClose') or 0))
                    if prev_close == 0:
                        return 0.0
                    change_pct = ((current_price - prev_close) / prev_close) * 100
            else:
                # Fallback to old structure
                # Try multiple possible field names for current price
                # First check 'values' dict (new API structure)
                current_price = 0.0
                if 'values' in gainer and isinstance(gainer['values'], dict):
                    values_data = gainer['values']
                    current_price = float((values_data.get('price') or 
                                         values_data.get('currentPrice') or 
                                         values_data.get('last') or 
                                         values_data.get('close') or 0))
                
                # Fallback to gainer directly
                if current_price == 0:
                    current_price = float((gainer.get('price') or 
                                         gainer.get('currentPrice') or 
                                         gainer.get('last') or 
                                         gainer.get('close') or 0))
                
                if current_price == 0:
                    return 0.0
                
                if rank_type == 'preMarket':
                    # Pre-market: Change from previous close
                    prev_close = float((gainer.get('preClose') or 
                                      gainer.get('previousClose') or 
                                      gainer.get('prevClose') or 0))
                    if prev_close == 0:
                        return 0.0
                    change_pct = ((current_price - prev_close) / prev_close) * 100
                    
                elif rank_type == 'afterMarket':
                    # After-hours: Change from today's regular close at 4:00 PM
                    # First check if 'values' dict exists (new API structure)
                    regular_close = 0.0
                    if 'values' in gainer and isinstance(gainer['values'], dict):
                        values_data = gainer['values']
                        regular_close = float((values_data.get('close') or 
                                             values_data.get('regularClose') or 
                                             values_data.get('dayClose') or 0))
                    
                    # Also check gainer directly
                    if regular_close == 0:
                        regular_close = float((gainer.get('close') or 
                                             gainer.get('regularClose') or 
                                             gainer.get('dayClose') or 0))
                    
                    # If still not found, fetch from minute data
                    if regular_close == 0:
                        # Extract symbol from gainer
                        symbol = (gainer.get('symbol') or 
                                 gainer.get('ticker') or 
                                 gainer.get('code') or 
                                 gainer.get('disSymbol') or '')
                        if symbol:
                            regular_close = self._get_today_regular_close(symbol=symbol, current_price=current_price, gainer=gainer)
                    
                    if regular_close == 0:
                        return 0.0
                    change_pct = ((current_price - regular_close) / regular_close) * 100
                    
                else:  # '1d' - regular hours
                    # Regular hours: Change from previous close
                    prev_close = float((gainer.get('preClose') or 
                                      gainer.get('previousClose') or 
                                      gainer.get('prevClose') or 0))
                    if prev_close == 0:
                        return 0.0
                    change_pct = ((current_price - prev_close) / prev_close) * 100
            
            return change_pct
            
        except Exception as e:
            logger.warning(f"Error calculating change pct: {e}")
            return 0.0
    
    def _get_today_regular_close(self, symbol: Optional[str] = None, current_price: float = 0.0, gainer: Optional[Dict] = None) -> float:
        """
        Get today's regular session close (4:00 PM) from minute data
        
        Args:
            symbol: Ticker symbol (if available)
            current_price: Current price (for logging)
            gainer: Gainer dict (to extract symbol if not provided)
            
        Returns:
            Regular session close price, or 0.0 if not found
        """
        try:
            # Extract symbol if not provided
            if not symbol and gainer:
                if 'ticker' in gainer and isinstance(gainer['ticker'], dict):
                    ticker_data = gainer['ticker']
                    symbol = (ticker_data.get('symbol') or 
                             ticker_data.get('ticker') or 
                             ticker_data.get('code') or 
                             ticker_data.get('disSymbol') or '')
                else:
                    symbol = (gainer.get('symbol') or 
                             gainer.get('ticker') or 
                             gainer.get('code') or '')
            
            if not symbol:
                return 0.0
            
            # First try quote API
            try:
                ticker_id = find_tickerid_for_symbol(symbol)
                if ticker_id:
                    quote = get_stock_quote(ticker_id)
                    if quote:
                        quote_close = (quote.get('close') or 
                                     quote.get('regularClose') or 
                                     quote.get('dayClose') or 0)
                        if quote_close:
                            try:
                                return float(quote_close)
                            except (ValueError, TypeError):
                                pass
            except Exception as e:
                logger.debug(f"Could not get quote for {symbol}: {e}")
            
            # If quote API doesn't have it, fetch from minute data
            try:
                ticker_id = find_tickerid_for_symbol(symbol)
                if ticker_id:
                    # Fetch today's minute data
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
                                    return regular_close
                                else:
                                    # If no regular hours data yet, try to get the last bar at exactly 16:00
                                    at_4pm = today_df[(today_df['timestamp'].dt.hour == 16) & (today_df['timestamp'].dt.minute == 0)]
                                    if not at_4pm.empty:
                                        regular_close = float(at_4pm.iloc[-1]['close'])
                                        logger.debug(f"Fetched regular close {regular_close} for {symbol} from minute data (exactly 4 PM)")
                                        return regular_close
            except Exception as e:
                logger.debug(f"Could not fetch regular close from minute data for {symbol}: {e}")
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"Error getting today's regular close for {symbol}: {e}")
            return 0.0
    
    def _analyze_single_gainer(self, gainer: Dict, rank_type: str) -> Optional[GainerData]:
        """Analyze a single gainer for quality and manipulation"""
        try:
            # Try multiple possible field names for symbol
            symbol = (gainer.get('symbol') or 
                     gainer.get('ticker') or 
                     gainer.get('code') or 
                     gainer.get('disSymbol') or 'Unknown')
            
            if not symbol or symbol == 'Unknown' or symbol in self.blacklist:
                return None
            
            # Try multiple possible field names for price
            price = float((gainer.get('price') or 
                         gainer.get('currentPrice') or 
                         gainer.get('last') or 
                         gainer.get('close') or 0))
            
            change_pct = self._calculate_change_pct(gainer, rank_type)  # Use new calculation method
            
            # Try multiple possible field names for volume
            volume = int((gainer.get('volume') or 
                         gainer.get('vol') or 
                         gainer.get('turnoverVolume') or 0))
            
            # Try multiple possible field names for average volume
            avg_volume = int((gainer.get('avgVol10D') or 
                             gainer.get('avgVolume') or 
                             gainer.get('averageVolume') or 0))
            
            # Try multiple possible field names for market cap
            market_cap = float((gainer.get('marketValue') or 
                              gainer.get('marketCap') or 
                              gainer.get('mktCap') or 0))
            
            # Basic price and volume filters
            if not (self.min_price <= price <= self.max_price):
                return None
            
            if volume < self.min_volume:
                return None
            
            # Calculate metrics
            volume_ratio = volume / max(avg_volume, 1)
            surge_score = self._calculate_surge_score(change_pct, volume_ratio)
            manipulation_score = self._calculate_manipulation_score(gainer, surge_score, rank_type)  # Pass rank_type
            quality_score = self._calculate_quality_score(gainer, surge_score, manipulation_score)
            
            return GainerData(
                symbol=symbol,
                price=price,
                change_pct=change_pct,
                volume=volume,
                avg_volume=avg_volume,
                volume_ratio=volume_ratio,
                market_cap=market_cap,
                sector=gainer.get('sector', ''),
                rank_type=rank_type,
                surge_score=surge_score,
                manipulation_score=manipulation_score,
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.warning(f"Error analyzing gainer data: {e}")
            return None
    
    def _calculate_surge_score(self, change_pct: float, volume_ratio: float) -> float:
        """
        Calculate surge score based on price change and volume
        
        Args:
            change_pct: Percentage change
            volume_ratio: Current volume / average volume
            
        Returns:
            Surge score (0-1, higher is better)
        """
        # Normalize change percentage (0-100% -> 0-1)
        price_score = min(change_pct / 50.0, 1.0)  # Cap at 50% for normalization
        
        # Normalize volume ratio (log scale)
        volume_score = min(np.log(volume_ratio + 1) / np.log(100), 1.0)  # Cap at 100x volume
        
        # Weighted combination
        surge_score = (price_score * 0.6) + (volume_score * 0.4)
        return surge_score
    
    def _calculate_manipulation_score(self, gainer: Dict, surge_score: float, rank_type: str) -> float:
        """
        Calculate manipulation detection score
        
        Higher score indicates higher manipulation risk
        
        Args:
            gainer: Raw gainer data
            surge_score: Previously calculated surge score
            rank_type: Current market session
            
        Returns:
            Manipulation score (0-1, lower is better)
        """
        manipulation_score = 0.0
        
        # Factor 1: Extremely high price change with low volume
        change_pct = self._calculate_change_pct(gainer, rank_type)  # Use new calculation
        volume_ratio = int(gainer.get('volume', 0)) / max(int(gainer.get('avgVol10D', 0)), 1)
        
        if change_pct > 100 and volume_ratio < 5:
            manipulation_score += 0.4
        elif change_pct > 200 and volume_ratio < 10:
            manipulation_score += 0.6
        
        # Factor 2: Very low market cap with huge move
        market_cap = float(gainer.get('marketValue', 0))
        if market_cap < 50e6 and change_pct > 50:  # <$50M market cap
            manipulation_score += 0.3
        
        # Factor 3: Suspicious trading patterns (check if we have historical data)
        symbol = gainer.get('symbol', '')
        if symbol in self.historical_gainers:
            recent_activity = self.historical_gainers[symbol]
            # Check for repeated pump-dump patterns
            if recent_activity.get('pump_dump_count', 0) > 2:
                manipulation_score += 0.5
        
        # Factor 4: Time-based manipulation (premarket/afterhours extreme moves)
        rank_type = self.get_current_rank_type()
        if rank_type in ['preMarket', 'afterMarket'] and change_pct > 100:
            manipulation_score += 0.2
        
        # Factor 5: Low float stocks (hard to detect from Webull data, use proxy)
        if market_cap < 100e6 and volume_ratio > 50:  # Small cap with huge volume
            manipulation_score += 0.2
        
        return min(manipulation_score, 1.0)
    
    def _calculate_quality_score(self, gainer: Dict, surge_score: float, manipulation_score: float) -> float:
        """
        Calculate overall quality score
        
        Args:
            gainer: Raw gainer data
            surge_score: Surge strength score
            manipulation_score: Manipulation risk score
            
        Returns:
            Quality score (0-1, higher is better)
        """
        # Base quality from surge score
        quality = surge_score * 0.4
        
        # Penalty for manipulation risk
        quality -= manipulation_score * 0.5
        
        # Bonus for legitimate indicators
        market_cap = float(gainer.market_cap)
        volume_ratio = int(gainer.volume) / max(int(gainer.avg_volume), 1)
        
        # Market cap quality (prefer established companies)
        if market_cap > 1e9:  # >$1B
            quality += 0.2
        elif market_cap > 100e6:  # >$100M
            quality += 0.1
        
        # Volume quality (sustained high volume is better than spike)
        if 5 <= volume_ratio <= 20:  # Reasonable volume increase
            quality += 0.1
        elif volume_ratio > 50:  # Extreme volume might be suspicious
            quality -= 0.1
        
        # Sector diversity (avoid certain speculative sectors)
        sector = gainer.get('sector', '').lower()
        speculative_sectors = ['cryptocurrency', 'blockchain', 'marijuana', 'penny stocks']
        if any(spec in sector for spec in speculative_sectors):
            quality -= 0.2
        
        return max(min(quality, 1.0), 0.0)
    
    def _passes_quality_filters(self, gainer: GainerData) -> bool:
        """Check if gainer passes all quality filters"""
        if gainer.manipulation_score > self.max_manipulation_score:
            return False
        
        if gainer.quality_score < self.min_quality_score:
            return False
        
        # Additional sanity checks
        if gainer.change_pct > 5000:  # >5000% change is likely reverse split (increased from 1000%)
            return False
        
        return True
    
    def update_historical_data(self, analyzed_gainers: List[GainerData]):
        """Update historical data for pattern detection"""
        current_time = datetime.now(self.et_timezone)
        
        for gainer in analyzed_gainers:
            symbol = gainer.symbol
            
            if symbol not in self.historical_gainers:
                self.historical_gainers[symbol] = {
                    'first_seen': current_time,
                    'appearances': 0,
                    'pump_dump_count': 0,
                    'last_change_pct': 0
                }
            
            # Update appearance count
            self.historical_gainers[symbol]['appearances'] += 1
            self.historical_gainers[symbol]['last_seen'] = current_time
            
            # Detect potential pump-dump patterns
            last_change = self.historical_gainers[symbol]['last_change_pct']
            if last_change > 50 and gainer.change_pct < -20:  # Big gain followed by big loss
                self.historical_gainers[symbol]['pump_dump_count'] += 1
            
            self.historical_gainers[symbol]['last_change_pct'] = gainer.change_pct
    
    def add_to_blacklist(self, symbol: str, reason: str = ""):
        """Add suspicious ticker to blacklist"""
        self.blacklist.add(symbol)
        logger.info(f"Added {symbol} to blacklist. Reason: {reason}")
    
    def get_top_quality_gainers(self, max_count: int = 20) -> List[GainerData]:
        """
        Get top quality gainers for trading
        
        Args:
            max_count: Maximum number of gainers to return
            
        Returns:
            List of top quality gainers
        """
        analyzed_gainers = self.fetch_and_analyze_gainers(page_size=max_count * 2)
        self.update_historical_data(analyzed_gainers)
        
        # Return top gainers by quality score
        return analyzed_gainers[:max_count]
    
    def get_session_summary(self) -> Dict:
        """Get summary of current scanning session"""
        return {
            'current_session': self.get_current_rank_type(),
            'blacklisted_count': len(self.blacklist),
            'tracked_symbols': len(self.historical_gainers),
            'min_quality_threshold': self.min_quality_score,
            'max_manipulation_threshold': self.max_manipulation_score
        }
