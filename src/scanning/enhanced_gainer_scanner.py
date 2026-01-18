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
from ..data.WebullUtil import fetch_top_gainers, get_rank_type

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
    
    def fetch_and_analyze_gainers(self, page_size: int = 50) -> List[GainerData]:
        """
        Fetch top gainers and apply comprehensive analysis
        
        Args:
            page_size: Number of gainers to fetch
            
        Returns:
            List of analyzed and filtered gainer data
        """
        try:
            rank_type = self.get_current_rank_type()
            raw_gainers = fetch_top_gainers(rankType=rank_type, pageSize=page_size)
            
            if not raw_gainers:
                logger.warning(f"No gainers returned for rank type: {rank_type}")
                return []
            
            analyzed_gainers = []
            for gainer in raw_gainers:
                try:
                    gainer_data = self._analyze_single_gainer(gainer, rank_type)
                    if gainer_data and self._passes_quality_filters(gainer_data):
                        analyzed_gainers.append(gainer_data)
                except Exception as e:
                    logger.warning(f"Error analyzing gainer {gainer.get('symbol', 'Unknown')}: {e}")
                    continue
            
            # Sort by quality score (highest first)
            analyzed_gainers.sort(key=lambda x: x.quality_score, reverse=True)
            
            logger.info(f"Analyzed {len(raw_gainers)} gainers, {len(analyzed_gainers)} passed quality filters")
            return analyzed_gainers
            
        except Exception as e:
            logger.error(f"Error fetching and analyzing gainers: {e}")
            return []
    
    def _analyze_single_gainer(self, gainer: Dict, rank_type: str) -> Optional[GainerData]:
        """Analyze a single gainer for quality and manipulation"""
        try:
            symbol = gainer.get('symbol', '')
            if not symbol or symbol in self.blacklist:
                return None
            
            price = float(gainer.get('price', 0))
            change_pct = float(gainer.get('changeRatio', 0)) * 100
            volume = int(gainer.get('volume', 0))
            avg_volume = int(gainer.get('avgVol10D', 0))
            market_cap = float(gainer.get('marketValue', 0))
            
            # Basic price and volume filters
            if not (self.min_price <= price <= self.max_price):
                return None
            
            if volume < self.min_volume:
                return None
            
            # Calculate metrics
            volume_ratio = volume / max(avg_volume, 1)
            surge_score = self._calculate_surge_score(change_pct, volume_ratio)
            manipulation_score = self._calculate_manipulation_score(gainer, surge_score)
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
    
    def _calculate_manipulation_score(self, gainer: Dict, surge_score: float) -> float:
        """
        Calculate manipulation detection score
        
        Higher score indicates higher manipulation risk
        
        Args:
            gainer: Raw gainer data
            surge_score: Previously calculated surge score
            
        Returns:
            Manipulation score (0-1, lower is better)
        """
        manipulation_score = 0.0
        
        # Factor 1: Extremely high price change with low volume
        change_pct = float(gainer.get('changeRatio', 0)) * 100
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
        if gainer.change_pct > 1000:  # >1000% change is likely reverse split
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
