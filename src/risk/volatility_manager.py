"""
Market Volatility Manager
Monitors market conditions and pauses/resumes trading during high volatility periods
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import pytz
import requests
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

class MarketCondition(Enum):
    """Market condition states"""
    NORMAL = "normal"
    VOLATILE = "volatile"
    EXTREME = "extreme"
    NEWS_DRIVEN = "news_driven"
    ECONOMIC_DATA = "economic_data"
    CLOSED = "closed"

class PauseReason(Enum):
    """Reasons for trading pause"""
    HIGH_VOLATILITY = "high_volatility"
    ECONOMIC_RELEASE = "economic_release"
    MAJOR_NEWS = "major_news"
    MARKET_STRESS = "market_stress"
    SYSTEM_RISK = "system_risk"
    MANUAL = "manual"

@dataclass
class VolatilityMetrics:
    """Current volatility metrics"""
    vix_level: float
    market_volatility: float
    index_volatility: float
    volume_spike: float
    price_range: float
    momentum_shift: float
    timestamp: datetime

@dataclass
class MarketEvent:
    """Significant market event"""
    event_type: str
    description: str
    impact_level: str  # 'low', 'medium', 'high', 'critical'
    timestamp: datetime
    expected_duration: timedelta
    affected_sectors: List[str]

class VolatilityManager:
    """
    Manages trading activity based on market volatility and conditions
    """
    
    def __init__(self):
        self.et_timezone = pytz.timezone('America/New_York')
        
        # Volatility thresholds
        self.thresholds = {
            'vix_normal': 20.0,
            'vix_volatile': 30.0,
            'vix_extreme': 40.0,
            'market_volatility_normal': 1.5,
            'market_volatility_volatile': 2.5,
            'market_volatility_extreme': 4.0,
            'volume_spike_threshold': 3.0,  # 3x normal volume
            'price_range_threshold': 3.0,  # 3% intraday range
        }
        
        # Current state
        self.current_condition = MarketCondition.NORMAL
        self.trading_paused = False
        self.pause_reason = None
        self.pause_start_time = None
        
        # Tracking
        self.volatility_history = []
        self.market_events = []
        self.pause_history = []
        
        # Economic calendar (simplified)
        self.economic_events = self._load_economic_calendar()
        
        # News sources (would need API keys for real implementation)
        self.news_sources = {
            'reuters': 'https://newsapi.org/v2/everything',
            'benzinga': 'https://api.benzinga.com/api/v2/news',
            'finviz': 'https://finnhub.io/api/v1/news'
        }
    
    def _load_economic_calendar(self) -> List[Dict]:
        """Load high-impact economic events calendar"""
        # Simplified calendar - in production would use economic data API
        return [
            {'name': 'FOMC Rate Decision', 'impact': 'critical', 'time': '14:00', 'days': ['wed']},
            {'name': 'CPI Data Release', 'impact': 'high', 'time': '08:30', 'days': ['tue', 'wed']},
            {'name': 'Non-Farm Payrolls', 'impact': 'critical', 'time': '08:30', 'days': ['fri']},
            {'name': 'GDP Report', 'impact': 'high', 'time': '08:30', 'days': ['thu']},
            {'name': 'Retail Sales', 'impact': 'medium', 'time': '08:30', 'days': ['tue']},
            {'name': 'Consumer Confidence', 'impact': 'medium', 'time': '10:00', 'days': ['tue']},
            {'name': 'ISM Manufacturing', 'impact': 'medium', 'time': '10:00', 'days': ['mon']},
            {'name': 'ADP Employment', 'impact': 'medium', 'time': '08:15', 'days': ['wed']},
            {'name': 'Unemployment Claims', 'impact': 'medium', 'time': '08:30', 'days': ['thu']},
        ]
    
    def check_market_conditions(self, market_data: Dict) -> MarketCondition:
        """
        Check current market conditions and determine if trading should be paused
        
        Args:
            market_data: Current market data including indices, volatility, etc.
            
        Returns:
            Current market condition
        """
        try:
            # Calculate volatility metrics
            volatility_metrics = self._calculate_volatility_metrics(market_data)
            self.volatility_history.append(volatility_metrics)
            
            # Keep history manageable
            if len(self.volatility_history) > 100:
                self.volatility_history = self.volatility_history[-100:]
            
            # Check for scheduled economic events
            economic_event = self._check_economic_events()
            if economic_event:
                self._handle_economic_event(economic_event)
                return MarketCondition.ECONOMIC_DATA
            
            # Check for major news events
            news_events = self._check_major_news()
            if news_events:
                self._handle_news_events(news_events)
                return MarketCondition.NEWS_DRIVEN
            
            # Assess overall market condition
            condition = self._assess_market_condition(volatility_metrics)
            self.current_condition = condition
            
            # Determine if trading should be paused
            self._evaluate_trading_pause(condition, volatility_metrics)
            
            return condition
            
        except Exception as e:
            logger.error(f"Error checking market conditions: {e}")
            return MarketCondition.NORMAL
    
    def _calculate_volatility_metrics(self, market_data: Dict) -> VolatilityMetrics:
        """Calculate current volatility metrics"""
        try:
            # VIX level (if available)
            vix_level = market_data.get('vix', 20.0)
            
            # Market volatility (based on major indices)
            spy_volatility = self._calculate_index_volatility(market_data.get('spy_data', {}))
            qqq_volatility = self._calculate_index_volatility(market_data.get('qqq_data', {}))
            index_volatility = (spy_volatility + qqq_volatility) / 2
            
            # Volume spike detection
            volume_spike = self._calculate_volume_spike(market_data)
            
            # Price range analysis
            price_range = self._calculate_price_range(market_data)
            
            # Momentum shift detection
            momentum_shift = self._calculate_momentum_shift(market_data)
            
            return VolatilityMetrics(
                vix_level=vix_level,
                market_volatility=index_volatility,
                index_volatility=index_volatility,
                volume_spike=volume_spike,
                price_range=price_range,
                momentum_shift=momentum_shift,
                timestamp=datetime.now(self.et_timezone)
            )
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {e}")
            # Return default metrics
            return VolatilityMetrics(
                vix_level=20.0,
                market_volatility=1.0,
                index_volatility=1.0,
                volume_spike=1.0,
                price_range=1.0,
                momentum_shift=0.0,
                timestamp=datetime.now(self.et_timezone)
            )
    
    def _calculate_index_volatility(self, index_data: Dict) -> float:
        """Calculate volatility for a major index"""
        try:
            if not index_data or 'historical_data' not in index_data:
                return 1.0
            
            historical_data = index_data['historical_data']
            if len(historical_data) < 20:
                return 1.0
            
            # Calculate daily returns volatility
            returns = historical_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating index volatility: {e}")
            return 1.0
    
    def _calculate_volume_spike(self, market_data: Dict) -> float:
        """Calculate volume spike compared to average"""
        try:
            current_volume = market_data.get('total_volume', 0)
            avg_volume = market_data.get('avg_volume', 1)
            
            if avg_volume > 0:
                return current_volume / avg_volume
            return 1.0
            
        except Exception as e:
            logger.error(f"Error calculating volume spike: {e}")
            return 1.0
    
    def _calculate_price_range(self, market_data: Dict) -> float:
        """Calculate intraday price range percentage"""
        try:
            high = market_data.get('high', 0)
            low = market_data.get('low', 0)
            
            if low > 0:
                return ((high - low) / low) * 100
            return 1.0
            
        except Exception as e:
            logger.error(f"Error calculating price range: {e}")
            return 1.0
    
    def _calculate_momentum_shift(self, market_data: Dict) -> float:
        """Calculate momentum shift indicator"""
        try:
            # Simple momentum shift based on recent price changes
            recent_changes = market_data.get('recent_changes', [])
            if len(recent_changes) < 10:
                return 0.0
            
            # Compare first half to second half
            first_half = recent_changes[:5]
            second_half = recent_changes[5:]
            
            first_avg = np.mean(first_half)
            second_avg = np.mean(second_half)
            
            momentum_shift = abs(second_avg - first_avg)
            return momentum_shift
            
        except Exception as e:
            logger.error(f"Error calculating momentum shift: {e}")
            return 0.0
    
    def _check_economic_events(self) -> Optional[MarketEvent]:
        """Check for scheduled economic events"""
        try:
            current_time = datetime.now(self.et_timezone)
            current_day = current_time.strftime('%a').lower()
            current_time_str = current_time.strftime('%H:%M')
            
            for event in self.economic_events:
                if current_day in event['days']:
                    # Check if we're within 30 minutes of the event
                    event_time = datetime.strptime(event['time'], '%H:%M').time()
                    event_datetime = current_time.replace(hour=event_time.hour, minute=event_time.minute, second=0, microsecond=0)
                    
                    time_diff = abs((current_time - event_datetime).total_seconds() / 60)  # minutes
                    
                    if time_diff <= 30:  # Within 30 minutes window
                        return MarketEvent(
                            event_type='economic_release',
                            description=f"{event['name']} release",
                            impact_level=event['impact'],
                            timestamp=current_time,
                            expected_duration=timedelta(minutes=30),
                            affected_sectors=['all']
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking economic events: {e}")
            return None
    
    def _check_major_news(self) -> List[MarketEvent]:
        """Check for major news events (simplified - would need news API)"""
        # In production, this would integrate with news APIs
        # For now, return empty list
        return []
    
    def _handle_economic_event(self, event: MarketEvent):
        """Handle scheduled economic event"""
        logger.info(f"Economic event detected: {event.description}")
        
        if event.impact_level in ['high', 'critical']:
            self.pause_trading(PauseReason.ECONOMIC_RELEASE, f"Economic event: {event.description}")
        
        self.market_events.append(event)
    
    def _handle_news_events(self, events: List[MarketEvent]):
        """Handle major news events"""
        for event in events:
            logger.info(f"Major news event: {event.description}")
            
            if event.impact_level in ['high', 'critical']:
                self.pause_trading(PauseReason.MAJOR_NEWS, f"Major news: {event.description}")
        
        self.market_events.extend(events)
    
    def _assess_market_condition(self, metrics: VolatilityMetrics) -> MarketCondition:
        """Assess overall market condition based on metrics"""
        try:
            # Check for extreme conditions first
            if (metrics.vix_level > self.thresholds['vix_extreme'] or
                metrics.market_volatility > self.thresholds['market_volatility_extreme']):
                return MarketCondition.EXTREME
            
            # Check for volatile conditions
            if (metrics.vix_level > self.thresholds['vix_volatile'] or
                metrics.market_volatility > self.thresholds['market_volatility_volatile'] or
                metrics.volume_spike > self.thresholds['volume_spike_threshold'] or
                metrics.price_range > self.thresholds['price_range_threshold']):
                return MarketCondition.VOLATILE
            
            # Default to normal
            return MarketCondition.NORMAL
            
        except Exception as e:
            logger.error(f"Error assessing market condition: {e}")
            return MarketCondition.NORMAL
    
    def _evaluate_trading_pause(self, condition: MarketCondition, metrics: VolatilityMetrics):
        """Evaluate if trading should be paused"""
        try:
            # Don't pause if already paused for same reason
            if self.trading_paused:
                return
            
            # Check conditions for pause
            if condition == MarketCondition.EXTREME:
                self.pause_trading(PauseReason.HIGH_VOLATILITY, "Extreme market volatility detected")
            elif condition == MarketCondition.VOLATILE:
                # Additional checks for volatile conditions
                if metrics.momentum_shift > 5.0:  # Large momentum shift
                    self.pause_trading(PauseReason.MARKET_STRESS, "High momentum shift detected")
                elif metrics.volume_spike > 5.0:  # Extreme volume spike
                    self.pause_trading(PauseReason.MARKET_STRESS, "Extreme volume spike detected")
            
        except Exception as e:
            logger.error(f"Error evaluating trading pause: {e}")
    
    def pause_trading(self, reason: PauseReason, description: str = ""):
        """Pause trading activity"""
        if not self.trading_paused:
            self.trading_paused = True
            self.pause_reason = reason
            self.pause_start_time = datetime.now(self.et_timezone)
            
            logger.warning(f"Trading PAUSED: {reason.value} - {description}")
            
            # Record pause
            self.pause_history.append({
                'reason': reason,
                'description': description,
                'start_time': self.pause_start_time,
                'end_time': None,
                'duration': None
            })
    
    def resume_trading(self, reason: str = ""):
        """Resume trading activity"""
        if self.trading_paused:
            self.trading_paused = False
            pause_end_time = datetime.now(self.et_timezone)
            
            # Update pause history
            if self.pause_history and self.pause_history[-1]['end_time'] is None:
                self.pause_history[-1]['end_time'] = pause_end_time
                self.pause_history[-1]['duration'] = pause_end_time - self.pause_history[-1]['start_time']
            
            logger.info(f"Trading RESUMED: {reason}")
            
            self.pause_reason = None
            self.pause_start_time = None
    
    def should_trade(self) -> Tuple[bool, str]:
        """
        Check if trading should be allowed
        
        Returns:
            Tuple of (should_trade, reason)
        """
        if self.trading_paused:
            return False, f"Trading paused: {self.pause_reason.value if self.pause_reason else 'Unknown'}"
        
        if self.current_condition == MarketCondition.EXTREME:
            return False, "Extreme market conditions"
        
        if self.current_condition == MarketCondition.CLOSED:
            return False, "Market closed"
        
        return True, "Trading allowed"
    
    def get_market_summary(self) -> Dict:
        """Get comprehensive market summary"""
        try:
            latest_metrics = self.volatility_history[-1] if self.volatility_history else None
            
            summary = {
                'current_condition': self.current_condition.value,
                'trading_paused': self.trading_paused,
                'pause_reason': self.pause_reason.value if self.pause_reason else None,
                'pause_duration': str(datetime.now(self.et_timezone) - self.pause_start_time) if self.pause_start_time and self.trading_paused else None,
                'volatility_metrics': {
                    'vix_level': latest_metrics.vix_level if latest_metrics else 0,
                    'market_volatility': latest_metrics.market_volatility if latest_metrics else 0,
                    'volume_spike': latest_metrics.volume_spike if latest_metrics else 0,
                    'price_range': latest_metrics.price_range if latest_metrics else 0,
                    'momentum_shift': latest_metrics.momentum_shift if latest_metrics else 0,
                } if latest_metrics else None,
                'recent_events': [
                    {
                        'type': event.event_type,
                        'description': event.description,
                        'impact': event.impact_level,
                        'time': event.timestamp.strftime('%H:%M:%S')
                    }
                    for event in self.market_events[-5:]  # Last 5 events
                ],
                'pause_history_count': len(self.pause_history),
                'total_pause_time': self._calculate_total_pause_time()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting market summary: {e}")
            return {}
    
    def _calculate_total_pause_time(self) -> timedelta:
        """Calculate total time trading has been paused today"""
        today = datetime.now(self.et_timezone).date()
        total_pause_time = timedelta(0)
        
        for pause in self.pause_history:
            if pause['start_time'].date() == today:
                if pause['duration']:
                    total_pause_time += pause['duration']
                elif pause['end_time'] is None:  # Currently paused
                    total_pause_time += datetime.now(self.et_timezone) - pause['start_time']
        
        return total_pause_time
    
    def auto_resume_check(self) -> bool:
        """
        Check if trading should be automatically resumed
        
        Returns:
            True if trading was resumed
        """
        if not self.trading_paused:
            return False
        
        try:
            # Auto-resume conditions
            current_time = datetime.now(self.et_timezone)
            pause_duration = current_time - self.pause_start_time
            
            # Check if pause reason has expired
            if self.pause_reason == PauseReason.ECONOMIC_RELEASE:
                if pause_duration > timedelta(minutes=45):  # Resume after 45 minutes
                    self.resume_trading("Economic event window passed")
                    return True
            
            elif self.pause_reason == PauseReason.MAJOR_NEWS:
                if pause_duration > timedelta(minutes=30):  # Resume after 30 minutes
                    self.resume_trading("News event window passed")
                    return True
            
            elif self.pause_reason == PauseReason.HIGH_VOLATILITY:
                # Check if volatility has subsided
                if len(self.volatility_history) >= 2:
                    latest = self.volatility_history[-1]
                    previous = self.volatility_history[-2]
                    
                    # If volatility decreased significantly
                    if (latest.vix_level < previous.vix_level * 0.8 and
                        latest.market_volatility < previous.market_volatility * 0.8):
                        self.resume_trading("Volatility has subsided")
                        return True
            
            # Default auto-resume after 2 hours
            if pause_duration > timedelta(hours=2):
                self.resume_trading("Maximum pause duration reached")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in auto-resume check: {e}")
            return False
    
    def update_thresholds(self, new_thresholds: Dict):
        """Update volatility thresholds"""
        try:
            for key, value in new_thresholds.items():
                if key in self.thresholds:
                    old_value = self.thresholds[key]
                    self.thresholds[key] = value
                    logger.info(f"Updated threshold {key}: {old_value} -> {value}")
                else:
                    logger.warning(f"Unknown threshold key: {key}")
                    
        except Exception as e:
            logger.error(f"Error updating thresholds: {e}")
    
    def get_volatility_trend(self, periods: int = 10) -> Dict:
        """Get volatility trend over recent periods"""
        try:
            if len(self.volatility_history) < periods:
                return {'trend': 'insufficient_data'}
            
            recent_metrics = self.volatility_history[-periods:]
            
            # Calculate trends
            vix_trend = np.polyfit(range(len(recent_metrics)), [m.vix_level for m in recent_metrics], 1)[0]
            volatility_trend = np.polyfit(range(len(recent_metrics)), [m.market_volatility for m in recent_metrics], 1)[0]
            
            # Determine trend direction
            vix_direction = 'rising' if vix_trend > 0.5 else 'falling' if vix_trend < -0.5 else 'stable'
            volatility_direction = 'rising' if volatility_trend > 0.1 else 'falling' if volatility_trend < -0.1 else 'stable'
            
            return {
                'vix_trend': vix_direction,
                'volatility_trend': volatility_direction,
                'vix_change': vix_trend,
                'volatility_change': volatility_trend,
                'periods_analyzed': periods
            }
            
        except Exception as e:
            logger.error(f"Error getting volatility trend: {e}")
            return {'trend': 'error'}
