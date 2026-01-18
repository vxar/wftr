"""
Advanced Manipulation Detection System
Detects price manipulation, pump-and-dump schemes, and false breakouts
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import re

logger = logging.getLogger(__name__)

class ManipulationType(Enum):
    """Types of market manipulation"""
    PUMP_AND_DUMP = "pump_and_dump"
    WASH_TRADING = "wash_trading"
    PAINTING_TAPE = "painting_tape"
    SPOOFING = "spoofing"
    LAYERING = "layering"
    REVERSE_SPLIT = "reverse_split"
    NEWS_MANIPULATION = "news_manipulation"
    PREMARKET_RAMP = "premarket_ramp"

@dataclass
class ManipulationSignal:
    """Detected manipulation signal"""
    manipulation_type: ManipulationType
    confidence: float  # 0-1, higher confidence
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    indicators: Dict[str, float]
    timestamp: datetime
    recommended_action: str  # 'avoid', 'caution', 'monitor'

class ManipulationDetector:
    """
    Advanced system to detect various forms of market manipulation
    """
    
    def __init__(self):
        # Detection thresholds
        self.thresholds = {
            'extreme_volume_ratio': 50.0,  # 50x average volume
            'extreme_price_change': 100.0,  # 100% price change
            'low_float_suspicion': 1000000,  # <$1M daily volume value
            'spread_threshold': 0.05,  # 5% bid-ask spread
            'volatility_spike': 10.0,  # 10x normal volatility
            'time_decay_factor': 0.1  # How quickly signals decay
        }
        
        # Historical tracking
        self.suspicious_tickers = {}  # Track suspicious activity over time
        self.manipulation_history = []  # Store detected manipulations
        
    def analyze_for_manipulation(self, 
                                ticker: str,
                                current_data: Dict,
                                historical_data: pd.DataFrame,
                                market_context: Dict) -> List[ManipulationSignal]:
        """
        Comprehensive manipulation analysis
        
        Args:
            ticker: Stock ticker
            current_data: Current market data
            historical_data: Historical price/volume data
            market_context: Overall market context
            
        Returns:
            List of detected manipulation signals
        """
        signals = []
        
        try:
            # 1. Pump and Dump Detection
            pump_dump_signal = self._detect_pump_and_dump(ticker, current_data, historical_data)
            if pump_dump_signal:
                signals.append(pump_dump_signal)
            
            # 2. Reverse Split Detection
            reverse_split_signal = self._detect_reverse_split(ticker, current_data, historical_data)
            if reverse_split_signal:
                signals.append(reverse_split_signal)
            
            # 3. Premarket Ramp Detection
            premarket_signal = self._detect_premarket_ramp(ticker, current_data, historical_data)
            if premarket_signal:
                signals.append(premarket_signal)
            
            # 4. Volume Anomaly Detection
            volume_signal = self._detect_volume_anomalies(ticker, current_data, historical_data)
            if volume_signal:
                signals.append(volume_signal)
            
            # 5. Price Pattern Anomalies
            pattern_signal = self._detect_pattern_anomalies(ticker, current_data, historical_data)
            if pattern_signal:
                signals.append(pattern_signal)
            
            # 6. Wash Trading Detection
            wash_signal = self._detect_wash_trading(ticker, current_data, historical_data)
            if wash_signal:
                signals.append(wash_signal)
            
            # Update suspicious ticker tracking
            self._update_suspicious_tracking(ticker, signals)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in manipulation analysis for {ticker}: {e}")
            return []
    
    def _detect_pump_and_dump(self, 
                            ticker: str,
                            current_data: Dict,
                            historical_data: pd.DataFrame) -> Optional[ManipulationSignal]:
        """Detect pump and dump patterns"""
        try:
            if len(historical_data) < 20:
                return None
            
            # Key indicators
            current_price = current_data.get('price', 0)
            volume_ratio = current_data.get('volume_ratio', 1.0)
            price_change_pct = current_data.get('change_pct', 0)
            
            # Check for extreme price increase with high volume
            if price_change_pct > 50 and volume_ratio > 10:
                # Look for classic pump pattern: rapid rise followed by distribution
                recent_data = historical_data.tail(20)
                
                # Price acceleration
                price_changes = recent_data['close'].pct_change().abs()
                avg_price_change = price_changes.mean()
                recent_volatility = price_changes.std()
                
                # Volume pattern
                volume_pattern = recent_data['volume'].values
                volume_trend = np.polyfit(range(len(volume_pattern)), volume_pattern, 1)[0]
                
                # Calculate pump score
                pump_score = 0.0
                
                # Extreme price move
                if price_change_pct > 100:
                    pump_score += 0.3
                elif price_change_pct > 50:
                    pump_score += 0.2
                
                # Volume explosion
                if volume_ratio > 50:
                    pump_score += 0.3
                elif volume_ratio > 20:
                    pump_score += 0.2
                
                # Accelerating volume (sign of coordinated buying)
                if volume_trend > 0:
                    pump_score += 0.2
                
                # High volatility (sign of manipulation)
                if recent_volatility > avg_price_change * 3:
                    pump_score += 0.2
                
                if pump_score > 0.5:
                    severity = "critical" if pump_score > 0.8 else "high" if pump_score > 0.6 else "medium"
                    
                    return ManipulationSignal(
                        manipulation_type=ManipulationType.PUMP_AND_DUMP,
                        confidence=pump_score,
                        severity=severity,
                        description=f"Pump and dump pattern detected: {price_change_pct:.1f}% gain with {volume_ratio:.1f}x volume",
                        indicators={
                            'price_change_pct': price_change_pct,
                            'volume_ratio': volume_ratio,
                            'volatility_score': recent_volatility / avg_price_change if avg_price_change > 0 else 0,
                            'volume_trend': volume_trend
                        },
                        timestamp=datetime.now(),
                        recommended_action="avoid" if pump_score > 0.7 else "caution"
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting pump and dump for {ticker}: {e}")
            return None
    
    def _detect_reverse_split(self, 
                            ticker: str,
                            current_data: Dict,
                            historical_data: pd.DataFrame) -> Optional[ManipulationSignal]:
        """Detect reverse stock splits (often mistaken for massive gains)"""
        try:
            if len(historical_data) < 5:
                return None
            
            # Look for massive overnight price jumps
            price_change_pct = current_data.get('change_pct', 0)
            
            if price_change_pct > 200:  # >200% gain is suspicious
                # Check for reverse split characteristics
                
                # 1. Clean price ratio (common reverse split ratios)
                current_price = current_data.get('price', 0)
                prev_close = current_data.get('prev_close', 0)
                
                if prev_close > 0:
                    price_ratio = current_price / prev_close
                    
                    # Common reverse split ratios
                    common_ratios = [2, 5, 10, 15, 20, 25, 50, 100, 200, 250, 500, 1000]
                    ratio_match = min(abs(price_ratio - ratio) / ratio for ratio in common_ratios)
                    
                    split_score = 0.0
                    
                    # Clean ratio match
                    if ratio_match < 0.1:  # Within 10% of common ratio
                        split_score += 0.4
                    elif ratio_match < 0.2:  # Within 20%
                        split_score += 0.2
                    
                    # Volume characteristics (reverse splits often have normal/low volume)
                    volume_ratio = current_data.get('volume_ratio', 1.0)
                    if volume_ratio < 5:  # Not extremely high volume
                        split_score += 0.3
                    
                    # Price pattern (gap up without gradual increase)
                    if len(historical_data) >= 2:
                        last_close = historical_data.iloc[-2]['close']
                        gap_pct = (current_price - last_close) / last_close * 100
                        
                        if gap_pct > 100:  # Massive gap
                            split_score += 0.3
                    
                    if split_score > 0.5:
                        return ManipulationSignal(
                            manipulation_type=ManipulationType.REVERSE_SPLIT,
                            confidence=split_score,
                            severity="high",
                            description=f"Reverse split detected: {price_change_pct:.1f}% gain with ratio {price_ratio:.1f}:1",
                            indicators={
                                'price_change_pct': price_change_pct,
                                'price_ratio': price_ratio,
                                'ratio_match': ratio_match,
                                'volume_ratio': volume_ratio
                            },
                            timestamp=datetime.now(),
                            recommended_action="avoid"
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting reverse split for {ticker}: {e}")
            return None
    
    def _detect_premarket_ramp(self, 
                             ticker: str,
                             current_data: Dict,
                             historical_data: pd.DataFrame) -> Optional[ManipulationSignal]:
        """Detect premarket price manipulation"""
        try:
            # Check if we're in premarket
            current_time = datetime.now()
            if not (4 <= current_time.hour < 9 or current_time.hour == 9 and current_time.minute < 30):
                return None
            
            if len(historical_data) < 10:
                return None
            
            # Look for suspicious premarket patterns
            premarket_data = historical_data.tail(10)
            price_change_pct = current_data.get('change_pct', 0)
            volume_ratio = current_data.get('volume_ratio', 1.0)
            
            # Premarket manipulation indicators
            ramp_score = 0.0
            
            # Extreme premarket move
            if price_change_pct > 30:
                ramp_score += 0.3
            elif price_change_pct > 15:
                ramp_score += 0.2
            
            # Low volume but high price move (suspicious)
            if volume_ratio < 2 and price_change_pct > 10:
                ramp_score += 0.4
            
            # Steady, unnatural price climb
            price_changes = premarket_data['close'].pct_change()
            positive_changes = sum(1 for change in price_changes if change > 0)
            
            if positive_changes >= 8:  # 8+ consecutive positive moves
                ramp_score += 0.3
            
            # Volume pattern (consistent low volume)
            volume_values = premarket_data['volume'].values
            volume_consistency = 1 - (np.std(volume_values) / np.mean(volume_values)) if np.mean(volume_values) > 0 else 0
            
            if volume_consistency > 0.8:  # Very consistent volume
                ramp_score += 0.2
            
            if ramp_score > 0.5:
                severity = "high" if ramp_score > 0.7 else "medium"
                
                return ManipulationSignal(
                    manipulation_type=ManipulationType.PREMARKET_RAMP,
                    confidence=ramp_score,
                    severity=severity,
                    description=f"Premarket ramp detected: {price_change_pct:.1f}% gain with {volume_ratio:.1f}x volume",
                    indicators={
                        'price_change_pct': price_change_pct,
                        'volume_ratio': volume_ratio,
                        'positive_changes': positive_changes,
                        'volume_consistency': volume_consistency
                    },
                    timestamp=datetime.now(),
                    recommended_action="caution"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting premarket ramp for {ticker}: {e}")
            return None
    
    def _detect_volume_anomalies(self, 
                               ticker: str,
                               current_data: Dict,
                               historical_data: pd.DataFrame) -> Optional[ManipulationSignal]:
        """Detect unusual volume patterns"""
        try:
            if len(historical_data) < 20:
                return None
            
            volume_ratio = current_data.get('volume_ratio', 1.0)
            current_volume = current_data.get('volume', 0)
            price_change_pct = current_data.get('change_pct', 0)
            
            anomaly_score = 0.0
            
            # Extreme volume spike
            if volume_ratio > 100:
                anomaly_score += 0.4
            elif volume_ratio > 50:
                anomaly_score += 0.3
            elif volume_ratio > 20:
                anomaly_score += 0.2
            
            # Volume without price movement (suspicious)
            if volume_ratio > 10 and abs(price_change_pct) < 2:
                anomaly_score += 0.3
            
            # Check for block trade patterns
            recent_volumes = historical_data.tail(10)['volume'].values
            avg_recent_volume = np.mean(recent_volumes)
            
            if current_volume > avg_recent_volume * 5:
                # Check if volume is in round numbers (sign of block trades)
                volume_str = str(int(current_volume))
                if volume_str.endswith('000') or volume_str.endswith('0000'):
                    anomaly_score += 0.2
            
            # Volume pattern analysis
            volume_pattern = historical_data.tail(20)['volume'].values
            volume_std = np.std(volume_pattern)
            volume_mean = np.mean(volume_pattern)
            
            if volume_std > volume_mean * 2:  # Highly irregular volume
                anomaly_score += 0.2
            
            if anomaly_score > 0.5:
                return ManipulationSignal(
                    manipulation_type=ManipulationType.WASH_TRADING,
                    confidence=anomaly_score,
                    severity="medium",
                    description=f"Volume anomaly detected: {volume_ratio:.1f}x normal volume",
                    indicators={
                        'volume_ratio': volume_ratio,
                        'price_change_pct': price_change_pct,
                        'volume_std_ratio': volume_std / volume_mean if volume_mean > 0 else 0
                    },
                    timestamp=datetime.now(),
                    recommended_action="monitor"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting volume anomalies for {ticker}: {e}")
            return None
    
    def _detect_pattern_anomalies(self, 
                                 ticker: str,
                                 current_data: Dict,
                                 historical_data: pd.DataFrame) -> Optional[ManipulationSignal]:
        """Detect unusual price patterns"""
        try:
            if len(historical_data) < 30:
                return None
            
            pattern_score = 0.0
            current_price = current_data.get('price', 0)
            
            # 1. Paint the tape detection
            recent_prices = historical_data.tail(10)['close'].values
            price_increments = np.diff(recent_prices)
            
            # Look for consistent small increments (painting)
            positive_increments = [inc for inc in price_increments if inc > 0]
            if len(positive_increments) >= 7:
                increment_std = np.std(positive_increments)
                increment_mean = np.mean(positive_increments)
                
                if increment_std < increment_mean * 0.3:  # Very consistent increments
                    pattern_score += 0.3
            
            # 2. Spoofing detection (large orders that disappear)
            # This would require order book data, so we'll use price action proxies
            price_volatility = historical_data['close'].pct_change().tail(20).std()
            avg_volatility = historical_data['close'].pct_change().std()
            
            if price_volatility > avg_volatility * 3:
                pattern_score += 0.2
            
            # 3. Unusual price levels
            price_levels = [int(p) for p in historical_data['close'].tail(50).values]
            round_numbers = [p for p in price_levels if p % 1 == 0 and p > 1]
            
            if len(round_numbers) > len(price_levels) * 0.8:  # Mostly round numbers
                pattern_score += 0.2
            
            # 4. Gap analysis
            gaps = []
            for i in range(1, len(historical_data)):
                prev_close = historical_data.iloc[i-1]['close']
                curr_open = historical_data.iloc[i]['open']
                gap_pct = abs(curr_open - prev_close) / prev_close * 100
                if gap_pct > 5:
                    gaps.append(gap_pct)
            
            if len(gaps) > 2:  # Multiple large gaps
                pattern_score += 0.3
            
            if pattern_score > 0.5:
                return ManipulationSignal(
                    manipulation_type=ManipulationType.PAINTING_TAPE,
                    confidence=pattern_score,
                    severity="medium",
                    description="Unusual price pattern detected",
                    indicators={
                        'increment_consistency': increment_std / increment_mean if len(positive_increments) > 0 else 0,
                        'volatility_spike': price_volatility / avg_volatility if avg_volatility > 0 else 0,
                        'round_number_ratio': len(round_numbers) / len(price_levels),
                        'large_gaps': len(gaps)
                    },
                    timestamp=datetime.now(),
                    recommended_action="monitor"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting pattern anomalies for {ticker}: {e}")
            return None
    
    def _detect_wash_trading(self, 
                           ticker: str,
                           current_data: Dict,
                           historical_data: pd.DataFrame) -> Optional[ManipulationSignal]:
        """Detect wash trading patterns"""
        try:
            if len(historical_data) < 50:
                return None
            
            wash_score = 0.0
            
            # Look for circular trading patterns
            # High volume with minimal price change
            volume_ratio = current_data.get('volume_ratio', 1.0)
            price_change_pct = current_data.get('change_pct', 0)
            
            if volume_ratio > 20 and abs(price_change_pct) < 1:
                wash_score += 0.4
            
            # Price range compression with high volume
            recent_data = historical_data.tail(20)
            price_range = (recent_data['high'].max() - recent_data['low'].min()) / recent_data['low'].min()
            avg_volume = recent_data['volume'].mean()
            
            if price_range < 0.02 and avg_volume > historical_data['volume'].mean() * 5:
                wash_score += 0.3
            
            # Repeating price patterns
            prices = recent_data['close'].values
            price_repeats = len(set(prices)) / len(prices)  # Lower means more repeats
            
            if price_repeats < 0.5:  # Less than 50% unique prices
                wash_score += 0.3
            
            if wash_score > 0.5:
                return ManipulationSignal(
                    manipulation_type=ManipulationType.WASH_TRADING,
                    confidence=wash_score,
                    severity="medium",
                    description="Wash trading pattern detected",
                    indicators={
                        'volume_ratio': volume_ratio,
                        'price_change_pct': price_change_pct,
                        'price_range_pct': price_range * 100,
                        'price_repeat_ratio': price_repeats
                    },
                    timestamp=datetime.now(),
                    recommended_action="avoid"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting wash trading for {ticker}: {e}")
            return None
    
    def _update_suspicious_tracking(self, ticker: str, signals: List[ManipulationSignal]):
        """Update tracking of suspicious tickers"""
        if ticker not in self.suspicious_tickers:
            self.suspicious_tickers[ticker] = {
                'first_detected': datetime.now(),
                'detection_count': 0,
                'manipulation_types': set(),
                'max_severity': 'low',
                'last_detected': None
            }
        
        # Update tracking data
        tracking = self.suspicious_tickers[ticker]
        tracking['detection_count'] += len(signals)
        tracking['last_detected'] = datetime.now()
        
        # Update manipulation types and severity
        for signal in signals:
            tracking['manipulation_types'].add(signal.manipulation_type)
            
            # Update max severity
            severity_order = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            current_severity_level = severity_order.get(tracking['max_severity'], 0)
            signal_severity_level = severity_order.get(signal.severity, 0)
            
            if signal_severity_level > current_severity_level:
                tracking['max_severity'] = signal.severity
        
        # Store in manipulation history
        for signal in signals:
            self.manipulation_history.append({
                'ticker': ticker,
                'signal': signal,
                'detected_at': datetime.now()
            })
    
    def get_manipulation_summary(self, ticker: str) -> Dict:
        """Get manipulation analysis summary for a ticker"""
        if ticker not in self.suspicious_tickers:
            return {
                'suspicious': False,
                'detection_count': 0,
                'manipulation_types': [],
                'max_severity': 'none',
                'recommendation': 'safe'
            }
        
        tracking = self.suspicious_tickers[ticker]
        
        # Determine recommendation based on severity and frequency
        recommendation = 'safe'
        if tracking['max_severity'] in ['high', 'critical']:
            recommendation = 'avoid'
        elif tracking['max_severity'] == 'medium' or tracking['detection_count'] > 3:
            recommendation = 'caution'
        elif tracking['detection_count'] > 1:
            recommendation = 'monitor'
        
        return {
            'suspicious': True,
            'detection_count': tracking['detection_count'],
            'manipulation_types': list(tracking['manipulation_types']),
            'max_severity': tracking['max_severity'],
            'first_detected': tracking['first_detected'],
            'last_detected': tracking['last_detected'],
            'recommendation': recommendation
        }
    
    def get_blacklist(self, min_severity: str = 'high') -> List[str]:
        """Get list of tickers to blacklist based on manipulation detection"""
        severity_order = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        min_level = severity_order.get(min_severity, 3)
        
        blacklist = []
        for ticker, tracking in self.suspicious_tickers.items():
            ticker_severity = severity_order.get(tracking['max_severity'], 0)
            if ticker_severity >= min_level:
                blacklist.append(ticker)
        
        return blacklist
    
    def clear_history(self, older_than_days: int = 30):
        """Clear manipulation history older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        # Clear manipulation history
        self.manipulation_history = [
            record for record in self.manipulation_history
            if record['detected_at'] > cutoff_date
        ]
        
        # Clear suspicious ticker tracking
        tickers_to_remove = []
        for ticker, tracking in self.suspicious_tickers.items():
            if tracking['last_detected'] and tracking['last_detected'] < cutoff_date:
                tickers_to_remove.append(ticker)
        
        for ticker in tickers_to_remove:
            del self.suspicious_tickers[ticker]
        
        logger.info(f"Cleared manipulation history older than {older_than_days} days")
