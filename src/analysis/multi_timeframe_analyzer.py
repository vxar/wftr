"""
Multi-Timeframe Analysis System
Validates trading signals across 1m, 5m, and 15m timeframes
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from ..data.api_interface import DataAPI

logger = logging.getLogger(__name__)

@dataclass
class TimeframeSignal:
    """Signal from a specific timeframe"""
    timeframe: str  # '1m', '5m', '15m'
    signal_type: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0-1, signal strength
    trend_direction: str  # 'up', 'down', 'sideways'
    momentum_score: float  # 0-1, momentum strength
    volume_confirmation: bool  # Volume supports the signal
    price_position: str  # 'oversold', 'neutral', 'overbought'

@dataclass
class MultiTimeframeAnalysis:
    """Combined analysis across all timeframes"""
    ticker: str
    timestamp: datetime
    primary_signal: str  # Overall signal direction
    confidence: float  # Overall confidence (0-1)
    timeframe_signals: Dict[str, TimeframeSignal]
    trend_alignment: float  # How well trends align across timeframes (0-1)
    volume_consistency: float  # Volume consistency across timeframes (0-1)
    entry_recommendation: str  # 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'
    risk_level: str  # 'low', 'medium', 'high'
    stop_loss_distance: float  # Recommended stop loss distance (%)
    target_distance: float  # Recommended target distance (%)

class MultiTimeframeAnalyzer:
    """
    Analyzes trading signals across multiple timeframes for robust validation
    """
    
    def __init__(self, data_api: DataAPI):
        """
        Args:
            data_api: Data API instance for fetching data
        """
        self.data_api = data_api
        
        # Timeframe configurations
        self.timeframe_configs = {
            '1m': {'periods': 200, 'lookback': 20},  # 200 minutes, 20 period analysis
            '5m': {'periods': 200, 'lookback': 20},  # 200 5-minute bars (16.7 hours)
            '15m': {'periods': 200, 'lookback': 20}  # 200 15-minute bars (50 hours)
        }
    
    def analyze_multi_timeframe(self, ticker: str) -> Optional[MultiTimeframeAnalysis]:
        """
        Perform comprehensive multi-timeframe analysis
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Multi-timeframe analysis or None if insufficient data
        """
        try:
            # Fetch data for all timeframes
            timeframe_data = {}
            for timeframe in ['1m', '5m', '15m']:
                data = self._fetch_timeframe_data(ticker, timeframe)
                if data is not None:
                    timeframe_data[timeframe] = data
                else:
                    logger.warning(f"No data available for {ticker} on {timeframe} timeframe")
                    return None
            
            # Analyze each timeframe
            timeframe_signals = {}
            for timeframe, data in timeframe_data.items():
                signal = self._analyze_single_timeframe(data, timeframe)
                timeframe_signals[timeframe] = signal
            
            # Combine signals for overall analysis
            combined_analysis = self._combine_timeframe_signals(ticker, timeframe_signals)
            
            return combined_analysis
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis for {ticker}: {e}")
            return None
    
    def _fetch_timeframe_data(self, ticker: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch data for specific timeframe"""
        try:
            if timeframe == '1m':
                minutes = self.timeframe_configs['1m']['periods']
                return self.data_api.get_1min_data(ticker, minutes=minutes)
            elif timeframe == '5m':
                # For 5m data, we'll resample from 1m data
                df_1m = self.data_api.get_1min_data(ticker, minutes=1000)
                if df_1m is not None:
                    return self._resample_to_timeframe(df_1m, '5m')
            elif timeframe == '15m':
                # For 15m data, we'll resample from 1m data
                df_1m = self.data_api.get_1min_data(ticker, minutes=1200)
                if df_1m is not None:
                    return self._resample_to_timeframe(df_1m, '15m')
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching {timeframe} data for {ticker}: {e}")
            return None
    
    def _resample_to_timeframe(self, df_1m: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """Resample 1-minute data to target timeframe"""
        try:
            df_1m = df_1m.copy()
            df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'])
            df_1m.set_index('timestamp', inplace=True)
            
            # Determine resampling frequency
            if target_timeframe == '5m':
                freq = '5T'
            elif target_timeframe == '15m':
                freq = '15T'
            else:
                raise ValueError(f"Unsupported timeframe: {target_timeframe}")
            
            # Resample with OHLCV aggregation
            df_resampled = df_1m.resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # Reset index to make timestamp a column
            df_resampled = df_resampled.reset_index()
            
            return df_resampled
            
        except Exception as e:
            logger.error(f"Error resampling to {target_timeframe}: {e}")
            return pd.DataFrame()
    
    def _analyze_single_timeframe(self, df: pd.DataFrame, timeframe: str) -> TimeframeSignal:
        """Analyze a single timeframe for signals"""
        try:
            if len(df) < 50:
                return TimeframeSignal(
                    timeframe=timeframe,
                    signal_type='neutral',
                    strength=0.0,
                    trend_direction='sideways',
                    momentum_score=0.0,
                    volume_confirmation=False,
                    price_position='neutral'
                )
            
            # Calculate indicators
            df = self._calculate_timeframe_indicators(df)
            
            # Get the most recent data point
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Determine trend direction
            trend_direction = self._determine_trend_direction(df)
            
            # Calculate signal strength
            signal_type, signal_strength = self._calculate_signal_strength(current, prev, df)
            
            # Calculate momentum
            momentum_score = self._calculate_momentum_score(df)
            
            # Volume confirmation
            volume_confirmation = self._check_volume_confirmation(df)
            
            # Price position
            price_position = self._determine_price_position(current)
            
            return TimeframeSignal(
                timeframe=timeframe,
                signal_type=signal_type,
                strength=signal_strength,
                trend_direction=trend_direction,
                momentum_score=momentum_score,
                volume_confirmation=volume_confirmation,
                price_position=price_position
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {timeframe}: {e}")
            return TimeframeSignal(
                timeframe=timeframe,
                signal_type='neutral',
                strength=0.0,
                trend_direction='sideways',
                momentum_score=0.0,
                volume_confirmation=False,
                price_position='neutral'
            )
    
    def _calculate_timeframe_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the timeframe"""
        df = df.copy()
        
        # Moving averages
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # EMA
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Price position relative to moving averages
        df['price_vs_sma10'] = (df['close'] - df['sma_10']) / df['sma_10'] * 100
        df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20'] * 100
        
        return df
    
    def _determine_trend_direction(self, df: pd.DataFrame) -> str:
        """Determine overall trend direction"""
        recent = df.tail(20)
        
        # Moving average alignment
        sma_alignment = (
            (recent['sma_10'].iloc[-1] > recent['sma_20'].iloc[-1]) and
            (recent['sma_20'].iloc[-1] > recent['sma_50'].iloc[-1])
        )
        
        # Price momentum
        price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
        
        # MACD direction
        macd_bullish = recent['macd'].iloc[-1] > recent['macd_signal'].iloc[-1]
        
        if sma_alignment and price_change > 0.02 and macd_bullish:
            return 'up'
        elif not sma_alignment and price_change < -0.02 and not macd_bullish:
            return 'down'
        else:
            return 'sideways'
    
    def _calculate_signal_strength(self, current: pd.Series, prev: pd.Series, df: pd.DataFrame) -> Tuple[str, float]:
        """Calculate signal type and strength"""
        strength = 0.0
        signal_type = 'neutral'
        
        # RSI-based signals
        rsi = current.get('rsi', 50)
        if rsi < 30:
            signal_type = 'bullish'
            strength += 0.3
        elif rsi > 70:
            signal_type = 'bearish'
            strength += 0.3
        
        # MACD signals
        macd_hist = current.get('macd_hist', 0)
        if macd_hist > 0:
            if signal_type != 'bearish':
                signal_type = 'bullish'
            strength += 0.2
        elif macd_hist < 0:
            if signal_type != 'bullish':
                signal_type = 'bearish'
            strength += 0.2
        
        # Moving average signals
        price_vs_sma20 = current.get('price_vs_sma20', 0)
        if price_vs_sma20 > 2:
            if signal_type != 'bearish':
                signal_type = 'bullish'
            strength += 0.2
        elif price_vs_sma20 < -2:
            if signal_type != 'bullish':
                signal_type = 'bearish'
            strength += 0.2
        
        # Volume confirmation
        volume_ratio = current.get('volume_ratio', 1)
        if volume_ratio > 1.5:
            strength += 0.2
        
        # Bollinger Band position
        bb_position = (current['close'] - current.get('bb_lower', current['close'])) / (current.get('bb_upper', current['close']) - current.get('bb_lower', current['close']))
        if bb_position < 0.1:  # Near lower band
            if signal_type != 'bearish':
                signal_type = 'bullish'
            strength += 0.1
        elif bb_position > 0.9:  # Near upper band
            if signal_type != 'bullish':
                signal_type = 'bearish'
            strength += 0.1
        
        return signal_type, min(strength, 1.0)
    
    def _calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """Calculate momentum score (0-1)"""
        try:
            recent = df.tail(10)
            
            # Price momentum
            price_momentum = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
            price_score = min(abs(price_momentum) * 10, 1.0)
            
            # Volume momentum
            volume_momentum = recent['volume_ratio'].mean()
            volume_score = min(volume_momentum / 3, 1.0)
            
            # MACD momentum
            macd_momentum = recent['macd_hist'].mean()
            macd_score = min(abs(macd_momentum) * 50, 1.0)
            
            # Combined momentum score
            momentum_score = (price_score * 0.4) + (volume_score * 0.3) + (macd_score * 0.3)
            
            return min(momentum_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating momentum score: {e}")
            return 0.0
    
    def _check_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Check if volume confirms the price action"""
        try:
            recent = df.tail(5)
            avg_volume_ratio = recent['volume_ratio'].mean()
            return avg_volume_ratio > 1.2
        except:
            return False
    
    def _determine_price_position(self, current: pd.Series) -> str:
        """Determine price position (oversold/neutral/overbought)"""
        rsi = current.get('rsi', 50)
        
        if rsi < 30:
            return 'oversold'
        elif rsi > 70:
            return 'overbought'
        else:
            return 'neutral'
    
    def _combine_timeframe_signals(self, ticker: str, signals: Dict[str, TimeframeSignal]) -> MultiTimeframeAnalysis:
        """Combine signals from all timeframes into final analysis"""
        try:
            # Count bullish vs bearish signals
            bullish_count = sum(1 for s in signals.values() if s.signal_type == 'bullish')
            bearish_count = sum(1 for s in signals.values() if s.signal_type == 'bearish')
            neutral_count = sum(1 for s in signals.values() if s.signal_type == 'neutral')
            
            # Determine primary signal
            if bullish_count >= 2:
                primary_signal = 'bullish'
            elif bearish_count >= 2:
                primary_signal = 'bearish'
            else:
                primary_signal = 'neutral'
            
            # Calculate overall confidence
            total_strength = sum(s.strength for s in signals.values())
            avg_strength = total_strength / len(signals)
            
            # Trend alignment (how well trends align across timeframes)
            trend_directions = [s.trend_direction for s in signals.values()]
            trend_alignment = max(trend_directions.count('up'), trend_directions.count('down')) / len(trend_directions)
            
            # Volume consistency
            volume_consistent = sum(1 for s in signals.values() if s.volume_confirmation) / len(signals)
            
            # Calculate overall confidence
            confidence = avg_strength * trend_alignment * volume_consistent
            
            # Entry recommendation
            entry_recommendation = self._get_entry_recommendation(primary_signal, confidence, signals)
            
            # Risk level
            risk_level = self._assess_risk_level(signals)
            
            # Stop loss and target distances
            stop_loss_distance, target_distance = self._calculate_risk_rewards(signals)
            
            return MultiTimeframeAnalysis(
                ticker=ticker,
                timestamp=datetime.now(),
                primary_signal=primary_signal,
                confidence=confidence,
                timeframe_signals=signals,
                trend_alignment=trend_alignment,
                volume_consistency=volume_consistent,
                entry_recommendation=entry_recommendation,
                risk_level=risk_level,
                stop_loss_distance=stop_loss_distance,
                target_distance=target_distance
            )
            
        except Exception as e:
            logger.error(f"Error combining timeframe signals: {e}")
            # Return neutral analysis on error
            return MultiTimeframeAnalysis(
                ticker=ticker,
                timestamp=datetime.now(),
                primary_signal='neutral',
                confidence=0.0,
                timeframe_signals=signals,
                trend_alignment=0.0,
                volume_consistency=0.0,
                entry_recommendation='hold',
                risk_level='high',
                stop_loss_distance=5.0,
                target_distance=5.0
            )
    
    def _get_entry_recommendation(self, primary_signal: str, confidence: float, signals: Dict[str, TimeframeAnalysis]) -> str:
        """Get entry recommendation based on signal and confidence"""
        if primary_signal == 'bullish' and confidence > 0.7:
            return 'strong_buy'
        elif primary_signal == 'bullish' and confidence > 0.5:
            return 'buy'
        elif primary_signal == 'bearish' and confidence > 0.7:
            return 'strong_sell'
        elif primary_signal == 'bearish' and confidence > 0.5:
            return 'sell'
        else:
            return 'hold'
    
    def _assess_risk_level(self, signals: Dict[str, TimeframeSignal]) -> str:
        """Assess overall risk level"""
        # High risk if signals are conflicting
        signal_types = [s.signal_type for s in signals.values()]
        unique_signals = set(signal_types)
        
        if len(unique_signals) > 2:
            return 'high'
        elif len(unique_signals) == 2:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_risk_rewards(self, signals: Dict[str, TimeframeSignal]) -> Tuple[float, float]:
        """Calculate recommended stop loss and target distances"""
        # Use 15m timeframe for primary risk/reward calculation
        if '15m' in signals:
            signal_15m = signals['15m']
            
            # Adjust based on RSI
            rsi = signal_15m.price_position
            if rsi == 'oversold':
                stop_loss_distance = 3.0  # Tighter stop when oversold
                target_distance = 8.0
            elif rsi == 'overbought':
                stop_loss_distance = 4.0  # Wider stop when overbought
                target_distance = 6.0
            else:
                stop_loss_distance = 3.5
                target_distance = 7.0
        else:
            # Default values
            stop_loss_distance = 3.5
            target_distance = 7.0
        
        return stop_loss_distance, target_distance
