"""
Pattern Detection System for Stock Data
Identifies bullish and bearish patterns that precede significant price movements
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class PatternSignal:
    """Represents a detected pattern signal"""
    ticker: str
    date: str
    pattern_type: str  # 'bullish' or 'bearish'
    pattern_name: str
    confidence: float  # 0-1
    entry_price: float
    target_price: float
    stop_loss: float
    timestamp: str  # Entry timestamp
    exit_timestamp: Optional[str] = None  # Exit timestamp
    indicators: Optional[Dict[str, float]] = None
    price_change_after: Optional[float] = None  # Actual price change after pattern
    max_price_reached: Optional[float] = None  # Maximum price reached after entry
    max_price_timestamp: Optional[str] = None  # Timestamp when max price was reached


class PatternDetector:
    """Detects bullish and bearish patterns in stock data"""
    
    def __init__(self, lookback_periods: int = 20, forward_periods: int = 10):
        """
        Args:
            lookback_periods: Number of periods to look back for pattern formation
            forward_periods: Number of periods to check forward for price movement
        """
        self.lookback_periods = lookback_periods
        self.forward_periods = forward_periods
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df = df.copy()
        
        # Moving Averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # VWAP (Volume Weighted Average Price) - calculate if not present
        if 'vwap' not in df.columns:
            # Calculate VWAP: cumulative (price * volume) / cumulative volume
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Price change indicators
        df['price_change'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(periods=5)
        df['price_change_10'] = df['close'].pct_change(periods=10)
        
        # Volatility
        df['atr'] = self._calculate_atr(df, period=14)
        df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
        
        # Support/Resistance levels
        df['recent_high'] = df['high'].rolling(window=20).max()
        df['recent_low'] = df['low'].rolling(window=20).min()
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['momentum_pct'] = df['momentum'] / df['close'].shift(10)
        
        # Advanced momentum indicators (for pattern detection)
        df['momentum_5'] = df['close'].pct_change(5) * 100
        df['momentum_10'] = df['close'].pct_change(10) * 100
        df['momentum_20'] = df['close'].pct_change(20) * 100
        df['momentum_30'] = df['close'].pct_change(30) * 100
        
        # Volume indicators
        df['volume_ma_10'] = df['volume'].rolling(10).mean()
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_trend_5'] = df['volume'] / df['volume'].rolling(5).mean()
        df['volume_trend_10'] = df['volume'] / df['volume_ma_10']
        df['volume_increasing'] = df['volume'] > df['volume'].shift(1)
        
        # Price position in range
        df['high_10'] = df['high'].rolling(10).max()
        df['low_10'] = df['low'].rolling(10).min()
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        df['price_position_10'] = (df['close'] - df['low_10']) / (df['high_10'] - df['low_10'] + 0.0001) * 100
        df['price_position_20'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 0.0001) * 100
        
        # Moving average relationships
        df['sma5_above_sma10'] = df['sma_5'] > df['sma_10']
        df['sma10_above_sma20'] = df['sma_10'] > df['sma_20']
        df['ma_bullish_order'] = df['sma5_above_sma10'] & df['sma10_above_sma20']
        df['price_above_all_ma'] = (df['close'] > df['sma_5']) & (df['close'] > df['sma_10']) & (df['close'] > df['sma_20'])
        
        # MACD indicators
        df['macd_bullish'] = df['macd'] > df['macd_signal']
        df['macd_hist_positive'] = df['macd_hist'] > 0
        df['macd_hist_increasing'] = df['macd_hist'] > df['macd_hist'].shift(1)
        df['macd_hist_accelerating'] = (df['macd_hist'] > df['macd_hist'].shift(1)) & (df['macd_hist'].shift(1) > df['macd_hist'].shift(2))
        
        # Higher highs/lower lows
        df['higher_high_10'] = df['high'] > df['high'].shift(1)
        df['higher_high_20'] = df['high'] > df['high'].rolling(10).max().shift(10)
        df['higher_low_10'] = df['low'] > df['low'].shift(1)
        
        # Breakout indicators
        df['breakout_10'] = df['close'] > df['high_10'].shift(1) * 1.02  # 2% above 10-period high
        df['breakout_20'] = df['close'] > df['high_20'].shift(1) * 1.02  # 2% above 20-period high
        
        # Consolidation detection
        df['range_10'] = (df['high_10'] - df['low_10']) / df['low_10'] * 100
        df['in_consolidation'] = df['range_10'] < 3.0  # Less than 3% range
        
        # RSI zones
        df['rsi_oversold'] = df['rsi'] < 30
        df['rsi_neutral'] = (df['rsi'] >= 30) & (df['rsi'] <= 70)
        df['rsi_accumulation'] = (df['rsi'] >= 50) & (df['rsi'] <= 65)
        df['rsi_overbought'] = df['rsi'] > 70
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    def _is_reverse_split(self, df: pd.DataFrame, signal_idx: int, signal: PatternSignal, 
                          forward_data: pd.DataFrame) -> bool:
        """
        Detect if a massive price increase is due to a reverse stock split.
        Reverse splits typically show:
        - Massive overnight/next-day price increases (>100%, >200%)
        - Price gap at the start of trading session
        - Price increase not sustained (often reverses quickly)
        - May have normal or lower volume
        """
        try:
            # Check if we have enough data to analyze
            if len(forward_data) < 2:
                return False
            
            # Calculate the price increase percentage
            if signal.max_price_reached:
                max_gain_pct = ((signal.max_price_reached - signal.entry_price) / signal.entry_price) * 100
            elif signal.price_change_after:
                max_gain_pct = signal.price_change_after
            else:
                return False
            
            # Only check for reverse splits if there's a massive price increase (>100%)
            if max_gain_pct < 100:
                return False
            
            # Extremely large price increases (>500%) are suspicious and often reverse splits
            # Normal trading rarely produces such massive moves overnight/next-day
            if max_gain_pct > 500:
                # Check if this is an overnight/next-day gap (common in reverse splits)
                entry_time = pd.to_datetime(signal.timestamp)
                if len(forward_data) > 0:
                    first_period_time = pd.to_datetime(forward_data.iloc[0]['timestamp'])
                    time_diff = (first_period_time - entry_time).total_seconds() / 3600  # hours
                    
                    # If there's a significant time gap (>4 hours, likely overnight)
                    # and massive price increase, likely reverse split
                    if time_diff > 4 and max_gain_pct > 500:
                        # Additional validation: check if the price ratio suggests a reverse split
                        # Reverse splits often have clean ratios like 1:10, 1:20, 1:50, etc.
                        price_ratio = signal.max_price_reached / signal.entry_price
                        
                        # Check if ratio is close to common reverse split ratios
                        common_ratios = [10, 20, 25, 50, 100, 200, 250, 500, 1000]
                        for ratio in common_ratios:
                            if abs(price_ratio - ratio) / ratio < 0.15:  # Within 15% of common ratio
                                return True
                        
                        # Even if not a clean ratio, if gain is >500% with overnight gap,
                        # it's likely a reverse split unless there's extremely high volume
                        first_period = forward_data.iloc[0]
                        avg_volume = df['volume'].rolling(window=20).mean().iloc[signal_idx] if signal_idx >= 20 else df['volume'].mean()
                        first_volume = first_period['volume']
                        
                        # If volume is not extremely high (>10x average), likely reverse split
                        if first_volume < avg_volume * 10:
                            return True
                
                # For gains >1000%, be even more aggressive
                if max_gain_pct > 1000:
                    # Additional validation: check if the price ratio suggests a reverse split
                    price_ratio = signal.max_price_reached / signal.entry_price
                    
                    # Check if ratio is close to common reverse split ratios
                    common_ratios = [10, 20, 25, 50, 100, 200, 250, 500, 1000]
                    for ratio in common_ratios:
                        if abs(price_ratio - ratio) / ratio < 0.1:  # Within 10% of common ratio
                            return True
                    
                    # Even if not a clean ratio, if gain is >1000%, it's likely a reverse split
                    # unless there's very high volume (which would suggest a real breakout)
                    if len(forward_data) > 0:
                        first_period = forward_data.iloc[0]
                        avg_volume = df['volume'].rolling(window=20).mean().iloc[signal_idx] if signal_idx >= 20 else df['volume'].mean()
                        first_volume = first_period['volume']
                        
                        # If volume is not extremely high (>5x average), likely reverse split
                        if first_volume < avg_volume * 5:
                            return True
            
            # Check for overnight/next-day gap (large price jump between periods)
            # Look at the first few periods after the signal
            if len(forward_data) >= 2:
                entry_price = signal.entry_price
                first_period = forward_data.iloc[0]
                second_period = forward_data.iloc[1] if len(forward_data) > 1 else None
                
                # Check for massive gap in the first period
                first_open = first_period['open']
                first_close = first_period['close']
                gap_pct = ((first_open - entry_price) / entry_price) * 100
                
                # If there's a massive gap (>50%) at the start, likely a reverse split
                if gap_pct > 50:
                    # Additional check: see if price reverses quickly after the gap
                    if len(forward_data) >= 3:
                        # Check if price drops significantly after the initial jump
                        max_price = forward_data['high'].max()
                        final_price = forward_data.iloc[-1]['close']
                        reversal_pct = ((max_price - final_price) / max_price) * 100
                        
                        # If price reverses by >30% after the jump, likely a reverse split
                        if reversal_pct > 30:
                            return True
                    
                    # If gap is >200%, almost certainly a reverse split
                    if gap_pct > 200:
                        return True
                
                # Check for massive single-period price increase (>150%)
                single_period_gain = ((first_close - entry_price) / entry_price) * 100
                if single_period_gain > 150:
                    # Check volume - reverse splits may have normal or lower volume
                    avg_volume = df['volume'].rolling(window=20).mean().iloc[signal_idx] if signal_idx >= 20 else df['volume'].mean()
                    first_volume = first_period['volume']
                    
                    # If massive gain with normal/low volume, likely reverse split
                    if first_volume < avg_volume * 1.5:  # Not a high-volume breakout
                        return True
            
            # Check for price pattern typical of reverse splits
            # Reverse splits often show: massive jump, then quick reversal
            if len(forward_data) >= 5:
                prices = forward_data['close'].values
                max_idx = prices.argmax()
                
                # If max price is early and price drops significantly after
                if max_idx < len(prices) // 2:  # Max price in first half
                    max_price = prices[max_idx]
                    final_price = prices[-1]
                    drop_from_max = ((max_price - final_price) / max_price) * 100
                    
                    # If dropped >40% from max, likely reverse split
                    if drop_from_max > 40 and max_gain_pct > 150:
                        return True
            
            return False
            
        except Exception as e:
            # If any error occurs, don't filter (be conservative)
            return False
    
    def _is_false_breakout(self, df: pd.DataFrame, signal_idx: int, signal: PatternSignal,
                          forward_data: pd.DataFrame) -> bool:
        """
        Detect if a breakout signal is a false breakout.
        False breakouts typically show:
        - Price breaks out but then quickly reverses back below breakout level
        - Breakout doesn't sustain for multiple periods
        - Volume may spike initially but then drop
        """
        try:
            # Only check bullish patterns for false breakouts
            if signal.pattern_type != 'bullish':
                return False
            
            if len(forward_data) < 3:
                return False
            
            entry_price = signal.entry_price
            
            # For breakout patterns, check if price sustains above entry
            breakout_patterns = ['Volume_Breakout', 'Consolidation_Breakout', 'BB_Lower_Bounce']
            
            if signal.pattern_name in breakout_patterns:
                # Check if price breaks above entry but then falls back below
                prices = forward_data['close'].values
                highs = forward_data['high'].values
                
                # Check if price went above entry (breakout occurred)
                broke_above = any(h > entry_price * 1.01 for h in highs)  # At least 1% above
                
                if broke_above:
                    # Check if price fell back below entry level
                    # Count how many periods price stayed above entry
                    periods_above = sum(1 for p in prices if p > entry_price)
                    total_periods = len(prices)
                    
                    # If price was above entry less than 30% of the time, it's a false breakout
                    if periods_above / total_periods < 0.3:
                        # Additional check: if final price is significantly below entry
                        final_price = prices[-1]
                        if final_price < entry_price * 0.98:  # 2% below entry
                            return True
                    
                    # Check for quick reversal pattern
                    # Price goes up, then quickly reverses
                    if len(prices) >= 3:
                        # Find the peak
                        peak_idx = prices.argmax()
                        peak_price = prices[peak_idx]
                        
                        # If peak is early and price drops significantly
                        if peak_idx < len(prices) // 2:
                            final_price = prices[-1]
                            drop_from_peak = ((peak_price - final_price) / peak_price) * 100
                            
                            # If dropped >15% from peak and ended below entry, false breakout
                            if drop_from_peak > 15 and final_price < entry_price:
                                return True
            
            # For other patterns, check if price doesn't sustain the move
            # If price increases but then reverses significantly
            if signal.max_price_reached:
                max_price = signal.max_price_reached
                final_price = forward_data.iloc[-1]['close']
                
                # Calculate how much of the gain was given back
                gain_from_entry = ((max_price - entry_price) / entry_price) * 100
                loss_from_max = ((max_price - final_price) / max_price) * 100
                
                # If significant gain (>20%) but then lost >50% of that gain, likely false signal
                if gain_from_entry > 20 and loss_from_max > 50:
                    # Additional check: if final price is near or below entry
                    if final_price <= entry_price * 1.05:  # Within 5% of entry
                        return True
            
            return False
            
        except Exception as e:
            # If any error occurs, don't filter (be conservative)
            return False
    
    def detect_patterns(self, df: pd.DataFrame, ticker: str, date: str) -> List[PatternSignal]:
        """Detect all patterns in the dataframe"""
        if len(df) < self.lookback_periods + self.forward_periods:
            return []
        
        df = self.calculate_indicators(df).reset_index(drop=True)
        signals = []
        
        # Check each point where we have enough forward data
        for i in range(self.lookback_periods, len(df) - self.forward_periods):
            window_df = df.iloc[i - self.lookback_periods:i + self.forward_periods].copy()
            current_row = df.iloc[i]
            
            # Detect bullish patterns
            bullish_patterns = self._detect_bullish_patterns(window_df, i, current_row, ticker, date)
            signals.extend(bullish_patterns)
            
            # Detect bearish patterns
            bearish_patterns = self._detect_bearish_patterns(window_df, i, current_row, ticker, date)
            signals.extend(bearish_patterns)
        
        # Calculate actual price changes and track exit points
        filtered_signals = []
        for signal in signals:
            # Find the index of the signal timestamp
            try:
                signal_timestamp = pd.to_datetime(signal.timestamp)
                mask = pd.to_datetime(df['timestamp']) == signal_timestamp
                matching_indices = df.index[mask].tolist()
                
                if matching_indices:
                    idx = matching_indices[0]
                    if idx + self.forward_periods < len(df):
                        # Check forward prices to find max gain and exit point
                        forward_data = df.iloc[idx:idx + self.forward_periods + 1]
                        
                        # Calculate price changes
                        future_price = df.iloc[idx + self.forward_periods]['close']
                        signal.price_change_after = ((future_price - signal.entry_price) / signal.entry_price) * 100
                        
                        # Find maximum price reached
                        max_price_idx = forward_data['high'].idxmax()
                        signal.max_price_reached = forward_data.loc[max_price_idx, 'high']
                        signal.max_price_timestamp = str(forward_data.loc[max_price_idx, 'timestamp'])
                        
                        # Set exit timestamp (when we check the result)
                        signal.exit_timestamp = str(df.iloc[idx + self.forward_periods]['timestamp'])
                        
                        # Calculate max gain percentage
                        max_gain = ((signal.max_price_reached - signal.entry_price) / signal.entry_price) * 100
                        
                        # Filter out reverse splits and false breakouts
                        if self._is_reverse_split(df, idx, signal, forward_data):
                            continue  # Skip reverse split scenarios
                        
                        if self._is_false_breakout(df, idx, signal, forward_data):
                            continue  # Skip false breakout scenarios
                        
                        filtered_signals.append(signal)
                        
            except Exception as e:
                # If timestamp matching fails, skip validation
                pass
        
        return filtered_signals
    
    def _detect_bullish_patterns(self, df: pd.DataFrame, idx: int, current: pd.Series, 
                                 ticker: str, date: str) -> List[PatternSignal]:
        """Detect bullish patterns based on user rules"""
        signals = []
        
        # Need at least 30 bars of data for pattern detection
        if idx < 30:
            return signals
        
        current_price = current['close']
        
        # Pattern 1: Volume_Breakout_Momentum (Score: 8, Confidence: 0.85)
        # Criteria: volume_ratio >= 1.8, momentum_10 >= 2.0%, breakout_10 == True, price_above_all_ma == True
        if (current.get('volume_ratio', 0) >= 1.8 and
            current.get('momentum_10', 0) >= 2.0 and
            current.get('breakout_10', False) and
            current.get('price_above_all_ma', False)):
            confidence = 0.85
            target = current_price * 1.20  # 20% target
            stop = current_price * 0.85   # 15% stop
            signals.append(PatternSignal(
                ticker=ticker, date=date, pattern_type='bullish',
                pattern_name='Volume_Breakout_Momentum', confidence=confidence,
                entry_price=current_price, target_price=target, stop_loss=stop,
                timestamp=current['timestamp'],
                indicators={
                    'volume_ratio': current.get('volume_ratio', 0),
                    'momentum_10': current.get('momentum_10', 0),
                    'breakout_10': current.get('breakout_10', False)
                }
            ))
        
        # Pattern 2: RSI_Accumulation_Entry (Score: 7, Confidence: 0.75)
        # Criteria: rsi_accumulation == True, momentum_10 >= 2.0%, volume_ratio >= 1.8,
        #          macd_hist_increasing == True, higher_high_20 == True
        if (current.get('rsi_accumulation', False) and
            current.get('momentum_10', 0) >= 2.0 and
            current.get('volume_ratio', 0) >= 1.8 and
            current.get('macd_hist_increasing', False) and
            current.get('higher_high_20', False)):
            confidence = 0.75
            target = current_price * 1.20  # 20% target
            stop = current_price * 0.85    # 15% stop
            signals.append(PatternSignal(
                ticker=ticker, date=date, pattern_type='bullish',
                pattern_name='RSI_Accumulation_Entry', confidence=confidence,
                entry_price=current_price, target_price=target, stop_loss=stop,
                timestamp=current['timestamp'],
                indicators={
                    'rsi': current.get('rsi', 0),
                    'momentum_10': current.get('momentum_10', 0),
                    'volume_ratio': current.get('volume_ratio', 0)
                }
            ))
        
        # Pattern 3: Golden_Cross_Volume (Score: 7, Confidence: 0.78)
        # Criteria: sma5_above_sma10 == True, sma10_above_sma20 == True,
        #          JUST crossed (sma10_above_sma20 was False, now True),
        #          volume_ratio >= 1.5, momentum_10 >= 1.5%
        if idx > 0 and len(df) > idx:
            prev = df.iloc[idx - 1]
            prev_sma10_above_sma20 = prev.get('sma10_above_sma20', False) if 'sma10_above_sma20' in prev else False
            if (current.get('sma5_above_sma10', False) and
                current.get('sma10_above_sma20', False) and
                not prev_sma10_above_sma20 and  # Just crossed
                current.get('volume_ratio', 0) >= 1.5 and
                current.get('momentum_10', 0) >= 1.5):
                confidence = 0.78
                target = current_price * 1.20  # 20% target
                stop = current_price * 0.85   # 15% stop
                signals.append(PatternSignal(
                    ticker=ticker, date=date, pattern_type='bullish',
                    pattern_name='Golden_Cross_Volume', confidence=confidence,
                    entry_price=current_price, target_price=target, stop_loss=stop,
                    timestamp=current['timestamp'],
                    indicators={
                        'sma5_above_sma10': current.get('sma5_above_sma10', False),
                        'sma10_above_sma20': current.get('sma10_above_sma20', False),
                        'volume_ratio': current.get('volume_ratio', 0)
                    }
                ))
        
        # Pattern 4: Slow_Accumulation (Score: 7, Confidence: 0.80)
        # Criteria: 1.8 <= volume_ratio < 3.5, momentum_10 >= 2.0%, momentum_20 >= 3.0%,
        #          volume_trend_10 >= 1.3, macd_hist_accelerating == True, price_position_20 >= 70
        volume_ratio = current.get('volume_ratio', 0)
        if (1.8 <= volume_ratio < 3.5 and
            current.get('momentum_10', 0) >= 2.0 and
            current.get('momentum_20', 0) >= 3.0 and
            current.get('volume_trend_10', 0) >= 1.3 and
            current.get('macd_hist_accelerating', False) and
            current.get('price_position_20', 0) >= 70):
            confidence = 0.80
            target = current_price * 1.20  # 20% target
            stop = current_price * 0.85   # 15% stop
            signals.append(PatternSignal(
                ticker=ticker, date=date, pattern_type='bullish',
                pattern_name='Slow_Accumulation', confidence=confidence,
                entry_price=current_price, target_price=target, stop_loss=stop,
                timestamp=current['timestamp'],
                indicators={
                    'volume_ratio': volume_ratio,
                    'momentum_10': current.get('momentum_10', 0),
                    'momentum_20': current.get('momentum_20', 0)
                }
            ))
        
        # Pattern 5: MACD_Acceleration_Breakout (Score: 8, Confidence: 0.82)
        # Criteria: macd_hist_accelerating == True, macd_bullish == True,
        #          breakout_20 == True, volume_ratio >= 2.0, momentum_20 >= 3.0%
        if (current.get('macd_hist_accelerating', False) and
            current.get('macd_bullish', False) and
            current.get('breakout_20', False) and
            current.get('volume_ratio', 0) >= 2.0 and
            current.get('momentum_20', 0) >= 3.0):
            confidence = 0.82
            target = current_price * 1.20  # 20% target
            stop = current_price * 0.85   # 15% stop
            signals.append(PatternSignal(
                ticker=ticker, date=date, pattern_type='bullish',
                pattern_name='MACD_Acceleration_Breakout', confidence=confidence,
                entry_price=current_price, target_price=target, stop_loss=stop,
                timestamp=current['timestamp'],
                indicators={
                    'macd_hist_accelerating': current.get('macd_hist_accelerating', False),
                    'macd_bullish': current.get('macd_bullish', False),
                    'breakout_20': current.get('breakout_20', False)
                }
            ))
        
        # Pattern 6: Consolidation_Breakout (Score: 8, Confidence: 0.83)
        # Criteria: in_consolidation == False (just broke out),
        #          was consolidating in last 5 bars (3+ periods),
        #          breakout_10 == True, volume_ratio >= 2.0, price_above_all_ma == True
        if idx >= 5:
            was_consolidating = df.iloc[idx-5:idx]['in_consolidation'].sum() >= 3
            if (not current.get('in_consolidation', False) and  # Just broke out
                was_consolidating and  # Was consolidating
                current.get('breakout_10', False) and
                current.get('volume_ratio', 0) >= 2.0 and
                current.get('price_above_all_ma', False)):
                confidence = 0.83
                target = current_price * 1.20  # 20% target
                stop = current_price * 0.85   # 15% stop
                signals.append(PatternSignal(
                    ticker=ticker, date=date, pattern_type='bullish',
                    pattern_name='Consolidation_Breakout', confidence=confidence,
                    entry_price=current_price, target_price=target, stop_loss=stop,
                    timestamp=current['timestamp'],
                    indicators={
                        'was_consolidating': was_consolidating,
                        'breakout_10': current.get('breakout_10', False),
                        'volume_ratio': current.get('volume_ratio', 0)
                    }
                ))
        
        return signals
    
    def _detect_bearish_patterns(self, df: pd.DataFrame, idx: int, current: pd.Series,
                                ticker: str, date: str) -> List[PatternSignal]:
        """Detect bearish patterns"""
        signals = []
        lookback = df.iloc[:self.lookback_periods]
        current_price = current['close']
        
        # Pattern 1: RSI Overbought Rejection
        if current['rsi'] > 70 and current['rsi'] < lookback['rsi'].iloc[-5:].max():
            confidence = (current['rsi'] - 70) / 30
            target = current_price * 0.95
            stop = current_price * 1.03
            signals.append(PatternSignal(
                ticker=ticker, date=date, pattern_type='bearish',
                pattern_name='RSI_Overbought_Rejection', confidence=confidence,
                entry_price=current_price, target_price=target, stop_loss=stop,
                timestamp=current['timestamp'],
                indicators={'rsi': current['rsi'], 'price': current_price}
            ))
        
        # Pattern 2: Death Cross (EMA crossover)
        if (current['ema_12'] < current['ema_26'] and 
            lookback['ema_12'].iloc[-2] >= lookback['ema_26'].iloc[-2]):
            confidence = 0.7
            target = current_price * 0.92
            stop = current_price * 1.05
            signals.append(PatternSignal(
                ticker=ticker, date=date, pattern_type='bearish',
                pattern_name='Death_Cross', confidence=confidence,
                entry_price=current_price, target_price=target, stop_loss=stop,
                timestamp=current['timestamp'],
                indicators={'ema_12': current['ema_12'], 'ema_26': current['ema_26']}
            ))
        
        return signals

