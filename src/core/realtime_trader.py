"""
Real-Time Trading System
Analyzes live 1-minute data and provides entry/exit signals for bullish trades
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pytz
from analysis.pattern_detector import PatternDetector, PatternSignal

logger = logging.getLogger(__name__)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) for volatility-based stop loss
    
    Args:
        df: DataFrame with high, low, close columns
        period: Period for ATR calculation (default: 14)
        
    Returns:
        Series with ATR values
    """
    if len(df) < period + 1:
        return pd.Series([0.0] * len(df), index=df.index)
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_volatility(df: pd.DataFrame, period: int = 20) -> float:
    """
    Calculate price volatility as percentage
    
    Args:
        df: DataFrame with high, low columns
        period: Number of periods to analyze
        
    Returns:
        Volatility percentage
    """
    if len(df) < period:
        return 0.0
    
    recent_df = df.tail(period)
    price_range = (recent_df['high'].max() - recent_df['low'].min()) / recent_df['low'].min() * 100
    return price_range


@dataclass
class TradeSignal:
    """Represents a trading signal (entry or exit)"""
    signal_type: str  # 'entry' or 'exit'
    ticker: str
    timestamp: datetime
    price: float
    pattern_name: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""  # Reason for the signal
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    indicators: Dict = field(default_factory=dict)


@dataclass
class ActivePosition:
    """Tracks an active trading position"""
    ticker: str
    entry_time: datetime
    entry_price: float
    entry_pattern: str
    entry_confidence: float
    target_price: float
    stop_loss: float
    current_price: float
    unrealized_pnl_pct: float = 0.0
    max_price_reached: float = 0.0
    trailing_stop_price: Optional[float] = None
    shares: float = 0.0  # Number of shares
    entry_value: float = 0.0  # Dollar value at entry
    partial_profit_taken: bool = False  # Track if first partial profit was taken (50% at +4%)
    partial_profit_taken_second: bool = False  # Track if second partial profit was taken (25% at +7%)
    original_shares: float = 0.0  # Original shares before partial exit
    is_slow_mover_entry: bool = False  # Flag to mark slow mover entries (uses different exit logic)


class RealtimeTrader:
    """Real-time trading analyzer with entry/exit logic"""
    
    def __init__(self, 
                 min_confidence: float = 0.72,  # BALANCED: 72% - high-quality trades with reasonable opportunities
                 min_entry_price_increase: float = 5.5,  # BALANCED: 5.5% - good quality setups
                 trailing_stop_pct: float = 2.5,  # REFINED: 2.5% - tighter stops, cut losses faster
                 profit_target_pct: float = 8.0,  # REFINED: 8% - realistic profit target
                 data_api=None):  # DataAPI instance for multi-timeframe analysis
        """
        Args:
            min_confidence: Minimum pattern confidence to enter trade (0-1)
            min_entry_price_increase: Minimum expected price increase to enter (%)
            trailing_stop_pct: Trailing stop loss percentage
            profit_target_pct: Profit target percentage
            data_api: DataAPI instance for fetching daily data (for multi-timeframe MACD)
        """
        self.pattern_detector = PatternDetector(lookback_periods=20, forward_periods=0)
        self.min_confidence = min_confidence
        self.min_entry_price_increase = min_entry_price_increase
        self.trailing_stop_pct = trailing_stop_pct
        self.profit_target_pct = profit_target_pct
        self.data_api = data_api  # For multi-timeframe analysis
        
        self.active_positions: Dict[str, ActivePosition] = {}
        self.trade_history: List[TradeSignal] = []
        self.last_rejection_reasons: Dict[str, List[str]] = {}  # Track rejection reasons per ticker
        self.last_fast_mover_status: Dict[str, Dict[str, float]] = {}  # Track fast mover status per ticker
        self.daily_macd_cache: Dict[str, Dict] = {}  # Cache daily MACD values
    
    def analyze_data(self, df: pd.DataFrame, ticker: str, current_price: Optional[float] = None) -> Tuple[Optional[TradeSignal], List[TradeSignal]]:
        """
        Analyze real-time data and return entry/exit signals
        
        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume
            ticker: Stock ticker symbol
            current_price: Optional current price from API (for premarket/real-time updates)
            
        Returns:
            Tuple of (entry_signal, exit_signals)
            - entry_signal: New entry opportunity (None if none found)
            - exit_signals: List of exit signals for active positions
        """
        if len(df) < 50:
            return None, []
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Check for exit signals on active positions
        # Pass current_price to ensure positions show current premarket price
        exit_signals = self._check_exit_signals(df, ticker, current_price=current_price)
        
        # Check for new entry signals (only if no active position)
        entry_signal = None
        if ticker not in self.active_positions:
            # FIRST: Try original entry logic
            entry_signal = self._check_entry_signal(df, ticker)
            
            # SECOND: If original logic found no entry, try slow mover logic
            if entry_signal is None:
                entry_signal = self._check_slow_mover_entry_signal(df, ticker)
        
        return entry_signal, exit_signals
    
    def _check_entry_signal(self, df: pd.DataFrame, ticker: str) -> Optional[TradeSignal]:
        """Check for valid entry signals"""
        # Calculate indicators
        df_with_indicators = self.pattern_detector.calculate_indicators(df)
        
        # Get the most recent data point
        if len(df_with_indicators) < 30:  # Need more history for confirmation
            return None
        
        current_idx = len(df_with_indicators) - 1
        current = df_with_indicators.iloc[current_idx]
        
        # PRIORITY 0: Minimum price filter - reject stocks below $0.50
        current_price = current.get('close', 0)
        if current_price < 0.50:
            self.last_rejection_reasons[ticker] = [f"Price ${current_price:.4f} below minimum $0.50"]
            return None
        
        # PRIORITY 0.5: Minimum volume filter - reject low volume stocks (time-based thresholds)
        # FIX: Use time-based volume thresholds (100K-500K based on time of day)
        current_time = pd.to_datetime(current.get('timestamp', datetime.now()))
        et = pytz.timezone('US/Eastern')
        if current_time.tz is None:
            current_time = et.localize(current_time)
        else:
            current_time = current_time.astimezone(et)
        
        hour = current_time.hour
        
        # Time-based volume thresholds
        if hour < 6:  # 4-6 AM
            min_daily_volume = 100000  # 100K
        elif hour < 8:  # 6-8 AM
            min_daily_volume = 200000  # 200K
        elif hour < 10:  # 8-10 AM
            min_daily_volume = 300000  # 300K
        else:  # 10 AM+
            min_daily_volume = 500000  # 500K
        
        # Check total volume over recent periods (simulating daily volume check)
        if len(df_with_indicators) >= 60:
            recent_volumes = df_with_indicators['volume'].tail(60).values
            total_volume_60min = recent_volumes.sum()
            if total_volume_60min < min_daily_volume:
                self.last_rejection_reasons[ticker] = [f"Low volume stock (total {total_volume_60min:,.0f} < {min_daily_volume:,.0f} over 60 min, threshold for hour {hour})"]
                return None
        elif len(df_with_indicators) >= 20:
            # If less than 60 minutes, check 20-minute total and extrapolate
            recent_volumes = df_with_indicators['volume'].tail(20).values
            total_volume_20min = recent_volumes.sum()
            # Extrapolate to 60 minutes: need at least min_daily_volume/3 over 20 min
            min_volume_20min = min_daily_volume // 3
            if total_volume_20min < min_volume_20min:
                self.last_rejection_reasons[ticker] = [f"Low volume stock (total {total_volume_20min:,.0f} < {min_volume_20min:,.0f} over 20 min, extrapolated, threshold for hour {hour})"]
                return None
        else:
            # If very little data, check current volume (should be at least 10K for single bar)
            current_volume = current.get('volume', 0)
            min_current_volume = 10000
            if current_volume < min_current_volume:
                self.last_rejection_reasons[ticker] = [f"Low volume stock ({current_volume:,.0f} < {min_current_volume:,.0f} required)"]
                return None
        
        # Detect patterns at current point
        lookback = df_with_indicators.iloc[:current_idx + 1]
        signals = self.pattern_detector._detect_bullish_patterns(
            lookback, current_idx, current, ticker, 
            datetime.now().strftime('%Y-%m-%d')
        )
        
        if not signals:
            logger.debug(f"[{ticker}] No patterns detected")
            return None
        
        logger.debug(f"[{ticker}] Found {len(signals)} pattern signal(s)")
        
        # Clear previous rejection reasons for this ticker
        self.last_rejection_reasons[ticker] = []
        
        # Filter and validate signals - VERY STRICT
        for signal in signals:
            # PRIORITY 0.5: Check entry price is above minimum
            if signal.entry_price < 0.50:
                self.last_rejection_reasons[ticker].append(f"Entry price ${signal.entry_price:.4f} below minimum $0.50")
                continue
            
            # PRIORITY 3 FIX: Lower confidence threshold for fast movers and early morning
            # Detect fast mover characteristics before confidence check
            volume_ratio = current.get('volume_ratio', 0)
            price_momentum_5 = ((current.get('close', 0) - df_with_indicators.iloc[max(0, current_idx-5)].get('close', 0)) / 
                               df_with_indicators.iloc[max(0, current_idx-5)].get('close', 0)) * 100 if current_idx >= 5 else 0
            
            is_fast_mover = volume_ratio >= 2.5 and price_momentum_5 >= 3.0
            
            # FIX: Time-based confidence threshold (70% before 10 AM, 72% after)
            current_time = pd.to_datetime(current.get('timestamp', datetime.now()))
            et = pytz.timezone('US/Eastern')
            if current_time.tz is None:
                current_time = et.localize(current_time)
            else:
                current_time = current_time.astimezone(et)
            
            hour = current_time.hour
            
            # Adjust confidence threshold
            effective_min_confidence = self.min_confidence
            if hour < 10:  # Before 10 AM - use 70% threshold
                effective_min_confidence = 0.70
                logger.debug(f"[{ticker}] EARLY MORNING: Using relaxed confidence threshold 70% (hour={hour})")
            elif is_fast_mover:
                # For very strong fast movers (high volume + high momentum), lower threshold to 65%
                if volume_ratio >= 4.0 and price_momentum_5 >= 10.0:  # Very strong
                    effective_min_confidence = 0.65  # Lower to 65% for explosive moves
                    logger.debug(f"[{ticker}] FAST MOVER (EXPLOSIVE): Using relaxed confidence threshold 65% (vol={volume_ratio:.2f}x, momentum={price_momentum_5:.1f}%)")
                else:
                    effective_min_confidence = 0.70  # Lower to 70% for fast movers
                    logger.debug(f"[{ticker}] FAST MOVER: Using relaxed confidence threshold 70% (vol={volume_ratio:.2f}x, momentum={price_momentum_5:.1f}%)")
            
            # Check minimum confidence with adjusted threshold
            if signal.confidence < effective_min_confidence:
                self.last_rejection_reasons[ticker].append(f"Confidence {signal.confidence*100:.1f}% < {effective_min_confidence*100:.0f}% required")
                continue
            
            # PRIORITY 1: Check for false breakouts FIRST (most important filter)
            # FIX: Relax false breakout for fast movers with 75%+ confidence OR any pattern with 75%+ confidence
            skip_false_breakout = False
            if signal.confidence >= 0.75:
                # Skip false breakout for high-confidence patterns
                skip_false_breakout = True
                logger.debug(f"[{ticker}] HIGH CONFIDENCE ({signal.confidence*100:.1f}%): Skipping false breakout check")
            elif is_fast_mover and signal.confidence >= 0.70:
                # Skip false breakout for fast movers with 70%+ confidence
                skip_false_breakout = True
                logger.debug(f"[{ticker}] FAST MOVER with 70%+ confidence: Skipping false breakout check")
            
            if not skip_false_breakout and self._is_false_breakout_realtime(df_with_indicators, current_idx, signal):
                self.last_rejection_reasons[ticker].append("False breakout detected")
                continue
            
            # PRIORITY 2: Check for reverse split (shouldn't happen in real-time, but check anyway)
            if self._is_reverse_split_realtime(df_with_indicators, current_idx, signal):
                self.last_rejection_reasons[ticker].append("Reverse split detected")
                continue
            
            # PRIORITY 3: Validate perfect setup (comprehensive check)
            validation_result, rejection_reason = self._validate_entry_signal(df_with_indicators, current_idx, signal, log_reasons=True)
            if not validation_result:
                if rejection_reason:
                    self.last_rejection_reasons[ticker].append(rejection_reason)
                continue
            
            # PRIORITY 4: Setup must be confirmed for multiple periods (not just appeared)
            # This ensures the setup is sustainable, not just a momentary spike
            # RELAXED for fast movers: Fast movers can have explosive moves that don't build over time
            is_fast_mover_check, fast_mover_metrics_check = self._is_fast_mover(df_with_indicators, current_idx)
            # For very strong fast movers (4x+ volume, 10%+ momentum), skip setup confirmation entirely
            if is_fast_mover_check and fast_mover_metrics_check.get('vol_ratio', 0) >= 4.0 and fast_mover_metrics_check.get('momentum', 0) >= 10.0:
                logger.info(f"[{ticker}] VERY STRONG FAST MOVER: Skipping setup confirmation (vol={fast_mover_metrics_check.get('vol_ratio', 0):.2f}x, momentum={fast_mover_metrics_check.get('momentum', 0):.2f}%)")
            elif not self._setup_confirmed_multiple_periods(df_with_indicators, current_idx, signal, is_fast_mover=is_fast_mover_check):
                self.last_rejection_reasons[ticker].append("Setup not confirmed for multiple periods")
                logger.debug(f"[{ticker}] Setup confirmation failed (fast_mover={is_fast_mover_check})")
                continue
            
            # PRIORITY 5: Check expected gain meets minimum
            expected_gain = ((signal.target_price - signal.entry_price) / signal.entry_price) * 100
            if expected_gain < self.min_entry_price_increase:
                continue
            
            # PRIORITY 6: Final confirmation - price must be confirming the signal NOW
            current_price = current.get('close', 0)
            if current_price < signal.entry_price * 0.98:  # Price already dropped 2% from signal
                continue  # Signal is stale or failing
            
            # IMPROVED: Store fast mover status in indicators for stop loss calculation
            indicators = signal.indicators or {}
            if is_fast_mover_check:
                indicators['is_fast_mover_entry'] = True
                indicators['fast_mover_vol_ratio'] = fast_mover_metrics_check.get('vol_ratio', 0)
                indicators['fast_mover_momentum'] = fast_mover_metrics_check.get('momentum', 0)
            
            # ALL CHECKS PASSED - This is a PERFECT SETUP
            return TradeSignal(
                signal_type='entry',
                ticker=ticker,
                timestamp=pd.to_datetime(current['timestamp']),
                price=signal.entry_price,
                pattern_name=signal.pattern_name,
                confidence=signal.confidence,
                reason=f"PERFECT SETUP: {signal.pattern_name} with {signal.confidence*100:.1f}% confidence, "
                       f"all confirmations passed, expected gain {expected_gain:.1f}%",
                target_price=signal.target_price,
                stop_loss=signal.stop_loss,
                indicators=indicators
            )
        
        return None
    
    def _check_slow_mover_entry_signal(self, df: pd.DataFrame, ticker: str) -> Optional[TradeSignal]:
        """
        Check for slow mover entry signals (alternative path when original logic fails)
        This method uses relaxed volume thresholds but strict quality criteria
        """
        # Calculate indicators
        df_with_indicators = self.pattern_detector.calculate_indicators(df)
        
        # Get the most recent data point
        if len(df_with_indicators) < 30:
            return None
        
        current_idx = len(df_with_indicators) - 1
        current = df_with_indicators.iloc[current_idx]
        
        # Calculate advanced indicators needed for slow mover detection
        df_with_indicators = self._calculate_advanced_indicators_for_slow_mover(df_with_indicators)
        current = df_with_indicators.iloc[current_idx]
        
        # PRIORITY 0: Minimum price filter
        current_price = current.get('close', 0)
        if current_price < 0.50:
            return None
        
        # PRIORITY 0.5: Slow mover volume check (lower threshold: 200K vs 500K normal)
        if len(df_with_indicators) >= 60:
            recent_volumes = df_with_indicators['volume'].tail(60).values
            total_volume_60min = recent_volumes.sum()
            min_slow_mover_volume = 200000  # 200K minimum for slow movers
            if total_volume_60min < min_slow_mover_volume:
                return None
        elif len(df_with_indicators) >= 20:
            recent_volumes = df_with_indicators['volume'].tail(20).values
            total_volume_20min = recent_volumes.sum()
            min_volume_20min = 67000  # 67K over 20 min (extrapolated from 200K/3)
            if total_volume_20min < min_volume_20min:
                return None
        else:
            return None
        
        # SLOW MOVER CRITERIA (ALL must pass)
        volume_ratio = current.get('volume_ratio', 0)
        momentum_10 = current.get('momentum_10', 0)
        momentum_20 = current.get('momentum_20', 0)
        
        # 1. Volume Ratio: 1.8x - 3.5x (moderate-high, not explosive)
        if volume_ratio < 1.8 or volume_ratio >= 3.5:
            return None
        
        # 2. Sustained Momentum: 10-min >= 2.0%, 20-min >= 3.0%
        if momentum_10 < 2.0 or momentum_20 < 3.0:
            return None
        
        # 3. Momentum consistency: 10-min >= 80% of 20-min (not decelerating)
        if momentum_20 > 0 and momentum_10 < (momentum_20 * 0.8):
            return None
        
        # 4. Volume building consistently
        if current_idx >= 10:
            recent_5_volumes = df_with_indicators['volume'].iloc[current_idx-4:current_idx+1].values
            prev_5_volumes = df_with_indicators['volume'].iloc[current_idx-9:current_idx-4].values
            if len(recent_5_volumes) >= 5 and len(prev_5_volumes) >= 5:
                recent_avg = np.mean(recent_5_volumes)
                prev_avg = np.mean(prev_5_volumes)
                if prev_avg > 0 and recent_avg < (prev_avg * 1.1):  # Last 5 periods >= 110% of previous 5
                    return None
            
            # Volume acceleration: Current volume >= 1.3x of 10-period average
            volume_ma_10 = current.get('volume_ma_10', 0)
            current_volume = current.get('volume', 0)
            if volume_ma_10 > 0 and current_volume < (volume_ma_10 * 1.3):
                return None
            
            # No declining volume for 3+ consecutive periods
            if current_idx >= 3:
                volumes_last_3 = df_with_indicators['volume'].iloc[current_idx-2:current_idx+1].values
                if len(volumes_last_3) >= 3 and all(volumes_last_3[i] >= volumes_last_3[i+1] for i in range(len(volumes_last_3)-1)):
                    # All volumes are declining
                    return None
        
        # 5. MACD Acceleration Pattern
        macd_hist_accelerating = current.get('macd_hist_accelerating', False)
        if not macd_hist_accelerating:
            return None
        
        # 6. Price Breaking Above Consolidation (breakout_10)
        breakout_10 = current.get('breakout_10', False)
        if not breakout_10:
            return None
        
        # 7. Higher Highs Pattern (20-period)
        higher_high_20 = current.get('higher_high_20', False)
        if not higher_high_20:
            return None
        
        # 8. RSI in Optimal Accumulation Zone (50-65)
        rsi = current.get('rsi', 50)
        rsi_accumulation = current.get('rsi_accumulation', False)
        if not rsi_accumulation or rsi < 50 or rsi > 65:
            return None
        
        # 9. Technical Setup: Price above all MAs, MACD bullish
        price_above_all_ma = current.get('price_above_all_ma', False)
        macd_bullish = current.get('macd_bullish', False)
        if not price_above_all_ma or not macd_bullish:
            return None
        
        # 10. Pattern Quality: Check if we have a valid pattern
        lookback = df_with_indicators.iloc[:current_idx + 1]
        signals = self.pattern_detector._detect_bullish_patterns(
            lookback, current_idx, current, ticker, 
            datetime.now().strftime('%Y-%m-%d')
        )
        
        if not signals:
            return None
        
        # Find best pattern (primary patterns preferred, or secondary with 80%+ confidence)
        best_signal = None
        for signal in signals:
            # Require confidence >= 80% for slow movers (higher bar)
            if signal.confidence >= 0.80:
                if best_signal is None or signal.confidence > best_signal.confidence:
                    best_signal = signal
        
        if best_signal is None:
            return None
        
        # Calculate stop loss and target (same as normal entries)
        stop_loss = current_price * 0.85  # 15% stop loss
        target_price = current_price * 1.20  # 20% target
        
        # Create TradeSignal with slow mover flag in indicators
        indicators = best_signal.indicators or {}
        indicators['is_slow_mover_entry'] = True
        indicators['slow_mover_volume_ratio'] = volume_ratio
        indicators['slow_mover_momentum_10'] = momentum_10
        indicators['slow_mover_momentum_20'] = momentum_20
        
        return TradeSignal(
            signal_type='entry',
            ticker=ticker,
            timestamp=pd.to_datetime(current['timestamp']),
            price=current_price,
            pattern_name=best_signal.pattern_name,
            confidence=best_signal.confidence,
            reason=f"SLOW MOVER: {best_signal.pattern_name} with {best_signal.confidence*100:.1f}% confidence, "
                   f"volume_ratio={volume_ratio:.2f}x, momentum_10={momentum_10:.1f}%, momentum_20={momentum_20:.1f}%",
            target_price=target_price,
            stop_loss=stop_loss,
            indicators=indicators
        )
    
    def _calculate_advanced_indicators_for_slow_mover(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced indicators needed for slow mover detection"""
        # Price momentum over different periods
        df['momentum_10'] = df['close'].pct_change(10) * 100
        df['momentum_20'] = df['close'].pct_change(20) * 100
        
        # Volume indicators
        df['volume_ma_10'] = df['volume'].rolling(10).mean()
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        
        # Price position in range
        df['high_10'] = df['high'].rolling(10).max()
        df['low_10'] = df['low'].rolling(10).min()
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        
        # Moving average relationships
        if 'sma_5' not in df.columns:
            df['sma_5'] = df['close'].rolling(5).mean()
        if 'sma_10' not in df.columns:
            df['sma_10'] = df['close'].rolling(10).mean()
        if 'sma_20' not in df.columns:
            df['sma_20'] = df['close'].rolling(20).mean()
        
        df['price_above_all_ma'] = (df['close'] > df['sma_5']) & (df['close'] > df['sma_10']) & (df['close'] > df['sma_20'])
        
        # MACD indicators
        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            # Calculate MACD if not already calculated
            df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        if 'macd_hist' not in df.columns:
            df['macd_hist'] = df['macd'] - df['macd_signal']
        
        df['macd_bullish'] = df['macd'] > df['macd_signal']
        df['macd_hist_accelerating'] = (df['macd_hist'] > df['macd_hist'].shift(1)) & (df['macd_hist'].shift(1) > df['macd_hist'].shift(2))
        
        # Higher highs/lower lows
        df['higher_high_20'] = df['high'] > df['high'].rolling(10).max().shift(10)
        
        # Breakout indicators
        df['breakout_10'] = df['close'] > df['high_10'].shift(1) * 1.02  # 2% above 10-period high
        
        # RSI zones (assuming RSI is already calculated)
        if 'rsi' in df.columns:
            df['rsi_accumulation'] = (df['rsi'] >= 50) & (df['rsi'] <= 65)
        else:
            df['rsi_accumulation'] = False
        
        return df
    
    def _setup_confirmed_multiple_periods(self, df: pd.DataFrame, idx: int, signal: PatternSignal, is_fast_mover: bool = False) -> bool:
        """
        Check that the setup has been valid for multiple periods (not just appeared)
        This ensures sustainability, not just a momentary spike
        RELAXED for fast movers: Fast movers can have explosive moves that don't build over time
        """
        if idx < 5:
            return False
        
        # RELAXED for fast movers: Only require 2 periods (was 4) for fast movers
        # Fast movers can have explosive moves that don't build gradually
        required_periods = 2 if is_fast_mover else 4
        
        # Check last 4-6 periods to ensure setup conditions have been building
        confirmation_periods = 0
        lookback_periods = 4 if is_fast_mover else 6  # Check fewer periods for fast movers
        
        for check_idx in range(max(0, idx-lookback_periods), idx):  # Check last N periods
            check_point = df.iloc[check_idx]
            
            # Check key conditions at this point
            conditions_met = 0
            
            # 1. Price above MAs
            if (check_point.get('close', 0) > check_point.get('sma_10', 0) and
                check_point.get('close', 0) > check_point.get('sma_20', 0)):
                conditions_met += 1
            
            # 2. MACD bullish
            if check_point.get('macd', 0) > check_point.get('macd_signal', 0):
                conditions_met += 1
            
            # 3. Volume above average (lower threshold for fast movers)
            volume_threshold = 1.0 if is_fast_mover else 1.2
            if check_point.get('volume_ratio', 0) > volume_threshold:
                conditions_met += 1
            
            # 4. Price momentum positive
            if check_idx >= 1:
                prev_close = df.iloc[check_idx-1].get('close', 0)
                if check_point.get('close', 0) > prev_close:
                    conditions_met += 1
            
            # RELAXED for fast movers: Only need 2 conditions (was 3) for fast movers
            required_conditions = 2 if is_fast_mover else 3
            if conditions_met >= required_conditions:
                confirmation_periods += 1
        
        # Setup must be confirmed for required periods
        return confirmation_periods >= required_periods
    
    def _check_exit_signals(self, df: pd.DataFrame, ticker: str, current_price: Optional[float] = None) -> List[TradeSignal]:
        """
        Check for exit signals on active positions
        
        Args:
            df: DataFrame with OHLCV data
            ticker: Stock ticker symbol
            current_price: Optional current price from API (for premarket/real-time updates)
        """
        exit_signals = []
        
        if ticker not in self.active_positions:
            return exit_signals
        
        position = self.active_positions[ticker]
        
        # Calculate indicators
        df_with_indicators = self.pattern_detector.calculate_indicators(df)
        
        if len(df_with_indicators) < 1:
            return exit_signals
        
        current = df_with_indicators.iloc[-1]
        # CRITICAL: Always use the latest 1-minute data close price for exit checks
        # The current_price parameter may be from get_current_price() which could return
        # previous day's close if market is closed or API returns stale data
        # For exit decisions, we MUST use the actual latest bar from the DataFrame
        current_price_from_df = current['close']
        current_time = pd.to_datetime(current['timestamp'])
        
        # Use DataFrame price for exit checks (most reliable)
        # Only use provided current_price for display/monitoring, not for exit decisions
        current_price = current_price_from_df
        
        # Log if there's a discrepancy between provided current_price and DataFrame price
        if current_price is not None and abs(current_price - current_price_from_df) > 0.01:
            logger.warning(f"[{ticker}] Price discrepancy: provided={current_price:.4f}, DataFrame={current_price_from_df:.4f} - using DataFrame price for exit check")
        
        # Update position
        position.current_price = current_price
        position.unrealized_pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
        
        if current_price > position.max_price_reached:
            position.max_price_reached = current_price
        
        # PRIORITY 1: Check if entry was during premarket and calculate hold time
        entry_time = position.entry_time
        et = pytz.timezone('US/Eastern')
        
        # Normalize entry_time to ET timezone
        if isinstance(entry_time, pd.Timestamp):
            if entry_time.tz is None:
                entry_time_et = entry_time.tz_localize('US/Eastern')
            else:
                entry_time_et = entry_time.tz_convert('US/Eastern')
        elif hasattr(entry_time, 'astimezone'):
            entry_time_et = entry_time.astimezone(et)
        else:
            entry_time_et = pd.to_datetime(entry_time)
            if entry_time_et.tz is None:
                entry_time_et = entry_time_et.tz_localize('US/Eastern')
            else:
                entry_time_et = entry_time_et.tz_convert('US/Eastern')
        
        entry_hour = entry_time_et.hour
        entry_minute = entry_time_et.minute
        is_premarket_entry = entry_hour < 9 or (entry_hour == 9 and entry_minute < 30)
        
        # Normalize current_time to ET timezone
        if isinstance(current_time, pd.Timestamp):
            if current_time.tz is None:
                current_time_et = current_time.tz_localize('US/Eastern')
            else:
                current_time_et = current_time.tz_convert('US/Eastern')
        elif hasattr(current_time, 'astimezone'):
            current_time_et = current_time.astimezone(et)
        else:
            current_time_et = pd.to_datetime(current_time)
            if current_time_et.tz is None:
                current_time_et = current_time_et.tz_localize('US/Eastern')
            else:
                current_time_et = current_time_et.tz_convert('US/Eastern')
        
        # Calculate minutes since entry
        time_diff = current_time_et - entry_time_et
        minutes_since_entry = time_diff.total_seconds() / 60
        
        # Minimum hold time for premarket entries (15 minutes)
        min_hold_time_premarket = 15
        
        # SLOW MOVER EXIT LOGIC: Use different logic if this is a slow mover entry
        is_slow_mover_entry = position.is_slow_mover_entry
        
        # Check exit conditions
        exit_reason = None
        
        # 0. IMMEDIATE EXIT: Setup failed right after entry (most important)
        # FIX: Only check setup failed after minimum hold time (90 minutes) to avoid premature exits
        min_hold_time_setup_failed = 90
        if minutes_since_entry >= min_hold_time_setup_failed:
            if self._setup_failed_after_entry(df_with_indicators, position, current_time):
                exit_reason = "Setup failed - multiple failure signals detected"
        else:
            logger.debug(f"[{ticker}] {minutes_since_entry:.1f} min since entry, skipping setup failed check (min {min_hold_time_setup_failed} min)")
        
        # 1. Stop loss hit
        if exit_reason is None and current_price <= position.stop_loss:
            exit_reason = f"Stop loss hit at ${position.stop_loss:.4f}"
        
        # 2. Target price reached
        # IMPROVED: For very strong fast movers, don't exit on profit target - let them run with trailing stops
        # Only exit on profit target for normal stocks or if showing clear reversal signs
        is_fast_mover_entry = getattr(position, 'indicators', {}).get('is_fast_mover_entry', False) if hasattr(position, 'indicators') else False
        fast_mover_vol_ratio = getattr(position, 'indicators', {}).get('fast_mover_vol_ratio', 0) if hasattr(position, 'indicators') else 0
        fast_mover_momentum = getattr(position, 'indicators', {}).get('fast_mover_momentum', 0) if hasattr(position, 'indicators') else 0
        is_very_strong_fast_mover = is_fast_mover_entry and (fast_mover_vol_ratio >= 5.0 or fast_mover_momentum >= 10.0)
        
        min_hold_time_profit_target = 20
        entry_hour = position.entry_time.hour
        current_hour = current_time.hour
        current_minute = current_time.minute
        
        # Generic logic: Early morning entries (before 9 AM) should be held longer when profit target is reached
        # This allows capturing more of the morning uptrend before exiting
        is_early_morning_entry = entry_hour < 9
        is_before_morning_peak = current_hour < 10 or (current_hour == 10 and current_minute < 40)
        should_hold_longer = is_early_morning_entry and is_before_morning_peak
        
        if exit_reason is None and minutes_since_entry >= min_hold_time_profit_target:
            if current_price >= position.target_price:
                # For very strong fast movers: Don't exit on profit target, let trailing stops handle it
                if is_very_strong_fast_mover:
                    logger.info(f"[{ticker}] VERY STRONG FAST MOVER: Profit target reached but continuing with trailing stops (vol={fast_mover_vol_ratio:.2f}x, momentum={fast_mover_momentum:.2f}%)")
                elif should_hold_longer:
                    logger.debug(f"[{ticker}] Profit target reached but holding longer (early entry at {entry_hour:02d}:{position.entry_time.minute:02d}, current: {current_hour:02d}:{current_minute:02d})")
                else:
                    exit_reason = f"Profit target reached at ${position.target_price:.4f}"
        elif exit_reason is None and current_price >= position.target_price:
            logger.debug(f"[{ticker}] {minutes_since_entry:.1f} min since entry, profit target reached but skipping (min {min_hold_time_profit_target} min)")
        
        # Time-based exit for early morning entries: exit at 10:40 if still holding
        if exit_reason is None and is_early_morning_entry and current_hour == 10 and current_minute == 40:
            exit_reason = f"Time-based exit at 10:40 (early morning entry held to capture full uptrend)"
        
        # 3. Trailing stop logic (different for slow movers vs normal)
        elif position.max_price_reached > 0 and position.unrealized_pnl_pct >= 3.0:
            if is_slow_mover_entry:
                # SLOW MOVER EXIT LOGIC: Wider trailing stops (5%), minimum hold time 10 minutes
                min_hold_time_slow_mover = 10
                
                # Skip trailing stop if within minimum hold time
                if minutes_since_entry < min_hold_time_slow_mover:
                    logger.debug(f"[{ticker}] Slow mover entry: {minutes_since_entry:.1f} min since entry, skipping trailing stop (min {min_hold_time_slow_mover} min)")
                else:
                    # Slow movers use wider trailing stop: 5% (fixed)
                    trailing_stop_pct = 5.0
                    
                    # Use ATR-based stop if available
                    atr = current.get('atr', 0)
                    if pd.notna(atr) and atr > 0:
                        trailing_stop = position.max_price_reached - (atr * 2.5)  # Wider for slow movers
                        logger.debug(f"[{ticker}] Slow mover ATR-based trailing stop: ${trailing_stop:.4f} (ATR: ${atr:.4f})")
                    else:
                        trailing_stop = position.max_price_reached * (1 - trailing_stop_pct / 100)
                    
                    # Ensure trailing stop never goes below entry price
                    trailing_stop = max(trailing_stop, position.entry_price)
                    
                    # Trailing stop only moves UP, never down
                    if position.trailing_stop_price is None:
                        position.trailing_stop_price = trailing_stop
                        logger.info(f"[{ticker}] Slow mover trailing stop activated at ${trailing_stop:.4f} (+{position.unrealized_pnl_pct:.2f}% profit)")
                    elif trailing_stop > position.trailing_stop_price:
                        position.trailing_stop_price = trailing_stop
                        logger.debug(f"[{ticker}] Slow mover trailing stop moved up to ${trailing_stop:.4f}")
                    
                    if current_price <= position.trailing_stop_price:
                        exit_reason = f"Slow mover trailing stop hit at ${position.trailing_stop_price:.4f} ({trailing_stop_pct:.1f}% from high)"
            else:
                # NORMAL EXIT LOGIC (original logic - unchanged)
                # PRIORITY 1: Minimum hold time protection for premarket entries
                if is_premarket_entry and minutes_since_entry < min_hold_time_premarket:
                    # Don't exit on trailing stop during minimum hold period
                    # Only allow hard stop loss, profit target, setup failure, or trend reversal exits
                    logger.debug(f"[{ticker}] Premarket entry: {minutes_since_entry:.1f} min since entry, skipping trailing stop (min {min_hold_time_premarket} min)")
                else:
                    # Calculate progressive trailing stop based on profit level and position size
                    unrealized_pnl_pct = position.unrealized_pnl_pct
                    
                    # Check if we have partial exits (position size reduced)
                    # After partial exits, use wider trailing stops to let remaining position run
                    has_partial_exits = (position.original_shares > 0 and 
                                       position.shares < position.original_shares)
                    
                    if has_partial_exits:
                        # After partial exits: Very wide trailing stops for remaining position
                        if unrealized_pnl_pct >= 100:
                            trailing_stop_pct = None  # Disable trailing stop for 100%+ profit (let it run)
                        elif unrealized_pnl_pct >= 50:
                            trailing_stop_pct = 30.0  # 30% trailing stop for 50%+ profit (very wide)
                        elif unrealized_pnl_pct >= 30:
                            trailing_stop_pct = 20.0  # 20% trailing stop for 30%+ profit
                        else:
                            trailing_stop_pct = 15.0  # 15% trailing stop
                    else:
                        # IMPROVED: Check if this is a fast mover entry - use wider trailing stops
                        is_fast_mover_entry = getattr(position, 'indicators', {}).get('is_fast_mover_entry', False) if hasattr(position, 'indicators') else False
                        
                        # Full position: Progressive trailing stop width - wider for bigger winners
                        # IMPROVED: Even wider stops for fast movers to capture big runs
                        if is_fast_mover_entry:
                            # Get fast mover strength for even wider stops on very strong movers
                            fast_mover_vol_ratio = getattr(position, 'indicators', {}).get('fast_mover_vol_ratio', 0) if hasattr(position, 'indicators') else 0
                            fast_mover_momentum = getattr(position, 'indicators', {}).get('fast_mover_momentum', 0) if hasattr(position, 'indicators') else 0
                            is_very_strong = fast_mover_vol_ratio >= 5.0 or fast_mover_momentum >= 10.0
                            
                            # Fast movers: Much wider trailing stops to allow for volatility and capture big runs
                            # Very strong fast movers get even wider stops
                            if unrealized_pnl_pct >= 50:
                                trailing_stop_pct = 40.0 if is_very_strong else 30.0  # 40% for very strong, 30% for regular
                            elif unrealized_pnl_pct >= 30:
                                trailing_stop_pct = 30.0 if is_very_strong else 20.0  # 30% for very strong, 20% for regular
                            elif unrealized_pnl_pct >= 20:
                                trailing_stop_pct = 20.0 if is_very_strong else 15.0  # 20% for very strong, 15% for regular
                            elif unrealized_pnl_pct >= 15:
                                trailing_stop_pct = 15.0 if is_very_strong else 12.0  # 15% for very strong, 12% for regular
                            elif unrealized_pnl_pct >= 10:
                                trailing_stop_pct = 12.0 if is_very_strong else 10.0  # 12% for very strong, 10% for regular
                            elif unrealized_pnl_pct >= 5:
                                trailing_stop_pct = 10.0 if is_very_strong else 8.0  # 10% for very strong, 8% for regular
                            else:
                                trailing_stop_pct = 8.0 if is_very_strong else 6.0  # 8% for very strong, 6% for regular
                            
                            # IMPROVED: Delay moving to breakeven for fast movers
                            # Very strong fast movers: Only after +30% profit (let them run)
                            # Regular fast movers: After +15% profit
                            breakeven_threshold = 30.0 if is_very_strong else 15.0
                            if unrealized_pnl_pct >= breakeven_threshold and position.stop_loss < position.entry_price:
                                position.stop_loss = position.entry_price
                                mover_type = "VERY STRONG FAST MOVER" if is_very_strong else "FAST MOVER"
                                logger.info(f"[{ticker}] {mover_type}: Stop moved to breakeven at ${position.entry_price:.4f} (+{unrealized_pnl_pct:.2f}% profit)")
                        else:
                            # Normal stocks: Standard progressive trailing stops
                            if unrealized_pnl_pct >= 50:
                                trailing_stop_pct = 20.0  # 20% trailing stop for 50%+ profit (very wide for massive moves)
                            elif unrealized_pnl_pct >= 30:
                                trailing_stop_pct = 15.0  # 15% trailing stop for 30%+ profit
                            elif unrealized_pnl_pct >= 20:
                                trailing_stop_pct = 12.0  # 12% trailing stop for 20%+ profit
                            elif unrealized_pnl_pct >= 15:
                                trailing_stop_pct = 10.0  # 10% trailing stop for 15%+ profit
                            elif unrealized_pnl_pct >= 10:
                                trailing_stop_pct = 7.0  # 7% trailing stop for 10%+ profit
                            elif unrealized_pnl_pct >= 5:
                                trailing_stop_pct = 5.0  # 5% trailing stop for 5%+ profit
                            else:
                                trailing_stop_pct = 5.0  # 5% initial trailing stop (only if profit >= 3%)
                            
                            # IMPROVED: Delay moving to breakeven - only move after +10% profit (was +5%)
                            # This prevents cutting off big runs too early
                            if unrealized_pnl_pct >= 10.0 and position.stop_loss < position.entry_price:
                                position.stop_loss = position.entry_price
                                logger.info(f"[{ticker}] Stop moved to breakeven at ${position.entry_price:.4f} (+{unrealized_pnl_pct:.2f}% profit)")
                    
                    # PRIORITY 2: Wider trailing stops during premarket (1.5x normal width)
                    # Check if we're still in premarket or if entry was premarket
                    if isinstance(current_time_et, pd.Timestamp):
                        current_hour = current_time_et.hour
                        current_minute = current_time_et.minute
                    else:
                        current_hour = current_time_et.hour
                        current_minute = current_time_et.minute
                    is_currently_premarket = current_hour < 9 or (current_hour == 9 and current_minute < 30)
                    
                    if is_premarket_entry or is_currently_premarket:
                        # Apply 1.5x multiplier for premarket (wider stops)
                        trailing_stop_pct = trailing_stop_pct * 1.5
                        # Cap at reasonable maximum (6%) to prevent excessive risk
                        trailing_stop_pct = min(trailing_stop_pct, 6.0)
                        logger.debug(f"[{ticker}] Premarket: Using wider trailing stop {trailing_stop_pct:.1f}% (1.5x normal)")
                    
                    # FIX: Use ATR-based stop if available (better for volatile stocks)
                    # Otherwise fallback to percentage-based stop
                    # Use the current row (already defined above as df_with_indicators.iloc[-1])
                    atr = current.get('atr', 0)
                    
                    if pd.notna(atr) and atr > 0:
                        # IMPROVED: Use wider ATR multiplier for fast movers (3x-5x vs 2x)
                        # Very strong fast movers get even wider ATR multipliers
                        is_fast_mover_entry = getattr(position, 'indicators', {}).get('is_fast_mover_entry', False) if hasattr(position, 'indicators') else False
                        if is_fast_mover_entry:
                            fast_mover_vol_ratio = getattr(position, 'indicators', {}).get('fast_mover_vol_ratio', 0) if hasattr(position, 'indicators') else 0
                            fast_mover_momentum = getattr(position, 'indicators', {}).get('fast_mover_momentum', 0) if hasattr(position, 'indicators') else 0
                            is_very_strong = fast_mover_vol_ratio >= 5.0 or fast_mover_momentum >= 10.0
                            atr_multiplier = 5.0 if is_very_strong else 3.0  # 5x for very strong, 3x for regular fast movers
                            entry_type = "VERY STRONG FAST MOVER" if is_very_strong else "FAST MOVER"
                        else:
                            atr_multiplier = 2.0
                            entry_type = "NORMAL"
                        trailing_stop = position.max_price_reached - (atr * atr_multiplier)
                        logger.debug(f"[{ticker}] {entry_type} ATR-based trailing stop: ${trailing_stop:.4f} (ATR: ${atr:.4f}, multiplier: {atr_multiplier}x)")
                    else:
                        # Fallback to percentage-based stop
                        trailing_stop = position.max_price_reached * (1 - trailing_stop_pct / 100)
                    
                        # FIX: Ensure trailing stop never goes below entry price
                        # This protects against setting stops that would cause losses
                        trailing_stop = max(trailing_stop, position.entry_price)
                        
                        # FIX: Trailing stop only moves UP, never down
                        # This ensures we protect profits and don't give back gains
                        if position.trailing_stop_price is None:
                            position.trailing_stop_price = trailing_stop
                            logger.info(f"[{ticker}] Trailing stop activated at ${trailing_stop:.4f} (+{unrealized_pnl_pct:.2f}% profit)")
                        elif trailing_stop > position.trailing_stop_price:
                            position.trailing_stop_price = trailing_stop
                            logger.debug(f"[{ticker}] Trailing stop moved up to ${trailing_stop:.4f}")
                        # Never move stop down - this protects profits
                        
                        if current_price <= position.trailing_stop_price:
                            exit_reason = f"Trailing stop hit at ${position.trailing_stop_price:.4f} ({trailing_stop_pct:.1f}% from high)"
        
        # 4. Trend weakness/reversal signals
        # FIX: Add minimum hold time before allowing trend weakness exit (100 minutes for better capture)
        # Also check if we're in a strong uptrend - if so, require even more signals
        min_hold_time_trend_weakness = 100
        if exit_reason is None and minutes_since_entry >= min_hold_time_trend_weakness:
            # Check if price is still well above entry (5%+ profit) - if so, be even more conservative
            if position.unrealized_pnl_pct >= 5.0:
                # For profitable trades, require even more confirmation
                if self._detect_trend_weakness(df_with_indicators, position):
                    exit_reason = "Trend weakness detected"
            elif self._detect_trend_weakness(df_with_indicators, position):
                exit_reason = "Trend weakness detected"
        elif exit_reason is None:
            logger.debug(f"[{ticker}] {minutes_since_entry:.1f} min since entry, skipping trend weakness check (min {min_hold_time_trend_weakness} min)")
        
        # 5. Bearish reversal pattern
        # FIX: Only exit on bearish reversal if price is at or below hard stop loss
        # Otherwise, let the trade continue to capture more profit
        min_hold_time_bearish_reversal = 60
        if exit_reason is None and minutes_since_entry >= min_hold_time_bearish_reversal:
            if self._detect_bearish_reversal(df_with_indicators, position):
                # Only exit if current price is at or below stop loss
                # This prevents premature exits when price is still well above stop loss
                if current_price <= position.stop_loss:
                    exit_reason = "Bearish reversal pattern detected (at stop loss)"
                else:
                    logger.debug(f"[{ticker}] Bearish reversal detected but price ${current_price:.4f} > stop loss ${position.stop_loss:.4f}, continuing trade")
        elif exit_reason is None:
            logger.debug(f"[{ticker}] {minutes_since_entry:.1f} min since entry, skipping bearish reversal check (min {min_hold_time_bearish_reversal} min)")
        
        # 6. Progressive partial profit taking - Lock in profits at multiple levels
        # Strategy: 50% at 20%, 25% at 40%, 12.5% at 80%, hold 12.5% with enhanced logic
        if not exit_reason:  # Only if no other exit reason
            hold_time_min = (current_time - position.entry_time).total_seconds() / 60
            
            # Only allow partial exits after minimum hold time (20 minutes)
            if hold_time_min >= 20:
                if not position.partial_profit_taken and position.unrealized_pnl_pct >= 20.0:
                    # First partial exit: 50% at 20% profit (lock in 10% gain)
                    partial_exit_signal = TradeSignal(
                        signal_type='partial_exit',
                        ticker=ticker,
                        timestamp=current_time,
                        price=current_price,
                        reason=f"Partial profit taking (50%) at +{position.unrealized_pnl_pct:.2f}%",
                        confidence=1.0
                    )
                    exit_signals.append(partial_exit_signal)
                    # Mark that first partial profit was taken (will be set in live_trading_bot)
                elif hasattr(position, 'partial_profit_taken_second') and not position.partial_profit_taken_second and position.unrealized_pnl_pct >= 40.0:
                    # Second partial exit: 25% at 40% profit (lock in additional 10% gain)
                    partial_exit_signal = TradeSignal(
                        signal_type='partial_exit',
                        ticker=ticker,
                        timestamp=current_time,
                        price=current_price,
                        reason=f"Partial profit taking (25%) at +{position.unrealized_pnl_pct:.2f}%",
                        confidence=1.0
                    )
                    exit_signals.append(partial_exit_signal)
                    # Mark that second partial profit was taken (will be set in live_trading_bot)
                elif hasattr(position, 'partial_profit_taken_third') and not position.partial_profit_taken_third and position.unrealized_pnl_pct >= 80.0:
                    # Third partial exit: 12.5% at 80% profit (lock in additional 10% gain)
                    partial_exit_signal = TradeSignal(
                        signal_type='partial_exit',
                        ticker=ticker,
                        timestamp=current_time,
                        price=current_price,
                        reason=f"Partial profit taking (12.5%) at +{position.unrealized_pnl_pct:.2f}%",
                        confidence=1.0
                    )
                    exit_signals.append(partial_exit_signal)
                    # Mark that third partial profit was taken (will be set in live_trading_bot)
        
        if exit_reason:
            exit_signal = TradeSignal(
                signal_type='exit',
                ticker=ticker,
                timestamp=current_time,
                price=current_price,
                reason=exit_reason,
                confidence=1.0
            )
            exit_signals.append(exit_signal)
        
        return exit_signals
    
    def _is_fast_mover(self, df: pd.DataFrame, idx: int) -> Tuple[bool, Dict[str, float]]:
        """
        Detect if a stock is a fast mover with exceptional volume and momentum.
        ADJUSTED: Lower thresholds to catch fast movers earlier (3x volume, 3% momentum)
        
        Args:
            df: DataFrame with indicators
            idx: Current index
            
        Returns:
            Tuple of (is_fast_mover: bool, metrics: dict with vol_ratio and momentum)
        """
        if idx < 5:
            return False, {}
        
        current = df.iloc[idx]
        lookback_10 = df.iloc[idx-10:idx] if idx >= 10 else df.iloc[:idx]
        
        # Check volume ratio - LOWERED from 5.0x to 3.0x to catch fast movers earlier
        volume_ratio = current.get('volume_ratio', 0)
        if volume_ratio < 3.0:  # Must be at least 3x average (was 5x)
            return False, {}
        
        # Check price momentum - LOWERED from 5% to 3% in last 5 periods
        if idx >= 5:
            price_change_5 = ((current.get('close', 0) - df.iloc[idx-5].get('close', 0)) / 
                             df.iloc[idx-5].get('close', 0)) * 100
            if price_change_5 < 3.0:  # Must be at least 3% gain (was 5%)
                return False, {}
        else:
            return False, {}
        
        # Check if volume is increasing (not declining)
        if len(lookback_10) >= 5:
            recent_volumes = lookback_10['volume'].tail(5).values
            if len(recent_volumes) >= 3:
                # Volume should be increasing, not declining
                if recent_volumes[-1] < recent_volumes[0] * 0.9:  # Declining 10%+
                    return False, {}
        
        # All conditions met - this is a fast mover
        metrics = {
            'vol_ratio': volume_ratio,
            'momentum': price_change_5
        }
        return True, metrics
    
    def _validate_entry_signal(self, df: pd.DataFrame, idx: int, signal: PatternSignal, log_reasons: bool = False) -> Tuple[bool, str]:
        """
        Validate that entry signal is a PERFECT setup - very strict criteria
        Only enter on confirmed, high-probability bullish signals
        
        Args:
            df: DataFrame with indicators
            idx: Current index
            signal: Pattern signal to validate
            log_reasons: If True, log detailed rejection reasons
        """
        rejection_reasons = []
        
        if idx < 30:  # Need more history for perfect setup validation
            reason = f"Insufficient history (idx={idx}, need 30+)"
            if log_reasons:
                rejection_reasons.append(reason)
                logger.debug(f"[{signal.ticker}] REJECTED: {', '.join(rejection_reasons)}")
            return False, reason
        
        # PRIORITY 2 FIX: Use best patterns, but allow others with strong confirmations
        best_patterns = [
            'Strong_Bullish_Setup',  # Multiple indicators align
            'Volume_Breakout'  # High volume with price breakout
        ]
        
        # Secondary patterns that are acceptable with strong confirmations
        acceptable_patterns_with_confirmation = [
            'Accumulation_Pattern',  # Volume accumulation with price action
            'MACD_Bullish_Cross',  # MACD crossover with momentum
            'Consolidation_Breakout',  # FIX: Accept Consolidation_Breakout with strong confirmations
            'Golden_Cross',  # FIX: Accept Golden_Cross with strong confirmations
        ]
        
        current = df.iloc[idx]
        
        if signal.pattern_name not in best_patterns:
            # Check if it's an acceptable pattern with strong confirmations
            if signal.pattern_name in acceptable_patterns_with_confirmation:
                # Require stronger confirmations for secondary patterns
                volume_ratio_check = current.get('volume_ratio', 0)
                price_momentum = ((current.get('close', 0) - df.iloc[max(0, idx-5)].get('close', 0)) / 
                                 df.iloc[max(0, idx-5)].get('close', 0)) * 100 if idx >= 5 else 0
                
                # Require: volume ratio > 2x AND (price momentum > 3% OR confidence > 75%)
                if volume_ratio_check >= 2.0 and (price_momentum > 3.0 or signal.confidence >= 0.75):
                    if log_reasons:
                        logger.info(f"[{signal.ticker}] Accepting secondary pattern '{signal.pattern_name}' with strong confirmations (vol={volume_ratio_check:.2f}x, momentum={price_momentum:.1f}%, conf={signal.confidence*100:.1f}%)")
                    # Pattern accepted, continue validation
                else:
                    reason = f"Pattern '{signal.pattern_name}' requires stronger confirmations (vol ratio {volume_ratio_check:.2f}x < 2.0x or momentum {price_momentum:.1f}% < 3% and confidence {signal.confidence*100:.1f}% < 75%)"
                    if log_reasons:
                        rejection_reasons.append(reason)
                        logger.debug(f"[{signal.ticker}] REJECTED: {', '.join(rejection_reasons)}")
                    return False, reason
            else:
                reason = f"Pattern '{signal.pattern_name}' not in best patterns"
                if log_reasons:
                    rejection_reasons.append(reason)
                    logger.debug(f"[{signal.ticker}] REJECTED: {', '.join(rejection_reasons)}")
                return False, reason
        
        lookback_20 = df.iloc[idx-20:idx]
        lookback_10 = df.iloc[idx-10:idx]
        
        # Check if this is a fast mover (exceptional volume and momentum)
        # MOVED EARLIER: Detect fast mover before critical requirements to apply relaxed rules
        is_fast_mover, fast_mover_metrics = self._is_fast_mover(df, idx)
        if is_fast_mover:
            logger.info(f"[{signal.ticker}] FAST MOVER detected: vol_ratio={fast_mover_metrics['vol_ratio']:.2f}x, momentum={fast_mover_metrics['momentum']:.2f}%")
            # Store fast mover status for monitoring
            self.last_fast_mover_status[signal.ticker] = fast_mover_metrics
        else:
            # Clear fast mover status if not detected
            if signal.ticker in self.last_fast_mover_status:
                del self.last_fast_mover_status[signal.ticker]
        
        # PERFECT SETUP REQUIREMENTS - CRITICAL: Must pass ALL core requirements
        # Then score-based system for additional confirmations
        # Fast movers get relaxed requirements for certain checks
        
        # === CRITICAL REQUIREMENTS (ALL MUST PASS) ===
        
        # 1. Price above ALL key moving averages (MANDATORY)
        close = current.get('close', 0)
        sma5 = current.get('sma_5', 0)
        sma10 = current.get('sma_10', 0)
        sma20 = current.get('sma_20', 0)
        if not (close > sma5 and close > sma10 and close > sma20):
            reason = f"Price ${close:.4f} not above all MAs"
            if log_reasons:
                rejection_reasons.append(reason)
            return False, reason  # REJECT if price not above all MAs
        
        # 2. Moving averages in bullish order (MANDATORY)
        # RELAXED for fast movers: Allow if price above all MAs even if not perfect order
        if is_fast_mover:
            # For fast movers: Only require price above all MAs, allow relaxed MA order
            # Check if at least 2 of 3 MA pairs are in bullish order
            ma_order_score = sum([sma5 > sma10, sma10 > sma20])
            if ma_order_score < 1:  # At least one pair must be in order
                reason = "MAs not showing bullish alignment (fast mover)"
                if log_reasons:
                    rejection_reasons.append(reason)
                return False, reason
            else:
                logger.debug(f"[{signal.ticker}] FAST MOVER: Relaxed MA order check (score: {ma_order_score}/2)")
        else:
            # Normal stocks: Strict MA order required
            if not (sma5 > sma10 and sma10 > sma20):
                reason = "MAs not in bullish order"
                if log_reasons:
                    rejection_reasons.append(reason)
                return False, reason  # REJECT if MAs not in bullish order
        
        # 3. Volume must be above average AND absolute volume must be sufficient (MANDATORY)
        volume_ratio = current.get('volume_ratio', 0)
        if volume_ratio < 1.5:  # Must be at least 1.5x average (increased for stronger confirmation)
            reason = f"Volume ratio {volume_ratio:.2f}x < 1.5x required"
            if log_reasons:
                rejection_reasons.append(reason)
            return False, reason  # REJECT if volume not strong
        
        # 3.5. Minimum absolute volume requirement (MANDATORY) - avoid low volume stocks
        # FIX: Use time-based volume thresholds (100K-500K based on time of day)
        current_time = pd.to_datetime(current.get('timestamp', datetime.now()))
        et = pytz.timezone('US/Eastern')
        if current_time.tz is None:
            current_time = et.localize(current_time)
        else:
            current_time = current_time.astimezone(et)
        
        hour = current_time.hour
        
        # Time-based volume thresholds
        if hour < 6:  # 4-6 AM
            min_daily_volume = 100000  # 100K
        elif hour < 8:  # 6-8 AM
            min_daily_volume = 200000  # 200K
        elif hour < 10:  # 8-10 AM
            min_daily_volume = 300000  # 300K
        else:  # 10 AM+
            min_daily_volume = 500000  # 500K
        
        # Check total volume over recent periods (simulating daily volume check)
        if len(df) >= 60:
            recent_volumes = df['volume'].tail(60).values
            total_volume_60min = recent_volumes.sum()
            if total_volume_60min < min_daily_volume:
                reason = f"Low volume stock (total {total_volume_60min:,.0f} < {min_daily_volume:,.0f} over 60 min, threshold for hour {hour})"
                if log_reasons:
                    rejection_reasons.append(reason)
                return False, reason  # REJECT if total volume too low
        elif len(df) >= 20:
            # If less than 60 minutes, check 20-minute total and extrapolate
            recent_volumes = df['volume'].tail(20).values
            total_volume_20min = recent_volumes.sum()
            # Extrapolate to 60 minutes: need at least min_daily_volume/3 over 20 min
            min_volume_20min = min_daily_volume // 3
            if total_volume_20min < min_volume_20min:
                reason = f"Low volume stock (total {total_volume_20min:,.0f} < {min_volume_20min:,.0f} over 20 min, extrapolated, threshold for hour {hour})"
                if log_reasons:
                    rejection_reasons.append(reason)
                return False, reason  # REJECT if volume too low
        else:
            # If very little data, check current volume (should be at least 10K for single bar)
            current_volume = current.get('volume', 0)
            min_current_volume = 10000
            if current_volume < min_current_volume:
                reason = f"Volume {current_volume:,.0f} < {min_current_volume:,.0f} minimum required"
                if log_reasons:
                    rejection_reasons.append(reason)
                return False, reason  # REJECT if volume too low
        
        # Check average volume over recent periods - additional liquidity check
        # Calculate average volume over 20 periods
        # PRIORITY 1 FIX: Use volume ratio as primary check, relax threshold for fast movers
        if len(df) >= 20:
            avg_volume_20 = df['volume'].tail(20).mean()
            
            # Calculate volume ratio (current vs historical average)
            # Use longer-term average for comparison (if available)
            if len(df) >= 100:
                historical_avg = df['volume'].tail(100).mean()
                volume_ratio_long = current.get('volume', 0) / historical_avg if historical_avg > 0 else 0
            else:
                historical_avg = avg_volume_20
                volume_ratio_long = volume_ratio  # Use current volume_ratio if available
            
            # For fast movers (volume ratio > 3x), relax absolute volume requirement
            is_fast_mover_volume = volume_ratio_long >= 3.0
            
            if is_fast_mover_volume:
                # Fast movers: Lower threshold to 30K/minute (was 100K)
                min_avg_volume = 30000
                if log_reasons:
                    logger.info(f"[{signal.ticker}] FAST MOVER VOLUME: Ratio {volume_ratio_long:.2f}x, using relaxed threshold {min_avg_volume:,}")
            else:
                # Normal stocks: Use market cap-adjusted threshold
                # Small cap (<$50M): 30K, Mid cap ($50M-$1B): 50K, Large cap (>$1B): 100K
                # For now, use 50K as default (can be enhanced with market cap data)
                min_avg_volume = 50000
            
            if avg_volume_20 < min_avg_volume:
                reason = f"Low volume stock (avg {avg_volume_20:,.0f} < {min_avg_volume:,} required)"
                if log_reasons:
                    rejection_reasons.append(reason)
                return False, reason  # REJECT if average volume too low
        
        # 4. MACD must be bullish (MANDATORY)
        macd = current.get('macd', 0)
        macd_signal = current.get('macd_signal', 0)
        if macd <= macd_signal:
            reason = "MACD not bullish (MACD <= Signal)"
            if log_reasons:
                rejection_reasons.append(reason)
            return False, reason  # REJECT if MACD not bullish
        
        # 5. Price must be making higher highs (MANDATORY - no recent rejection)
        if len(lookback_20) >= 10:
            recent_highs = lookback_20['high'].tail(10).values
            current_high = current.get('high', 0)
            # Price should be at or near recent high (not rejected)
            if current_high < max(recent_highs) * 0.95:  # More than 5% below recent high
                reason = "Price rejected from recent highs"
                if log_reasons:
                    rejection_reasons.append(reason)
                return False, reason  # REJECT if price was rejected from highs
        
        # 6. No recent price weakness (MANDATORY)
        if len(lookback_10) >= 5:
            recent_closes = lookback_10['close'].tail(5).values
            if len(recent_closes) >= 3:
                # Check if price has been declining
                declining_periods = sum(1 for i in range(1, len(recent_closes)) 
                                     if recent_closes[i] < recent_closes[i-1])
                if declining_periods >= 3:  # 3+ declining periods
                    reason = "Price showing weakness (3+ declining periods)"
                    if log_reasons:
                        rejection_reasons.append(reason)
                    return False, reason  # REJECT if price showing weakness
        
        # 7. Price must be in a longer-term uptrend (MANDATORY - not just recent)
        if len(lookback_20) >= 15:
            # Check price trend over last 15 periods
            older_price = lookback_20.iloc[0].get('close', 0)
            current_price = current.get('close', 0)
            if older_price > 0:
                long_term_change = ((current_price - older_price) / older_price) * 100
                if long_term_change < 2.0:  # Must be up at least 2% over 15 periods
                    reason = f"Not in longer-term uptrend ({long_term_change:.1f}% < 2% required)"
                    if log_reasons:
                        rejection_reasons.append(reason)
                    return False, reason  # REJECT if not in longer-term uptrend
        
        # 8. Price must be making consistent higher lows (MANDATORY - uptrend confirmation)
        # RELAXED for fast movers: Check overall trend, not just recent lows
        if len(lookback_20) >= 10:
            recent_lows = lookback_20['low'].tail(10).values
            if len(recent_lows) >= 5:
                # Check if lows are generally increasing
                older_lows = recent_lows[:5]
                newer_lows = recent_lows[5:]
                avg_older_low = min(older_lows) if len(older_lows) > 0 else 0
                avg_newer_low = min(newer_lows) if len(newer_lows) > 0 else 0
                if avg_older_low > 0 and avg_newer_low < avg_older_low * 0.98:  # Lower lows
                    # For fast movers: Check if overall price trend is still up
                    if is_fast_mover:
                        # Check if current price is still significantly above older price
                        older_price = lookback_20.iloc[0].get('close', 0)
                        current_price = current.get('close', 0)
                        if older_price > 0 and current_price > older_price * 1.05:  # Still up 5%+
                            # Allow if overall trend is up despite temporary lower lows
                            logger.debug(f"[{signal.ticker}] FAST MOVER: Lower lows detected but overall trend up ({((current_price - older_price) / older_price) * 100:.1f}%)")
                        else:
                            reason = "Making lower lows (downtrend)"
                            if log_reasons:
                                rejection_reasons.append(reason)
                            return False, reason
                    else:
                        reason = "Making lower lows (downtrend)"
                        if log_reasons:
                            rejection_reasons.append(reason)
                        return False, reason  # REJECT if making lower lows
        
        # 9. Price momentum must be positive (actively moving up, not at peak) (MANDATORY)
        if idx >= 3:
            recent_closes = df.iloc[idx-3:idx+1]['close'].values
            if len(recent_closes) >= 3:
                # Price should be making higher closes in recent periods
                if recent_closes[-1] <= recent_closes[-2] * 0.995:  # Not moving up
                    reason = "Price not showing upward momentum"
                    if log_reasons:
                        rejection_reasons.append(reason)
                    return False, reason  # REJECT if price not showing upward momentum
        
        # 10. Price stability check (avoid high volatility entries) (MANDATORY)
        # Fast movers bypass this check - high volatility is expected for breakouts
        # INCREASED threshold from 8% to 15% for normal stocks to allow more volatile breakouts
        if not is_fast_mover:
            if len(lookback_10) >= 5:
                recent_highs = lookback_10['high'].tail(5).values
                recent_lows = lookback_10['low'].tail(5).values
                if len(recent_highs) > 0 and len(recent_lows) > 0:
                    price_range_pct = ((max(recent_highs) - min(recent_lows)) / min(recent_lows)) * 100
                    if price_range_pct > 15.0:  # Too volatile (15%+ range in 5 periods, was 8%)
                        reason = f"Too volatile ({price_range_pct:.1f}% range in 5 periods)"
                        if log_reasons:
                            rejection_reasons.append(reason)
                        return False, reason  # REJECT if too volatile
        else:
            logger.info(f"[{signal.ticker}] FAST MOVER: Bypassing volatility check")
        
        # === SCORING SYSTEM FOR ADDITIONAL CONFIRMATIONS ===
        perfect_setup_score = 0
        # BALANCED: Quality trades with reasonable opportunities (6/8 for normal, 5/8 for fast movers)
        # Adjusted from 7/8 to 6/8 to allow more quality trades while maintaining standards
        required_score = 5 if is_fast_mover else 6  # Out of 8 possible points
        if is_fast_mover:
            logger.info(f"[{signal.ticker}] FAST MOVER: Required score {required_score}/8")
        
        # === PRICE ACTION CONFIRMATIONS ===
        
        # 1. Price momentum is strong (recent price increase)
        
        # 1. Price momentum is strong (recent price increase)
        price_change_5 = ((current.get('close', 0) - df.iloc[idx-5].get('close', 0)) / 
                         df.iloc[idx-5].get('close', 0)) * 100 if idx >= 5 else 0
        if price_change_5 > 2.5:  # At least 2.5% gain in last 5 periods (balanced)
            perfect_setup_score += 1
        elif price_change_5 > 1.0:  # Moderate gain
            perfect_setup_score += 0.5
        
        # === VOLUME CONFIRMATIONS ===
        
        # 2. Volume is significantly above average (strong buying interest)
        if volume_ratio > 2.5:  # High volume requirement (balanced)
            perfect_setup_score += 1
        elif volume_ratio > 1.8:  # Good volume
            perfect_setup_score += 0.5
        
        # 3. Volume trend is increasing (not declining)
        if len(lookback_10) >= 5:
            recent_volumes = lookback_10['volume'].tail(5).values
            if len(recent_volumes) >= 3:
                # Check if volume is increasing
                if recent_volumes[-1] > recent_volumes[0] * 1.3:  # 30%+ increase
                    perfect_setup_score += 1
                elif recent_volumes[-1] > recent_volumes[0] * 1.1:  # 10%+ increase
                    perfect_setup_score += 0.5
        
        # Require volume to be increasing (not declining) - MANDATORY
        if len(lookback_10) >= 5:
            recent_volumes = lookback_10['volume'].tail(5).values
            if len(recent_volumes) >= 3:
                # Volume should be increasing, not declining
                if recent_volumes[-1] < recent_volumes[0] * 0.9:  # Declining 10%+
                    reason = "Volume declining"
                    if log_reasons:
                        rejection_reasons.append(reason)
                    return False, reason  # REJECT if volume declining
        
        # === MOMENTUM INDICATORS ===
        
        # 4. RSI in optimal range (not overbought, not oversold)
        rsi = current.get('rsi', 50)
        if 45 < rsi < 70:  # Strong but not overbought (stricter range)
            perfect_setup_score += 1
        elif 40 < rsi <= 45 or 70 <= rsi < 75:  # Acceptable but not ideal
            perfect_setup_score += 0.5
        
        # 5. MACD histogram must be positive AND accelerating (MANDATORY)
        macd_hist = current.get('macd_hist', 0)
        if macd_hist <= 0:
            reason = "MACD histogram not positive"
            if log_reasons:
                rejection_reasons.append(reason)
            return False, reason  # REJECT if histogram not positive
        
        # Require acceleration (not just positive)
        # Reduced requirements: 3% for normal stocks, 2% for fast movers (was 5% and 2%)
        if idx >= 2:
            prev_hist = df.iloc[idx-1].get('macd_hist', 0)
            acceleration_threshold = 1.02 if is_fast_mover else 1.03  # 2% for fast movers, 3% for normal
            if macd_hist <= prev_hist * acceleration_threshold:
                reason = f"MACD histogram not accelerating (need {((acceleration_threshold-1)*100):.1f}% increase)"
                if log_reasons:
                    rejection_reasons.append(reason)
                if is_fast_mover:
                    logger.info(f"[{signal.ticker}] FAST MOVER: Relaxed MACD acceleration requirement to 2%")
                return False, reason  # REJECT if not accelerating
            elif is_fast_mover:
                logger.info(f"[{signal.ticker}] FAST MOVER: Relaxed MACD acceleration requirement to 2%")
            
            # Award points for strong acceleration (already checked above)
            if macd_hist > prev_hist * 1.1:  # 10%+ increase
                perfect_setup_score += 1
            elif macd_hist > prev_hist * 1.05:  # 5-10% increase
                perfect_setup_score += 0.5
        
        # === TREND STRENGTH ===
        
        # 6. Price is breaking out of consolidation (not just a bounce)
        if len(lookback_20) >= 10:
            recent_highs = lookback_20['high'].tail(10).values
            # If price is breaking above recent consolidation
            if current.get('close', 0) > max(recent_highs) * 0.99:  # At or above recent high (stricter)
                perfect_setup_score += 1
            elif current.get('close', 0) > max(recent_highs) * 0.97:  # Near recent high
                perfect_setup_score += 0.5
        
        # 11. Avoid entering at local peak (price at recent high without momentum) (MANDATORY)
        # Fast movers get relaxed threshold (3% momentum instead of 1%) but still checked
        if len(lookback_10) >= 5:
            recent_highs = lookback_10['high'].tail(5).values
            current_high = current.get('high', 0)
            if current_high >= max(recent_highs) * 0.99:  # At or near recent high
                # Check if momentum is still strong
                if idx >= 2:
                    price_change_2 = ((current.get('close', 0) - df.iloc[idx-2].get('close', 0)) / 
                                    df.iloc[idx-2].get('close', 0)) * 100
                    # Fast movers need 3% momentum, normal stocks need 1%
                    required_momentum = 3.0 if is_fast_mover else 1.0
                    if price_change_2 < required_momentum:  # Not enough momentum
                        reason = f"At peak without momentum ({price_change_2:.1f}% < {required_momentum:.1f}% required)"
                        if log_reasons:
                            rejection_reasons.append(reason)
                        return False, reason  # REJECT if at peak without momentum
        if is_fast_mover:
            logger.info(f"[{signal.ticker}] FAST MOVER: Using relaxed peak detection (3% momentum required)")
        
        # 7. Price is making higher highs (uptrend confirmation)
        if len(lookback_20) >= 15:
            older_highs = lookback_20['high'].head(10).values
            newer_highs = lookback_20['high'].tail(5).values
            if len(older_highs) > 0 and len(newer_highs) > 0:
                avg_older_high = max(older_highs)
                avg_newer_high = max(newer_highs)
                if avg_newer_high > avg_older_high * 1.02:  # Higher highs
                    perfect_setup_score += 1
        
        # 8. Bollinger Bands confirm breakout (price near or above upper band on expansion)
        bb_position = current.get('bb_position', 0.5)
        bb_width = current.get('bb_width', 0)
        if bb_position > 0.75 and bb_width > 0.025:  # Price in upper band, bands expanding (stricter)
            perfect_setup_score += 1
        elif bb_position > 0.65 and bb_width > 0.02:
            perfect_setup_score += 0.5
        
        # === ADDITIONAL SAFETY CHECKS (AUTO-REJECT) ===
        
        # Reject if price is too extended (recent massive move)
        # RELAXED for fast movers: Allow up to 20% move in 5 periods (was 10% for all)
        max_extended_pct = 20.0 if is_fast_mover else 10.0
        if price_change_5 > max_extended_pct:
            reason = f"Price too extended ({price_change_5:.1f}% in 5 periods, max {max_extended_pct}% allowed)"
            if log_reasons:
                rejection_reasons.append(reason)
            return False, reason
        
        # Reject if volume is declining significantly (lack of interest)
        if len(lookback_10) >= 5:
            recent_vols = lookback_10['volume'].tail(5).values
            if len(recent_vols) >= 3:
                # Check if volume is consistently declining
                declining_vol_periods = sum(1 for i in range(1, len(recent_vols)) 
                                          if recent_vols[i] < recent_vols[i-1])
                if declining_vol_periods >= 3 and recent_vols[-1] < recent_vols[0] * 0.6:  # 40%+ drop
                    reason = "Volume declining significantly"
                    if log_reasons:
                        rejection_reasons.append(reason)
                    return False, reason  # Volume declining significantly
        
        # Reject if RSI is extremely overbought (>85) or oversold (<25)
        if rsi > 85 or rsi < 25:  # More extreme thresholds
            reason = f"RSI {rsi:.1f} out of range (overbought/oversold)"
            if log_reasons:
                rejection_reasons.append(reason)
            return False, reason
        
        # Reject if price shows strong recent rejection (long upper wick with high volume)
        current_candle = current
        candle_range = current_candle.get('high', 0) - current_candle.get('low', 0)
        if candle_range > 0:
            upper_wick = current_candle.get('high', 0) - max(current_candle.get('open', 0), 
                                                             current_candle.get('close', 0))
            wick_ratio = upper_wick / candle_range
            if wick_ratio > 0.5 and volume_ratio > 2.0:  # 50%+ upper wick with high volume = strong rejection
                reason = "Strong rejection (long upper wick with high volume)"
                if log_reasons:
                    rejection_reasons.append(reason)
                return False, reason
        
        # Reject if price closed significantly below open (strong bearish candle) on high volume
        if (current_candle.get('close', 0) < current_candle.get('open', 0)):
            candle_body_pct = abs(current_candle.get('close', 0) - current_candle.get('open', 0)) / current_candle.get('open', 0) * 100
            if candle_body_pct > 2.0 and volume_ratio > 2.0:  # 2%+ bearish body on high volume
                reason = "Strong bearish candle on high volume"
                if log_reasons:
                    rejection_reasons.append(reason)
                return False, reason
        
        # Reject if there's been a recent failed breakout attempt (more strict)
        if len(lookback_10) >= 7:
            recent_closes = lookback_10['close'].values
            current_price = current.get('close', 0)
            if len(recent_closes) >= 7:
                max_recent = max(recent_closes)
                # If price was significantly higher recently but now lower
                if max_recent > current_price * 1.05:  # Was 5%+ higher (stricter)
                    # Also check if volume was high during the failed attempt
                    max_idx = recent_closes.argmax()
                    if max_idx < len(lookback_10) and lookback_10.iloc[max_idx].get('volume_ratio', 0) > 2.0:
                        reason = "Failed breakout attempt with high volume"
                        if log_reasons:
                            rejection_reasons.append(reason)
                        return False, reason  # Failed breakout attempt with high volume
        
        # Final check: Must meet minimum score for perfect setup
        if perfect_setup_score < required_score:
            reason = f"Perfect setup score {perfect_setup_score:.1f} < required {required_score}"
            if log_reasons:
                rejection_reasons.append(reason)
                logger.info(f"[{signal.ticker}] REJECTED: {', '.join(rejection_reasons)}")
            return False, reason
        
        # All checks passed!
        if log_reasons:
            logger.info(f"[{signal.ticker}] VALIDATED: All checks passed (score: {perfect_setup_score:.1f}/{required_score}, price=${close:.4f}, vol_ratio={volume_ratio:.2f}x)")
        return True, ""
    
    def _is_false_breakout_realtime(self, df: pd.DataFrame, idx: int, signal: PatternSignal) -> bool:
        """
        Enhanced false breakout detection - multiple checks
        """
        if idx < 10:
            return False
        
        entry_price = signal.entry_price
        current_price = df.iloc[idx]['close']
        recent_data = df.iloc[max(0, idx-10):idx+1]
        
        # Check 1: Price broke out but immediately reversed
        if len(recent_data) >= 5:
            highs = recent_data['high'].values
            closes = recent_data['close'].values
            
            # Check if price broke above entry but then closed below
            broke_above = any(h > entry_price * 1.02 for h in highs)  # 2% above entry
            if broke_above:
                # If current close is below entry, it's a false breakout
                if current_price < entry_price * 0.995:  # Below entry
                    return True
                
                # If price broke out but then gave back most of the gain
                max_high = max(highs)
                if max_high > entry_price * 1.05:  # Broke 5% above
                    if current_price < entry_price * 1.01:  # But now only 1% above
                        return True
        
        # Check 2: Volume spike without price follow-through
        if len(recent_data) >= 5:
            volumes = recent_data['volume'].values
            prices = recent_data['close'].values
            
            # Find volume spike
            avg_volume = volumes[:-1].mean() if len(volumes) > 1 else volumes[0]
            if len(volumes) > 0 and volumes[-1] > avg_volume * 2:
                # Volume spiked, but did price move?
                price_change = ((prices[-1] - prices[0]) / prices[0]) * 100
                if price_change < 1.0:  # Less than 1% move despite volume spike
                    return True
        
        # Check 3: Multiple failed breakout attempts
        if len(recent_data) >= 8:
            highs = recent_data['high'].values
            closes = recent_data['close'].values
            
            # Count how many times price tried to break above entry but failed
            failed_attempts = 0
            for i in range(len(highs) - 1):
                if highs[i] > entry_price * 1.01:  # Tried to break out
                    if closes[i+1] < entry_price:  # But closed below entry
                        failed_attempts += 1
            
            if failed_attempts >= 2:  # Multiple failed attempts = false signal
                return True
        
        # Check 4: Price is in a downtrend despite the signal
        if len(recent_data) >= 5:
            recent_closes = recent_data['close'].tail(5).values
            if len(recent_closes) >= 3:
                # Check if price is declining
                if recent_closes[-1] < recent_closes[0] * 0.97:  # Down 3%+
                    return True
        
        return False
    
    def _is_reverse_split_realtime(self, df: pd.DataFrame, idx: int, signal: PatternSignal) -> bool:
        """Check if this looks like a reverse split (real-time version)"""
        if idx < 2:
            return False
        
        # Check for massive price jump
        current_price = df.iloc[idx]['close']
        prev_price = df.iloc[idx-1]['close']
        
        price_jump_pct = ((current_price - prev_price) / prev_price) * 100
        
        # If price jumped >50% in one period, suspicious
        if price_jump_pct > 50:
            # Check volume - reverse splits often have normal volume
            current_volume = df.iloc[idx]['volume']
            avg_volume = df['volume'].rolling(window=20).mean().iloc[idx] if idx >= 20 else df['volume'].mean()
            
            if current_volume < avg_volume * 2:
                return True
        
        return False
    
    def _setup_failed_after_entry(self, df: pd.DataFrame, position: ActivePosition, current_time: datetime) -> bool:
        """
        Check if the setup conditions have failed immediately after entry
        Exit quickly if the perfect setup is no longer valid
        More conservative - requires multiple confirmations to avoid false exits
        """
        # Calculate time since entry
        time_since_entry = (current_time - position.entry_time).total_seconds() / 60  # minutes
        
        # FIX: Only check for first 60 minutes after entry (give trades much more time to develop)
        if time_since_entry > 60:
            return False
        
        if len(df) < 5:
            return False
        
        current = df.iloc[-1]
        current_price = current.get('close', 0)
        
        # Require MULTIPLE failure signals to confirm setup failure (more conservative)
        failure_signals = 0
        critical_failures = 0
        
        # CRITICAL: Price dropped significantly from entry (5%+ drop)
        price_drop = ((position.entry_price - current_price) / position.entry_price) * 100
        if price_drop > 5.0:  # Down more than 5% from entry (more conservative - give trades room)
            critical_failures += 1
        elif price_drop > 4.0:  # 4-5% drop
            failure_signals += 1
        
        # CRITICAL: Price broke below key moving average AND MACD turned bearish
        if current_price < current.get('sma_10', 0):
            macd = current.get('macd', 0)
            macd_signal = current.get('macd_signal', 0)
            if macd < macd_signal:  # Both conditions
                critical_failures += 1
            else:
                failure_signals += 0.5
        
        # MACD turned bearish (momentum lost) - only if price also declining
        macd = current.get('macd', 0)
        macd_signal = current.get('macd_signal', 0)
        if macd < macd_signal:
            if price_drop > 1.0:  # Price also declining
                failure_signals += 1
        
        # Volume dried up significantly (no interest) - only if price also declining
        volume_ratio = current.get('volume_ratio', 0)
        if volume_ratio < 0.6:  # Volume well below average (stricter)
            if price_drop > 1.0:  # Price also declining
                failure_signals += 1
        
        # Price made lower low (reversal starting) - only if very significant
        if len(df) >= 7:
            recent_lows = df['low'].tail(7).values
            if len(recent_lows) >= 4:
                entry_low = min(recent_lows[:3])  # Low near entry
                current_low = min(recent_lows[-2:])  # Recent low
                if entry_low > 0 and current_low < entry_low * 0.95:  # 5%+ lower low (stricter)
                    failure_signals += 1
        
        # FIX: Require at least 1 critical failure OR 7+ regular failure signals (very conservative - give trades maximum room)
        return critical_failures >= 1 or failure_signals >= 7
    
    def _detect_trend_weakness(self, df: pd.DataFrame, position: ActivePosition) -> bool:
        """
        Enhanced trend weakness detection - multiple confirmations required
        Only exit if trend is clearly weakening, not just a small pullback
        """
        if len(df) < 15:
            return False
        
        current = df.iloc[-1]
        recent = df.iloc[-10:]
        
        weakness_signals = 0
        critical_signals = 0  # Critical signals that alone can trigger exit
        required_signals = 5  # FIX: Increased from 4 to 5 - require even more confirmation before exiting
        
        # CRITICAL: Price broke below key moving averages (strong sell signal)
        if current.get('close', 0) < current.get('sma_5', 0):
            critical_signals += 1
        if current.get('close', 0) < current.get('sma_10', 0):
            critical_signals += 1
        
        # CRITICAL: MACD bearish crossover (momentum reversal)
        current_macd = current.get('macd', 0)
        current_macd_signal = current.get('macd_signal', 0)
        current_macd_hist = current.get('macd_hist', 0)
        if len(df) >= 2:
            prev_macd = df.iloc[-2].get('macd', 0)
            prev_signal = df.iloc[-2].get('macd_signal', 0)
            if prev_macd > prev_signal and current_macd < current_macd_signal:  # Just crossed bearish
                critical_signals += 1
        
        # 1. Price is declining from peak (less sensitive - give trades room)
        if position.max_price_reached > 0:
            decline_from_peak = ((position.max_price_reached - current['close']) / position.max_price_reached) * 100
            if decline_from_peak > 4.5:  # Down 4.5%+ from peak (less sensitive)
                weakness_signals += 1
        
        # 2. Volume is declining (lack of interest)
        if len(recent) >= 5:
            recent_volumes = recent['volume'].tail(5).values
            if len(recent_volumes) >= 3:
                if recent_volumes[-1] < recent_volumes[0] * 0.7:  # Volume down 30%+
                    weakness_signals += 1
        
        # 3. Price broke below key moving average (already checked as critical)
        if current.get('close', 0) < current.get('sma_10', 0):
            weakness_signals += 1
        
        # 4. RSI showing bearish divergence or declining
        current_rsi = current.get('rsi', 50)
        if len(recent) >= 5:
            recent_rsi = recent['rsi'].tail(5).values
            if len(recent_rsi) >= 3:
                if recent_rsi[-1] < recent_rsi[0] - 5:  # RSI declining 5+ points
                    weakness_signals += 1
        
        # 5. MACD showing bearish crossover or weakening (already checked as critical)
        if current_macd < current_macd_signal:  # MACD below signal
            weakness_signals += 1
        elif current_macd_hist < 0:  # Histogram negative
            weakness_signals += 0.5
        
        # 6. Price momentum is negative (multiple down periods)
        if len(recent) >= 5:
            recent_closes = recent['close'].tail(5).values
            if len(recent_closes) >= 3:
                down_periods = sum(1 for i in range(1, len(recent_closes)) 
                                 if recent_closes[i] < recent_closes[i-1])
                if down_periods >= 3:  # 3+ down periods
                    weakness_signals += 1
        
        # 7. Price is making lower highs (bearish pattern)
        if len(recent) >= 7:
            recent_highs = recent['high'].tail(7).values
            if len(recent_highs) >= 4:
                # Check if recent highs are declining
                first_half_max = max(recent_highs[:len(recent_highs)//2])
                second_half_max = max(recent_highs[len(recent_highs)//2:])
                if second_half_max < first_half_max * 0.98:  # Lower highs
                    weakness_signals += 1
        
        # FIX: Require at least 3 critical signals OR 7+ weakness signals (much more conservative)
        if critical_signals >= 3:
            return True
        
        # Otherwise, require even more signals for less aggressive exits
        return weakness_signals >= (required_signals + 2)  # Need 7+ signals (was 5+)
    
    def _detect_bearish_reversal(self, df: pd.DataFrame, position: ActivePosition) -> bool:
        """
        Enhanced bearish reversal detection - strong reversal signals only
        Don't exit on minor pullbacks, only on clear reversals
        """
        if len(df) < 15:
            return False
        
        current = df.iloc[-1]
        recent = df.iloc[-10:]
        
        reversal_signals = 0
        required_signals = 3  # FIX: Increased from 2 to 3 - need even clearer reversal signals
        
        # 1. Strong bearish candle pattern
        if len(recent) >= 3:
            # Check for bearish engulfing or strong down candle
            current_candle = current
            prev_candle = df.iloc[-2]
            
            # Bearish engulfing: current opens above prev close, closes below prev open
            if (current_candle['open'] > prev_candle['close'] and 
                current_candle['close'] < prev_candle['open']):
                reversal_signals += 1
            
            # Strong down candle: large red candle
            candle_body = abs(current_candle['close'] - current_candle['open'])
            candle_range = current_candle['high'] - current_candle['low']
            if candle_range > 0:
                body_ratio = candle_body / candle_range
                if body_ratio > 0.7 and current_candle['close'] < current_candle['open']:
                    # Large bearish body
                    if candle_body > current_candle['close'] * 0.03:  # 3%+ body
                        reversal_signals += 1
        
        # 2. MACD bearish crossover (strong signal)
        current_macd = current.get('macd', 0)
        current_macd_signal = current.get('macd_signal', 0)
        prev_macd = df.iloc[-2].get('macd', 0) if len(df) >= 2 else 0
        prev_macd_signal = df.iloc[-2].get('macd_signal', 0) if len(df) >= 2 else 0
        
        # MACD crossed below signal
        if prev_macd > prev_macd_signal and current_macd < current_macd_signal:
            reversal_signals += 1
        
        # 3. RSI showing overbought then reversal
        current_rsi = current.get('rsi', 50)
        if len(recent) >= 5:
            recent_rsi = recent['rsi'].tail(5).values
            max_rsi = max(recent_rsi)
            if max_rsi > 75 and current_rsi < max_rsi - 10:  # Was overbought, now declining
                reversal_signals += 1
        
        # 4. Price broke below support (key moving average or recent low)
        if len(recent) >= 5:
            recent_lows = recent['low'].tail(5).values
            support_level = min(recent_lows)
            if current['close'] < support_level * 0.995:  # Broke below support
                reversal_signals += 1
        
        # 5. Volume spike on down move (distribution)
        if len(recent) >= 5:
            recent_volumes = recent['volume'].tail(5).values
            recent_closes = recent['close'].tail(5).values
            avg_volume = recent_volumes[:-1].mean() if len(recent_volumes) > 1 else recent_volumes[0]
            
            # High volume on down move
            if recent_volumes[-1] > avg_volume * 1.5:  # Volume spike
                if recent_closes[-1] < recent_closes[-2]:  # Price down
                    reversal_signals += 1
        
        # 6. Price is in clear downtrend (below all MAs)
        if (current.get('close', 0) < current.get('sma_5', 0) and
            current.get('close', 0) < current.get('sma_10', 0) and
            current.get('close', 0) < current.get('sma_20', 0)):
            reversal_signals += 1
        
        # Require 2+ reversal signals
        return reversal_signals >= required_signals
    
    def enter_position(self, signal: TradeSignal, df: Optional[pd.DataFrame] = None) -> ActivePosition:
        """
        Enter a new position based on entry signal
        IMPROVED: Dynamic stop loss based on entry type (fast mover vs normal) and volatility
        
        Args:
            signal: Entry signal
            df: Optional DataFrame to calculate ATR-based stop loss
        """
        # Check if this is a fast mover entry - use indicators from signal first, then check df if needed
        is_fast_mover_entry = False
        fast_mover_vol_ratio = 0
        fast_mover_momentum = 0
        
        # First check signal indicators (set during _check_entry_signal)
        if signal.indicators and signal.indicators.get('is_fast_mover_entry', False):
            is_fast_mover_entry = True
            fast_mover_vol_ratio = signal.indicators.get('fast_mover_vol_ratio', 0)
            fast_mover_momentum = signal.indicators.get('fast_mover_momentum', 0)
        elif df is not None and len(df) >= 5:
            # Fallback: detect fast mover from dataframe if not in indicators
            try:
                current_idx = len(df) - 1
                is_fast_mover_entry, fast_mover_metrics = self._is_fast_mover(df, current_idx)
                if is_fast_mover_entry:
                    fast_mover_vol_ratio = fast_mover_metrics.get('vol_ratio', 0)
                    fast_mover_momentum = fast_mover_metrics.get('momentum', 0)
            except Exception as e:
                logger.debug(f"Could not check fast mover status: {e}")
        
        # IMPROVED: Calculate dynamic stop loss using ATR multipliers for better volatility handling
        stop_loss = signal.stop_loss
        if df is not None and len(df) >= 14:
            try:
                atr_series = calculate_atr(df, period=14)
                if len(atr_series) > 0 and atr_series.iloc[-1] > 0:
                    atr = atr_series.iloc[-1]
                    atr_pct = (atr / signal.price) * 100
                    
                    # IMPROVED: Use ATR multipliers (2x-5x ATR) with additional factors for fast movers
                    # This gives more room for volatile stocks while being tighter for calm stocks
                    if is_fast_mover_entry:
                        # Fast movers: Use 3x-5x ATR (wider stops for high volatility)
                        # Base multiplier on ATR percentage
                        if atr_pct > 8:
                            base_multiplier = 4.5  # 4.5x ATR for very high volatility fast movers
                        elif atr_pct > 6:
                            base_multiplier = 4.0  # 4x ATR for high volatility fast movers
                        elif atr_pct > 4:
                            base_multiplier = 3.5  # 3.5x ATR for medium volatility fast movers
                        elif atr_pct > 2:
                            base_multiplier = 3.0  # 3x ATR for low-medium volatility fast movers
                        else:
                            base_multiplier = 2.5  # 2.5x ATR for low volatility fast movers
                        
                        # IMPROVED: Adjust multiplier based on fast mover strength (volume ratio and momentum)
                        # Very strong fast movers need wider stops due to higher volatility potential
                        if fast_mover_vol_ratio >= 5.0 or fast_mover_momentum >= 10.0:
                            # Very strong fast mover: add 0.5x-1.0x to multiplier
                            strength_bonus = 1.0
                        elif fast_mover_vol_ratio >= 4.0 or fast_mover_momentum >= 7.0:
                            # Strong fast mover: add 0.5x to multiplier
                            strength_bonus = 0.5
                        else:
                            strength_bonus = 0.0
                        
                        atr_multiplier = base_multiplier + strength_bonus
                        # Cap at 5x ATR for very extreme cases
                        atr_multiplier = min(atr_multiplier, 5.0)
                    else:
                        # Normal stocks: Use 2x-3x ATR
                        if atr_pct > 6:
                            atr_multiplier = 3.0  # 3x ATR for high volatility
                        elif atr_pct > 4:
                            atr_multiplier = 2.5  # 2.5x ATR for medium volatility
                        else:
                            atr_multiplier = 2.0  # 2x ATR for low volatility
                    
                    # Calculate stop loss using ATR multiplier
                    stop_loss = signal.price - (atr * atr_multiplier)
                    stop_loss_pct = ((signal.price - stop_loss) / signal.price) * 100
                    
                    # IMPROVED: Also check recent volatility (5-period range) and use the wider of ATR-based or recent volatility
                    if len(df) >= 5:
                        recent_highs = df['high'].tail(5).values
                        recent_lows = df['low'].tail(5).values
                        if len(recent_highs) > 0 and len(recent_lows) > 0:
                            recent_range = max(recent_highs) - min(recent_lows)
                            recent_range_pct = (recent_range / min(recent_lows)) * 100
                            
                            # For very volatile stocks, use recent range as additional buffer
                            if recent_range_pct > 10:
                                # Use 50% of recent range as additional buffer for very volatile stocks
                                volatility_buffer = recent_range * 0.5
                                stop_loss = min(stop_loss, signal.price - volatility_buffer)
                                stop_loss_pct = ((signal.price - stop_loss) / signal.price) * 100
                    
                    # IMPROVED: Ensure minimum stop distance for fast movers and penny stocks
                    # Fast movers need wider stops due to higher volatility potential
                    if is_fast_mover_entry:
                        # Fast movers: minimum 8-10% stop depending on strength
                        if fast_mover_vol_ratio >= 5.0 or fast_mover_momentum >= 10.0:
                            min_stop_pct = 10.0  # Very strong fast movers need at least 10%
                        elif fast_mover_vol_ratio >= 4.0 or fast_mover_momentum >= 7.0:
                            min_stop_pct = 9.0  # Strong fast movers need at least 9%
                        else:
                            min_stop_pct = 8.0  # Regular fast movers need at least 8%
                    elif signal.price < 1.0:
                        # Penny stocks: minimum 6-8% stop
                        min_stop_pct = 8.0 if is_fast_mover_entry else 6.0
                    else:
                        # Normal stocks: no minimum (use ATR-based calculation)
                        min_stop_pct = None
                    
                    if min_stop_pct is not None:
                        min_stop = signal.price * (1 - min_stop_pct / 100)
                        if stop_loss > min_stop:  # stop_loss is lower price, so if it's higher, it's closer to entry
                            stop_loss = min_stop
                            stop_loss_pct = min_stop_pct
                    
                    # Ensure stop loss is reasonable (not more than 20% for normal, 25% for fast movers)
                    max_stop_pct = 25.0 if is_fast_mover_entry else 20.0
                    max_stop = signal.price * (1 - max_stop_pct / 100)
                    if stop_loss < max_stop:  # stop_loss is lower price, so if it's lower than max, it's too far
                        stop_loss = max_stop
                        stop_loss_pct = max_stop_pct
                    
                    entry_type = "FAST MOVER" if is_fast_mover_entry else "NORMAL"
                    logger.info(f"[{signal.ticker}] {entry_type} ATR-based stop loss: {stop_loss_pct:.2f}% (ATR: {atr_pct:.2f}%, multiplier: {atr_multiplier:.1f}x)")
            except Exception as e:
                logger.warning(f"Error calculating ATR for {signal.ticker}: {e}")
        
        # Fallback to signal stop_loss or default based on entry type
        if stop_loss is None:
            if is_fast_mover_entry:
                stop_loss = signal.price * 0.92  # 8% default for fast movers
            else:
                stop_loss = signal.price * 0.97  # 3% default for normal stocks
        
        # Check if this is a slow mover entry
        is_slow_mover_entry = signal.indicators.get('is_slow_mover_entry', False) if signal.indicators else False
        
        # IMPROVED: Set dynamic profit targets based on fast mover strength
        # Very strong fast movers need much higher targets to capture big runs
        if signal.target_price:
            target_price = signal.target_price
        elif is_fast_mover_entry:
            # Fast movers: Use higher profit targets based on strength
            if fast_mover_vol_ratio >= 5.0 or fast_mover_momentum >= 10.0:
                # Very strong fast movers: 100% target (let them run to capture massive moves)
                target_price = signal.price * 2.0
                logger.info(f"[{signal.ticker}] VERY STRONG FAST MOVER: Setting 100% profit target (vol={fast_mover_vol_ratio:.2f}x, momentum={fast_mover_momentum:.2f}%)")
            elif fast_mover_vol_ratio >= 4.0 or fast_mover_momentum >= 7.0:
                # Strong fast movers: 50% target
                target_price = signal.price * 1.5
                logger.info(f"[{signal.ticker}] STRONG FAST MOVER: Setting 50% profit target (vol={fast_mover_vol_ratio:.2f}x, momentum={fast_mover_momentum:.2f}%)")
            else:
                # Regular fast movers: 30% target
                target_price = signal.price * 1.3
                logger.info(f"[{signal.ticker}] FAST MOVER: Setting 30% profit target (vol={fast_mover_vol_ratio:.2f}x, momentum={fast_mover_momentum:.2f}%)")
        else:
            # Normal stocks: Use default profit target
            target_price = signal.price * (1 + self.profit_target_pct / 100)
        
        # Store fast mover status in position for exit logic
        position = ActivePosition(
            ticker=signal.ticker,
            entry_time=signal.timestamp,
            entry_price=signal.price,
            entry_pattern=signal.pattern_name or "Unknown",
            entry_confidence=signal.confidence,
            target_price=target_price,
            stop_loss=stop_loss,
            current_price=signal.price,
            max_price_reached=signal.price,
            original_shares=0.0,  # Will be set when shares are assigned
            is_slow_mover_entry=is_slow_mover_entry
        )
        
        # Store fast mover metrics in position for exit logic (use indicators dict if available)
        if is_fast_mover_entry:
            if not hasattr(position, 'indicators'):
                position.indicators = {}
            position.indicators['is_fast_mover_entry'] = True
            position.indicators['fast_mover_vol_ratio'] = fast_mover_vol_ratio
            position.indicators['fast_mover_momentum'] = fast_mover_momentum
        
        self.active_positions[signal.ticker] = position
        self.trade_history.append(signal)
        
        return position
    
    def exit_position(self, signal: TradeSignal) -> Optional[ActivePosition]:
        """Exit an active position"""
        if signal.ticker not in self.active_positions:
            return None
        
        position = self.active_positions.pop(signal.ticker)
        self.trade_history.append(signal)
        
        return position
    
    def get_position_status(self, ticker: str) -> Optional[ActivePosition]:
        """Get current status of active position"""
        return self.active_positions.get(ticker)
    
    def get_all_positions(self) -> Dict[str, ActivePosition]:
        """Get all active positions"""
        return self.active_positions.copy()

