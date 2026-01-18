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
from ..analysis.pattern_detector import PatternDetector, PatternSignal

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
    is_surge_entry: bool = False  # Flag to mark surge entries (uses different exit logic)


class RealtimeTrader:
    """Real-time trading analyzer with entry/exit logic"""
    
    def __init__(self, 
                 min_confidence: float = 0.72,  # BALANCED: 72% - high-quality trades with reasonable opportunities
                 min_entry_price_increase: float = 5.5,  # BALANCED: 5.5% - good quality setups
                 trailing_stop_pct: float = 2.5,  # REFINED: 2.5% - tighter stops, cut losses faster
                 profit_target_pct: float = 8.0,  # REFINED: 8% - realistic profit target
                 data_api=None,  # DataAPI instance for multi-timeframe analysis
                 rejection_callback=None):  # Callback function to save rejections to database
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
        self.rejection_callback = rejection_callback  # Callback to save rejections to database
        
        self.active_positions: Dict[str, ActivePosition] = {}
        self.trade_history: List[TradeSignal] = []
        self.last_rejection_reasons: Dict[str, List[str]] = {}  # Track rejection reasons per ticker
        self.last_fast_mover_status: Dict[str, Dict[str, float]] = {}  # Track fast mover status per ticker
        self.daily_macd_cache: Dict[str, Dict] = {}  # Cache daily MACD values
        
        # Surge detection configuration
        self.surge_detection_enabled: bool = True
        self.surge_min_volume: int = 50000
        self.surge_min_volume_ratio: float = 100.0
        self.surge_min_price_increase: float = 30.0
        self.surge_continuation_min_volume: int = 500000
        self.surge_exit_min_hold_minutes: int = 5
        self.surge_exit_max_hold_minutes: int = 30
        self.surge_exit_trailing_stop_pct: float = 10.0
        self.surge_exit_hard_stop_pct: float = 12.0
    
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
        # CRITICAL FIX: Allow surge detection with minimal data (4 bars)
        # The original 50-bar requirement was blocking early morning surges
        # Surge detection needs current_idx >= 3, which means at least 4 bars total
        # This allows catching surges at 8:46 AM when we only have 4 bars of data
        min_required_bars = 4 if self.surge_detection_enabled else 50
        
        if len(df) < min_required_bars:
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
            # PRIORITY 0: SURGE DETECTION - Check for surges with minimal data (4 bars)
            # This allows catching surges even when we don't have enough data for full pattern detection
            # Surge detection needs current_idx >= 3, which means at least 4 bars total
            # This is critical for stocks that suddenly surge without much trading history (e.g., 8:46 AM)
            if self.surge_detection_enabled and len(df) >= 4:
                # Calculate basic indicators for surge detection (minimal requirements)
                df_with_indicators = self.pattern_detector.calculate_indicators(df)
                if len(df_with_indicators) >= 5:
                    current_idx = len(df_with_indicators) - 1
                    surge_signal = self._detect_price_volume_surge(df_with_indicators, current_idx, ticker)
                    if surge_signal:
                        # Surge detected - create entry signal immediately
                        current = df_with_indicators.iloc[current_idx]
                        current_price = current.get('close', 0)
                        current_time = pd.to_datetime(current.get('timestamp', datetime.now()))
                        if not pd.api.types.is_datetime64_any_dtype(type(current_time)):
                            current_time = datetime.now()
                        
                        # Minimal validation for surge trades
                        if current_price >= 0.50:  # Price filter
                            # Quick reverse split check (only if we have enough data)
                            reverse_split_check_passed = True
                            if current_idx >= 5:
                                prev_5_prices = df_with_indicators.iloc[current_idx-5:current_idx]['close'].values
                                if len(prev_5_prices) > 0 and current_price > max(prev_5_prices) * 3:
                                    # Price more than 3x recent prices - likely reverse split
                                    reverse_split_check_passed = False
                                    logger.warning(f"[{ticker}] SURGE REJECTED: Possible reverse split (price ${current_price:.4f} > 3x recent max ${max(prev_5_prices):.4f})")
                            
                            # Basic price above recent low check (only if we have enough data)
                            dead_cat_bounce_check_passed = True
                            if current_idx >= 10:
                                recent_low = df_with_indicators.iloc[max(0, current_idx-10):current_idx]['low'].min()
                                if current_price < recent_low * 0.8:
                                    dead_cat_bounce_check_passed = False
                                    logger.warning(f"[{ticker}] SURGE REJECTED: Price ${current_price:.4f} below recent low ${recent_low:.4f} (possible dead cat bounce)")
                            
                            # Surge validated if all checks passed
                            if reverse_split_check_passed and dead_cat_bounce_check_passed:
                                # Surge validated - create entry signal
                                surge_type = surge_signal['surge_type']
                                surge_confidence = surge_signal['confidence']
                                
                                # Calculate target and stop loss for surge trades
                                target_price = current_price * 1.25  # 25% target
                                stop_loss = current_price * (1 - self.surge_exit_hard_stop_pct / 100)  # 12% stop
                                
                                entry_signal = TradeSignal(
                                    signal_type='entry',
                                    ticker=ticker,
                                    timestamp=current_time,
                                    price=current_price,
                                    pattern_name='PRICE_VOLUME_SURGE',
                                    confidence=surge_confidence,
                                    reason=f"{surge_type}: Price +{surge_signal['price_change_pct']:.1f}%, Volume {surge_signal['volume_ratio']:.1f}x",
                                    target_price=target_price,
                                    stop_loss=stop_loss,
                                    indicators={
                                        'surge_type': surge_type,
                                        'price_change_pct': surge_signal['price_change_pct'],
                                        'volume_ratio': surge_signal['volume_ratio'],
                                        'baseline_price': surge_signal['baseline_price'],
                                        'baseline_volume': surge_signal['baseline_volume']
                                    }
                                )
                                
                                logger.info(f"[{ticker}] SURGE ENTRY SIGNAL (EARLY): {surge_type} @ ${current_price:.4f} (Confidence: {surge_confidence*100:.0f}%)")
            
            # FIRST: Try original entry logic (requires 30+ bars)
            if entry_signal is None:
                entry_signal = self._check_entry_signal(df, ticker)
            
            # SECOND: If original logic found no entry, try slow mover logic
            if entry_signal is None:
                entry_signal = self._check_slow_mover_entry_signal(df, ticker)
        
        return entry_signal, exit_signals
    
    def _calculate_baseline_metrics(self, df: pd.DataFrame, current_idx: int, lookback_minutes: int = 15) -> Optional[Dict]:
        """
        Calculate baseline price and volume from recent history
        
        Args:
            df: DataFrame with price/volume data
            current_idx: Current index (exclude from baseline)
            lookback_minutes: Minutes to look back (default 15)
        
        Returns:
            Dict with avg_price, avg_volume, min_volume, periods or None if insufficient data
        """
        # Get data from lookback_minutes ago to current_idx-1
        start_idx = max(0, current_idx - lookback_minutes)
        end_idx = current_idx  # Exclude current
        
        if start_idx >= end_idx:
            # Not enough history, use all available
            baseline_df = df.iloc[:current_idx]
        else:
            baseline_df = df.iloc[start_idx:end_idx]
        
        if len(baseline_df) == 0:
            return None
        
        avg_price = baseline_df['close'].mean()
        avg_volume = baseline_df['volume'].mean()
        min_volume = baseline_df['volume'].min()
        
        # Handle zero/very low volume
        if avg_volume < 100:
            avg_volume = 100  # Minimum baseline
        if min_volume < 10:
            min_volume = 10
        
        return {
            'avg_price': avg_price,
            'avg_volume': avg_volume,
            'min_volume': min_volume,
            'periods': len(baseline_df)
        }
    
    def _detect_price_volume_surge(self, df: pd.DataFrame, current_idx: int, ticker: str) -> Optional[Dict]:
        """
        Detect massive price and volume surges
        Requires only 20 bars of data (vs 50 for full pattern detection)
        
        Returns:
            Dict with surge details if detected, None otherwise
        """
        if not self.surge_detection_enabled:
            return None
        
        # Surge detection only needs 3 bars (much less than pattern detection's 50 bars)
        # This allows catching surges early even with limited historical data
        # For stocks with sudden surges, we may not have much history
        # Minimum 3 bars allows: [baseline1, baseline2, baseline3, SURGE]
        if current_idx < 3:  # Minimum 3 bars for basic surge detection
            return None
        
        current = df.iloc[current_idx]
        current_price = current.get('close', 0)
        current_volume = current.get('volume', 0)
        
        # Calculate baseline - use available data (minimum 3 bars, prefer 10-15)
        # For limited data, use shorter lookback to catch surges early
        # This is critical for stocks that suddenly surge without much trading history
        available_bars = current_idx
        if available_bars >= 15:
            lookback_minutes = 15
        elif available_bars >= 10:
            lookback_minutes = 10
        elif available_bars >= 3:
            lookback_minutes = available_bars  # Use all available data (minimum 3 bars)
        else:
            return None  # Not enough data
        
        baseline = self._calculate_baseline_metrics(df, current_idx, lookback_minutes=lookback_minutes)
        if not baseline:
            return None
        
        # Calculate changes
        price_change_pct = ((current_price - baseline['avg_price']) / baseline['avg_price']) * 100
        volume_ratio = current_volume / baseline['avg_volume'] if baseline['avg_volume'] > 0 else 0
        
        # Debug logging for surge detection
        logger.debug(f"[{ticker}] Surge check at idx {current_idx}: price=${current_price:.4f} (baseline=${baseline['avg_price']:.4f}, +{price_change_pct:.1f}%), vol={current_volume:,.0f} (baseline={baseline['avg_volume']:.0f}, {volume_ratio:.1f}x)")
        
        # PRIMARY SURGE: Extreme volume and price increase
        is_primary_surge = (
            current_volume >= self.surge_min_volume and  # Absolute minimum
            (
                (volume_ratio >= self.surge_min_volume_ratio and price_change_pct >= self.surge_min_price_increase) or  # 100x volume + 30% price
                (current_volume >= 200000 and price_change_pct >= 20)  # 200K volume + 20% price
            )
        )
        
        if not is_primary_surge and current_volume >= 50000:
            logger.debug(f"[{ticker}] Surge not detected: vol_ok={current_volume >= self.surge_min_volume}, vol_ratio_ok={volume_ratio >= self.surge_min_volume_ratio}, price_ok={price_change_pct >= self.surge_min_price_increase}, alt_vol_ok={current_volume >= 200000}, alt_price_ok={price_change_pct >= 20}")
        
        # CONTINUATION SURGE: High volume continuation
        is_continuation_surge = False
        price_increase_pct = 0
        volume_increase_pct = 0
        prev_5min_avg_price = 0
        prev_5min_avg_volume = 0
        if not is_primary_surge and current_idx >= 5:
            prev_5min_avg_volume = df.iloc[max(0, current_idx-5):current_idx]['volume'].mean()
            prev_5min_avg_price = df.iloc[max(0, current_idx-5):current_idx]['close'].mean()
            
            volume_increase_pct = ((current_volume - prev_5min_avg_volume) / prev_5min_avg_volume) * 100 if prev_5min_avg_volume > 0 else 0
            price_increase_pct = ((current_price - prev_5min_avg_price) / prev_5min_avg_price) * 100 if prev_5min_avg_price > 0 else 0
            
            prev_volume = df.iloc[current_idx-1].get('volume', 0) if current_idx > 0 else 0
            
            is_continuation_surge = (
                current_volume >= self.surge_continuation_min_volume and
                volume_increase_pct >= 50 and
                current_volume >= (prev_volume * 2) and
                (
                    price_increase_pct >= 10 or
                    (price_increase_pct >= 5 and current_volume >= self.surge_continuation_min_volume)
                )
            )
        
        # CRITICAL: Validate uptrend - ensure price is actually going UP, not down
        # This prevents entering on pullbacks or downtrends after a surge
        is_uptrend = True
        uptrend_reason = ""
        
        # Check 1: Current price should be at or near recent high (within 5% tolerance)
        if current_idx >= 5:
            # Check last 5 minutes for recent high
            recent_5min_prices = df.iloc[max(0, current_idx-5):current_idx]['close'].values
            if len(recent_5min_prices) >= 3:
                recent_high = max(recent_5min_prices)
                # Reject if current price is more than 5% below recent high (significant pullback)
                if current_price < recent_high * 0.95:
                    is_uptrend = False
                    uptrend_reason = f"Price ${current_price:.4f} is {((recent_high - current_price) / recent_high * 100):.1f}% below recent high ${recent_high:.4f}"
        
        # Check 2: Ensure price is not declining from previous bar (significant drop)
        if is_uptrend and current_idx >= 1:
            prev_price = df.iloc[current_idx-1].get('close', 0)
            if prev_price > 0:
                price_change_from_prev = ((current_price - prev_price) / prev_price) * 100
                # Reject if price dropped more than 2% from previous bar
                if price_change_from_prev < -2.0:
                    is_uptrend = False
                    uptrend_reason = f"Price ${current_price:.4f} dropped {abs(price_change_from_prev):.1f}% from previous ${prev_price:.4f}"
        
        # Check 3: For continuation surge, ensure price is actually increasing (not just high volume on decline)
        if is_uptrend and is_continuation_surge:
            if price_increase_pct <= 0:
                is_uptrend = False
                uptrend_reason = f"Continuation surge rejected: Price ${current_price:.4f} is down {abs(price_increase_pct):.1f}% from 5-min avg ${prev_5min_avg_price:.4f}"
        
        if not is_uptrend:
            logger.warning(f"[{ticker}] SURGE REJECTED (downtrend): {uptrend_reason}")
        
        if is_primary_surge and is_uptrend:
            logger.info(f"[{ticker}] PRIMARY SURGE DETECTED: Price ${current_price:.4f} (+{price_change_pct:.1f}%), Volume {current_volume:,.0f} ({volume_ratio:.1f}x baseline)")
            return {
                'surge_type': 'PRIMARY_SURGE',
                'confidence': 0.85,
                'baseline_price': baseline['avg_price'],
                'baseline_volume': baseline['avg_volume'],
                'current_price': current_price,
                'current_volume': current_volume,
                'price_change_pct': price_change_pct,
                'volume_ratio': volume_ratio
            }
        elif is_continuation_surge and is_uptrend:
            logger.info(f"[{ticker}] CONTINUATION SURGE DETECTED: Price ${current_price:.4f} (+{price_increase_pct:.1f}%), Volume {current_volume:,.0f} ({volume_increase_pct:.1f}% increase)")
            return {
                'surge_type': 'CONTINUATION_SURGE',
                'confidence': 0.80,
                'baseline_price': prev_5min_avg_price,
                'baseline_volume': prev_5min_avg_volume,
                'current_price': current_price,
                'current_volume': current_volume,
                'price_change_pct': price_increase_pct,
                'volume_ratio': current_volume / prev_5min_avg_volume if prev_5min_avg_volume > 0 else 0
            }
        
        return None
    
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
        
        # PRIORITY 0.25: SURGE DETECTION - Check for massive price/volume surges BEFORE normal validation
        # This allows immediate entry for explosive moves without waiting for pattern confirmation
        surge_signal = self._detect_price_volume_surge(df_with_indicators, current_idx, ticker)
        if surge_signal:
            # Surge detected - create entry signal immediately with minimal validation
            surge_type = surge_signal['surge_type']
            surge_confidence = surge_signal['confidence']
            
            # Minimal validation for surge trades
            # 1. Price already checked (>= $0.50)
            # 2. Quick reverse split check
            if current_idx >= 5:
                prev_5_prices = df_with_indicators.iloc[current_idx-5:current_idx]['close'].values
                if len(prev_5_prices) > 0 and current_price > max(prev_5_prices) * 3:
                    # Price more than 3x recent prices - likely reverse split
                    logger.warning(f"[{ticker}] SURGE REJECTED: Possible reverse split (price ${current_price:.4f} > 3x recent max ${max(prev_5_prices):.4f})")
                    return None
            
            # 3. Basic price above recent low check (avoid dead cat bounce)
            if current_idx >= 10:
                recent_low = df_with_indicators.iloc[max(0, current_idx-10):current_idx]['low'].min()
                if current_price < recent_low * 0.8:
                    logger.warning(f"[{ticker}] SURGE REJECTED: Price ${current_price:.4f} below recent low ${recent_low:.4f} (possible dead cat bounce)")
                    return None
            
            # Surge validated - create entry signal
            current_time = pd.to_datetime(current.get('timestamp', datetime.now()))
            if not pd.api.types.is_datetime64_any_dtype(type(current_time)):
                current_time = datetime.now()
            
            # Calculate target and stop loss for surge trades
            # More aggressive targets for surges
            target_price = current_price * 1.25  # 25% target
            stop_loss = current_price * (1 - self.surge_exit_hard_stop_pct / 100)  # 12% stop
            
            entry_signal = TradeSignal(
                signal_type='entry',
                ticker=ticker,
                timestamp=current_time,
                price=current_price,
                pattern_name='PRICE_VOLUME_SURGE',
                confidence=surge_confidence,
                reason=f"{surge_type}: Price +{surge_signal['price_change_pct']:.1f}%, Volume {surge_signal['volume_ratio']:.1f}x",
                target_price=target_price,
                stop_loss=stop_loss,
                indicators={
                    'surge_type': surge_type,
                    'price_change_pct': surge_signal['price_change_pct'],
                    'volume_ratio': surge_signal['volume_ratio'],
                    'baseline_price': surge_signal['baseline_price'],
                    'baseline_volume': surge_signal['baseline_volume']
                }
            )
            
            logger.info(f"[{ticker}] SURGE ENTRY SIGNAL: {surge_type} @ ${current_price:.4f} (Confidence: {surge_confidence*100:.0f}%)")
            return entry_signal
        
        # PRIORITY 0.5: Minimum volume filter - reject low volume and extremely slow moving stocks
        # ENHANCED: Use calculated volume thresholds based on stock's historical average volume
        # Purpose: Filter out low-volume/no-movement stocks (liquidity filter)
        
        # Calculate historical average volume (use longer period for better baseline)
        if len(df_with_indicators) >= 100:
            historical_avg_volume = df_with_indicators['volume'].tail(100).mean()
        elif len(df_with_indicators) >= 50:
            historical_avg_volume = df_with_indicators['volume'].tail(50).mean()
        else:
            historical_avg_volume = df_with_indicators['volume'].mean() if len(df_with_indicators) > 0 else 0
        
        # Calculate volume ratio (current vs historical average)
        current_volume = current.get('volume', 0)
        if len(df_with_indicators) >= 100:
            volume_ratio_long = current_volume / historical_avg_volume if historical_avg_volume > 0 else 0
        else:
            avg_volume_20 = df_with_indicators['volume'].tail(20).mean() if len(df_with_indicators) >= 20 else historical_avg_volume
            volume_ratio_long = current_volume / avg_volume_20 if avg_volume_20 > 0 else 0
        
        # PRIORITY: Use volume ratio as primary indicator
        # If volume ratio is exceptional (>= 5x), skip daily volume check (stock is clearly moving)
        if volume_ratio_long >= 5.0:
            logger.debug(f"[{ticker}] EXCEPTIONAL VOLUME: Volume ratio {volume_ratio_long:.2f}x >= 5x, skipping daily volume check")
            # Skip daily volume check - exceptional volume ratio indicates sufficient liquidity
        else:
            # For stocks with moderate volume ratio, check daily volume to filter out low-volume stocks
            # Calculate threshold based on historical average and time of day
            current_time = pd.to_datetime(current.get('timestamp', datetime.now()))
            et = pytz.timezone('US/Eastern')
            if current_time.tz is None:
                current_time = et.localize(current_time)
            else:
                current_time = current_time.astimezone(et)
            
            hour = current_time.hour
            
            # Calculate minimum daily volume based on historical average and time of day
            # Early morning: Lower threshold (volume hasn't accumulated yet)
            # Regular hours: Higher threshold (full trading day)
            # After-hours: Lower threshold (lower absolute volume but high ratio)
            
            if hour < 6:  # 4-6 AM (premarket)
                # Use 20x historical average (very low threshold for early morning)
                min_daily_volume = max(historical_avg_volume * 20, 50000)  # At least 50K absolute minimum
            elif hour < 8:  # 6-8 AM (early morning)
                # Use 40x historical average
                min_daily_volume = max(historical_avg_volume * 40, 100000)  # At least 100K absolute minimum
            elif hour < 10:  # 8-10 AM (mid-morning)
                # Use 60x historical average
                min_daily_volume = max(historical_avg_volume * 60, 150000)  # At least 150K absolute minimum
            elif hour >= 16 and hour < 20:  # After-hours (4 PM - 8 PM)
                # Use 30x historical average (lower for after-hours)
                min_daily_volume = max(historical_avg_volume * 30, 100000)  # At least 100K absolute minimum
                logger.debug(f"[{ticker}] AFTER-HOURS: Using calculated threshold {min_daily_volume:,.0f} (historical avg: {historical_avg_volume:,.0f})")
            elif hour >= 15 and hour < 16:  # Late-day (3 PM - 4 PM)
                # Use 50x historical average
                min_daily_volume = max(historical_avg_volume * 50, 150000)  # At least 150K absolute minimum
                logger.debug(f"[{ticker}] LATE-DAY: Using calculated threshold {min_daily_volume:,.0f} (historical avg: {historical_avg_volume:,.0f})")
            else:  # 10 AM - 3 PM (regular hours)
                # Use 100x historical average (full trading day)
                min_daily_volume = max(historical_avg_volume * 100, 200000)  # At least 200K absolute minimum
            
            # Check total volume over recent periods (simulating daily volume check)
            if len(df_with_indicators) >= 60:
                recent_volumes = df_with_indicators['volume'].tail(60).values
                total_volume_60min = recent_volumes.sum()
                if total_volume_60min < min_daily_volume:
                    self.last_rejection_reasons[ticker] = [f"Low volume stock (total {total_volume_60min:,.0f} < {min_daily_volume:,.0f} over 60 min, calculated from historical avg {historical_avg_volume:,.0f})"]
                    return None
            elif len(df_with_indicators) >= 20:
                # If less than 60 minutes, check 20-minute total and extrapolate
                recent_volumes = df_with_indicators['volume'].tail(20).values
                total_volume_20min = recent_volumes.sum()
                # Extrapolate to 60 minutes: need at least min_daily_volume/3 over 20 min
                min_volume_20min = min_daily_volume // 3
                if total_volume_20min < min_volume_20min:
                    self.last_rejection_reasons[ticker] = [f"Low volume stock (total {total_volume_20min:,.0f} < {min_volume_20min:,.0f} over 20 min, extrapolated, calculated from historical avg {historical_avg_volume:,.0f})"]
                    return None
            else:
                # If very little data, check current volume (should be at least 2x historical average)
                min_current_volume = max(historical_avg_volume * 2, 10000)  # At least 2x average or 10K absolute minimum
                if current_volume < min_current_volume:
                    self.last_rejection_reasons[ticker] = [f"Low volume stock ({current_volume:,.0f} < {min_current_volume:,.0f} required, calculated from historical avg {historical_avg_volume:,.0f})"]
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
        
        pattern_names = [s.pattern_name for s in signals]
        logger.info(f"[{ticker}] Found {len(signals)} pattern signal(s): {', '.join(pattern_names)}")
        
        # Clear previous rejection reasons for this ticker
        self.last_rejection_reasons[ticker] = []
        
        # Filter and validate signals - VERY STRICT
        for signal in signals:
            # PRIORITY 0.5: Check entry price is above minimum
            if signal.entry_price < 0.50:
                reason = f"Entry price ${signal.entry_price:.4f} below minimum $0.50"
                self.last_rejection_reasons[ticker].append(reason)
                # FIX: Save rejection to database via callback
                if self.rejection_callback:
                    try:
                        self.rejection_callback(ticker, signal.entry_price, reason)
                    except Exception as e:
                        logger.error(f"Error saving rejection to database for {ticker}: {e}")
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
            
            # ENHANCED: Adjust confidence threshold based on time and fast mover status
            effective_min_confidence = self.min_confidence  # Default 72%
            
            # Time-based thresholds
            if hour < 10:  # Before 10 AM
                effective_min_confidence = 0.70
                logger.debug(f"[{ticker}] EARLY MORNING: Using relaxed confidence threshold 70% (hour={hour})")
            elif hour >= 16 and hour < 20:  # After-hours (4 PM - 8 PM)
                effective_min_confidence = 0.70
                logger.debug(f"[{ticker}] AFTER-HOURS: Using relaxed confidence threshold 70% (hour={hour})")
            elif hour >= 15 and hour < 16:  # Late-day (3 PM - 4 PM)
                effective_min_confidence = 0.70
                logger.debug(f"[{ticker}] LATE-DAY: Using relaxed confidence threshold 70% (hour={hour})")
            
            # PRIORITY FIX: Fast mover thresholds (override time-based if applicable)
            # Ensure fast mover status is recognized before confidence check
            if is_fast_mover:
                # For exceptional volume (10x+), reduce to 68%
                if volume_ratio >= 10.0:
                    effective_min_confidence = 0.68
                    logger.debug(f"[{ticker}] EXCEPTIONAL VOLUME: Volume ratio {volume_ratio:.2f}x >= 10x, using confidence threshold 68%")
                # For very strong fast movers (6x+ OR momentum >= 10%), reduce to 68%
                elif volume_ratio >= 6.0 or price_momentum_5 >= 10.0:
                    effective_min_confidence = 0.68
                    logger.debug(f"[{ticker}] VERY STRONG FAST MOVER: Vol {volume_ratio:.2f}x OR Momentum {price_momentum_5:.1f}%, using confidence threshold 68%")
                # For strong fast movers (4x+ OR momentum >= 5%), use 70%
                elif volume_ratio >= 4.0 or price_momentum_5 >= 5.0:
                    effective_min_confidence = 0.70
                    logger.debug(f"[{ticker}] STRONG FAST MOVER: Vol {volume_ratio:.2f}x OR Momentum {price_momentum_5:.1f}%, using confidence threshold 70%")
                # For regular fast movers (3x+ AND momentum >= 3%), use 70%
                else:
                    effective_min_confidence = 0.70
                    logger.debug(f"[{ticker}] FAST MOVER: Using confidence threshold 70% (vol={volume_ratio:.2f}x, momentum={price_momentum_5:.1f}%)")
            
            # Check minimum confidence with adjusted threshold
            if signal.confidence < effective_min_confidence:
                reason = f"Confidence {signal.confidence*100:.1f}% < {effective_min_confidence*100:.0f}% required"
                self.last_rejection_reasons[ticker].append(reason)
                # FIX: Save rejection to database via callback
                if self.rejection_callback:
                    try:
                        self.rejection_callback(ticker, signal.entry_price, reason)
                    except Exception as e:
                        logger.error(f"Error saving rejection to database for {ticker}: {e}")
                continue
            
            # PRIORITY 1: Check for false breakouts FIRST (most important filter)
            # ENHANCED: Skip false breakout for fast movers, high-confidence patterns, and exceptional volume
            skip_false_breakout = False
            
            # Skip for high-confidence patterns (75%+)
            if signal.confidence >= 0.75:
                skip_false_breakout = True
                logger.debug(f"[{ticker}] HIGH CONFIDENCE ({signal.confidence*100:.1f}%): Skipping false breakout check")
            
            # Skip for fast movers with strong volume and momentum
            elif is_fast_mover and volume_ratio >= 2.0 and price_momentum_5 >= 3.0:
                skip_false_breakout = True
                logger.debug(f"[{ticker}] FAST MOVER: Skipping false breakout check (vol={volume_ratio:.2f}x, momentum={price_momentum_5:.1f}%)")
            
            # Skip for exceptional volume ratio (5x+)
            elif volume_ratio >= 5.0:
                skip_false_breakout = True
                logger.debug(f"[{ticker}] EXCEPTIONAL VOLUME: Skipping false breakout check (vol={volume_ratio:.2f}x)")
            
            # Skip for fast movers with 70%+ confidence (fallback)
            elif is_fast_mover and signal.confidence >= 0.70:
                skip_false_breakout = True
                logger.debug(f"[{ticker}] FAST MOVER with 70%+ confidence: Skipping false breakout check")
            
            if not skip_false_breakout and self._is_false_breakout_realtime(df_with_indicators, current_idx, signal):
                reason = "False breakout detected"
                self.last_rejection_reasons[ticker].append(reason)
                # FIX: Save rejection to database via callback
                if self.rejection_callback:
                    try:
                        self.rejection_callback(ticker, signal.entry_price, reason)
                    except Exception as e:
                        logger.error(f"Error saving rejection to database for {ticker}: {e}")
                continue
            
            # PRIORITY 2: Check for reverse split (shouldn't happen in real-time, but check anyway)
            if self._is_reverse_split_realtime(df_with_indicators, current_idx, signal):
                reason = "Reverse split detected"
                self.last_rejection_reasons[ticker].append(reason)
                # FIX: Save rejection to database via callback
                if self.rejection_callback:
                    try:
                        self.rejection_callback(ticker, signal.entry_price, reason)
                    except Exception as e:
                        logger.error(f"Error saving rejection to database for {ticker}: {e}")
                continue
            
            # PRIORITY 3: Validate perfect setup (comprehensive check)
            validation_result, rejection_reason = self._validate_entry_signal(df_with_indicators, current_idx, signal, log_reasons=True)
            if not validation_result:
                if rejection_reason:
                    self.last_rejection_reasons[ticker].append(rejection_reason)
                    # FIX: Save rejection to database via callback
                    if self.rejection_callback:
                        try:
                            self.rejection_callback(ticker, signal.entry_price, rejection_reason)
                        except Exception as e:
                            logger.error(f"Error saving rejection to database for {ticker}: {e}")
                continue
            
            # PRIORITY 4: Setup must be confirmed for multiple periods (not just appeared)
            # ENHANCED: Relaxed for fast movers, after-hours, and late-day
            is_fast_mover_check, fast_mover_metrics_check = self._is_fast_mover(df_with_indicators, current_idx)
            vol_ratio_check = fast_mover_metrics_check.get('vol_ratio', 0)
            
            # For very strong fast movers (6x+ volume), reduce to 1 period
            # For very strong fast movers (4x+ volume, 10%+ momentum), skip setup confirmation entirely
            if is_fast_mover_check and vol_ratio_check >= 4.0 and fast_mover_metrics_check.get('momentum', 0) >= 10.0:
                logger.info(f"[{ticker}] VERY STRONG FAST MOVER: Skipping setup confirmation (vol={vol_ratio_check:.2f}x, momentum={fast_mover_metrics_check.get('momentum', 0):.2f}%)")
            else:
                # Determine time-based requirements
                is_after_hours = hour >= 16 and hour < 20
                is_late_day = hour >= 15 and hour < 16
                
                # Set min/max periods based on context
                if is_fast_mover_check and vol_ratio_check >= 6.0:
                    min_periods = 1
                    max_periods = 2
                    logger.debug(f"[{ticker}] VERY STRONG FAST MOVER: Reduced confirmation to 1 period (vol={vol_ratio_check:.2f}x)")
                elif is_after_hours or is_late_day:
                    min_periods = 1 if is_fast_mover_check else 2
                    max_periods = 4
                    logger.debug(f"[{ticker}] AFTER-HOURS/LATE-DAY: Reduced confirmation to {min_periods} period(s)")
                else:
                    min_periods = None  # Use default
                    max_periods = None
                
                if not self._setup_confirmed_multiple_periods(df_with_indicators, current_idx, signal, 
                                                               is_fast_mover=is_fast_mover_check,
                                                               min_periods=min_periods,
                                                               max_periods=max_periods):
                    reason = "Setup not confirmed for multiple periods"
                    self.last_rejection_reasons[ticker].append(reason)
                    # FIX: Save rejection to database via callback
                    if self.rejection_callback:
                        try:
                            self.rejection_callback(ticker, signal.entry_price, reason)
                        except Exception as e:
                            logger.error(f"Error saving rejection to database for {ticker}: {e}")
                    logger.debug(f"[{ticker}] Setup confirmation failed (fast_mover={is_fast_mover_check})")
                    continue
            
            # PRIORITY 5: Check expected gain meets minimum
            expected_gain = ((signal.target_price - signal.entry_price) / signal.entry_price) * 100
            if expected_gain < self.min_entry_price_increase:
                reason = f"Expected gain {expected_gain:.2f}% < minimum {self.min_entry_price_increase:.2f}%"
                self.last_rejection_reasons[ticker].append(reason)
                # FIX: Save rejection to database via callback
                if self.rejection_callback:
                    try:
                        self.rejection_callback(ticker, signal.entry_price, reason)
                    except Exception as e:
                        logger.error(f"Error saving rejection to database for {ticker}: {e}")
                logger.debug(f"[{ticker}] Expected gain {expected_gain:.2f}% < minimum {self.min_entry_price_increase:.2f}%")
                continue
            
            # PRIORITY 6: Final confirmation - price must be confirming the signal NOW
            current_price = current.get('close', 0)
            if current_price < signal.entry_price * 0.98:  # Price already dropped 2% from signal
                reason = f"Price dropped {((signal.entry_price - current_price) / signal.entry_price * 100):.2f}% from signal entry price"
                self.last_rejection_reasons[ticker].append(reason)
                # FIX: Save rejection to database via callback
                if self.rejection_callback:
                    try:
                        self.rejection_callback(ticker, signal.entry_price, reason)
                    except Exception as e:
                        logger.error(f"Error saving rejection to database for {ticker}: {e}")
                logger.debug(f"[{ticker}] Price dropped from ${signal.entry_price:.4f} to ${current_price:.4f}")
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
        
        # PRIORITY 0.5: Slow mover volume check - ENHANCED: Use calculated thresholds based on historical average
        # Purpose: Filter out low-volume/no-movement stocks (liquidity filter)
        
        # Calculate historical average volume (use longer period for better baseline)
        if len(df_with_indicators) >= 100:
            historical_avg_volume = df_with_indicators['volume'].tail(100).mean()
        elif len(df_with_indicators) >= 50:
            historical_avg_volume = df_with_indicators['volume'].tail(50).mean()
        else:
            historical_avg_volume = df_with_indicators['volume'].mean() if len(df_with_indicators) > 0 else 0
        
        # Calculate volume ratio (current vs historical average)
        current_volume = current.get('volume', 0)
        if len(df_with_indicators) >= 100:
            volume_ratio_long = current_volume / historical_avg_volume if historical_avg_volume > 0 else 0
        else:
            avg_volume_20 = df_with_indicators['volume'].tail(20).mean() if len(df_with_indicators) >= 20 else historical_avg_volume
            volume_ratio_long = current_volume / avg_volume_20 if avg_volume_20 > 0 else 0
        
        # PRIORITY: Use volume ratio as primary indicator
        # If volume ratio is exceptional (>= 5x), skip daily volume check (stock is clearly moving)
        if volume_ratio_long >= 5.0:
            logger.debug(f"[{ticker}] SLOW MOVER: EXCEPTIONAL VOLUME: Volume ratio {volume_ratio_long:.2f}x >= 5x, skipping daily volume check")
            # Skip daily volume check - exceptional volume ratio indicates sufficient liquidity
        else:
            # For stocks with moderate volume ratio, check daily volume to filter out low-volume stocks
            # Calculate threshold based on historical average and time of day
            current_time = pd.to_datetime(current.get('timestamp', datetime.now()))
            et = pytz.timezone('US/Eastern')
            if current_time.tz is None:
                current_time = et.localize(current_time)
            else:
                current_time = current_time.astimezone(et)
            
            hour = current_time.hour
            
            # Calculate minimum daily volume based on historical average and time of day
            # Slow movers use lower thresholds than normal entries (40% of normal thresholds)
            if hour < 6:  # 4-6 AM (premarket)
                # Use 8x historical average (40% of normal 20x)
                min_daily_volume = max(historical_avg_volume * 8, 20000)  # At least 20K absolute minimum
            elif hour < 8:  # 6-8 AM (early morning)
                # Use 16x historical average (40% of normal 40x)
                min_daily_volume = max(historical_avg_volume * 16, 40000)  # At least 40K absolute minimum
            elif hour < 10:  # 8-10 AM (mid-morning)
                # Use 24x historical average (40% of normal 60x)
                min_daily_volume = max(historical_avg_volume * 24, 60000)  # At least 60K absolute minimum
            elif hour >= 16 and hour < 20:  # After-hours (4 PM - 8 PM)
                # Use 12x historical average (40% of normal 30x)
                min_daily_volume = max(historical_avg_volume * 12, 40000)  # At least 40K absolute minimum
            elif hour >= 15 and hour < 16:  # Late-day (3 PM - 4 PM)
                # Use 20x historical average (40% of normal 50x)
                min_daily_volume = max(historical_avg_volume * 20, 60000)  # At least 60K absolute minimum
            else:  # 10 AM - 3 PM (regular hours)
                # Use 40x historical average (40% of normal 100x)
                min_daily_volume = max(historical_avg_volume * 40, 80000)  # At least 80K absolute minimum
            
            # Check total volume over recent periods (simulating daily volume check)
            if len(df_with_indicators) >= 60:
                recent_volumes = df_with_indicators['volume'].tail(60).values
                total_volume_60min = recent_volumes.sum()
                if total_volume_60min < min_daily_volume:
                    logger.debug(f"[{ticker}] SLOW MOVER: Low volume stock (total {total_volume_60min:,.0f} < {min_daily_volume:,.0f} over 60 min, calculated from historical avg {historical_avg_volume:,.0f})")
                    return None
            elif len(df_with_indicators) >= 20:
                recent_volumes = df_with_indicators['volume'].tail(20).values
                total_volume_20min = recent_volumes.sum()
                # Extrapolate to 60 minutes: need at least min_daily_volume/3 over 20 min
                min_volume_20min = min_daily_volume // 3
                if total_volume_20min < min_volume_20min:
                    logger.debug(f"[{ticker}] SLOW MOVER: Low volume stock (total {total_volume_20min:,.0f} < {min_volume_20min:,.0f} over 20 min, extrapolated, calculated from historical avg {historical_avg_volume:,.0f})")
                    return None
            else:
                # If very little data, check current volume (should be at least 1.5x historical average for slow movers)
                min_current_volume = max(historical_avg_volume * 1.5, 5000)  # At least 1.5x average or 5K absolute minimum
                if current_volume < min_current_volume:
                    logger.debug(f"[{ticker}] SLOW MOVER: Low volume stock ({current_volume:,.0f} < {min_current_volume:,.0f} required, calculated from historical avg {historical_avg_volume:,.0f})")
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
    
    def _setup_confirmed_multiple_periods(self, df: pd.DataFrame, idx: int, signal: PatternSignal, 
                                          is_fast_mover: bool = False,
                                          min_periods: Optional[int] = None,
                                          max_periods: Optional[int] = None) -> bool:
        """
        Check that the setup has been valid for multiple periods (not just appeared)
        This ensures sustainability, not just a momentary spike
        ENHANCED: Relaxed for fast movers, after-hours, and late-day
        
        Args:
            min_periods: Minimum periods required (default: 2 for fast movers, 4 for normal)
            max_periods: Maximum periods to check (default: 4 for fast movers, 6 for normal)
        """
        if idx < 5:
            return False
        
        # Set default periods if not provided
        if min_periods is None:
            min_periods = 2 if is_fast_mover else 4
        if max_periods is None:
            max_periods = 4 if is_fast_mover else 6
        
        # For very strong fast movers (6x+ volume), reduce to 1 period
        if is_fast_mover and min_periods == 1:
            max_periods = min(max_periods, 2)  # Only check 2 periods max
        
        # Check last N periods to ensure setup conditions have been building
        confirmation_periods = 0
        lookback_periods = max_periods
        
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
        return confirmation_periods >= min_periods
    
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
        
        # SURGE EXIT LOGIC: Use different logic if this is a surge entry
        is_surge_entry = position.is_surge_entry
        
        # Check exit conditions
        exit_reason = None
        
        # SURGE-SPECIFIC EXIT LOGIC (check before normal exit logic)
        if is_surge_entry:
            # Surge trades have tighter stops and shorter hold times
            min_hold_time_surge = self.surge_exit_min_hold_minutes
            max_hold_time_surge = self.surge_exit_max_hold_minutes
            
            # 1. Maximum hold time (30 minutes) - surges can reverse quickly
            if minutes_since_entry >= max_hold_time_surge:
                exit_reason = f"Surge max hold time reached ({max_hold_time_surge} minutes)"
            
            # 2. Hard stop loss (12% - tighter than normal 15%)
            # OPTIMIZED: Don't exit on stop loss if price dropped too rapidly (suggests recovery) or volume is still strong
            elif current_price <= position.stop_loss:
                # Check if price dropped too rapidly from peak (suggests it might recover)
                rapid_drop = False
                price_recovering = False
                volume_still_strong = False
                price_above_entry = current_price > position.entry_price
                
                if len(df_with_indicators) >= 5 and position.max_price_reached > 0:
                    recent_bars = df_with_indicators.iloc[-5:]
                    
                    # Check if price dropped very rapidly from peak (more than 20% in short time)
                    drop_from_peak = ((position.max_price_reached - current_price) / position.max_price_reached) * 100
                    if drop_from_peak > 20.0:  # Dropped more than 20% from peak
                        rapid_drop = True
                    
                    # Check if last 2 bars are showing recovery (higher closes)
                    if len(recent_bars) >= 2:
                        prev_close = recent_bars.iloc[-2]['close']
                        curr_close = recent_bars.iloc[-1]['close']
                        if curr_close > prev_close:
                            price_recovering = True
                    
                    # Check if volume is still strong (above average of recent bars)
                    current_vol = recent_bars.iloc[-1].get('volume', 0)
                    avg_vol = recent_bars['volume'].mean()
                    if current_vol > avg_vol * 0.7:  # Still 70%+ of average volume
                        volume_still_strong = True
                
                # OPTIMIZED: Don't exit on stop loss if:
                # - (Price is still above entry OR rapid drop suggests recovery) AND
                # - (Price is recovering OR volume is still strong) AND
                # - We haven't hit max hold time yet AND
                # - We're still early in the trade (less than 20 minutes)
                should_continue = False
                if minutes_since_entry < 20:  # Still early in trade
                    if (price_above_entry or rapid_drop) and (price_recovering or volume_still_strong):
                        should_continue = True
                elif price_above_entry and (price_recovering or volume_still_strong) and minutes_since_entry < max_hold_time_surge:
                    should_continue = True
                
                if should_continue:
                    logger.info(f"[{ticker}] SURGE: Stop loss hit but continuing (recovering={price_recovering}, vol_strong={volume_still_strong}, rapid_drop={rapid_drop}, above_entry={price_above_entry})")
                else:
                    exit_reason = f"Surge stop loss hit at ${position.stop_loss:.4f} ({self.surge_exit_hard_stop_pct:.1f}%)"
            
            # 3. Progressive profit taking (only after minimum hold time)
            elif minutes_since_entry >= min_hold_time_surge:
                profit_pct = position.unrealized_pnl_pct
                
                # 20% profit: Take 50% of position (extended from 15%)
                if profit_pct >= 20.0 and not position.partial_profit_taken:
                    exit_reason = f"Surge profit target 1: 20% profit - taking 50% of position"
                    position.partial_profit_taken = True
                    position.original_shares = position.shares
                    position.shares = position.shares * 0.5
                    logger.info(f"[{ticker}] SURGE: Taking 50% profit at +{profit_pct:.1f}% (${current_price:.4f})")
                
                # 30% profit: Take another 25% of original position (50% of remaining)
                elif profit_pct >= 30.0 and position.partial_profit_taken and not position.partial_profit_taken_second:
                    exit_reason = f"Surge profit target 2: 30% profit - taking 25% of original position"
                    position.partial_profit_taken_second = True
                    if position.original_shares > 0:
                        # Take 50% of remaining shares (which is 25% of original)
                        position.shares = position.shares * 0.5
                    logger.info(f"[{ticker}] SURGE: Taking 25% of original position at +{profit_pct:.1f}% (${current_price:.4f})")
                
                # 50% profit: Take remaining position (extended from 40%)
                elif profit_pct >= 50.0:
                    exit_reason = f"Surge profit target 3: 50% profit - taking remaining position"
                    logger.info(f"[{ticker}] SURGE: Taking remaining position at +{profit_pct:.1f}% (${current_price:.4f})")
                
                # 4. Trailing stop (10% - tighter than normal)
                # OPTIMIZED: Use wider trailing stops for strong surge moves to capture more
                elif position.max_price_reached > 0 and position.unrealized_pnl_pct >= 3.0:
                    # Use wider trailing stops for stronger moves
                    if position.unrealized_pnl_pct >= 15.0:
                        trailing_stop_pct = 15.0  # 15% trailing stop for 15%+ profit (wider)
                    elif position.unrealized_pnl_pct >= 10.0:
                        trailing_stop_pct = 12.0  # 12% trailing stop for 10%+ profit
                    else:
                        trailing_stop_pct = self.surge_exit_trailing_stop_pct  # 10% default
                    
                    trailing_stop = position.max_price_reached * (1 - trailing_stop_pct / 100)
                    
                    # Ensure trailing stop never goes below entry price
                    trailing_stop = max(trailing_stop, position.entry_price)
                    
                    # Trailing stop only moves UP, never down
                    if position.trailing_stop_price is None:
                        position.trailing_stop_price = trailing_stop
                        logger.info(f"[{ticker}] SURGE trailing stop activated at ${trailing_stop:.4f} (+{position.unrealized_pnl_pct:.2f}% profit, {trailing_stop_pct:.1f}% width)")
                    elif trailing_stop > position.trailing_stop_price:
                        position.trailing_stop_price = trailing_stop
                        logger.debug(f"[{ticker}] SURGE trailing stop moved up to ${trailing_stop:.4f}")
                    
                    # OPTIMIZED: Check if price is recovering before exiting on trailing stop
                    if current_price <= position.trailing_stop_price:
                        # Check if price is recovering (last bar higher than previous)
                        price_recovering = False
                        if len(df_with_indicators) >= 2:
                            prev_close = df_with_indicators.iloc[-2]['close']
                            if current_price > prev_close:
                                price_recovering = True
                        
                        # Don't exit if price is recovering and still above entry
                        if price_recovering and current_price > position.entry_price:
                            logger.debug(f"[{ticker}] SURGE: Trailing stop hit but price recovering (${current_price:.4f}), continuing")
                        else:
                            exit_reason = f"Surge trailing stop hit at ${position.trailing_stop_price:.4f} ({trailing_stop_pct:.1f}% from high)"
                
                # 4.5. SURGE MOMENTUM HOLDING LOGIC (Key Improvement)
                # Continue holding surge trades if strong momentum persists
                # This captures the full potential of high-momentum surge trades
                if profit_pct >= 8.0:  # Base 8% profit threshold
                    # Get current indicators for momentum check
                    current_rsi = current.get('rsi', 50)
                    current_volume = current.get('volume', 0)
                    
                    # Calculate volume ratio vs recent average
                    if len(df_with_indicators) >= 10:
                        recent_avg_volume = df_with_indicators.iloc[-10:]['volume'].mean()
                        volume_ratio_current = current_volume / recent_avg_volume if recent_avg_volume > 0 else 1.0
                    else:
                        volume_ratio_current = 1.0
                    
                    # IMPROVED: More permissive surge conditions (matching simulator optimization)
                    # Volume threshold: 2.0 → 1.8 (more sensitive)
                    # RSI threshold: 75 → 80 (allows stronger surges)
                    if (volume_ratio_current > 1.8 and  # Lower volume threshold (more sensitive)
                        current_rsi < 80):  # Higher RSI threshold (allows stronger surges)
                        logger.info(f"[{ticker}] SURGE: Strong momentum persists - continuing hold (vol={volume_ratio_current:.1f}x > 1.8, RSI={current_rsi:.1f} < 80, profit={profit_pct:.1f}%)")
                        # Don't exit - let surge continue running
                        # This captures extended gains from strong momentum moves
                    elif profit_pct >= 20.0:
                        # Extended take profit for surge (20% target from simulator)
                        exit_reason = f"Surge extended target reached: 20% profit - taking full position"
                        logger.info(f"[{ticker}] SURGE: Extended target 20% reached at ${current_price:.4f}")
                    else:
                        # Momentum weakening but still above base - consider trailing stop
                        pass
                
                # 5. Strong reversal detection (more conservative - require stronger signals)
                elif minutes_since_entry >= min_hold_time_surge:
                    reversal_signals = 0
                    required_signals = 3  # Default requirement
                    
                    if len(df_with_indicators) >= 5:
                        # Additional check: If we're in a strong uptrend with significant profit,
                        # require even more reversal signals to avoid exiting on minor pullbacks
                        profit_pct = position.unrealized_pnl_pct
                        price_above_entry_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                        
                        # CRITICAL FIX: Require more signals for early exits (first 30 minutes)
                        # This prevents premature exits during normal pullbacks in strong uptrends
                        if minutes_since_entry < 30:
                            # Very early in trade - require 6+ signals to avoid false exits
                            required_signals = 6
                        elif minutes_since_entry < 60:
                            # Early in trade - require 5+ signals
                            required_signals = 5
                        elif profit_pct >= 20.0 and price_above_entry_pct >= 15.0:
                            # Strong uptrend with good profit - require 5+ signals instead of 3+
                            required_signals = 5
                        elif profit_pct >= 10.0 and price_above_entry_pct >= 8.0:
                            # Moderate uptrend - require 4+ signals
                            required_signals = 4
                        
                        # Check for price recovery after pullback
                        # If price is recovering, don't count reversal signals
                        recent_bars = df_with_indicators.iloc[-5:]
                        current_bar = recent_bars.iloc[-1]
                        prev_bar = recent_bars.iloc[-2] if len(recent_bars) >= 2 else current_bar
                        
                        # Recovery check: If current price is higher than previous bar, we're recovering
                        price_recovering = current_price > prev_bar['close']
                        
                        # Check if price is still above entry (critical for early exits)
                        price_still_above_entry = current_price > position.entry_price
                        
                        # Check if we're still in uptrend (price above recent MAs)
                        sma_10 = current_bar.get('sma_10', 0)
                        sma_20 = current_bar.get('sma_20', 0)
                        price_above_mas = (sma_10 > 0 and current_price > sma_10) or (sma_20 > 0 and current_price > sma_20)
                        
                        # CRITICAL: If price is recovering, skip reversal detection entirely
                        # This prevents exits during minor pullbacks that are already recovering
                        if price_recovering:
                            # Price is recovering - don't count any reversal signals
                            # This gives trades room to recover from pullbacks
                            reversal_signals = 0
                            required_signals = 999  # Effectively disable exit
                        elif price_still_above_entry and price_above_mas:
                            # Price still above entry and MAs - require more signals but still count them
                            # This gives trades room to breathe during minor pullbacks in uptrends
                            required_signals = max(int(required_signals * 1.5), 6)
                            
                            # Check for price declining, volume declining, MACD turning negative
                            # But be more conservative - require significant declines, not just minor pullbacks
                            for i in range(1, len(recent_bars)):
                                prev_check_bar = recent_bars.iloc[i-1]
                                curr_check_bar = recent_bars.iloc[i]
                                # Price decline: require >2% drop, not just any decline
                                price_drop_pct = ((prev_check_bar['close'] - curr_check_bar['close']) / prev_check_bar['close']) * 100
                                if price_drop_pct > 2.0:
                                    reversal_signals += 1
                                # Volume declining: require >30% drop, not just 20%
                                if curr_check_bar['volume'] < prev_check_bar['volume'] * 0.7:
                                    reversal_signals += 1
                                # MACD histogram: require significant decline
                                macd_prev = prev_check_bar.get('macd_hist', 0)
                                macd_curr = curr_check_bar.get('macd_hist', 0)
                                if macd_prev > 0 and macd_curr < macd_prev * 0.5:  # MACD cut in half
                                    reversal_signals += 1
                        else:
                            # Price below entry or MAs - count reversal signals normally
                            # Check for price declining, volume declining, MACD turning negative
                            for i in range(1, len(recent_bars)):
                                prev_check_bar = recent_bars.iloc[i-1]
                                curr_check_bar = recent_bars.iloc[i]
                                # Price decline: require >2% drop, not just any decline
                                price_drop_pct = ((prev_check_bar['close'] - curr_check_bar['close']) / prev_check_bar['close']) * 100
                                if price_drop_pct > 2.0:
                                    reversal_signals += 1
                                # Volume declining: require >30% drop, not just 20%
                                if curr_check_bar['volume'] < prev_check_bar['volume'] * 0.7:
                                    reversal_signals += 1
                                # MACD histogram: require significant decline
                                macd_prev = prev_check_bar.get('macd_hist', 0)
                                macd_curr = curr_check_bar.get('macd_hist', 0)
                                if macd_prev > 0 and macd_curr < macd_prev * 0.5:  # MACD cut in half
                                    reversal_signals += 1
                    
                    if reversal_signals >= required_signals:
                        exit_reason = f"Surge strong reversal detected ({reversal_signals} reversal signals, required {required_signals})"
            
            # If no exit reason yet and within minimum hold time, don't exit
            if exit_reason is None and minutes_since_entry < min_hold_time_surge:
                logger.debug(f"[{ticker}] SURGE entry: {minutes_since_entry:.1f} min since entry, skipping exit checks (min {min_hold_time_surge} min)")
                return exit_signals  # Return early, don't check normal exit logic
        
        # STRATEGY-SPECIFIC EXIT LOGIC (Key Improvement)
        # Different exit strategies for different position types
        strategy_exit_reason = None
        
        # Get position type from pattern name or indicators
        position_pattern = position.entry_pattern.lower() if position.entry_pattern else ""
        is_breakout_entry = any(term in position_pattern for term in ['breakout', 'consolidation_breakout'])
        is_scalp_entry = any(term in position_pattern for term in ['volume_breakout']) and position.unrealized_pnl_pct < 5.0
        is_swing_entry = not (is_surge_entry or is_breakout_entry or is_scalp_entry)
        
        # Get current indicators for strategy decisions
        current_rsi = current.get('rsi', 50)
        current_macd_hist = current.get('macd_hist', 0)
        current_volume = current.get('volume', 0)
        
        # Calculate volume ratio vs recent average
        if len(df_with_indicators) >= 10:
            recent_avg_volume = df_with_indicators.iloc[-10:]['volume'].mean()
            volume_ratio_current = current_volume / recent_avg_volume if recent_avg_volume > 0 else 1.0
        else:
            volume_ratio_current = 1.0
        
        # BREAKOUT STRATEGY: Conservative trailing stops, 12% target
        if is_breakout_entry and exit_reason is None:
            if position.unrealized_pnl_pct >= 12.0:
                strategy_exit_reason = "Breakout target reached: 12% profit"
                logger.info(f"[{ticker}] BREAKOUT: Target 12% reached at ${current_price:.4f}")
            # Trailing stop after 8 minutes for breakout
            elif minutes_since_entry >= 8 and position.unrealized_pnl_pct >= 6.0:
                if position.max_price_reached > 0:
                    drop_from_peak = (position.max_price_reached - current_price) / position.max_price_reached
                    if drop_from_peak > 0.05:  # 5% trailing from 6% profit level
                        strategy_exit_reason = f"Breakout trailing stop: 5% drop from peak (profit: {position.unrealized_pnl_pct:.1f}%)"
            # Standard reversal conditions for breakout
            elif current_rsi > 75 and current_macd_hist < 0:
                strategy_exit_reason = "Breakout reversal: RSI > 75 and MACD negative"
        
        # SCALP STRATEGY: Quick exits, 3% target
        elif is_scalp_entry and exit_reason is None:
            if position.unrealized_pnl_pct >= 3.0:
                strategy_exit_reason = "Scalp target reached: 3% profit"
                logger.info(f"[{ticker}] SCALP: Target 3% reached at ${current_price:.4f}")
            # Tight trailing stop for scalp
            elif position.unrealized_pnl_pct >= 1.5:
                if position.max_price_reached > 0:
                    drop_from_peak = (position.max_price_reached - current_price) / position.max_price_reached
                    if drop_from_peak > 0.02:  # 2% trailing stop
                        strategy_exit_reason = f"Scalp trailing stop: 2% drop from peak"
            # Quick exit on reversal
            elif current_rsi > 70 and current_macd_hist < 0:
                strategy_exit_reason = "Scalp reversal: RSI > 70 and MACD negative"
        
        # SWING STRATEGY: Standard logic, 10% target
        elif is_swing_entry and exit_reason is None:
            if position.unrealized_pnl_pct >= 10.0:
                strategy_exit_reason = "Swing target reached: 10% profit"
                logger.info(f"[{ticker}] SWING: Target 10% reached at ${current_price:.4f}")
            # Standard trailing stop after 15 minutes
            elif minutes_since_entry >= 15 and position.unrealized_pnl_pct >= 5.0:
                if position.max_price_reached > 0:
                    drop_from_peak = (position.max_price_reached - current_price) / position.max_price_reached
                    if drop_from_peak > 0.04:  # 4% trailing stop
                        strategy_exit_reason = f"Swing trailing stop: 4% drop from peak"
            # Standard reversal conditions
            elif current_rsi > 72 and current_macd_hist < 0:
                strategy_exit_reason = "Swing reversal: RSI > 72 and MACD negative"
        
        # Use strategy-specific exit reason if determined and no other exit reason exists
        if strategy_exit_reason and exit_reason is None:
            exit_reason = strategy_exit_reason
        
        # 0. IMMEDIATE EXIT: Setup failed right after entry (most important)
        # FIX: Only check setup failed after minimum hold time (90 minutes) to avoid premature exits
        min_hold_time_setup_failed = 90
        if minutes_since_entry >= min_hold_time_setup_failed:
            if self._setup_failed_after_entry(df_with_indicators, position, current_time):
                exit_reason = "Setup failed - multiple failure signals detected"
        else:
            logger.debug(f"[{ticker}] {minutes_since_entry:.1f} min since entry, skipping setup failed check (min {min_hold_time_setup_failed} min)")
        
        # 1. Stop loss hit
        # OPTIMIZED: For normal entries, check if price is recovering or dropped too rapidly before exiting
        if exit_reason is None and current_price <= position.stop_loss:
            # Check if price is recovering (recent bars showing upward movement)
            price_recovering = False
            rapid_drop = False
            volume_still_strong = False
            
            if len(df_with_indicators) >= 5 and position.max_price_reached > 0:
                recent_bars = df_with_indicators.iloc[-5:]
                
                # Check if price dropped very rapidly from peak (more than 15% in short time)
                drop_from_peak = ((position.max_price_reached - current_price) / position.max_price_reached) * 100
                if drop_from_peak > 15.0:  # Dropped more than 15% from peak
                    rapid_drop = True
                
                if len(recent_bars) >= 2:
                    prev_close = recent_bars.iloc[-2]['close']
                    curr_close = recent_bars.iloc[-1]['close']
                    if curr_close > prev_close:
                        price_recovering = True
                
                # Check if volume is still strong
                current_vol = recent_bars.iloc[-1].get('volume', 0)
                avg_vol = recent_bars['volume'].mean()
                if current_vol > avg_vol * 0.7:  # Still 70%+ of average volume
                    volume_still_strong = True
            
            # OPTIMIZED: Don't exit on stop loss if:
            # - Price is recovering AND (still above entry OR rapid drop suggests recovery) AND
            # - (Volume still strong OR we're early in trade < 30 minutes)
            should_continue = False
            if price_recovering:
                if (current_price > position.entry_price or rapid_drop) and (volume_still_strong or minutes_since_entry < 30):
                    should_continue = True
            
            if should_continue:
                logger.info(f"[{ticker}] Stop loss hit but continuing (recovering={price_recovering}, vol_strong={volume_still_strong}, rapid_drop={rapid_drop})")
            else:
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
                        # OPTIMIZED: Even wider stops to capture more of strong moves
                        if unrealized_pnl_pct >= 150:
                            trailing_stop_pct = None  # Disable trailing stop for 150%+ profit (let it run)
                        elif unrealized_pnl_pct >= 100:
                            trailing_stop_pct = 40.0  # 40% trailing stop for 100%+ profit (very wide)
                        elif unrealized_pnl_pct >= 50:
                            trailing_stop_pct = 35.0  # 35% trailing stop for 50%+ profit (wider than before)
                        elif unrealized_pnl_pct >= 30:
                            trailing_stop_pct = 25.0  # 25% trailing stop for 30%+ profit (wider than before)
                        else:
                            trailing_stop_pct = 20.0  # 20% trailing stop (wider than before)
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
        # OPTIMIZED: More conservative for strong moves - require higher profit threshold
        # Also check if we have partial exits - if so, be even more conservative
        min_hold_time_trend_weakness = 100
        has_partial_exits = (position.original_shares > 0 and position.shares < position.original_shares)
        
        if exit_reason is None and minutes_since_entry >= min_hold_time_trend_weakness:
            # OPTIMIZED: For strong moves (50%+ profit) or after partial exits, require price to drop significantly
            # This prevents premature exits during normal pullbacks in strong uptrends
            if position.unrealized_pnl_pct >= 50.0 or has_partial_exits:
                # For very strong moves: Only exit on trend weakness if price drops below recent high by significant amount
                # This allows the trade to recover from pullbacks
                recent_high = position.max_price_reached
                price_drop_from_high = ((recent_high - current_price) / recent_high) * 100
                
                # Require at least 15% drop from high before considering trend weakness exit
                if price_drop_from_high >= 15.0 and self._detect_trend_weakness(df_with_indicators, position):
                    exit_reason = f"Trend weakness detected (price dropped {price_drop_from_high:.1f}% from high)"
                else:
                    logger.debug(f"[{ticker}] Strong move (+{position.unrealized_pnl_pct:.1f}%): Price drop {price_drop_from_high:.1f}% from high, not enough for trend weakness exit")
            elif position.unrealized_pnl_pct >= 5.0:
                # For profitable trades, require confirmation
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
        # CRITICAL FIX: Only generate partial exit signals if position has shares > 0
        if not exit_reason and position.shares > 0:  # Only if no other exit reason AND position has shares
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
    
    def _get_daily_macd(self, ticker: str, current_date: datetime) -> Optional[Dict[str, float]]:
        """
        Get daily MACD values for multi-timeframe analysis
        
        Args:
            ticker: Stock ticker symbol
            current_date: Current date/time for caching
            
        Returns:
            Dict with 'macd', 'macd_signal', 'macd_hist' or None if unavailable
        """
        if not self.data_api:
            return None
        
        # Check cache first (cache for the day)
        cache_key = f"{ticker}_{current_date.strftime('%Y-%m-%d')}"
        if cache_key in self.daily_macd_cache:
            return self.daily_macd_cache[cache_key]
        
        try:
            # Fetch daily data (aggregate from 1-minute data or fetch daily if available)
            # For now, we'll calculate from 1-minute data by aggregating to daily
            df_1min = self.data_api.get_1min_data(ticker, minutes=390)  # Get full trading day
            
            if df_1min is None or len(df_1min) < 50:
                return None
            
            # Aggregate to daily: use last day's data
            # Group by date and calculate daily OHLCV
            df_1min['date'] = pd.to_datetime(df_1min['timestamp']).dt.date
            daily_data = df_1min.groupby('date').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).reset_index()
            
            if len(daily_data) < 26:  # Need at least 26 days for MACD calculation
                return None
            
            # Calculate daily MACD
            daily_data['macd'] = daily_data['close'].ewm(span=12, adjust=False).mean() - daily_data['close'].ewm(span=26, adjust=False).mean()
            daily_data['macd_signal'] = daily_data['macd'].ewm(span=9, adjust=False).mean()
            daily_data['macd_hist'] = daily_data['macd'] - daily_data['macd_signal']
            
            # Get most recent daily MACD values
            latest = daily_data.iloc[-1]
            daily_macd = {
                'macd': latest['macd'],
                'macd_signal': latest['macd_signal'],
                'macd_hist': latest['macd_hist']
            }
            
            # Cache the result
            self.daily_macd_cache[cache_key] = daily_macd
            return daily_macd
            
        except Exception as e:
            logger.debug(f"[{ticker}] Error fetching daily MACD: {e}")
            return None
    
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
            'Volume_Breakout',  # High volume with price breakout
            'Volume_Breakout_Momentum'  # FIX: Accept Volume_Breakout_Momentum (variant of Volume_Breakout)
        ]
        
        # Secondary patterns that are acceptable with strong confirmations
        acceptable_patterns_with_confirmation = [
            'Accumulation_Pattern',  # Volume accumulation with price action
            'Slow_Accumulation',  # FIX: Accept Slow_Accumulation (variant of Accumulation_Pattern)
            'MACD_Bullish_Cross',  # MACD crossover with momentum
            'MACD_Acceleration_Breakout',  # FIX: Accept MACD_Acceleration_Breakout (variant of MACD_Bullish_Cross)
            'Consolidation_Breakout',  # FIX: Accept Consolidation_Breakout with strong confirmations
            'Golden_Cross',  # FIX: Accept Golden_Cross with strong confirmations
            'Golden_Cross_Volume',  # FIX: Accept Golden_Cross_Volume (variant of Golden_Cross)
        ]
        
        current = df.iloc[idx]
        
        # FIX: Detect fast mover status for relaxed MA validation
        is_fast_mover, fast_mover_metrics = self._is_fast_mover(df, idx)
        volume_ratio = current.get('volume_ratio', 0)
        fast_mover_momentum = fast_mover_metrics.get('momentum', 0) if is_fast_mover else 0
        
        if signal.pattern_name not in best_patterns:
            # Check if it's an acceptable pattern with strong confirmations
            if signal.pattern_name in acceptable_patterns_with_confirmation:
                # Require stronger confirmations for secondary patterns
                volume_ratio_check = current.get('volume_ratio', 0)
                price_momentum = ((current.get('close', 0) - df.iloc[max(0, idx-5)].get('close', 0)) / 
                                 df.iloc[max(0, idx-5)].get('close', 0)) * 100 if idx >= 5 else 0
                
                # Require: volume ratio >= 2x AND (price momentum > 3% OR confidence >= 75%)
                strong_confirmation = (volume_ratio_check >= 2.0 and 
                                      (price_momentum > 3.0 or signal.confidence >= 0.75))
                
                if strong_confirmation:
                    if log_reasons:
                        logger.info(f"[{signal.ticker}] PATTERN ACCEPTED: {signal.pattern_name} with strong confirmations (vol={volume_ratio_check:.2f}x, momentum={price_momentum:.1f}%, conf={signal.confidence*100:.1f}%)")
                    # Pattern accepted, continue validation
                else:
                    reason = f"Pattern '{signal.pattern_name}' requires strong confirmations (vol >= 2x AND (momentum > 3% OR conf >= 75%))"
                    if log_reasons:
                        rejection_reasons.append(reason)
                        logger.debug(f"[{signal.ticker}] REJECTED: {', '.join(rejection_reasons)}")
                    return False, reason  # Note: This rejection is handled by caller
            else:
                reason = f"Pattern '{signal.pattern_name}' not in best patterns"
                if log_reasons:
                    rejection_reasons.append(reason)
                    logger.debug(f"[{signal.ticker}] REJECTED: {', '.join(rejection_reasons)}")
                return False, reason
        
        lookback_20 = df.iloc[idx-20:idx]
        lookback_10 = df.iloc[idx-10:idx]
        
        # PRIORITY FIX: Calculate volume ratio and historical average FIRST (needed for fast mover detection and volume checks)
        current_volume = current.get('volume', 0)
        volume_ratio = current.get('volume_ratio', 0)
        
        # Calculate historical average volume for volume ratio calculation
        if len(df) >= 100:
            historical_avg_volume = df['volume'].tail(100).mean()
            volume_ratio_long = current_volume / historical_avg_volume if historical_avg_volume > 0 else 0
        else:
            avg_volume_20 = df['volume'].tail(20).mean() if len(df) >= 20 else df['volume'].mean()
            historical_avg_volume = avg_volume_20
            volume_ratio_long = current_volume / avg_volume_20 if avg_volume_20 > 0 else 0
        
        # PRIORITY FIX: Detect fast mover FIRST (before all other checks)
        # This ensures fast mover status is recognized before volatility and confidence checks
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
        
        # 1. ENHANCED: Price above key moving averages (MANDATORY, with relaxed rules for sustained moves/fast movers)
        close = current.get('close', 0)
        sma5 = current.get('sma_5', 0)
        sma10 = current.get('sma_10', 0)
        sma20 = current.get('sma_20', 0)
        
        # Check price above all MAs (preferred)
        price_above_all = (close > sma5 and close > sma10 and close > sma20)
        
        if not price_above_all:
            # For sustained moves or fast movers: Allow if price above longer MAs
            price_above_longer = (close > sma10 and close > sma20)
            price_near_sma5 = abs(close - sma5) / sma5 < 0.01 if sma5 > 0 else False  # Within 1% of SMA5
            
            # FIX: Relaxed for fast movers - allow if volume >= 4x OR momentum >= 5%
            fast_mover_momentum = fast_mover_metrics.get('momentum', 0) if is_fast_mover else 0
            is_strong_fast_mover = is_fast_mover and (volume_ratio >= 4.0 or fast_mover_momentum >= 5.0)
            
            # For fast movers with strong volume/momentum: Relax MA requirement
            if is_strong_fast_mover:
                if price_above_longer:
                    logger.info(f"[{signal.ticker}] FAST MOVER: Price above longer MAs (SMA10/SMA20), allowing entry (vol={volume_ratio:.2f}x, momentum={fast_mover_momentum:.2f}%)")
                    # Allow entry - price is above SMA10 and SMA20, which is sufficient for fast movers
                else:
                    reason = f"Price ${close:.4f} not above all MAs and not above longer MAs (SMA10=${sma10:.4f}, SMA20=${sma20:.4f})"
                    if log_reasons:
                        rejection_reasons.append(reason)
                        logger.info(f"[{signal.ticker}] REJECTED: {reason}")
                    return False, reason
            # For sustained moves: Allow if price above longer MAs and within 1% of SMA5
            elif price_above_longer and price_near_sma5:
                logger.info(f"[{signal.ticker}] SUSTAINED MOVE: Price above longer MAs and within 1% of SMA5, allowing entry")
                # Allow entry
            else:
                reason = f"Price ${close:.4f} not above all MAs (SMA5=${sma5:.4f}, SMA10=${sma10:.4f}, SMA20=${sma20:.4f})"
                if log_reasons:
                    rejection_reasons.append(reason)
                    logger.info(f"[{signal.ticker}] REJECTED: {reason}")
                return False, reason  # REJECT if price not above all MAs
        
        # 2. Moving averages in bullish order (MANDATORY)
        # FIX: RELAXED for fast movers with strong momentum - allow entry even if MAs aren't perfectly aligned
        if is_fast_mover:
            # For fast movers with very strong momentum (>50%): Allow entry even if MAs aren't perfectly aligned
            if fast_mover_momentum >= 50.0:
                logger.info(f"[{signal.ticker}] FAST MOVER with very strong momentum ({fast_mover_momentum:.2f}%): Relaxing MA alignment requirement")
                # Allow entry - momentum is so strong that MA alignment is less critical
            else:
                # For regular fast movers: Check if at least one MA pair is in bullish order
                ma_order_score = sum([sma5 > sma10, sma10 > sma20])
                if ma_order_score < 1:  # At least one pair must be in order
                    reason = f"MAs not in bullish order (SMA5=${sma5:.4f}, SMA10=${sma10:.4f}, SMA20=${sma20:.4f})"
                    if log_reasons:
                        rejection_reasons.append(reason)
                        logger.info(f"[{signal.ticker}] REJECTED: {reason}")
                    return False, reason
                else:
                    logger.debug(f"[{signal.ticker}] FAST MOVER: Relaxed MA order check (score: {ma_order_score}/2)")
        else:
            # Normal stocks: Strict MA order required
            if not (sma5 > sma10 and sma10 > sma20):
                reason = f"MAs not in bullish order (SMA5=${sma5:.4f}, SMA10=${sma10:.4f}, SMA20=${sma20:.4f})"
                if log_reasons:
                    rejection_reasons.append(reason)
                    logger.info(f"[{signal.ticker}] REJECTED: {reason}")
                return False, reason  # REJECT if MAs not in bullish order
        
        # 3. Volume must be above average (MANDATORY)
        # FIX: Relaxed for fast movers - lower threshold if already detected as fast mover
        # Volume ratio check happens first - if volume ratio is high enough, stock is clearly moving
        min_volume_ratio = 1.5
        if is_fast_mover and fast_mover_momentum >= 5.0:
            # For fast movers with strong momentum, allow slightly lower volume ratio
            min_volume_ratio = 1.4
            logger.debug(f"[{signal.ticker}] FAST MOVER: Using relaxed volume ratio threshold {min_volume_ratio}x (momentum={fast_mover_momentum:.2f}%)")
        
        if volume_ratio < min_volume_ratio:
            reason = f"Volume ratio {volume_ratio:.2f}x < {min_volume_ratio}x required"
            if log_reasons:
                rejection_reasons.append(reason)
                logger.info(f"[{signal.ticker}] REJECTED: {reason}")
            return False, reason  # REJECT if volume not strong
        
        # 3.5. Minimum absolute volume requirement (MANDATORY) - avoid low volume and extremely slow moving stocks
        # ENHANCED: Use calculated volume thresholds based on stock's historical average volume
        # Purpose: Filter out low-volume/no-movement stocks (liquidity filter)
        
        # Calculate historical average volume (use longer period for better baseline)
        if len(df) >= 100:
            historical_avg_volume = df['volume'].tail(100).mean()
        elif len(df) >= 50:
            historical_avg_volume = df['volume'].tail(50).mean()
        else:
            historical_avg_volume = df['volume'].mean() if len(df) > 0 else 0
        
        # Calculate volume ratio (current vs historical average)
        current_volume = current.get('volume', 0)
        if len(df) >= 100:
            volume_ratio_long = current_volume / historical_avg_volume if historical_avg_volume > 0 else 0
        else:
            avg_volume_20 = df['volume'].tail(20).mean() if len(df) >= 20 else historical_avg_volume
            volume_ratio_long = current_volume / avg_volume_20 if avg_volume_20 > 0 else 0
        
        # PRIORITY: Use volume ratio as primary indicator
        # If volume ratio is exceptional (>= 5x), skip daily volume check (stock is clearly moving)
        if volume_ratio_long >= 5.0:
            if log_reasons:
                logger.info(f"[{signal.ticker}] EXCEPTIONAL VOLUME: Volume ratio {volume_ratio_long:.2f}x >= 5x, skipping daily volume check")
            # Skip daily volume check - exceptional volume ratio indicates sufficient liquidity
        else:
            # For stocks with moderate volume ratio, check daily volume to filter out low-volume stocks
            # Calculate threshold based on historical average and time of day
            current_time = pd.to_datetime(current.get('timestamp', datetime.now()))
            et = pytz.timezone('US/Eastern')
            if current_time.tz is None:
                current_time = et.localize(current_time)
            else:
                current_time = current_time.astimezone(et)
            
            hour = current_time.hour
            
            # Calculate minimum daily volume based on historical average and time of day
            # Early morning: Lower threshold (volume hasn't accumulated yet)
            # Regular hours: Higher threshold (full trading day)
            # After-hours: Lower threshold (lower absolute volume but high ratio)
            
            if hour < 6:  # 4-6 AM (premarket)
                # Use 20x historical average (very low threshold for early morning)
                min_daily_volume = max(historical_avg_volume * 20, 50000)  # At least 50K absolute minimum
            elif hour < 8:  # 6-8 AM (early morning)
                # Use 40x historical average
                min_daily_volume = max(historical_avg_volume * 40, 100000)  # At least 100K absolute minimum
            elif hour < 10:  # 8-10 AM (mid-morning)
                # Use 60x historical average
                min_daily_volume = max(historical_avg_volume * 60, 150000)  # At least 150K absolute minimum
            elif hour >= 16 and hour < 20:  # After-hours (4 PM - 8 PM)
                # Use 30x historical average (lower for after-hours)
                min_daily_volume = max(historical_avg_volume * 30, 100000)  # At least 100K absolute minimum
                if log_reasons:
                    logger.debug(f"[{signal.ticker}] AFTER-HOURS: Using calculated threshold {min_daily_volume:,.0f} (historical avg: {historical_avg_volume:,.0f})")
            elif hour >= 15 and hour < 16:  # Late-day (3 PM - 4 PM)
                # Use 50x historical average
                min_daily_volume = max(historical_avg_volume * 50, 150000)  # At least 150K absolute minimum
                if log_reasons:
                    logger.debug(f"[{signal.ticker}] LATE-DAY: Using calculated threshold {min_daily_volume:,.0f} (historical avg: {historical_avg_volume:,.0f})")
            else:  # 10 AM - 3 PM (regular hours)
                # Use 100x historical average (full trading day)
                min_daily_volume = max(historical_avg_volume * 100, 200000)  # At least 200K absolute minimum
            
            # Check total volume over recent periods (simulating daily volume check)
            if len(df) >= 60:
                recent_volumes = df['volume'].tail(60).values
                total_volume_60min = recent_volumes.sum()
                if total_volume_60min < min_daily_volume:
                    reason = f"Low volume stock (total {total_volume_60min:,.0f} < {min_daily_volume:,.0f} over 60 min, calculated from historical avg {historical_avg_volume:,.0f})"
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
                    reason = f"Low volume stock (total {total_volume_20min:,.0f} < {min_volume_20min:,.0f} over 20 min, extrapolated, calculated from historical avg {historical_avg_volume:,.0f})"
                    if log_reasons:
                        rejection_reasons.append(reason)
                    return False, reason  # REJECT if volume too low
            else:
                # If very little data, check current volume (should be at least 2x historical average)
                min_current_volume = max(historical_avg_volume * 2, 10000)  # At least 2x average or 10K absolute minimum
                if current_volume < min_current_volume:
                    reason = f"Volume {current_volume:,.0f} < {min_current_volume:,.0f} minimum required (calculated from historical avg {historical_avg_volume:,.0f})"
                    if log_reasons:
                        rejection_reasons.append(reason)
                    return False, reason  # REJECT if volume too low
        
        # ENHANCED: Check average volume - use rolling average from move start instead of fixed window
        # This avoids diluting average with pre-move low-volume periods
        if len(df) >= 20:
            # Find when volume ratio first exceeded 2x (move start)
            move_start_idx = None
            for i in range(max(0, idx-20), idx+1):
                if i < len(df):
                    vol_ratio_at_i = df.iloc[i].get('volume_ratio', 0)
                    if vol_ratio_at_i >= 2.0:
                        move_start_idx = i
                        break
            
            # Calculate volume ratio (current vs historical average)
            if len(df) >= 100:
                historical_avg = df['volume'].tail(100).mean()
                volume_ratio_long = current.get('volume', 0) / historical_avg if historical_avg > 0 else 0
            else:
                avg_volume_20 = df['volume'].tail(20).mean()
                volume_ratio_long = current.get('volume', 0) / avg_volume_20 if avg_volume_20 > 0 else 0
            
            # For fast movers (volume ratio >= 3x), skip per-minute average check
            is_fast_mover_volume = volume_ratio_long >= 3.0
            
            if is_fast_mover_volume:
                # Fast movers: Skip per-minute average check (daily volume check is sufficient)
                if log_reasons:
                    logger.info(f"[{signal.ticker}] FAST MOVER: Skipping per-minute average check (vol_ratio={volume_ratio_long:.2f}x)")
            else:
                # Use rolling average from move start if found, otherwise fallback to fixed window
                if move_start_idx is not None:
                    volumes_since_move = df.iloc[move_start_idx:idx+1]['volume'].values
                    avg_volume_since_move = volumes_since_move.mean() if len(volumes_since_move) > 0 else 0
                    
                    # Calculate minimum based on historical average (use 0.5x historical average as minimum)
                    # This ensures we're filtering out truly low-volume stocks, not just stocks with lower baseline volume
                    min_avg_volume = max(historical_avg_volume * 0.5, 30000)  # At least 0.5x historical average or 30K absolute minimum
                    
                    if avg_volume_since_move < min_avg_volume:
                        reason = f"Low volume since move start (avg {avg_volume_since_move:,.0f} < {min_avg_volume:,.0f} required, calculated from historical avg {historical_avg_volume:,.0f})"
                        if log_reasons:
                            rejection_reasons.append(reason)
                        return False, reason
                else:
                    # Fallback to fixed window if move start not found
                    avg_volume_20 = df['volume'].tail(20).mean()
                    # Calculate minimum based on historical average
                    min_avg_volume = max(historical_avg_volume * 0.5, 30000)  # At least 0.5x historical average or 30K absolute minimum
                    
                    if avg_volume_20 < min_avg_volume:
                        reason = f"Low volume stock (avg {avg_volume_20:,.0f} < {min_avg_volume:,.0f} required, calculated from historical avg {historical_avg_volume:,.0f})"
                        if log_reasons:
                            rejection_reasons.append(reason)
                        return False, reason
        
        # 4. MACD must be bullish (MANDATORY)
        # ENHANCED: Multi-timeframe MACD analysis - check daily MACD when 1-minute MACD is bearish
        macd = current.get('macd', 0)
        macd_signal = current.get('macd_signal', 0)
        macd_hist = current.get('macd_hist', 0)
        
        # Check if 1-minute MACD is bearish
        one_min_macd_bearish = macd <= macd_signal
        
        if one_min_macd_bearish:
            # PRIORITY FIX: Multi-timeframe MACD analysis
            # If 1-minute MACD is bearish, check daily MACD
            # If daily MACD is bullish AND volume exceptional (6x+), relax 1-minute requirement
            
            # Get volume ratio for exceptional volume check
            vol_ratio = fast_mover_metrics.get('vol_ratio', volume_ratio_long) if is_fast_mover else volume_ratio_long
            
            # Check daily MACD if volume is exceptional (6x+) or fast mover
            if vol_ratio >= 6.0 or is_fast_mover:
                current_time = pd.to_datetime(current.get('timestamp', datetime.now()))
                daily_macd = self._get_daily_macd(signal.ticker, current_time)
                
                if daily_macd:
                    daily_macd_bullish = daily_macd['macd'] > daily_macd['macd_signal']
                    daily_hist_positive = daily_macd['macd_hist'] > 0
                    
                    if daily_macd_bullish and vol_ratio >= 6.0:
                        # Daily MACD is bullish AND volume exceptional - allow entry
                        # Check if 1-minute MACD is improving (trending up) even if not crossed
                        if idx >= 2:
                            prev_macd = df.iloc[idx-1].get('macd', 0)
                            prev_signal = df.iloc[idx-1].get('macd_signal', 0)
                            # Check if MACD is improving (getting closer to signal or histogram improving)
                            macd_improving = (macd > prev_macd) or (macd_hist > df.iloc[idx-1].get('macd_hist', 0))
                            
                            if macd_improving:
                                logger.info(f"[{signal.ticker}] MULTI-TIMEFRAME MACD: Daily MACD bullish, 1-min improving (vol={vol_ratio:.2f}x), allowing entry")
                                # Allow entry - daily MACD confirms trend, 1-minute is improving
                            else:
                                reason = f"MACD not bullish (1-min: {macd:.4f} <= {macd_signal:.4f}, daily bullish but 1-min not improving)"
                                if log_reasons:
                                    rejection_reasons.append(reason)
                                return False, reason
                        else:
                            # Not enough history to check improvement, but daily MACD is bullish
                            logger.info(f"[{signal.ticker}] MULTI-TIMEFRAME MACD: Daily MACD bullish (vol={vol_ratio:.2f}x), allowing entry (insufficient history for improvement check)")
                            # Allow entry - daily MACD confirms trend
                    elif daily_macd_bullish and is_fast_mover:
                        # Daily MACD is bullish AND fast mover - check if 1-minute MACD is improving
                        if idx >= 2:
                            prev_macd = df.iloc[idx-1].get('macd', 0)
                            macd_improving = (macd > prev_macd) or (macd_hist > df.iloc[idx-1].get('macd_hist', 0))
                            
                            if macd_improving:
                                logger.info(f"[{signal.ticker}] MULTI-TIMEFRAME MACD: Daily MACD bullish, 1-min improving (fast mover), allowing entry")
                                # Allow entry
                            else:
                                reason = f"MACD not bullish (1-min: {macd:.4f} <= {macd_signal:.4f}, daily bullish but 1-min not improving)"
                                if log_reasons:
                                    rejection_reasons.append(reason)
                                return False, reason
                        else:
                            logger.info(f"[{signal.ticker}] MULTI-TIMEFRAME MACD: Daily MACD bullish (fast mover), allowing entry")
                            # Allow entry
                    else:
                        # Daily MACD not bullish or volume not exceptional
                        reason = f"MACD not bullish (1-min: {macd:.4f} <= {macd_signal:.4f}, daily MACD not bullish or volume not exceptional)"
                        if log_reasons:
                            rejection_reasons.append(reason)
                        return False, reason
                else:
                    # Could not fetch daily MACD - use strict 1-minute check
                    reason = f"MACD not bullish (MACD {macd:.4f} <= Signal {macd_signal:.4f}) and daily MACD unavailable"
                    if log_reasons:
                        rejection_reasons.append(reason)
                    return False, reason
            else:
                # Volume not exceptional and 1-minute MACD is bearish - reject
                reason = f"MACD not bullish (MACD {macd:.4f} <= Signal {macd_signal:.4f}) and volume not exceptional (vol={vol_ratio:.2f}x < 6x)"
                if log_reasons:
                    rejection_reasons.append(reason)
                return False, reason
        
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
        # PRIORITY FIX: Check volatility AFTER fast mover detection
        # Fast movers bypass this check - high volatility is expected for breakouts
        # For extreme fast movers (volume >= 10x OR momentum >= 50%), skip volatility check entirely
        # INCREASED threshold from 8% to 15% for normal stocks to allow more volatile breakouts
        
        # Check if extreme fast mover (volume >= 10x OR momentum >= 50%)
        is_extreme_fast_mover = False
        if is_fast_mover:
            vol_ratio = fast_mover_metrics.get('vol_ratio', 0)
            momentum = fast_mover_metrics.get('momentum', 0)
            if vol_ratio >= 10.0 or momentum >= 50.0:
                is_extreme_fast_mover = True
                logger.info(f"[{signal.ticker}] EXTREME FAST MOVER: Vol {vol_ratio:.2f}x OR Momentum {momentum:.2f}%, skipping volatility check")
        
        if is_fast_mover or is_extreme_fast_mover:
            # Fast movers: Bypass volatility check (high volatility is expected)
            logger.info(f"[{signal.ticker}] FAST MOVER: Bypassing volatility check")
        else:
            # Normal stocks: Check volatility
            if len(lookback_10) >= 5:
                recent_highs = lookback_10['high'].tail(5).values
                recent_lows = lookback_10['low'].tail(5).values
                if len(recent_highs) > 0 and len(recent_lows) > 0:
                    price_range_pct = ((max(recent_highs) - min(recent_lows)) / min(recent_lows)) * 100
                    # For normal stocks: 15% threshold
                    # For stocks with volume ratio 2x-3x: 20% threshold (moderate volatility acceptable)
                    volatility_threshold = 20.0 if volume_ratio >= 2.0 else 15.0
                    if price_range_pct > volatility_threshold:
                        reason = f"Too volatile ({price_range_pct:.1f}% range in 5 periods, threshold {volatility_threshold:.1f}%)"
                        if log_reasons:
                            rejection_reasons.append(reason)
                        return False, reason  # REJECT if too volatile
        
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
        # ENHANCED: Multi-timeframe MACD analysis - allow histogram to be improving (trending up) even if negative
        macd_hist = current.get('macd_hist', 0)
        histogram_positive = macd_hist > 0
        
        if not histogram_positive:
            # PRIORITY FIX: Multi-timeframe MACD analysis for histogram
            # If 1-minute histogram is negative, check daily MACD
            # If daily MACD is bullish AND volume exceptional (6x+), allow histogram to be improving (trending up)
            
            # Get volume ratio for exceptional volume check
            vol_ratio = fast_mover_metrics.get('vol_ratio', volume_ratio_long) if is_fast_mover else volume_ratio_long
            
            # Check daily MACD if volume is exceptional (6x+) or fast mover
            if vol_ratio >= 6.0 or is_fast_mover:
                current_time = pd.to_datetime(current.get('timestamp', datetime.now()))
                daily_macd = self._get_daily_macd(signal.ticker, current_time)
                
                if daily_macd:
                    daily_macd_bullish = daily_macd['macd'] > daily_macd['macd_signal']
                    daily_hist_positive = daily_macd['macd_hist'] > 0
                    
                    # Check if 1-minute histogram is improving (trending up) even if negative
                    histogram_improving = False
                    if idx >= 2:
                        prev_hist = df.iloc[idx-1].get('macd_hist', 0)
                        prev_prev_hist = df.iloc[idx-2].get('macd_hist', 0) if idx >= 2 else prev_hist
                        # Histogram is improving if it's increasing (trending up)
                        histogram_improving = (macd_hist > prev_hist) and (prev_hist > prev_prev_hist)
                    
                    if daily_macd_bullish and (daily_hist_positive or histogram_improving) and vol_ratio >= 6.0:
                        # Daily MACD bullish, histogram improving, volume exceptional - allow entry
                        logger.info(f"[{signal.ticker}] MULTI-TIMEFRAME MACD HIST: Daily MACD bullish, 1-min hist improving (vol={vol_ratio:.2f}x), allowing entry")
                        # Allow entry - histogram is improving even if negative
                    elif daily_macd_bullish and histogram_improving and is_fast_mover:
                        # Daily MACD bullish, histogram improving, fast mover - allow entry
                        logger.info(f"[{signal.ticker}] MULTI-TIMEFRAME MACD HIST: Daily MACD bullish, 1-min hist improving (fast mover), allowing entry")
                        # Allow entry
                    else:
                        reason = f"MACD histogram not positive ({macd_hist:.4f}) and not improving with daily MACD confirmation"
                        if log_reasons:
                            rejection_reasons.append(reason)
                        return False, reason
                else:
                    # Could not fetch daily MACD - use strict 1-minute check
                    reason = f"MACD histogram not positive ({macd_hist:.4f}) and daily MACD unavailable"
                    if log_reasons:
                        rejection_reasons.append(reason)
                    return False, reason
            else:
                # Volume not exceptional and histogram is negative - reject
                reason = f"MACD histogram not positive ({macd_hist:.4f}) and volume not exceptional (vol={vol_ratio:.2f}x < 6x)"
                if log_reasons:
                    rejection_reasons.append(reason)
                return False, reason
        
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
        # FIX: Surge entries bypass this check entirely (they are explosive moves by definition)
        # FIX: For fast movers, allow up to 20% move in 5 periods (was 10% for all)
        # FIX: For surge entries, allow up to 50% move in 5 periods
        is_surge_entry = signal.pattern_name == 'PRICE_VOLUME_SURGE'
        if is_surge_entry:
            # Surge entries bypass "price too extended" check - these are explosive moves by definition
            # Surge detection already validates that the move is legitimate (volume >= 5x AND price change >= 30%)
            pass  # Skip this check for surge entries
        else:
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
        # FIX: Relax RSI overbought rejection for fast movers with exceptional volume
        # FIX: Allow RSI up to 90 for fast movers with volume >= 5x (explosive moves can stay overbought)
        rsi_threshold = 90.0 if (is_fast_mover and volume_ratio >= 5.0) else 85.0
        if rsi > rsi_threshold or rsi < 25:  # More extreme thresholds
            reason = f"RSI {rsi:.1f} out of range (overbought/oversold, threshold {rsi_threshold:.1f})"
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
        
        # Check if this is a surge entry
        is_surge_entry = signal.pattern_name == 'PRICE_VOLUME_SURGE'
        
        # SURGE ENTRIES: Use surge-specific stop loss (12% instead of default)
        if is_surge_entry:
            stop_loss = signal.price * (1 - self.surge_exit_hard_stop_pct / 100)
            logger.info(f"[{signal.ticker}] SURGE ENTRY: Using surge-specific stop loss {self.surge_exit_hard_stop_pct:.1f}% at ${stop_loss:.4f}")
        
        # IMPROVED: Set dynamic profit targets based on fast mover strength
        # Very strong fast movers need much higher targets to capture big runs
        if signal.target_price:
            target_price = signal.target_price
        elif is_surge_entry:
            # Surge entries: Use 25% target (aggressive for explosive moves)
            target_price = signal.price * 1.25
            logger.info(f"[{signal.ticker}] SURGE ENTRY: Setting 25% profit target")
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
            is_slow_mover_entry=is_slow_mover_entry,
            is_surge_entry=is_surge_entry
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

