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
            entry_signal = self._check_entry_signal(df, ticker)
        
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
        
        # PRIORITY 0.5: Minimum volume filter - reject low volume stocks (500K daily minimum)
        # Check total volume over recent periods (simulating daily volume check)
        if len(df_with_indicators) >= 60:
            recent_volumes = df_with_indicators['volume'].tail(60).values
            total_volume_60min = recent_volumes.sum()
            min_daily_volume = 500000  # Minimum 500K shares over 60 minutes
            if total_volume_60min < min_daily_volume:
                self.last_rejection_reasons[ticker] = [f"Low volume stock (total {total_volume_60min:,.0f} < {min_daily_volume:,.0f} over 60 min)"]
                return None
        elif len(df_with_indicators) >= 20:
            # If less than 60 minutes, check 20-minute total and extrapolate
            recent_volumes = df_with_indicators['volume'].tail(20).values
            total_volume_20min = recent_volumes.sum()
            # Extrapolate to 60 minutes: need at least 500K/3 = 167K over 20 min
            min_volume_20min = 167000
            if total_volume_20min < min_volume_20min:
                self.last_rejection_reasons[ticker] = [f"Low volume stock (total {total_volume_20min:,.0f} < {min_volume_20min:,.0f} over 20 min, extrapolated)"]
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
            # Check minimum confidence (must be high)
            if signal.confidence < self.min_confidence:
                self.last_rejection_reasons[ticker].append(f"Confidence {signal.confidence*100:.1f}% < {self.min_confidence*100:.0f}% required")
                continue
            
            # PRIORITY 1: Check for false breakouts FIRST (most important filter)
            if self._is_false_breakout_realtime(df_with_indicators, current_idx, signal):
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
            if not self._setup_confirmed_multiple_periods(df_with_indicators, current_idx, signal):
                self.last_rejection_reasons[ticker].append("Setup not confirmed for multiple periods")
                continue
            
            # PRIORITY 5: Check expected gain meets minimum
            expected_gain = ((signal.target_price - signal.entry_price) / signal.entry_price) * 100
            if expected_gain < self.min_entry_price_increase:
                continue
            
            # PRIORITY 6: Final confirmation - price must be confirming the signal NOW
            current_price = current.get('close', 0)
            if current_price < signal.entry_price * 0.98:  # Price already dropped 2% from signal
                continue  # Signal is stale or failing
            
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
                indicators=signal.indicators or {}
            )
        
        return None
    
    def _setup_confirmed_multiple_periods(self, df: pd.DataFrame, idx: int, signal: PatternSignal) -> bool:
        """
        Check that the setup has been valid for multiple periods (not just appeared)
        This ensures sustainability, not just a momentary spike
        """
        if idx < 5:
            return False
        
        # Check last 4-6 periods to ensure setup conditions have been building
        confirmation_periods = 0
        required_periods = 4  # Setup must be valid for at least 4 periods
        
        for check_idx in range(max(0, idx-5), idx):  # Check last 6 periods
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
            
            # 3. Volume above average
            if check_point.get('volume_ratio', 0) > 1.2:
                conditions_met += 1
            
            # 4. Price momentum positive
            if check_idx >= 1:
                prev_close = df.iloc[check_idx-1].get('close', 0)
                if check_point.get('close', 0) > prev_close:
                    conditions_met += 1
            
            # If 3+ conditions met, this period confirms the setup
            if conditions_met >= 3:
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
        
        # Check exit conditions
        exit_reason = None
        
        # 0. IMMEDIATE EXIT: Setup failed right after entry (most important)
        # If setup conditions are no longer met within first few periods, exit immediately
        # Only exit if multiple failure signals confirm (more conservative)
        if self._setup_failed_after_entry(df_with_indicators, position, current_time):
            exit_reason = "Setup failed - multiple failure signals detected"
        
        # 1. Stop loss hit
        elif current_price <= position.stop_loss:
            exit_reason = f"Stop loss hit at ${position.stop_loss:.4f}"
        
        # 2. Target price reached
        elif current_price >= position.target_price:
            exit_reason = f"Profit target reached at ${position.target_price:.4f}"
        
        # 3. Progressive trailing stop (tightens as profit increases)
        # FIX: Only activate trailing stop after minimum profit threshold (3%)
        # This prevents premature exits on small price movements
        elif position.max_price_reached > 0 and position.unrealized_pnl_pct >= 3.0:
            # Calculate progressive trailing stop based on profit level
            unrealized_pnl_pct = position.unrealized_pnl_pct
            
            # Progressive trailing stop width - wider for bigger winners
            if unrealized_pnl_pct >= 15:
                trailing_stop_pct = 5.0  # Very wide for big winners (let them run)
            elif unrealized_pnl_pct >= 10:
                trailing_stop_pct = 4.0  # Wider stop for big winners
            elif unrealized_pnl_pct >= 7:
                trailing_stop_pct = 3.5  # Medium stop
            elif unrealized_pnl_pct >= 5:
                trailing_stop_pct = 3.0  # Wider stop
                # Move stop to breakeven after +5% profit (partial exit handles +4%)
                if position.stop_loss < position.entry_price:
                    position.stop_loss = position.entry_price
                    logger.info(f"[{ticker}] Stop moved to breakeven at ${position.entry_price:.4f} (+{unrealized_pnl_pct:.2f}% profit)")
            else:
                trailing_stop_pct = 2.5  # Initial trailing stop (only if profit >= 3%)
            
            # FIX: Use ATR-based stop if available (better for volatile stocks)
            # Otherwise fallback to percentage-based stop
            # Use the current row (already defined above as df_with_indicators.iloc[-1])
            atr = current.get('atr', 0)
            
            if pd.notna(atr) and atr > 0:
                # Use 2x ATR for trailing stop (wider for volatile stocks)
                trailing_stop = position.max_price_reached - (atr * 2)
                logger.debug(f"[{ticker}] ATR-based trailing stop: ${trailing_stop:.4f} (ATR: ${atr:.4f})")
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
        elif self._detect_trend_weakness(df_with_indicators, position):
            exit_reason = "Trend weakness detected"
        
        # 5. Bearish reversal pattern
        elif self._detect_bearish_reversal(df_with_indicators, position):
            exit_reason = "Bearish reversal pattern detected"
        
        # 6. Progressive partial profit taking - Lock in profits at multiple levels
        # Strategy: 50% at +4%, 25% at +7%, hold 25% to target
        if not exit_reason:  # Only if no other exit reason
            if not position.partial_profit_taken and position.unrealized_pnl_pct >= 4.0:
                # First partial exit: 50% at +4%
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
            elif hasattr(position, 'partial_profit_taken_second') and not position.partial_profit_taken_second and position.unrealized_pnl_pct >= 7.0:
                # Second partial exit: 25% at +7% (of remaining position)
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
        
        # Check volume ratio
        volume_ratio = current.get('volume_ratio', 0)
        if volume_ratio < 5.0:  # Must be at least 5x average
            return False, {}
        
        # Check price momentum (5%+ in last 5 periods)
        if idx >= 5:
            price_change_5 = ((current.get('close', 0) - df.iloc[idx-5].get('close', 0)) / 
                             df.iloc[idx-5].get('close', 0)) * 100
            if price_change_5 < 5.0:  # Must be at least 5% gain
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
        
        # Only use the absolute best patterns
        best_patterns = [
            'Strong_Bullish_Setup',  # Multiple indicators align
            'Volume_Breakout'  # High volume with price breakout
        ]
        
        if signal.pattern_name not in best_patterns:
            reason = f"Pattern '{signal.pattern_name}' not in best patterns"
            if log_reasons:
                rejection_reasons.append(reason)
                logger.debug(f"[{signal.ticker}] REJECTED: {', '.join(rejection_reasons)}")
            return False, reason
        
        current = df.iloc[idx]
        lookback_20 = df.iloc[idx-20:idx]
        lookback_10 = df.iloc[idx-10:idx]
        
        # Check if this is a fast mover (exceptional volume and momentum)
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
        # Check total volume over recent periods (simulating daily volume check)
        # User requirement: Minimum 500K shares daily volume to avoid stocks that don't move much
        # Calculate total volume over last 60 minutes (approximately 1 hour of trading)
        if len(df) >= 60:
            recent_volumes = df['volume'].tail(60).values
            total_volume_60min = recent_volumes.sum()
            min_daily_volume = 500000  # Minimum 500K shares over 60 minutes
            if total_volume_60min < min_daily_volume:
                reason = f"Low volume stock (total {total_volume_60min:,.0f} < {min_daily_volume:,.0f} over 60 min)"
                if log_reasons:
                    rejection_reasons.append(reason)
                return False, reason  # REJECT if total volume too low
        elif len(df) >= 20:
            # If less than 60 minutes, check 20-minute total and extrapolate
            recent_volumes = df['volume'].tail(20).values
            total_volume_20min = recent_volumes.sum()
            # Extrapolate to 60 minutes: need at least 500K/3 = 167K over 20 min
            min_volume_20min = 167000
            if total_volume_20min < min_volume_20min:
                reason = f"Low volume stock (total {total_volume_20min:,.0f} < {min_volume_20min:,.0f} over 20 min, extrapolated)"
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
        if len(df) >= 20:
            avg_volume_20 = df['volume'].tail(20).mean()
            
            # Minimum average volume threshold: 100,000 shares per minute average
            # This ensures the stock has sufficient liquidity for trading
            min_avg_volume = 100000  # 100K shares/minute average
            
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
        if len(lookback_20) >= 10:
            recent_lows = lookback_20['low'].tail(10).values
            if len(recent_lows) >= 5:
                # Check if lows are generally increasing
                older_lows = recent_lows[:5]
                newer_lows = recent_lows[5:]
                avg_older_low = min(older_lows) if len(older_lows) > 0 else 0
                avg_newer_low = min(newer_lows) if len(newer_lows) > 0 else 0
                if avg_older_low > 0 and avg_newer_low < avg_older_low * 0.98:  # Lower lows
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
        if not is_fast_mover:
            if len(lookback_10) >= 5:
                recent_highs = lookback_10['high'].tail(5).values
                recent_lows = lookback_10['low'].tail(5).values
                if len(recent_highs) > 0 and len(recent_lows) > 0:
                    price_range_pct = ((max(recent_highs) - min(recent_lows)) / min(recent_lows)) * 100
                    if price_range_pct > 8.0:  # Too volatile (8%+ range in 5 periods)
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
        # Reduced from 15% to 10% to avoid entering at peaks after big moves
        if price_change_5 > 10:  # More than 10% in 5 periods - too extended
            reason = f"Price too extended ({price_change_5:.1f}% in 5 periods, max 10% allowed)"
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
        
        # Only check for first 10 minutes after entry (give trades more time to develop)
        if time_since_entry > 10:
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
        
        # Require at least 1 critical failure OR 4+ regular failure signals (more conservative - give trades room)
        return critical_failures >= 1 or failure_signals >= 4
    
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
        required_signals = 3  # Increased from 2 to 3 - require more confirmation before exiting
        
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
        
        # If any critical signal, exit immediately
        if critical_signals >= 1:
            return True
        
        # Otherwise, require more signals for less aggressive exits
        return weakness_signals >= (required_signals + 1)  # Need 3+ signals instead of 2
    
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
        required_signals = 2  # Need clear reversal signals
        
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
        
        Args:
            signal: Entry signal
            df: Optional DataFrame to calculate ATR-based stop loss
        """
        # Calculate dynamic stop loss based on ATR if DataFrame provided
        stop_loss = signal.stop_loss
        if df is not None and len(df) >= 14:
            try:
                atr_series = calculate_atr(df, period=14)
                if len(atr_series) > 0 and atr_series.iloc[-1] > 0:
                    atr = atr_series.iloc[-1]
                    atr_pct = (atr / signal.price) * 100
                    
                    # Set stop loss based on ATR - increased ranges to give more room
                    if atr_pct > 6:
                        stop_loss_pct = 8.0  # High volatility (was 6.0%)
                    elif atr_pct > 4:
                        stop_loss_pct = 6.0  # Medium volatility (was 4.5%)
                    else:
                        stop_loss_pct = 4.0  # Low volatility (was 3.0%)
                    
                    # For penny stocks (< $1), add extra buffer
                    if signal.price < 1.0:
                        stop_loss_pct = max(stop_loss_pct, 6.0)  # Minimum 6% for penny stocks (was 5%)
                    
                    stop_loss = signal.price * (1 - stop_loss_pct / 100)
                    logger.info(f"[{signal.ticker}] ATR-based stop loss: {stop_loss_pct:.2f}% (ATR: {atr_pct:.2f}%)")
            except Exception as e:
                logger.warning(f"Error calculating ATR for {signal.ticker}: {e}")
        
        # Fallback to signal stop_loss or default 3%
        if stop_loss is None:
            stop_loss = signal.price * 0.97  # 3% default
        
        position = ActivePosition(
            ticker=signal.ticker,
            entry_time=signal.timestamp,
            entry_price=signal.price,
            entry_pattern=signal.pattern_name or "Unknown",
            entry_confidence=signal.confidence,
            target_price=signal.target_price or signal.price * (1 + self.profit_target_pct / 100),
            stop_loss=stop_loss,
            current_price=signal.price,
            max_price_reached=signal.price,
            original_shares=0.0  # Will be set when shares are assigned
        )
        
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

