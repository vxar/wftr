"""
Intelligent Position Manager
Handles smart entry decisions, partial profit taking, and dynamic exit management
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
import pytz

logger = logging.getLogger(__name__)

class ExitReason(Enum):
    """Reasons for position exit"""
    PROFIT_TARGET = "profit_target"
    PARTIAL_PROFIT_1 = "partial_profit_1"  # First partial profit (50% at +4%)
    PARTIAL_PROFIT_2 = "partial_profit_2"  # Second partial profit (25% at +7%)
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TREND_REVERSAL = "trend_reversal"
    VOLATILITY_STOP = "volatility_stop"
    TIME_EXIT = "time_exit"
    MANUAL_EXIT = "manual_exit"
    END_OF_DAY = "end_of_day"

class PositionType(Enum):
    """Types of positions with different exit strategies"""
    SCALP = "scalp"  # Quick trades, 1-5% targets
    SWING = "swing"  # Medium-term trades, 5-15% targets
    SURGE = "surge"  # High momentum, 10-25% targets
    SLOW_MOVER = "slow_mover"  # Gradual movers, 3-8% targets
    BREAKOUT = "breakout"  # Breakout trades, 8-15% targets

@dataclass
class ExitPlan:
    """Structured exit plan for a position"""
    position_type: PositionType
    initial_stop_loss: float  # Percentage
    trailing_stop_enabled: bool
    partial_profit_levels: List[Tuple[float, float]]  # [(profit_pct, sell_pct), ...]
    final_target: float  # Percentage
    max_hold_time: Optional[timedelta] = None
    volatility_adjusted_stop: bool = False

@dataclass
class ActivePosition:
    """Enhanced active position with intelligent management"""
    ticker: str
    entry_time: datetime
    entry_price: float
    shares: float
    original_shares: float
    entry_value: float
    position_type: PositionType
    exit_plan: ExitPlan
    
    # Current state
    current_price: float
    unrealized_pnl_pct: float
    unrealized_pnl_dollars: float
    
    # Exit tracking
    partial_profits_taken: List[float] = field(default_factory=list)
    current_stop_loss: Optional[float] = None
    current_target: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    
    # Performance tracking
    max_price_reached: float = 0.0
    max_unrealized_pct: float = 0.0
    time_at_max_price: Optional[datetime] = None
    
    # Exit information
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    realized_pnl: float = 0.0
    
    # Risk management
    risk_score: float = 0.0  # 0-1, higher is riskier
    volatility_score: float = 0.0  # 0-1, current volatility
    
    # Metadata
    entry_pattern: str = ""
    entry_confidence: float = 0.0
    multi_timeframe_confidence: float = 0.0

class IntelligentPositionManager:
    """
    Manages positions with intelligent entry/exit logic and partial profit taking
    """
    
    def __init__(self, 
                 max_positions: int = 3,
                 position_size_pct: float = 0.33,
                 risk_per_trade: float = 0.02):
        """
        Args:
            max_positions: Maximum concurrent positions
            position_size_pct: Position size as percentage of capital
            risk_per_trade: Risk per trade as percentage of capital
        """
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.risk_per_trade = risk_per_trade
        self.et_timezone = pytz.timezone('America/New_York')
        
        # Active positions
        self.active_positions: Dict[str, ActivePosition] = {}
        
        # Position history for completed trades
        self.position_history: Dict[str, ActivePosition] = {}
        
        # Performance tracking
        self.daily_trades: List[Dict] = []
        self.performance_metrics = {
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0
        }
        
        # Position type configurations
        self.position_configs = self._initialize_position_configs()
    
    def _initialize_position_configs(self) -> Dict[PositionType, ExitPlan]:
        """Initialize exit plans for different position types"""
        return {
            PositionType.SCALP: ExitPlan(
                position_type=PositionType.SCALP,
                initial_stop_loss=3.0,  # Increased from 2.0% to 3.0%
                trailing_stop_enabled=True,
                partial_profit_levels=[(1.5, 0.5), (2.5, 0.5)],  # 50% at 1.5%, 50% at 2.5%
                final_target=3.0,
                max_hold_time=timedelta(minutes=30)
            ),
            PositionType.SWING: ExitPlan(
                position_type=PositionType.SWING,
                initial_stop_loss=4.0,  # Increased from 3.0% to 4.0%
                trailing_stop_enabled=True,
                partial_profit_levels=[(3.0, 0.3), (6.0, 0.4)],  # 30% at 3%, 40% at 6%
                final_target=10.0,
                max_hold_time=timedelta(hours=4)
            ),
            PositionType.SURGE: ExitPlan(
                position_type=PositionType.SURGE,
                initial_stop_loss=12.0,  # Increased from 6.0% to 12.0% to allow surge to develop (matches realtime_trader surge logic)
                trailing_stop_enabled=True,
                partial_profit_levels=[(4.0, 0.25), (8.0, 0.25), (15.0, 0.25)],  # 25% at 4%, 8%, 15%
                final_target=25.0,
                max_hold_time=timedelta(hours=2)
            ),
            PositionType.BREAKOUT: ExitPlan(
                position_type=PositionType.BREAKOUT,
                initial_stop_loss=4.0,  # Conservative stop loss for breakout
                trailing_stop_enabled=True,
                partial_profit_levels=[(6.0, 0.5), (10.0, 0.5)],  # 50% at 6%, 50% at 10%
                final_target=12.0,  # 12% target (matching simulator)
                max_hold_time=timedelta(hours=1)
            ),
            PositionType.SLOW_MOVER: ExitPlan(
                position_type=PositionType.SLOW_MOVER,
                initial_stop_loss=4.0,  # Increased from 2.5% to 4.0%
                trailing_stop_enabled=False,
                partial_profit_levels=[(3.0, 0.5), (6.0, 0.5)],  # 50% at 3%, 50% at 6%
                final_target=8.0,
                max_hold_time=timedelta(hours=6)
            )
        }
    
    def evaluate_entry_signal(self, 
                            ticker: str,
                            current_price: float,
                            signal_strength: float,
                            multi_timeframe_analysis: Dict,
                            volume_data: Dict,
                            pattern_info: Dict) -> Tuple[bool, Optional[PositionType], str]:
        """
        Evaluate whether to enter a position
        
        Returns:
            Tuple of (should_enter, position_type, reason)
        """
        try:
            # Check minimum entry price
            if current_price < 1.0:
                return False, None, f"Entry price ${current_price:.4f} below minimum $1.00"
            
            # Check if we have capacity
            if len(self.active_positions) >= self.max_positions:
                return False, None, "Maximum positions reached"
            
            # Check if already in position
            if ticker in self.active_positions:
                return False, None, "Already in position"
            
            # Determine position type based on signal characteristics
            position_type = self._determine_position_type(
                signal_strength, multi_timeframe_analysis, volume_data, pattern_info
            )
            
            # Calculate risk score
            risk_score = self._calculate_entry_risk_score(
                ticker, current_price, multi_timeframe_analysis, volume_data
            )
            
            # Risk-based filtering
            if risk_score > 0.8:
                return False, None, "Risk score too high"
            
            # Confidence threshold based on position type
            confidence_thresholds = {
                PositionType.SCALP: 0.5,  # Reduced from 0.6
                PositionType.SWING: 0.6,  # Reduced from 0.7
                PositionType.SURGE: 0.7,  # Reduced from 0.8
                PositionType.BREAKOUT: 0.6,  # Reduced from 0.7
                PositionType.SLOW_MOVER: 0.55  # Reduced from 0.65
            }
            
            min_confidence = confidence_thresholds[position_type]
            if signal_strength < min_confidence:
                return False, None, f"Signal strength {signal_strength:.2f} below threshold {min_confidence}"
            
            # Volume confirmation - DISABLED for now to allow entries
            # return True  # Always pass volume confirmation for now
            
            # Trend confirmation - DISABLED for now to allow entries
            # trend_confirmed = self._has_trend_confirmation(multi_timeframe_analysis, current_price)
            # if not trend_confirmed:
            #     return False, None, "Insufficient trend confirmation"
            trend_confirmed = True  # Always pass trend confirmation for now
            
            # Multi-timeframe alignment - only check if data is available
            # If multi_timeframe_analysis is empty or doesn't have trend_alignment, allow entry
            # (since we can't properly evaluate alignment without the data)
            if multi_timeframe_analysis and 'trend_alignment' in multi_timeframe_analysis:
                if not self._has_timeframe_alignment(multi_timeframe_analysis):
                    return False, None, "Poor multi-timeframe alignment"
            
            # Always return proper tuple format: (should_enter, position_type, reason)
            return True, position_type, f"Valid {position_type.value} entry signal"
            
        except Exception as e:
            logger.error(f"Error evaluating entry signal for {ticker}: {e}")
            return False, None, f"Error: {str(e)}"
    
    def _determine_position_type(self, 
                               signal_strength: float,
                               multi_timeframe_analysis: Dict,
                               volume_data: Dict,
                               pattern_info: Dict) -> PositionType:
        """Determine the appropriate position type"""
        
        # Enhanced surge detection (FIX 1: More inclusive surge detection)
        volume_ratio = volume_data.get('volume_ratio', 1.0)
        price_momentum = multi_timeframe_analysis.get('momentum_score', 0.0)
        pattern_name = pattern_info.get('pattern_name', '')
        
        # More inclusive surge detection - catch more surge scenarios
        if (pattern_name == 'PRICE_VOLUME_SURGE' or
            pattern_name.startswith('CONTINUATION_SURGE') or
            (volume_ratio > 5 and price_momentum > 0.5) or
            (volume_ratio > 3 and signal_strength > 0.75) or
            (volume_ratio > 8 and price_momentum > 0.7)):  # Original criteria as fallback
            return PositionType.SURGE
        
        # Breakout detection
        if (pattern_info.get('pattern_name', '').startswith('Breakout') and 
            signal_strength > 0.7 and
            volume_ratio > 2.0):
            return PositionType.BREAKOUT
        
        # Scalp detection
        if (pattern_info.get('pattern_name', '').startswith('Volume_Breakout') and 
            signal_strength > 0.8):
            return PositionType.SCALP
        
        # Slow mover detection
        if (volume_ratio < 3 and 
            multi_timeframe_analysis.get('trend_alignment', 0.0) > 0.7 and
            pattern_info.get('pattern_name', '').startswith('Slow_Accumulation')):
            return PositionType.SLOW_MOVER
        
        # Default to swing
        return PositionType.SWING
    
    def _calculate_entry_risk_score(self, 
                                   ticker: str,
                                   current_price: float,
                                   multi_timeframe_analysis: Dict,
                                   volume_data: Dict) -> float:
        """Calculate risk score for entry (0-1, higher is riskier)"""
        risk_score = 0.0
        
        # Volatility risk
        volatility = multi_timeframe_analysis.get('volatility_score', 0.0)
        risk_score += volatility * 0.3
        
        # Volume risk (very high volume can be risky)
        volume_ratio = volume_data.get('volume_ratio', 1.0)
        if volume_ratio > 50:
            risk_score += 0.2
        elif volume_ratio < 1.5:
            risk_score += 0.1
        
        # Trend alignment risk
        trend_alignment = multi_timeframe_analysis.get('trend_alignment', 0.0)
        risk_score += (1 - trend_alignment) * 0.3
        
        # Price position risk
        rsi = multi_timeframe_analysis.get('rsi', 50)
        if rsi > 80:
            risk_score += 0.2
        elif rsi < 20:
            risk_score += 0.1
        
        return min(risk_score, 1.0)
    
    def _has_volume_confirmation(self, volume_data: Dict, position_type: PositionType) -> bool:
        """Check if volume confirms the entry"""
        volume_ratio = volume_data.get('volume_ratio', 1.0)
        
        # Different volume requirements for different position types (optimized from simulator)
        volume_requirements = {
            PositionType.SCALP: 3.0,  # Increased from 2.0 to 3.0
            PositionType.SWING: 2.0,  # Increased from 1.5 to 2.0
            PositionType.SURGE: 2.0,  # Reduced from 10.0 to 2.0 to match real data
            PositionType.BREAKOUT: 2.0,  # Moderate volume for breakout
            PositionType.SLOW_MOVER: 1.0  # Reduced from 1.5 to 1.0 to match real data
        }
        
        return volume_ratio >= volume_requirements[position_type]
    
    def _has_trend_confirmation(self, multi_timeframe_analysis: Dict, current_price: float) -> bool:
        """Check if trend is confirmed with moving averages (optimized from simulator)"""
        try:
            # Get moving average data from multi-timeframe analysis
            # This would need to be populated by the data source
            sma_5 = multi_timeframe_analysis.get('sma_5')
            sma_15 = multi_timeframe_analysis.get('sma_15') 
            sma_50 = multi_timeframe_analysis.get('sma_50')
            
            # Convert to scalar values if they are Series
            if hasattr(sma_5, 'item'):
                sma_5 = sma_5.item()
            if hasattr(sma_15, 'item'):
                sma_15 = sma_15.item()
            if hasattr(sma_50, 'item'):
                sma_50 = sma_50.item()
            
            # Trend confirmation - only trade if price is above key moving averages
            if (sma_5 is not None and sma_15 is not None and sma_50 is not None and
                not pd.isna(sma_5) and not pd.isna(sma_15) and not pd.isna(sma_50)):
                
                # Ensure current_price is also a scalar
                if hasattr(current_price, 'item'):
                    current_price = current_price.item()
                
                return (current_price > sma_5 and sma_5 > sma_15 and sma_15 > sma_50)
            
            # Fallback to trend alignment if moving averages not available
            return multi_timeframe_analysis.get('trend_alignment', 0.0) >= 0.57
            
        except Exception as e:
            logger.error(f"Error checking trend confirmation: {e}")
            return False
    
    def _has_timeframe_alignment(self, multi_timeframe_analysis: Dict) -> bool:
        """Check if multiple timeframes align"""
        trend_alignment = multi_timeframe_analysis.get('trend_alignment', 0.0)
        return trend_alignment >= 0.6
    
    def enter_position(self, 
                      ticker: str,
                      entry_price: float,
                      shares: float,
                      position_type: PositionType,
                      entry_pattern: str,
                      entry_confidence: float,
                      multi_timeframe_confidence: float) -> bool:
        """Enter a new position"""
        try:
            # Create active position
            exit_plan = self.position_configs[position_type]
            
            position = ActivePosition(
                ticker=ticker,
                entry_time=datetime.now(self.et_timezone),
                entry_price=entry_price,
                shares=shares,
                original_shares=shares,
                entry_value=entry_price * shares,
                position_type=position_type,
                exit_plan=exit_plan,
                current_price=entry_price,
                unrealized_pnl_pct=0.0,
                unrealized_pnl_dollars=0.0,
                entry_pattern=entry_pattern,
                entry_confidence=entry_confidence,
                multi_timeframe_confidence=multi_timeframe_confidence,
                max_price_reached=entry_price
            )
            
            # Set initial stop loss
            position.current_stop_loss = entry_price * (1 - exit_plan.initial_stop_loss / 100)
            position.current_target = entry_price * (1 + exit_plan.final_target / 100)
            
            # Add to active positions
            self.active_positions[ticker] = position
            
            logger.info(f"Entered {position_type.value} position: {ticker} - {shares:.2f} shares @ ${entry_price:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error entering position {ticker}: {e}")
            return False
    
    def update_positions(self, market_data: Dict[str, Dict]) -> List[Dict]:
        """
        Update all active positions and check for exit conditions
        
        Args:
            market_data: Dictionary of current market data per ticker
            
        Returns:
            List of exit decisions with expected format for simulator
        """
        exits = []
        active_count = len(self.active_positions)
        
        if active_count == 0:
            logger.debug("No active positions to update")
            return exits
        
        logger.info(f"Updating {active_count} active position(s) for exit conditions")
        
        for ticker, position in list(self.active_positions.items()):
            try:
                # Use market_data if available, otherwise use position's current price
                if ticker not in market_data:
                    logger.warning(f"[{ticker}] No market data available - using position current price ${position.current_price:.4f} for exit check")
                    # Create minimal market_data from position
                    current_data = {
                        'price': position.current_price,
                        'volume_ratio': 1.0,
                        'rsi': 50.0,
                        'macd_hist': 0.0,
                        'volatility_score': 0.0
                    }
                else:
                    current_data = market_data[ticker]
                
                current_price = current_data.get('price', position.current_price)
                
                # Update position
                self._update_position_data(position, current_price, current_data)
                
                # Check exit conditions
                exit_decision = self._check_exit_conditions(position, current_data)
                
                if exit_decision:
                    # Calculate shares sold for partial exits (before executing exit)
                    shares_sold = None
                    if exit_decision['reason'] in [ExitReason.PARTIAL_PROFIT_1, ExitReason.PARTIAL_PROFIT_2]:
                        sell_pct = exit_decision.get('sell_percentage', 0.5)
                        shares_sold = position.shares * sell_pct
                    
                    # Execute exit
                    if self._execute_exit(position, exit_decision):
                        # Format exit info for simulator
                        exit_info = {
                            'exited': True,
                            'position_id': ticker,
                            'exit_price': exit_decision['price'],
                            'exit_reason': exit_decision['reason'].value if hasattr(exit_decision['reason'], 'value') else str(exit_decision['reason']),
                            'sell_percentage': exit_decision.get('sell_percentage'),  # For partial exits
                            'shares_sold': shares_sold  # Actual shares sold (for partial exits)
                        }
                        exits.append(exit_info)
                        
                        # Remove from active positions only if completely exited
                        if position.shares <= 0.01:
                            del self.active_positions[ticker]
                    
            except Exception as e:
                logger.error(f"Error updating position {ticker}: {e}")
                continue
        
        return exits
    
    def exit_position(self, ticker: str, exit_reason: ExitReason) -> bool:
        """Force exit a position (used by simulator for end-of-day)"""
        try:
            if ticker not in self.active_positions:
                return False
            
            position = self.active_positions[ticker]
            exit_price = position.current_price
            
            # Create exit decision
            exit_decision = {
                'reason': exit_reason,
                'price': exit_price,
                'message': f"Force exit: {exit_reason.value}"
            }
            
            # Execute the exit
            success = self._execute_exit(position, exit_decision)
            if success:
                # Remove from active positions
                del self.active_positions[ticker]
                logger.info(f"Force exited {ticker} position at ${exit_price:.4f} - {exit_reason.value}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error force exiting position {ticker}: {e}")
            return False
    
    def _update_position_data(self, position: ActivePosition, current_price: float, market_data: Dict):
        """Update position data with current market information"""
        position.current_price = current_price
        
        # Calculate unrealized P&L
        price_change_pct = (current_price - position.entry_price) / position.entry_price * 100
        position.unrealized_pnl_pct = price_change_pct
        position.unrealized_pnl_dollars = (current_price - position.entry_price) * position.shares
        
        # Update max price reached
        if current_price > position.max_price_reached:
            position.max_price_reached = current_price
            position.time_at_max_price = datetime.now(self.et_timezone)
        
        # Update max unrealized P&L
        if price_change_pct > position.max_unrealized_pct:
            position.max_unrealized_pct = price_change_pct
        
        # Update volatility score
        position.volatility_score = market_data.get('volatility_score', 0.0)
        
        # Update trailing stop if enabled
        if position.exit_plan.trailing_stop_enabled and position.unrealized_pnl_pct > 2.0:
            self._update_trailing_stop(position)
    
    def _update_trailing_stop(self, position: ActivePosition):
        """Update trailing stop loss"""
        if position.shares <= 0:
            return
        
        # Calculate trailing stop distance based on position type
        trailing_distances = {
            PositionType.SCALP: 1.5,
            PositionType.SWING: 2.5,
            PositionType.SURGE: 4.0,
            PositionType.BREAKOUT: 3.0,  # Moderate trailing for breakout
            PositionType.SLOW_MOVER: 2.0
        }
        
        trail_distance = trailing_distances[position.position_type]
        
        # Set trailing stop at max price - trail distance
        new_trailing_stop = position.max_price_reached * (1 - trail_distance / 100)
        
        # Only move stop up, never down
        if position.trailing_stop_price is None or new_trailing_stop > position.trailing_stop_price:
            position.trailing_stop_price = new_trailing_stop
            position.current_stop_loss = new_trailing_stop
    
    def _check_exit_conditions(self, position: ActivePosition, market_data: Dict) -> Optional[Dict]:
        """Check if position should be exited"""
        current_time = datetime.now(self.et_timezone)
        time_in_position = current_time - position.entry_time
        minutes_in_position = time_in_position.total_seconds() / 60
        
        stop_loss_str = f"${position.current_stop_loss:.4f}" if position.current_stop_loss else "None"
        logger.info(f"[{position.ticker}] Checking exit conditions: P&L={position.unrealized_pnl_pct:.2f}%, Price=${position.current_price:.4f}, Stop={stop_loss_str}, Time={minutes_in_position:.1f}min")
        
        # Check time-based exit
        if position.exit_plan.max_hold_time:
            if time_in_position > position.exit_plan.max_hold_time:
                logger.info(f"[{position.ticker}] TIME EXIT triggered: {time_in_position} > {position.exit_plan.max_hold_time}")
                return {
                    'reason': ExitReason.TIME_EXIT,
                    'price': position.current_price,
                    'message': f"Max hold time exceeded: {time_in_position}"
                }
        
        # Check stop loss
        # FIX 2: Ensure surge positions get special treatment
        if position.current_stop_loss and position.current_price <= position.current_stop_loss:
            logger.info(f"[{position.ticker}] STOP LOSS triggered: ${position.current_price:.4f} <= ${position.current_stop_loss:.4f}")
            
            # For SURGE positions, use enhanced recovery logic
            if position.position_type == PositionType.SURGE:
                return self._check_surge_exit_conditions(position, market_data, current_time)
            else:
                # For non-SURGE positions, use standard stop loss logic
                stop_type = "trailing" if position.trailing_stop_price else "initial"
                return {
                    'reason': ExitReason.STOP_LOSS,
                    'price': position.current_stop_loss,
                    'message': f"{stop_type.title()} stop loss hit"
                }
        
        # Check partial profit levels
        # FIX 4: Apply dynamic adjustments and exit delays
        self._adjust_exit_thresholds_by_momentum(position, market_data)
        
        for profit_pct, sell_pct in position.exit_plan.partial_profit_levels:
            if profit_pct not in position.partial_profits_taken:
                if position.unrealized_pnl_pct >= profit_pct:
                    # Check if we should delay the exit
                    if self._should_delay_exit(position, market_data, current_time):
                        logger.info(f"[{position.ticker}] PARTIAL PROFIT DELAYED: {position.unrealized_pnl_pct:.2f}% >= {profit_pct}% (strong momentum)")
                        continue
                    
                    logger.info(f"[{position.ticker}] PARTIAL PROFIT triggered: {position.unrealized_pnl_pct:.2f}% >= {profit_pct}%")
                    return {
                        'reason': ExitReason.PARTIAL_PROFIT_1 if len(position.partial_profits_taken) == 0 else ExitReason.PARTIAL_PROFIT_2,
                        'price': position.current_price,
                        'sell_percentage': sell_pct,
                        'profit_level': profit_pct,
                        'message': f"Partial profit at {profit_pct}%"
                    }
        
        # Check final target
        if position.exit_plan.final_target and position.unrealized_pnl_pct >= position.exit_plan.final_target:
            logger.info(f"[{position.ticker}] PROFIT TARGET triggered: {position.unrealized_pnl_pct:.2f}% >= {position.exit_plan.final_target}%")
            return {
                'reason': ExitReason.PROFIT_TARGET,
                'price': position.current_price,
                'message': f"Final target {position.exit_plan.final_target}% reached"
            }
        
        # Check trend reversal (for swing, surge, and breakout positions)
        if position.position_type in [PositionType.SWING, PositionType.SURGE, PositionType.BREAKOUT]:
            if self._detect_trend_reversal(position, market_data):
                logger.info(f"[{position.ticker}] TREND REVERSAL triggered")
                return {
                    'reason': ExitReason.TREND_REVERSAL,
                    'price': position.current_price,
                    'message': "Trend reversal detected"
                }
        
        # Check volatility stop (for scalp positions)
        if position.position_type == PositionType.SCALP:
            if position.volatility_score > 0.8:
                return {
                    'reason': ExitReason.VOLATILITY_STOP,
                    'price': position.current_price,
                    'message': "High volatility exit"
                }
        
        return None
    
    def _check_surge_exit_conditions(self, position: ActivePosition, market_data: Dict, current_time: datetime) -> Optional[Dict]:
        """FIX 2: Enhanced surge-specific exit conditions with recovery checks"""
        time_in_position = (current_time - position.entry_time).total_seconds() / 60
        
        # Dynamic stop loss for surge: 20% max loss or 10% after 30 minutes
        if time_in_position >= 30:
            # After 30 minutes, use 10% stop loss
            if position.unrealized_pnl_pct <= -10.0:
                # Check recovery signs before exiting
                if not self._is_recovering(position, market_data):
                    stop_type = "trailing" if position.trailing_stop_price else "initial"
                    return {
                        'reason': ExitReason.STOP_LOSS,
                        'price': position.current_stop_loss,
                        'message': f"{stop_type.title()} stop loss hit (SURGE: {time_in_position:.1f} min, {position.unrealized_pnl_pct:.1f}% P&L)"
                    }
        else:
            # First 30 minutes, use 20% stop loss only (allow surge to develop)
            if position.unrealized_pnl_pct <= -20.0:  # 20% maximum loss
                # Check recovery signs before exiting
                if not self._is_recovering(position, market_data):
                    stop_type = "trailing" if position.trailing_stop_price else "initial"
                    return {
                        'reason': ExitReason.STOP_LOSS,
                        'price': position.current_stop_loss,
                        'message': f"{stop_type.title()} stop loss hit (SURGE: {time_in_position:.1f} min, {position.unrealized_pnl_pct:.1f}% P&L)"
                    }
        
        # If recovering or within limits, don't exit yet
        logger.info(f"[{position.ticker}] SURGE: Stop loss hit but continuing (recovering check passed, {time_in_position:.1f} min)")
        return None
    
    def _is_recovering(self, position: ActivePosition, market_data: Dict) -> bool:
        """FIX 3: Check if position shows signs of recovery"""
        volume_ratio = market_data.get('volume_ratio', 1.0)
        price_change = market_data.get('price_change_pct', 0)
        price_above_entry = position.current_price > position.entry_price
        
        # Recovery if:
        # 1. Price is still above entry AND volume is strong
        # 2. Price is turning up (positive price change) AND volume is moderate
        # 3. Volume is very strong (over 3x) indicating continued interest
        
        if price_above_entry and volume_ratio > 2.0:
            return True
        
        if price_change > -1.0 and volume_ratio > 1.5:
            return True
        
        if volume_ratio > 3.0:
            return True
        
        return False
    
    def _adjust_exit_thresholds_by_momentum(self, position: ActivePosition, market_data: Dict) -> None:
        """FIX 4: Adjust exit thresholds based on current momentum"""
        momentum = market_data.get('momentum_5min', 0.0)
        volume_ratio = market_data.get('volume_ratio', 1.0)
        
        # Only adjust for very strong momentum
        if momentum > 25.0 or (momentum > 20.0 and volume_ratio > 5.0):
            # Increase profit targets by 50% for extreme momentum
            original_levels = list(position.exit_plan.partial_profit_levels)
            adjusted_levels = []
            
            for profit_pct, sell_pct in original_levels:
                # Increase profit threshold by 50%
                adjusted_profit = profit_pct * 1.5
                adjusted_levels.append((adjusted_profit, sell_pct))
                logger.info(f"[{position.ticker}] DYNAMIC ADJUSTMENT: {profit_pct}% -> {adjusted_profit:.1f}% (momentum={momentum:.1f}%)")
            
            position.exit_plan.partial_profit_levels = adjusted_levels
            position.exit_plan.final_target *= 1.5
            
            # Also increase stop loss tolerance for extreme moves
            if position.position_type == PositionType.SURGE:
                position.exit_plan.initial_stop_loss *= 1.5
    
    def _should_delay_exit(self, position: ActivePosition, market_data: Dict, current_time: datetime) -> bool:
        """FIX 4: Delay exits for strong movers with high momentum"""
        time_in_position = (current_time - position.entry_time).total_seconds() / 60
        momentum = market_data.get('momentum_5min', 0.0)
        volume_ratio = market_data.get('volume_ratio', 1.0)
        
        # Delay partial exits if:
        # - Strong momentum (>20%) AND less than 3 minutes AND profit < 20%
        # - Very strong volume (>5x) AND momentum > 15% AND less than 5 minutes
        if ((momentum > 20.0 and time_in_position < 3.0 and position.unrealized_pnl_pct < 20.0) or
            (volume_ratio > 5.0 and momentum > 15.0 and time_in_position < 5.0)):
            logger.info(f"[{position.ticker}] EXIT DELAYED: momentum={momentum:.1f}%, vol={volume_ratio:.1f}x, time={time_in_position:.1f}min")
            return True
        return False
    
    def _detect_trend_reversal(self, position: ActivePosition, market_data: Dict, current_time: Optional[datetime] = None) -> bool:
        """Detect if trend is reversing with strategy-specific logic"""
        rsi = market_data.get('rsi', 50)
        macd_hist = market_data.get('macd_hist', 0)
        volume_ratio = market_data.get('volume_ratio', 1.0)
        
        # Use provided current_time or actual current time
        if current_time is None:
            time_in_position = (datetime.now(self.et_timezone) - position.entry_time).total_seconds() / 60
        else:
            time_in_position = (current_time - position.entry_time).total_seconds() / 60
        
        # For surge positions, use dynamic stop loss - protect against catastrophic drops
        if position.position_type == PositionType.SURGE:
            # Dynamic stop loss for surge: 20% max loss or 10% after 30 minutes
            if time_in_position >= 30:
                # After 30 minutes, use 10% stop loss
                if position.unrealized_pnl_pct <= -10.0:
                    return True
            else:
                # First 30 minutes, use 20% stop loss only (allow surge to develop)
                if position.unrealized_pnl_pct <= -20.0:  # 20% maximum loss
                    return True
            # Don't use other trend reversal logic for surge
            return False
        
        # Strategy-specific reversal detection for other position types
        if position.position_type == PositionType.BREAKOUT:
            # Breakout strategy: more conservative than surge
            
            # 12% target for breakout (matching simulator)
            if position.unrealized_pnl_pct >= 12.0:
                return True
            
            # Trailing stop after 8 minutes for breakout
            if time_in_position >= 8 and position.unrealized_pnl_pct >= 6.0:
                if position.max_price_reached > 0:
                    drop_from_peak = (position.max_price_reached - position.current_price) / position.max_price_reached
                    if drop_from_peak > 0.05:  # 5% trailing from 6% profit level
                        return True
            
            # Standard reversal conditions
            if rsi > 75 and macd_hist < 0:
                return True
                
        else:
            # Standard reversal conditions for other position types
            # Overbought with negative MACD histogram
            if rsi > 70 and macd_hist < 0:
                return True
        
        # Price dropped significantly from max (applies to all strategies except surge below base profit)
        if position.max_price_reached > 0:
            drop_from_max = (position.max_price_reached - position.current_price) / position.max_price_reached
            # For surge positions, only apply 5% drop rule after reaching 8% profit
            if position.position_type == PositionType.SURGE:
                if position.unrealized_pnl_pct >= 8.0 and drop_from_max > 0.05:
                    return True
            elif drop_from_max > 0.05:  # Apply to all other strategies
                return True
        
        return False
    
    def _execute_exit(self, position: ActivePosition, exit_decision: Dict) -> bool:
        """Execute position exit"""
        try:
            exit_price = exit_decision['price']
            exit_reason = exit_decision['reason']
            
            # Handle partial exits
            if exit_reason in [ExitReason.PARTIAL_PROFIT_1, ExitReason.PARTIAL_PROFIT_2]:
                sell_pct = exit_decision.get('sell_percentage', 0.5)
                shares_to_sell = position.shares * sell_pct
                
                # Calculate realized P&L for partial exit
                pnl_per_share = exit_price - position.entry_price
                realized_pnl = pnl_per_share * shares_to_sell
                
                # Update position
                position.shares -= shares_to_sell
                position.partial_profits_taken.append(exit_decision.get('profit_level', 0))
                position.realized_pnl += realized_pnl
                
                logger.info(f"Partial exit {position.ticker}: Sold {shares_to_sell:.2f} shares @ ${exit_price:.4f}, P&L: ${realized_pnl:.2f}")
                
                # If all shares sold, complete the exit
                if position.shares <= 0.01:
                    return self._complete_exit(position, exit_price, exit_reason)
                
                return True
            
            # Complete exit
            return self._complete_exit(position, exit_price, exit_reason)
            
        except Exception as e:
            logger.error(f"Error executing exit for {position.ticker}: {e}")
            return False
    
    def _complete_exit(self, position: ActivePosition, exit_price: float, exit_reason: ExitReason) -> bool:
        """Complete position exit"""
        try:
            # Calculate final P&L
            total_pnl_per_share = exit_price - position.entry_price
            total_pnl = total_pnl_per_share * position.original_shares
            
            # Update position
            position.exit_time = datetime.now(self.et_timezone)
            position.exit_price = exit_price
            position.exit_reason = exit_reason
            position.realized_pnl += total_pnl_per_share * position.shares  # Add remaining P&L
            position.shares = 0
            
            # Add to position history
            self.position_history[position.ticker] = position
            
            # Record trade
            self._record_completed_trade(position)
            
            logger.info(f"Complete exit {position.ticker}: {exit_reason.value} @ ${exit_price:.4f}, Total P&L: ${position.realized_pnl:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error completing exit for {position.ticker}: {e}")
            return False
    
    def _record_completed_trade(self, position: ActivePosition):
        """Record completed trade for performance analysis"""
        try:
            trade_record = {
                'ticker': position.ticker,
                'entry_time': position.entry_time,
                'exit_time': position.exit_time,
                'entry_price': position.entry_price,
                'exit_price': position.exit_price,
                'shares': position.original_shares,
                'entry_value': position.entry_value,
                'exit_value': position.exit_price * position.original_shares,
                'realized_pnl': position.realized_pnl,
                'pnl_pct': (position.exit_price - position.entry_price) / position.entry_price * 100,
                'position_type': position.position_type.value,
                'entry_pattern': position.entry_pattern,
                'exit_reason': position.exit_reason.value,
                'entry_confidence': position.entry_confidence,
                'multi_timeframe_confidence': position.multi_timeframe_confidence,
                'max_unrealized_pct': position.max_unrealized_pct,
                'partial_profits_taken': position.partial_profits_taken,
                'hold_time_minutes': (position.exit_time - position.entry_time).total_seconds() / 60
            }
            
            self.daily_trades.append(trade_record)
            
            # Update performance metrics
            self._update_performance_metrics()
            
        except Exception as e:
            logger.error(f"Error recording completed trade: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics from recent trades"""
        try:
            if not self.daily_trades:
                return
            
            recent_trades = self.daily_trades[-50:]  # Last 50 trades
            
            # Win rate
            wins = sum(1 for trade in recent_trades if trade['realized_pnl'] > 0)
            self.performance_metrics['win_rate'] = wins / len(recent_trades)
            
            # Average win/loss
            winning_trades = [t for t in recent_trades if t['realized_pnl'] > 0]
            losing_trades = [t for t in recent_trades if t['realized_pnl'] < 0]
            
            if winning_trades:
                self.performance_metrics['avg_win'] = np.mean([t['realized_pnl'] for t in winning_trades])
            
            if losing_trades:
                self.performance_metrics['avg_loss'] = np.mean([t['realized_pnl'] for t in losing_trades])
            
            # Profit factor
            total_wins = sum(t['realized_pnl'] for t in winning_trades)
            total_losses = abs(sum(t['realized_pnl'] for t in losing_trades))
            
            if total_losses > 0:
                self.performance_metrics['profit_factor'] = total_wins / total_losses
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def get_position_summary(self) -> Dict:
        """Get summary of current positions and performance"""
        try:
            active_summary = {}
            total_unrealized = 0.0
            
            for ticker, position in self.active_positions.items():
                position_info = {
                    'ticker': ticker,
                    'position_type': position.position_type.value,
                    'shares': position.shares,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'unrealized_pnl_pct': position.unrealized_pnl_pct,
                    'unrealized_pnl_dollars': position.unrealized_pnl_dollars,
                    'max_unrealized_pct': position.max_unrealized_pct,
                    'current_stop_loss': position.current_stop_loss,
                    'time_in_minutes': (datetime.now(self.et_timezone) - position.entry_time).total_seconds() / 60,
                    'partial_profits_taken': position.partial_profits_taken
                }
                active_summary[ticker] = position_info
                total_unrealized += position.unrealized_pnl_dollars
            
            return {
                'active_positions': active_summary,
                'total_active_positions': len(self.active_positions),
                'total_unrealized_pnl': total_unrealized,
                'performance_metrics': self.performance_metrics,
                'daily_trade_count': len(self.daily_trades)
            }
            
        except Exception as e:
            logger.error(f"Error getting position summary: {e}")
            return {}
