"""
Shared trade processing utilities for bot and simulator
Ensures both use the same logic for processing entries and exits
"""
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
import logging
import pytz

logger = logging.getLogger(__name__)

def process_exit_to_trade_data(
    exit_info: Dict,
    position_manager,
    timestamp: Optional[datetime] = None
) -> Optional[Dict]:
    """
    Process an exit decision into standardized trade data.
    This is shared logic used by both autonomous bot and simulator.
    
    Args:
        exit_info: Exit decision from position manager (contains position_id, exit_price, exit_reason, etc.)
        position_manager: IntelligentPositionManager instance
        timestamp: Optional timestamp for exit (defaults to now)
        
    Returns:
        Dictionary with trade data: {
            'ticker': str,
            'entry_time': datetime,
            'exit_time': datetime,
            'entry_price': float,
            'exit_price': float,
            'shares': float,  # Actual shares for this exit (partial or full)
            'entry_value': float,
            'exit_value': float,
            'pnl': float,
            'pnl_pct': float,
            'entry_pattern': str,
            'exit_reason': str,
            'confidence': float,
            'is_partial_exit': bool
        }
        Returns None if position data cannot be retrieved
    """
    try:
        if timestamp is None:
            timestamp = datetime.now(pytz.timezone('America/New_York'))
        
        # Get ticker from exit_info
        ticker = exit_info.get('position_id') or exit_info.get('ticker')
        if not ticker:
            logger.error(f"Exit info missing ticker/position_id: {exit_info}")
            return None
        
        exit_price = exit_info.get('exit_price') or exit_info.get('price')
        if exit_price is None:
            logger.error(f"Exit info missing price: {exit_info}")
            return None
        
        exit_reason = exit_info.get('exit_reason', 'unknown')
        
        # Check if this is a partial exit
        is_partial_exit = exit_reason in ['partial_profit_1', 'partial_profit_2'] or str(exit_reason).startswith('partial_profit')
        
        # Get position details from position manager
        entry_price = None
        shares = None
        entry_pattern = 'Unknown'
        confidence = 0.0
        entry_time = None
        
        # Try to get from active_positions first (for partial exits or recent complete exits)
        if hasattr(position_manager, 'active_positions') and ticker in position_manager.active_positions:
            pos = position_manager.active_positions[ticker]
            entry_price = pos.entry_price
            entry_time = pos.entry_time
            
            # For partial exits, use the shares actually sold
            if is_partial_exit:
                shares_sold = exit_info.get('shares_sold')
                if shares_sold is None:
                    # Fallback: calculate from sell_percentage
                    sell_pct = exit_info.get('sell_percentage', 0.5)
                    # Note: position.shares has already been reduced by the exit
                    # So we need to reverse-calculate: shares_before = shares_after / (1 - sell_pct)
                    shares_before_exit = pos.shares / (1 - sell_pct) if sell_pct < 1.0 else pos.shares
                    shares_sold = shares_before_exit * sell_pct
                shares = shares_sold
            else:
                shares = pos.original_shares
            
            entry_pattern = getattr(pos, 'entry_pattern', 'Unknown')
            confidence = getattr(pos, 'entry_confidence', 0.0)
        
        # Fallback: try position_history (for complete exits)
        elif hasattr(position_manager, 'position_history') and ticker in position_manager.position_history:
            pos = position_manager.position_history[ticker]
            entry_price = pos.entry_price
            entry_time = pos.entry_time
            shares = pos.original_shares  # Complete exit uses original shares
            entry_pattern = getattr(pos, 'entry_pattern', 'Unknown')
            confidence = getattr(pos, 'entry_confidence', 0.0)
        
        # Last resort: try position summary
        else:
            position_summary = position_manager.get_position_summary()
            if ticker in position_summary.get('active_positions', {}):
                position_info = position_summary['active_positions'][ticker]
                entry_price = position_info['entry_price']
                entry_time = getattr(position_manager.active_positions.get(ticker), 'entry_time', timestamp)
                
                if is_partial_exit:
                    shares_sold = exit_info.get('shares_sold')
                    if shares_sold is None:
                        sell_pct = exit_info.get('sell_percentage', 0.5)
                        if hasattr(position_manager, 'active_positions') and ticker in position_manager.active_positions:
                            original_shares = position_manager.active_positions[ticker].original_shares
                            shares_sold = original_shares * sell_pct
                        else:
                            current_shares = position_info['shares']
                            original_shares = current_shares / (1 - sell_pct) if sell_pct < 1.0 else current_shares
                            shares_sold = original_shares * sell_pct
                    shares = shares_sold
                else:
                    shares = position_info['shares']
                
                # Try to get pattern and confidence from position manager
                if hasattr(position_manager, 'active_positions') and ticker in position_manager.active_positions:
                    pos = position_manager.active_positions[ticker]
                    entry_pattern = getattr(pos, 'entry_pattern', 'Unknown')
                    confidence = getattr(pos, 'entry_confidence', 0.0)
            else:
                logger.error(f"[{ticker}] Cannot find position data for exit processing")
                return None
        
        # Validate we have required data
        if entry_price is None or shares is None or entry_price <= 0 or shares <= 0:
            logger.error(f"[{ticker}] Cannot process exit - missing entry_price or shares (entry_price={entry_price}, shares={shares})")
            return None
        
        if entry_time is None:
            entry_time = timestamp
        
        # Calculate P&L
        entry_value = entry_price * shares
        exit_value = exit_price * shares
        pnl = exit_value - entry_value
        pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        
        # Ensure confidence is between 0.0-1.0
        if confidence > 1.0:
            confidence = confidence / 100.0
        elif confidence < 0.0:
            confidence = 0.0
        
        return {
            'ticker': ticker,
            'entry_time': entry_time,
            'exit_time': timestamp,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'shares': shares,
            'entry_value': entry_value,
            'exit_value': exit_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'entry_pattern': entry_pattern,
            'exit_reason': str(exit_reason),
            'confidence': confidence,
            'is_partial_exit': is_partial_exit
        }
        
    except Exception as e:
        logger.error(f"Error processing exit to trade data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def process_entry_signal(
    entry_signal: Any,  # TradeSignal from RealtimeTrader
    position_manager,
    config: Dict,
    current_capital: Optional[float] = None,
    timestamp: Optional[datetime] = None
) -> Optional[Dict]:
    """
    Process an entry signal into a position entry.
    This is shared logic used by both autonomous bot and simulator.
    
    Args:
        entry_signal: Entry signal from RealtimeTrader (has price, confidence, pattern_name, etc.)
        position_manager: IntelligentPositionManager instance
        config: Configuration dict with 'max_positions', 'position_size_pct', etc.
        current_capital: Current available capital (if None, uses config['initial_capital'])
        timestamp: Optional timestamp for entry (defaults to now)
        
    Returns:
        Dictionary with entry result: {
            'success': bool,
            'ticker': str,
            'entry_price': float,
            'shares': float,
            'position_value': float,
            'position_type': PositionType,
            'entry_pattern': str,
            'confidence': float,
            'reason': str,  # Reason if rejected
            'timestamp': datetime
        }
        Returns None if entry signal is invalid
    """
    try:
        if timestamp is None:
            timestamp = datetime.now(pytz.timezone('America/New_York'))
        
        # Get ticker from entry signal (TradeSignal always has ticker attribute)
        ticker = getattr(entry_signal, 'ticker', None)
        if not ticker:
            logger.error(f"Entry signal missing ticker: {entry_signal}")
            return None
        
        # Check max positions
        position_summary = position_manager.get_position_summary()
        max_positions = config.get('max_positions', 3)
        if len(position_summary.get('active_positions', {})) >= max_positions:
            return {
                'success': False,
                'ticker': ticker,
                'reason': f"Max positions reached ({max_positions})",
                'timestamp': timestamp
            }
        
        # Check capital
        if current_capital is None:
            current_capital = config.get('initial_capital', 10000.0)
        
        min_capital = config.get('min_capital', 100.0)
        if current_capital < min_capital:
            return {
                'success': False,
                'ticker': ticker,
                'reason': f"Insufficient capital (${current_capital:.2f} < ${min_capital:.2f})",
                'timestamp': timestamp
            }
        
        # Get volume data from entry signal or use defaults
        volume_ratio = getattr(entry_signal, 'volume_ratio', 1.0)
        if hasattr(entry_signal, 'indicators') and isinstance(entry_signal.indicators, dict):
            volume_ratio = entry_signal.indicators.get('volume_ratio', volume_ratio)
        
        # Evaluate entry signal using position manager
        should_enter, position_type, reason = position_manager.evaluate_entry_signal(
            ticker=ticker,
            current_price=entry_signal.price,
            signal_strength=entry_signal.confidence,
            multi_timeframe_analysis={},  # Can be populated by caller if needed
            volume_data={'volume_ratio': volume_ratio},
            pattern_info={'pattern_name': entry_signal.pattern_name if hasattr(entry_signal, 'pattern_name') else 'Unknown'}
        )
        
        if not should_enter:
            return {
                'success': False,
                'ticker': ticker,
                'reason': reason or "Entry evaluation failed",
                'timestamp': timestamp
            }
        
        # Calculate position size
        position_size_pct = config.get('position_size_pct', 0.33)
        position_value = current_capital * position_size_pct
        shares = position_value / entry_signal.price
        
        # Ensure confidence is between 0.0-1.0
        confidence = entry_signal.confidence
        if confidence > 1.0:
            confidence = confidence / 100.0
        elif confidence < 0.0:
            confidence = 0.0
        
        # Enter position
        success = position_manager.enter_position(
            ticker=ticker,
            entry_price=entry_signal.price,
            shares=shares,
            position_type=position_type,
            entry_pattern=entry_signal.pattern_name if hasattr(entry_signal, 'pattern_name') else 'Unknown',
            entry_confidence=confidence,
            multi_timeframe_confidence=confidence
        )
        
        if success:
            return {
                'success': True,
                'ticker': ticker,
                'entry_price': entry_signal.price,
                'shares': shares,
                'position_value': position_value,
                'position_type': position_type,
                'entry_pattern': entry_signal.pattern_name if hasattr(entry_signal, 'pattern_name') else 'Unknown',
                'confidence': confidence,
                'target_price': getattr(entry_signal, 'target_price', None),
                'stop_loss': getattr(entry_signal, 'stop_loss', None),
                'reason': 'Position entered successfully',
                'timestamp': timestamp
            }
        else:
            return {
                'success': False,
                'ticker': ticker,
                'reason': 'Position entry failed (position manager returned False)',
                'timestamp': timestamp
            }
        
    except Exception as e:
        logger.error(f"Error processing entry signal: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'ticker': getattr(entry_signal, 'ticker', 'UNKNOWN') if entry_signal else 'UNKNOWN',
            'reason': f"Error: {str(e)}",
            'timestamp': timestamp or datetime.now(pytz.timezone('America/New_York'))
        }
