"""
Comprehensive analysis of 5 stocks (GNPX, MLTX, VLN, INBS, ANPA)
to identify common patterns and optimal entry/exit strategies
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from data.webull_data_api import WebullDataAPI
from analysis.pattern_detector import PatternDetector
from collections import defaultdict

def calculate_advanced_indicators(df):
    """Calculate advanced indicators for pattern detection"""
    
    # Price momentum over different periods
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
    df['price_above_all_ma'] = (df['close'] > df['sma_5']) & (df['close'] > df['sma_10']) & (df['close'] > df['sma_20'])
    df['ma_bullish_order'] = df['sma5_above_sma10'] & df['sma10_above_sma20']
    
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

def identify_entry_patterns(df, idx):
    """Identify entry patterns at a given index"""
    
    if idx < 30:
        return []
    
    current = df.iloc[idx]
    patterns = []
    
    # Pattern 1: Volume Breakout with Momentum
    if (current.get('volume_ratio', 0) >= 1.8 and
        current.get('momentum_10', 0) >= 2.0 and
        current.get('breakout_10', False) and
        current.get('price_above_all_ma', False)):
        patterns.append({
            'name': 'Volume_Breakout_Momentum',
            'score': 8,
            'confidence': 0.85
        })
    
    # Pattern 2: Slow Accumulation
    if (1.8 <= current.get('volume_ratio', 0) < 3.5 and
        current.get('momentum_10', 0) >= 2.0 and
        current.get('momentum_20', 0) >= 3.0 and
        current.get('volume_trend_10', 0) >= 1.3 and
        current.get('macd_hist_accelerating', False) and
        current.get('price_position_20', 0) >= 70):
        patterns.append({
            'name': 'Slow_Accumulation',
            'score': 7,
            'confidence': 0.80
        })
    
    # Pattern 3: MACD Acceleration Breakout
    if (current.get('macd_hist_accelerating', False) and
        current.get('macd_bullish', False) and
        current.get('breakout_20', False) and
        current.get('volume_ratio', 0) >= 2.0 and
        current.get('momentum_20', 0) >= 3.0):
        patterns.append({
            'name': 'MACD_Acceleration_Breakout',
            'score': 8,
            'confidence': 0.82
        })
    
    # Pattern 4: Golden Cross with Volume
    if (current.get('sma5_above_sma10', False) and
        current.get('sma10_above_sma20', False) and
        df.iloc[idx-1].get('sma10_above_sma20', False) == False and  # Just crossed
        current.get('volume_ratio', 0) >= 1.5 and
        current.get('momentum_10', 0) >= 1.5):
        patterns.append({
            'name': 'Golden_Cross_Volume',
            'score': 7,
            'confidence': 0.78
        })
    
    # Pattern 5: Consolidation Breakout
    if (current.get('in_consolidation', False) == False and  # Just broke out
        df.iloc[idx-5:idx]['in_consolidation'].sum() >= 3 and  # Was consolidating
        current.get('breakout_10', False) and
        current.get('volume_ratio', 0) >= 2.0 and
        current.get('price_above_all_ma', False)):
        patterns.append({
            'name': 'Consolidation_Breakout',
            'score': 8,
            'confidence': 0.83
        })
    
    # Pattern 6: RSI Accumulation Zone Entry
    if (current.get('rsi_accumulation', False) and
        current.get('momentum_10', 0) >= 2.0 and
        current.get('volume_ratio', 0) >= 1.8 and
        current.get('macd_hist_increasing', False) and
        current.get('higher_high_20', False)):
        patterns.append({
            'name': 'RSI_Accumulation_Entry',
            'score': 7,
            'confidence': 0.75
        })
    
    return patterns

def simulate_trades(df, ticker, entry_patterns_func, min_score=6):
    """Simulate trades based on entry patterns"""
    
    opportunities = []
    completed_trades = []
    current_position = None
    
    for idx in range(30, len(df)):
        current = df.iloc[idx]
        current_time = pd.to_datetime(current['timestamp'])
        current_price = current.get('close', 0)
        
        # Check for entry
        if current_position is None:
            patterns = entry_patterns_func(df, idx)
            
            # Find best pattern
            best_pattern = None
            for pattern in patterns:
                if pattern['score'] >= min_score:
                    if best_pattern is None or pattern['score'] > best_pattern['score']:
                        best_pattern = pattern
            
            if best_pattern:
                # Calculate stop loss and target
                stop_loss = current_price * 0.85  # 15% stop
                target = current_price * 1.20  # 20% target
                
                current_position = {
                    'entry_time': current_time,
                    'entry_price': current_price,
                    'entry_idx': idx,
                    'pattern': best_pattern['name'],
                    'score': best_pattern['score'],
                    'confidence': best_pattern['confidence'],
                    'stop_loss': stop_loss,
                    'target': target,
                    'volume_ratio': current.get('volume_ratio', 0),
                    'momentum_10': current.get('momentum_10', 0),
                    'momentum_20': current.get('momentum_20', 0),
                    'position_size': 1.0,  # 100% of original position (for partial exits)
                    'partial_exits': []  # Track partial exits
                }
        
        # Check for exit - IMPROVED LOGIC: Only exit on hard stop or strong reversal
        if current_position is not None:
            entry_idx = current_position['entry_idx']
            entry_price = current_position['entry_price']
            stop_loss = current_position['stop_loss']  # Hard stop: 15% from entry
            target = current_position['target']
            
            current_high = current.get('high', 0)
            exit_reason = None
            exit_price = current_price
            
            # Calculate hold time and profit
            hold_time_min = (current_time - current_position['entry_time']).total_seconds() / 60
            max_price_during = df.iloc[entry_idx:idx+1]['high'].max()
            current_profit_pct = ((current_price - entry_price) / entry_price) * 100
            max_profit_pct = ((max_price_during - entry_price) / entry_price) * 100
            
            # 1. HARD STOP LOSS (always active, no exceptions)
            if current_price <= stop_loss:
                exit_reason = "Hard Stop Loss (15%)"
                exit_price = stop_loss
            
            # 2. MINIMUM HOLD TIME: Don't allow exits (except hard stop) for first 20 minutes
            elif hold_time_min < 20:
                # Only allow hard stop, nothing else
                exit_reason = None
            
            # 2.5. PARTIAL EXITS (scale-out strategy for massive moves)
            # Check for partial exits - CONTINUE position, don't exit fully
            partial_exit_taken = False
            if hold_time_min >= 20 and exit_reason is None:  # Only after minimum hold time and no other exit
                if current_position['position_size'] > 0.5 and current_profit_pct >= 20:
                    # First partial: 50% at 20% profit (lock in 10% gain)
                    partial_exit = {
                        'time': current_time,
                        'price': current_price,
                        'size': 0.5,  # 50% of position
                        'profit_pct': current_profit_pct,
                        'entry_price': entry_price,
                        'reason': 'Partial Exit 50% at 20% profit'
                    }
                    current_position['partial_exits'].append(partial_exit)
                    current_position['position_size'] = 0.5  # Reduce to 50%
                    partial_exit_taken = True
                    # CONTINUE position - don't exit fully
                    
                elif current_position['position_size'] > 0.25 and current_profit_pct >= 40:
                    # Second partial: 25% at 40% profit (lock in additional 10% gain)
                    partial_exit = {
                        'time': current_time,
                        'price': current_price,
                        'size': 0.25,  # 25% of position
                        'profit_pct': current_profit_pct,
                        'entry_price': entry_price,
                        'reason': 'Partial Exit 25% at 40% profit'
                    }
                    current_position['partial_exits'].append(partial_exit)
                    current_position['position_size'] = 0.25  # Reduce to 25%
                    partial_exit_taken = True
                    # CONTINUE position - don't exit fully
                    
                elif current_position['position_size'] > 0.125 and current_profit_pct >= 80:
                    # Third partial: 12.5% at 80% profit (lock in additional 10% gain)
                    partial_exit = {
                        'time': current_time,
                        'price': current_price,
                        'size': 0.125,  # 12.5% of position
                        'profit_pct': current_profit_pct,
                        'entry_price': entry_price,
                        'reason': 'Partial Exit 12.5% at 80% profit'
                    }
                    current_position['partial_exits'].append(partial_exit)
                    current_position['position_size'] = 0.125  # Reduce to 12.5%
                    partial_exit_taken = True
                    # CONTINUE position - don't exit fully
            
            # 3. DYNAMIC TRAILING STOP (only after minimum hold time and no partial exit just taken)
            # OPTIMIZED: Widen significantly for strong moves, especially after partial exits
            elif hold_time_min >= 20:
                # Check if we have partial exits (smaller position size = wider stops)
                position_size = current_position.get('position_size', 1.0)
                
                if position_size <= 0.25:  # After partial exits (remaining 25% or less)
                    # Very wide trailing stops for remaining position after partial exits
                    if current_profit_pct >= 100:
                        trailing_pct = None  # Disable trailing stop for 100%+ profit (let it run)
                    elif current_profit_pct >= 50:
                        trailing_pct = 0.30  # 30% trailing stop for 50%+ profit (very wide)
                    elif current_profit_pct >= 30:
                        trailing_pct = 0.20  # 20% trailing stop for 30%+ profit
                    else:
                        trailing_pct = 0.15  # 15% trailing stop
                else:
                    # Full position - use standard optimized trailing stops
                    # Base trailing stop based on hold time
                    if hold_time_min < 30:
                        base_trailing_pct = 0.07  # 7% base trailing stop
                    else:
                        base_trailing_pct = 0.10  # 10% base trailing stop after 30 min
                    
                    # Widen significantly for strong moves (allows massive gains to run)
                    if current_profit_pct >= 50:
                        trailing_pct = 0.20  # 20% trailing stop for 50%+ profit (very wide for massive moves)
                    elif current_profit_pct >= 30:
                        trailing_pct = 0.15  # 15% trailing stop for 30%+ profit
                    elif current_profit_pct >= 20:
                        trailing_pct = 0.12  # 12% trailing stop for 20%+ profit
                    elif current_profit_pct >= 10:
                        trailing_pct = 0.10  # 10% trailing stop for 10%+ profit
                    elif current_profit_pct >= 5:
                        trailing_pct = max(base_trailing_pct, 0.07)  # 7% for 5%+ profit
                    else:
                        trailing_pct = 0.05  # 5% trailing stop for <5% profit (tighter for small gains)
                
                # Only apply trailing stop if not disabled
                if trailing_pct is not None:
                    max_price = df.iloc[entry_idx:idx+1]['high'].max()
                    trailing_stop = max_price * (1 - trailing_pct)
                    
                    if current_price <= trailing_stop:
                        exit_reason = f"Trailing Stop ({trailing_pct*100:.0f}%)"
                        exit_price = trailing_stop
            
            # 4. STRONG REVERSAL SIGNALS (require multiple confirmations)
            if exit_reason is None and hold_time_min >= 20:
                reversal_signals = 0
                
                # Signal 1: Price below SMA10 AND SMA20
                if (current.get('close', 0) < current.get('sma_10', 0) and 
                    current.get('close', 0) < current.get('sma_20', 0)):
                    reversal_signals += 1
                
                # Signal 2: MACD bearish crossover
                if idx > 0:
                    prev_macd_bullish = df.iloc[idx-1].get('macd_bullish', False)
                    if prev_macd_bullish and not current.get('macd_bullish', False):
                        reversal_signals += 1
                
                # Signal 3: MACD histogram negative
                if current.get('macd_hist', 0) < 0:
                    reversal_signals += 1
                
                # Signal 4: Volume declining 30%+ AND price declining
                if idx >= 5:
                    recent_volumes = df.iloc[idx-5:idx+1]['volume'].values
                    if len(recent_volumes) >= 3:
                        volume_decline = (recent_volumes[-1] < recent_volumes[0] * 0.7)
                        price_decline = current_price < df.iloc[idx-3].get('close', 0)
                        if volume_decline and price_decline:
                            reversal_signals += 1
                
                # Signal 5: Price 5%+ below recent high
                if max_price_during > 0:
                    decline_from_high = ((max_price_during - current_price) / max_price_during) * 100
                    if decline_from_high >= 5.0:
                        reversal_signals += 1
                
                # Signal 6: RSI dropping below 50 from overbought
                if idx >= 3:
                    prev_rsi = df.iloc[idx-3].get('rsi', 50)
                    current_rsi = current.get('rsi', 50)
                    if prev_rsi > 70 and current_rsi < 50:
                        reversal_signals += 1
                
                # OPTIMIZED: Require more signals for high-profit trades (less sensitive for strong moves)
                # After partial exits, require even more signals for remaining position
                position_size = current_position.get('position_size', 1.0)
                
                if position_size <= 0.25:  # After partial exits (remaining 25% or less)
                    # Very conservative for remaining position
                    if current_profit_pct >= 200:
                        required_signals = 6  # 6+ signals for 200%+ profit (very conservative)
                    elif current_profit_pct >= 100:
                        required_signals = 5  # 5+ signals for 100%+ profit
                    elif current_profit_pct >= 50:
                        required_signals = 4  # 4+ signals for 50%+ profit
                    else:
                        required_signals = 3  # 3+ signals
                else:
                    # Full position - use standard optimized logic
                    if current_profit_pct >= 50:
                        required_signals = 5  # 5+ signals for 50%+ profit (very conservative)
                    elif current_profit_pct >= 20:
                        required_signals = 4  # 4+ signals for 20%+ profit (conservative)
                    else:
                        required_signals = 3  # 3+ signals for <20% profit (current)
                
                # Exit only if required signals met (strong reversal)
                if reversal_signals >= required_signals:
                    exit_reason = f"Strong Reversal ({reversal_signals} signals, required {required_signals}+)"
                    exit_price = current_price
            
            # 5. PROFIT TARGET (optional - can be disabled for trending stocks)
            # Only take profit if we've held for 30+ minutes and profit > 20%
            # DISABLED if partial exits have been taken (let remaining position run)
            has_partial_exits = len(current_position.get('partial_exits', [])) > 0
            if exit_reason is None and not has_partial_exits and hold_time_min >= 30 and current_profit_pct >= 20:
                exit_reason = "Profit Target (20%+)"
                exit_price = target if current_high >= target else current_price
            
            if exit_reason:
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                hold_time = (current_time - current_position['entry_time']).total_seconds() / 60
                
                trade = {
                    'entry_time': current_position['entry_time'],
                    'entry_price': entry_price,
                    'entry_idx': entry_idx,
                    'exit_time': current_time,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl_pct': pnl_pct,
                    'hold_time_min': hold_time,
                    'pattern': current_position['pattern'],
                    'score': current_position['score'],
                    'confidence': current_position['confidence'],
                    'volume_ratio': current_position.get('volume_ratio', 0),
                    'momentum_10': current_position.get('momentum_10', 0),
                    'momentum_20': current_position.get('momentum_20', 0),
                }
                
                completed_trades.append(trade)
                current_position = None
    
    # Handle open position at end
    if current_position is not None:
        final_price = df.iloc[-1].get('close', 0)
        entry_price = current_position['entry_price']
        
        # Calculate P&L from partial exits
        partial_pnl_pct = 0
        for partial in current_position.get('partial_exits', []):
            partial_pnl = ((partial['price'] - entry_price) / entry_price) * 100 * partial['size']
            partial_pnl_pct += partial_pnl
        
        # Calculate P&L from remaining position
        position_size = current_position.get('position_size', 1.0)
        remaining_pnl_pct = ((final_price - entry_price) / entry_price) * 100 * position_size
        
        # Total P&L (partial exits + remaining position)
        total_pnl_pct = partial_pnl_pct + remaining_pnl_pct
        
        hold_time = (df.iloc[-1]['timestamp'] - current_position['entry_time']).total_seconds() / 60
        
        # Build exit reason string with partial exit info
        exit_reason = "End of Day"
        if len(current_position.get('partial_exits', [])) > 0:
            exit_reason += f" ({len(current_position['partial_exits'])} partial exits taken)"
        
        completed_trades.append({
            'entry_time': current_position['entry_time'],
            'entry_price': entry_price,
            'entry_idx': current_position['entry_idx'],
            'exit_time': df.iloc[-1]['timestamp'],
            'exit_price': final_price,
            'exit_reason': exit_reason,
            'pnl_pct': total_pnl_pct,  # Total P&L including partial exits
            'hold_time_min': hold_time,
            'pattern': current_position['pattern'],
            'score': current_position['score'],
            'confidence': current_position['confidence'],
            'volume_ratio': current_position.get('volume_ratio', 0),
            'momentum_10': current_position.get('momentum_10', 0),
            'momentum_20': current_position.get('momentum_20', 0),
            'partial_exits': current_position.get('partial_exits', []),  # Store partial exit info
            'final_position_size': position_size  # Final position size
        })
    
    return completed_trades

def analyze_stock(ticker, start_hour=4, verbose=True):
    """Analyze a single stock"""
    
    print(f"\n{'='*80}")
    print(f"ANALYZING {ticker}")
    print(f"{'='*80}\n")
    
    data_api = WebullDataAPI()
    pattern_detector = PatternDetector()
    
    # Fetch data
    print(f"Fetching {ticker} data...")
    df_1min = data_api.get_1min_data(ticker, minutes=800)
    
    if df_1min is None or len(df_1min) == 0:
        print(f"ERROR: No data for {ticker}")
        return None
    
    # Filter from start hour
    df_1min['timestamp'] = pd.to_datetime(df_1min['timestamp'])
    if df_1min['timestamp'].dt.tz is None:
        df_1min['timestamp'] = df_1min['timestamp'].dt.tz_localize('US/Eastern')
    
    et = pytz.timezone('US/Eastern')
    today = datetime.now(et).date()
    start_time = et.localize(datetime.combine(today, datetime.min.time().replace(hour=start_hour, minute=0)))
    
    df_filtered = df_1min[df_1min['timestamp'] >= start_time].copy()
    if len(df_filtered) == 0:
        df_filtered = df_1min.copy()
    
    # Calculate indicators
    print("Calculating indicators...")
    df_with_indicators = pattern_detector.calculate_indicators(df_filtered)
    df_with_indicators = calculate_advanced_indicators(df_with_indicators)
    
    if len(df_with_indicators) < 30:
        print(f"ERROR: Insufficient data")
        return None
    
    print(f"Data points: {len(df_with_indicators)}\n")
    
    # Simulate trades
    print("Simulating trades...")
    trades = simulate_trades(df_with_indicators, ticker, identify_entry_patterns, min_score=6)
    
    # Price analysis
    starting_price = df_with_indicators.iloc[0].get('close', 0)
    final_price = df_with_indicators.iloc[-1].get('close', 0)
    max_price = df_with_indicators['high'].max()
    min_price = df_with_indicators['low'].min()
    
    max_gain = ((max_price - starting_price) / starting_price) * 100
    total_change = ((final_price - starting_price) / starting_price) * 100
    
    # Results
    winning = [t for t in trades if t['pnl_pct'] > 0]
    losing = [t for t in trades if t['pnl_pct'] <= 0]
    total_pnl = sum(t['pnl_pct'] for t in trades)
    
    result = {
        'ticker': ticker,
        'trades': trades,
        'starting_price': starting_price,
        'final_price': final_price,
        'max_price': max_price,
        'min_price': min_price,
        'max_gain': max_gain,
        'total_change': total_change,
        'total_pnl': total_pnl,
        'win_rate': len(winning) / len(trades) * 100 if len(trades) > 0 else 0,
        'avg_pnl': total_pnl / len(trades) if len(trades) > 0 else 0,
        'df': df_with_indicators
    }
    
    print(f"Trades: {len(trades)}")
    print(f"Winning: {len(winning)} ({result['win_rate']:.1f}%)")
    print(f"Total P&L: {total_pnl:.2f}%")
    print(f"Max Gain Available: {max_gain:.2f}%")
    print(f"Capture Rate: {total_pnl/max_gain*100:.1f}%" if max_gain > 0 else "N/A")
    
    # Print detailed entry/exit information
    if verbose and len(trades) > 0:
        print(f"\n{'='*80}")
        print(f"DETAILED TRADES - {ticker}")
        print(f"{'='*80}\n")
        
        for i, trade in enumerate(trades, 1):
            entry_time = pd.to_datetime(trade['entry_time'])
            exit_time = pd.to_datetime(trade['exit_time'])
            
            # Get entry context
            entry_idx = trade.get('entry_idx', None)
            
            entry_context = {}
            if entry_idx is not None and entry_idx < len(df_with_indicators):
                entry_row = df_with_indicators.iloc[entry_idx]
                entry_context = {
                    'volume_ratio': entry_row.get('volume_ratio', 0),
                    'momentum_10': entry_row.get('momentum_10', 0),
                    'momentum_20': entry_row.get('momentum_20', 0),
                    'rsi': entry_row.get('rsi', 0),
                    'macd_hist': entry_row.get('macd_hist', 0),
                    'price_position_20': entry_row.get('price_position_20', 0),
                    'sma5': entry_row.get('sma_5', 0),
                    'sma10': entry_row.get('sma_10', 0),
                    'sma20': entry_row.get('sma_20', 0),
                }
            else:
                # Fallback to stored values
                entry_context = {
                    'volume_ratio': trade.get('volume_ratio', 0),
                    'momentum_10': trade.get('momentum_10', 0),
                    'momentum_20': trade.get('momentum_20', 0),
                    'rsi': 0,
                    'macd_hist': 0,
                    'price_position_20': 0,
                    'sma5': 0,
                    'sma10': 0,
                    'sma20': 0,
                }
            
            print(f"TRADE #{i}")
            print(f"  Entry Time: {entry_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"  Entry Price: ${trade['entry_price']:.4f}")
            print(f"  Pattern: {trade['pattern']}")
            print(f"  Score: {trade['score']}/8")
            print(f"  Confidence: {trade['confidence']*100:.1f}%")
            print(f"  Entry Context:")
            print(f"    Volume Ratio: {entry_context.get('volume_ratio', 0):.2f}x")
            print(f"    Momentum: 10m={entry_context.get('momentum_10', 0):.1f}%, 20m={entry_context.get('momentum_20', 0):.1f}%")
            print(f"    RSI: {entry_context.get('rsi', 0):.1f}")
            print(f"    MACD Hist: {entry_context.get('macd_hist', 0):.4f}")
            print(f"    Price Position (20): {entry_context.get('price_position_20', 0):.1f}%")
            print(f"    MAs: SMA5=${entry_context.get('sma5', 0):.2f}, SMA10=${entry_context.get('sma10', 0):.2f}, SMA20=${entry_context.get('sma20', 0):.2f}")
            print(f"  Exit Time: {exit_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"  Exit Price: ${trade['exit_price']:.4f}")
            print(f"  Exit Reason: {trade['exit_reason']}")
            print(f"  Hold Time: {trade['hold_time_min']:.1f} minutes")
            print(f"  P&L: {trade['pnl_pct']:.2f}%")
            print(f"  P&L $: ${(trade['exit_price'] - trade['entry_price']):.2f} per share")
            
            # Calculate max price during hold
            if entry_idx is not None:
                exit_idx = None
                for idx in range(len(df_with_indicators)):
                    if pd.to_datetime(df_with_indicators.iloc[idx]['timestamp']) >= exit_time:
                        exit_idx = idx
                        break
                
                if exit_idx is not None and exit_idx > entry_idx:
                    max_price_during = df_with_indicators.iloc[entry_idx:exit_idx+1]['high'].max()
                    max_gain_during = ((max_price_during - trade['entry_price']) / trade['entry_price']) * 100
                    print(f"  Max Price During Hold: ${max_price_during:.4f} ({max_gain_during:.2f}% gain)")
            
            print()
    
    return result

def find_common_patterns(all_results):
    """Find common patterns across all stocks"""
    
    print(f"\n{'='*80}")
    print("COMMON PATTERNS ANALYSIS")
    print(f"{'='*80}\n")
    
    pattern_stats = defaultdict(lambda: {'count': 0, 'total_pnl': 0, 'wins': 0, 'losses': 0})
    
    for result in all_results:
        for trade in result['trades']:
            pattern = trade['pattern']
            pattern_stats[pattern]['count'] += 1
            pattern_stats[pattern]['total_pnl'] += trade['pnl_pct']
            if trade['pnl_pct'] > 0:
                pattern_stats[pattern]['wins'] += 1
            else:
                pattern_stats[pattern]['losses'] += 1
    
    print("Pattern Performance:")
    print("-" * 80)
    
    for pattern, stats in sorted(pattern_stats.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
        avg_pnl = stats['total_pnl'] / stats['count'] if stats['count'] > 0 else 0
        win_rate = stats['wins'] / stats['count'] * 100 if stats['count'] > 0 else 0
        
        print(f"\n{pattern}:")
        print(f"  Count: {stats['count']}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Total P&L: {stats['total_pnl']:.2f}%")
        print(f"  Average P&L: {avg_pnl:.2f}%")
    
    return pattern_stats

def main():
    """Main analysis function"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE STOCK ANALYSIS")
    print("Analyzing: GNPX, MLTX, VLN, INBS, ANPA")
    print("="*80)
    
    tickers = ['GNPX', 'MLTX', 'VLN', 'INBS', 'ANPA']
    all_results = []
    
    for ticker in tickers:
        result = analyze_stock(ticker, start_hour=4, verbose=True)
        if result:
            all_results.append(result)
    
    # Find common patterns
    pattern_stats = find_common_patterns(all_results)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    for result in all_results:
        print(f"{result['ticker']}:")
        print(f"  Max Gain: {result['max_gain']:.2f}%")
        print(f"  Trades: {len(result['trades'])}")
        print(f"  Win Rate: {result['win_rate']:.1f}%")
        print(f"  Total P&L: {result['total_pnl']:.2f}%")
        print(f"  Capture Rate: {result['total_pnl']/result['max_gain']*100:.1f}%" if result['max_gain'] > 0 else "N/A")
        print()
    
    # Export detailed CSV for each stock
    print(f"\n{'='*80}")
    print("EXPORTING DETAILED TRADES TO CSV")
    print(f"{'='*80}\n")
    
    for result in all_results:
        if len(result['trades']) > 0:
            # Create detailed trade records
            trade_records = []
            for trade in result['trades']:
                entry_time = pd.to_datetime(trade['entry_time'])
                exit_time = pd.to_datetime(trade['exit_time'])
                
                trade_records.append({
                    'Ticker': result['ticker'],
                    'Entry_Time': entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'Entry_Price': trade['entry_price'],
                    'Exit_Time': exit_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'Exit_Price': trade['exit_price'],
                    'Pattern': trade['pattern'],
                    'Score': trade['score'],
                    'Confidence': f"{trade['confidence']*100:.1f}%",
                    'Exit_Reason': trade['exit_reason'],
                    'Hold_Time_Min': f"{trade['hold_time_min']:.1f}",
                    'PnL_Pct': f"{trade['pnl_pct']:.2f}%",
                    'PnL_Dollar': f"${trade['exit_price'] - trade['entry_price']:.2f}",
                })
            
            df_trades = pd.DataFrame(trade_records)
            filename = f"analysis/{result['ticker']}_detailed_trades.csv"
            df_trades.to_csv(filename, index=False)
            print(f"Exported {len(trade_records)} trades for {result['ticker']} to {filename}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
