"""
Analyze ANPA using the bot (RealtimeTrader) from 4AM on Friday
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from data.webull_data_api import WebullDataAPI
from core.realtime_trader import RealtimeTrader, ActivePosition
from analysis.pattern_detector import PatternDetector
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_with_bot(ticker, start_date='2026-01-09', start_hour=4, verbose=True):
    """
    Analyze a stock using RealtimeTrader from a specific start time
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        start_hour: Start hour (4 = 4 AM ET)
        verbose: Print detailed information
    """
    et = pytz.timezone('US/Eastern')
    
    # Parse start date
    start_date_obj = pd.to_datetime(start_date)
    start_time = start_date_obj.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    start_time_et = et.localize(start_time) if start_time.tz is None else start_time.tz_convert(et)
    
    print(f"\n{'='*80}")
    print(f"ANALYZING {ticker} WITH BOT FROM {start_time_et.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"{'='*80}\n")
    
    # Initialize API and trader
    api = WebullDataAPI()
    pattern_detector = PatternDetector()
    
    # Create trader instance with appropriate settings
    trader = RealtimeTrader(
        min_confidence=0.72,
        min_entry_price_increase=5.5,
        trailing_stop_pct=2.5,
        profit_target_pct=8.0,
        data_api=api
    )
    
    # Fetch minute data (fetch enough for a full trading day)
    print(f"Fetching {ticker} data...")
    try:
        df = api.get_1min_data(ticker, minutes=960)  # 16 hours of data (4 AM to 8 PM)
        if df is None or len(df) == 0:
            print(f"Error: No data returned for {ticker}")
            return None
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            df.reset_index(inplace=True)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                df['timestamp'] = pd.to_datetime(df.index)
        
        # Filter data from start time onwards
        df = df[df['timestamp'] >= start_time_et].copy()
        
        if len(df) == 0:
            print(f"Error: No data after {start_time_et.strftime('%Y-%m-%d %H:%M %Z')}")
            return None
        
        print(f"Data points: {len(df)}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Calculate indicators
    print("Calculating indicators...")
    df_with_indicators = pattern_detector.calculate_indicators(df)
    
    # Simulate bot behavior
    print("Running bot simulation...\n")
    
    completed_trades = []
    current_position = None
    
    for idx in range(30, len(df_with_indicators)):
        current = df_with_indicators.iloc[idx]
        current_time = pd.to_datetime(current['timestamp'])
        current_price = current.get('close', 0)
        
        # Check for entry
        if current_position is None:
            # Use trader's analyze_data to check for entry
            df_slice = df_with_indicators.iloc[:idx+1]
            entry_signal, exit_signals = trader.analyze_data(df_slice, ticker, current_price)
            
            if entry_signal:
                # Enter position
                current_position = ActivePosition(
                    ticker=ticker,
                    entry_time=current_time,
                    entry_price=current_price,
                    entry_pattern=entry_signal.reason if hasattr(entry_signal, 'reason') else entry_signal.pattern if hasattr(entry_signal, 'pattern') else 'Unknown',
                    entry_confidence=entry_signal.confidence,
                    target_price=current_price * 1.20,  # 20% target
                    stop_loss=current_price * 0.85,  # 15% stop
                    current_price=current_price,
                    unrealized_pnl_pct=0.0,
                    max_price_reached=current_price,
                    shares=100.0,  # Assume 100 shares
                    entry_value=current_price * 100.0,
                    original_shares=100.0,
                    is_slow_mover_entry=entry_signal.indicators.get('is_slow_mover_entry', False) if hasattr(entry_signal, 'indicators') else False
                )
                
                # Add position to trader's active_positions so exit logic can find it
                trader.active_positions[ticker] = current_position
                
                if verbose:
                    print(f"ENTRY: {current_time.strftime('%Y-%m-%d %H:%M:%S')} - {ticker} @ ${current_price:.2f}")
                    print(f"  Pattern: {current_position.entry_pattern}, Confidence: {current_position.entry_confidence*100:.1f}%")
        else:
            # Check for exit using the trader's exit logic
            df_slice = df_with_indicators.iloc[:idx+1]
            
            # Ensure position is in trader's active_positions
            if ticker not in trader.active_positions:
                trader.active_positions[ticker] = current_position
            
            # Use the trader's _check_exit_signals method to check for exits
            try:
                exit_signals = trader._check_exit_signals(
                    df_slice,
                    ticker,
                    current_price
                )
            except Exception as e:
                if verbose:
                    print(f"Error checking exit signals: {e}")
                exit_signals = []
            
            # Handle exit signals (both partial and full exits)
            if exit_signals:
                full_exit = None
                partial_exits_taken = []
                
                for exit_signal in exit_signals:
                    if exit_signal.signal_type == 'exit':
                        full_exit = exit_signal
                    elif exit_signal.signal_type == 'partial_exit':
                        partial_exits_taken.append(exit_signal)
                        # Handle partial exit - reduce position size
                        if '50%' in exit_signal.reason:
                            current_position.shares = current_position.shares * 0.5
                            current_position.partial_profit_taken = True
                        elif '25%' in exit_signal.reason:
                            current_position.shares = current_position.shares * 0.25
                            current_position.partial_profit_taken_second = True
                        elif '12.5%' in exit_signal.reason:
                            current_position.shares = current_position.shares * 0.125
                            current_position.partial_profit_taken_third = True
                        
                        if verbose:
                            print(f"PARTIAL EXIT: {current_time.strftime('%Y-%m-%d %H:%M:%S')} - {ticker} @ ${exit_signal.price:.2f}")
                            print(f"  Reason: {exit_signal.reason}")
                            print(f"  Remaining Shares: {current_position.shares:.1f}\n")
                
                # Handle full exit
                if full_exit:
                    # Calculate P&L including partial exits
                    exit_price = full_exit.price
                    pnl_pct = ((exit_price - current_position.entry_price) / current_position.entry_price) * 100
                    
                    # Adjust P&L for partial exits (only count remaining position)
                    position_ratio = current_position.shares / current_position.original_shares if current_position.original_shares > 0 else 1.0
                    remaining_pnl_pct = pnl_pct * position_ratio
                    
                    # Calculate partial exit P&L
                    partial_pnl_pct = 0.0
                    for partial_exit in partial_exits_taken:
                        partial_pnl = ((partial_exit.price - current_position.entry_price) / current_position.entry_price) * 100
                        if '50%' in partial_exit.reason:
                            partial_pnl_pct += partial_pnl * 0.5
                        elif '25%' in partial_exit.reason:
                            partial_pnl_pct += partial_pnl * 0.25
                        elif '12.5%' in partial_exit.reason:
                            partial_pnl_pct += partial_pnl * 0.125
                    
                    total_pnl_pct = partial_pnl_pct + remaining_pnl_pct
                    hold_time = (current_time - current_position.entry_time).total_seconds() / 60
                    
                    trade = {
                        'entry_time': current_position.entry_time,
                        'entry_price': current_position.entry_price,
                        'exit_time': current_time,
                        'exit_price': exit_price,
                        'exit_reason': full_exit.reason,
                        'pnl_pct': total_pnl_pct,
                        'hold_time_min': hold_time,
                        'pattern': current_position.entry_pattern,
                        'confidence': current_position.entry_confidence,
                        'max_price_during': current_position.max_price_reached,
                        'partial_exits': len(partial_exits_taken)
                    }
                    
                    completed_trades.append(trade)
                    
                    if verbose:
                        print(f"EXIT: {current_time.strftime('%Y-%m-%d %H:%M:%S')} - {ticker} @ ${exit_price:.2f}")
                        print(f"  Reason: {full_exit.reason}")
                        print(f"  P&L: {total_pnl_pct:.2f}%, Hold: {hold_time:.1f} min")
                        print(f"  Max Price: ${current_position.max_price_reached:.2f} ({((current_position.max_price_reached - current_position.entry_price) / current_position.entry_price) * 100:.2f}%)")
                        if len(partial_exits_taken) > 0:
                            print(f"  Partial Exits: {len(partial_exits_taken)}\n")
                        else:
                            print()
                    
                    # Remove position from trader's active_positions
                    if ticker in trader.active_positions:
                        del trader.active_positions[ticker]
                    current_position = None
    
    # Handle open position at end
    if current_position is not None:
        final_price = df_with_indicators.iloc[-1].get('close', 0)
        pnl_pct = ((final_price - current_position.entry_price) / current_position.entry_price) * 100
        hold_time = (df_with_indicators.iloc[-1]['timestamp'] - current_position.entry_time).total_seconds() / 60
        
        trade = {
            'entry_time': current_position.entry_time,
            'entry_price': current_position.entry_price,
            'exit_time': df_with_indicators.iloc[-1]['timestamp'],
            'exit_price': final_price,
            'exit_reason': 'End of Day',
            'pnl_pct': pnl_pct,
            'hold_time_min': hold_time,
            'pattern': current_position.entry_pattern,
            'confidence': current_position.entry_confidence,
            'max_price_during': current_position.max_price_reached
        }
        
        completed_trades.append(trade)
        
        if verbose:
            print(f"END OF DAY EXIT: {df_with_indicators.iloc[-1]['timestamp']} - {ticker} @ ${final_price:.2f}")
            print(f"  P&L: {pnl_pct:.2f}%, Hold: {hold_time:.1f} min\n")
    
    # Calculate summary
    if len(completed_trades) > 0:
        winning = [t for t in completed_trades if t['pnl_pct'] > 0]
        losing = [t for t in completed_trades if t['pnl_pct'] <= 0]
        total_pnl = sum(t['pnl_pct'] for t in completed_trades)
        avg_pnl = total_pnl / len(completed_trades) if len(completed_trades) > 0 else 0
        win_rate = (len(winning) / len(completed_trades) * 100) if len(completed_trades) > 0 else 0
        
        # Calculate max gain available
        max_price = df_with_indicators['high'].max()
        min_price_in_period = df_with_indicators.iloc[:len(df_with_indicators)//2]['low'].min() if len(df_with_indicators) > 0 else 0
        if min_price_in_period > 0:
            max_gain = ((max_price - min_price_in_period) / min_price_in_period) * 100
        else:
            max_gain = 0
        
        capture_rate = (total_pnl / max_gain * 100) if max_gain > 0 else 0
        
        print(f"{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}\n")
        print(f"Trades: {len(completed_trades)}")
        print(f"Winning: {len(winning)} ({win_rate:.1f}%)")
        print(f"Losing: {len(losing)}")
        print(f"Total P&L: {total_pnl:.2f}%")
        print(f"Average P&L: {avg_pnl:.2f}%")
        print(f"Max Gain Available: {max_gain:.2f}%")
        print(f"Capture Rate: {capture_rate:.1f}%\n")
        
        # Print detailed trades
        print(f"{'='*80}")
        print("DETAILED TRADES")
        print(f"{'='*80}\n")
        
        for i, trade in enumerate(completed_trades, 1):
            entry_time = pd.to_datetime(trade['entry_time'])
            exit_time = pd.to_datetime(trade['exit_time'])
            pnl_dollar = (trade['exit_price'] - trade['entry_price']) * 100  # Assuming 100 shares
            
            print(f"TRADE #{i}")
            print(f"  Entry: {entry_time.strftime('%Y-%m-%d %H:%M:%S')} @ ${trade['entry_price']:.2f}")
            print(f"  Exit: {exit_time.strftime('%Y-%m-%d %H:%M:%S')} @ ${trade['exit_price']:.2f}")
            print(f"  Pattern: {trade['pattern']}, Confidence: {trade['confidence']*100:.1f}%")
            print(f"  Exit Reason: {trade['exit_reason']}")
            print(f"  Hold Time: {trade['hold_time_min']:.1f} minutes")
            print(f"  P&L: {trade['pnl_pct']:.2f}% (${pnl_dollar:.2f})")
            print(f"  Max Price: ${trade['max_price_during']:.2f} ({((trade['max_price_during'] - trade['entry_price']) / trade['entry_price']) * 100:.2f}%)\n")
        
        return {
            'ticker': ticker,
            'trades': completed_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'max_gain': max_gain,
            'capture_rate': capture_rate
        }
    else:
        print("No trades executed.\n")
        return None

def main():
    """Run analysis for ANPA from 4 AM on Friday"""
    
    # Friday is 2026-01-09 based on the CSV data
    result = analyze_with_bot('ANPA', start_date='2026-01-09', start_hour=4, verbose=True)
    
    if result:
        print(f"\nAnalysis complete for {result['ticker']}!")
        print(f"Total P&L: {result['total_pnl']:.2f}%")
        print(f"Capture Rate: {result['capture_rate']:.1f}%")

if __name__ == "__main__":
    main()
