"""
Run analysis for all 5 stocks (ANPA, INBS, GNPX, MLTX, VLN) from 4 AM using fixed implementation
This uses the RealtimeTrader with the fixes to match original comprehensive_stock_analysis.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from data.webull_data_api import WebullDataAPI
from core.realtime_trader import RealtimeTrader
from collections import defaultdict

def fetch_minute_data_from_4am(ticker, date_str):
    """Fetch 1-minute data from 4 AM ET for the given date"""
    api = WebullDataAPI()
    
    print(f"Fetching 1-minute data for {ticker} (max 1200 minutes)...")
    
    try:
        # Fetch maximum available data (1200 minutes)
        df = api.get_1min_data(ticker, minutes=1200)
        if df is None or len(df) == 0:
            print(f"No data returned for {ticker}")
            return None
        
        print(f"Fetched {len(df)} bars of data for {ticker}")
        
        # Ensure timestamp is datetime and in ET timezone
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        et = pytz.timezone('US/Eastern')
        
        # Convert to ET timezone for filtering
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        df['timestamp'] = df['timestamp'].dt.tz_convert(et)
        
        # Filter from 4 AM ET on the given date
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        start_time = et.localize(datetime(date_obj.year, date_obj.month, date_obj.day, 4, 0, 0))
        end_time = et.localize(datetime(date_obj.year, date_obj.month, date_obj.day, 20, 0, 0))
        
        # Filter data from 4 AM onwards
        df_filtered = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)].copy()
        
        if len(df_filtered) == 0:
            print(f"No data available from 4 AM for {ticker} on {date_str}")
            return None
        
        print(f"Filtered to {len(df_filtered)} bars from 4 AM onwards")
        print(f"Data range: {df_filtered.iloc[0]['timestamp']} to {df_filtered.iloc[-1]['timestamp']}")
        
        # Sort by timestamp
        df_filtered = df_filtered.sort_values('timestamp').reset_index(drop=True)
        
        return df_filtered
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_trades_with_bot(ticker, date_str):
    """Analyze trades using the actual RealtimeTrader with FIXED implementation"""
    
    # Fetch data from 4 AM
    df = fetch_minute_data_from_4am(ticker, date_str)
    if df is None or len(df) < 50:
        print(f"Insufficient data for {ticker}")
        return None
    
    # Data is already filtered from 4 AM onwards
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    if len(df) < 50:
        print(f"Insufficient data after 4 AM for {ticker} (got {len(df)} bars, need at least 50)")
        return None
    
    print(f"\n{'='*80}")
    print(f"Analyzing {ticker} with FIXED BOT IMPLEMENTATION")
    print(f"Data range: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
    print(f"Total bars: {len(df)}")
    print(f"{'='*80}\n")
    
    # Create RealtimeTrader instance with FIXED implementation
    trader = RealtimeTrader(
        min_confidence=0.72,  # Equivalent to score >= 6 in original
        min_entry_price_increase=5.5,
        trailing_stop_pct=2.5,  # Base trailing stop (but exit logic uses dynamic stops 7%/10%)
        profit_target_pct=8.0
    )
    
    # Track all trades and entry opportunities
    all_trades = []
    entry_opportunities = []
    pattern_performance = defaultdict(lambda: {'trades': 0, 'wins': 0, 'total_pnl': 0, 'total_pnl_pct': 0})
    
    print(f"Starting simulation from index 50 (need at least 50 bars for indicators)...")
    print(f"Checking every minute from {df.iloc[50]['timestamp']} to {df.iloc[-1]['timestamp']}\n")
    
    # Simulate minute-by-minute
    for idx in range(50, len(df)):
        current = df.iloc[idx]
        current_time = pd.to_datetime(current['timestamp'])
        current_price = current.get('close', 0)
        
        # Get data up to current point
        df_up_to_now = df.iloc[:idx+1].copy()
        
        # Analyze with bot
        entry_signal, exit_signals = trader.analyze_data(df_up_to_now, ticker, current_price)
        
        # Handle entry signal
        if entry_signal and ticker not in trader.active_positions:
            print(f"\n{'='*80}")
            print(f"ENTRY OPPORTUNITY #{len(entry_opportunities) + 1}")
            print(f"{'='*80}")
            print(f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"Price: ${current_price:.4f}")
            print(f"Pattern: {entry_signal.pattern_name}")
            print(f"Confidence: {entry_signal.confidence:.2%}")
            print(f"Reason: {entry_signal.reason}")
            
            entry_opportunities.append({
                'time': current_time,
                'price': current_price,
                'pattern': entry_signal.pattern_name,
                'confidence': entry_signal.confidence,
                'reason': entry_signal.reason
            })
            
            # Enter position
            trader.enter_position(entry_signal)
            position = trader.active_positions.get(ticker)
            
            if position:
                print(f"\nPOSITION ENTERED:")
                print(f"  Entry Price: ${position.entry_price:.4f}")
                print(f"  Stop Loss: ${position.stop_loss:.4f} ({((position.stop_loss - position.entry_price) / position.entry_price) * 100:.2f}%)")
                print(f"  Target: ${position.target_price:.4f} ({((position.target_price - position.entry_price) / position.entry_price) * 100:.2f}%)")
                print(f"  Pattern: {position.entry_pattern}")
                print(f"  Confidence: {position.entry_confidence:.2%}")
        
        # Handle exit signals
        if exit_signals:
            for exit_signal in exit_signals:
                if exit_signal.ticker == ticker and ticker in trader.active_positions:
                    position = trader.active_positions.get(ticker)
                    
                    if position:
                        # Calculate P&L
                        exit_price = exit_signal.price
                        pnl_pct = ((exit_price - position.entry_price) / position.entry_price) * 100
                        hold_time = (current_time - position.entry_time).total_seconds() / 60
                        
                        # Get max price during hold
                        entry_idx = idx - int(hold_time) if int(hold_time) < idx else 0
                        max_price = df.iloc[entry_idx:idx+1]['high'].max() if entry_idx < idx else position.entry_price
                        max_profit_pct = ((max_price - position.entry_price) / position.entry_price) * 100
                        
                        print(f"\n{'='*80}")
                        print(f"EXIT SIGNAL - TRADE #{len(all_trades) + 1}")
                        print(f"{'='*80}")
                        print(f"Exit Time: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                        print(f"Exit Price: ${exit_price:.4f}")
                        print(f"Exit Reason: {exit_signal.reason}")
                        print(f"Entry Price: ${position.entry_price:.4f}")
                        print(f"Entry Time: {position.entry_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                        print(f"Hold Time: {hold_time:.1f} minutes")
                        print(f"P&L: {pnl_pct:.2f}%")
                        print(f"Max Price During Hold: ${max_price:.4f} ({max_profit_pct:.2f}% gain)")
                        print(f"Capture Rate: {(pnl_pct / max_profit_pct * 100) if max_profit_pct > 0 else 0:.1f}%")
                        
                        # Record trade
                        trade = {
                            'entry_time': position.entry_time,
                            'entry_price': position.entry_price,
                            'exit_time': current_time,
                            'exit_price': exit_price,
                            'exit_reason': exit_signal.reason,
                            'pattern': position.entry_pattern,
                            'confidence': position.entry_confidence,
                            'hold_time_min': hold_time,
                            'pnl_pct': pnl_pct,
                            'pnl_dollars': (exit_price - position.entry_price),  # Per share
                            'max_price': max_price,
                            'max_profit_pct': max_profit_pct,
                            'capture_rate': (pnl_pct / max_profit_pct * 100) if max_profit_pct > 0 else 0
                        }
                        all_trades.append(trade)
                        
                        # Update pattern performance
                        pattern_performance[position.entry_pattern]['trades'] += 1
                        if pnl_pct > 0:
                            pattern_performance[position.entry_pattern]['wins'] += 1
                        pattern_performance[position.entry_pattern]['total_pnl'] += pnl_pct
                        pattern_performance[position.entry_pattern]['total_pnl_pct'] += pnl_pct
                        
                        # Exit position
                        trader.exit_position(exit_signal)
                        
                        print(f"\nTRADE COMPLETED:")
                        print(f"  Total P&L: {pnl_pct:.2f}%")
                        print(f"  Win: {'Yes' if pnl_pct > 0 else 'No'}")
                        print(f"{'='*80}\n")
    
    # Handle open position at end
    if ticker in trader.active_positions:
        position = trader.active_positions[ticker]
        final_price = df.iloc[-1].get('close', 0)
        final_time = pd.to_datetime(df.iloc[-1]['timestamp'])
        pnl_pct = ((final_price - position.entry_price) / position.entry_price) * 100
        hold_time = (final_time - position.entry_time).total_seconds() / 60
        
        print(f"\n{'='*80}")
        print(f"OPEN POSITION AT END - TRADE #{len(all_trades) + 1}")
        print(f"{'='*80}")
        print(f"Entry Time: {position.entry_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Entry Price: ${position.entry_price:.4f}")
        print(f"Final Price: ${final_price:.4f}")
        print(f"Hold Time: {hold_time:.1f} minutes")
        print(f"P&L: {pnl_pct:.2f}%")
        print(f"{'='*80}\n")
        
        trade = {
            'entry_time': position.entry_time,
            'entry_price': position.entry_price,
            'exit_time': final_time,
            'exit_price': final_price,
            'exit_reason': 'End of Day',
            'pattern': position.entry_pattern,
            'confidence': position.entry_confidence,
            'hold_time_min': hold_time,
            'pnl_pct': pnl_pct,
            'pnl_dollars': (final_price - position.entry_price),
            'max_price': position.max_price_reached if position.max_price_reached > 0 else position.entry_price,
            'max_profit_pct': ((position.max_price_reached - position.entry_price) / position.entry_price * 100) if position.max_price_reached > 0 else 0,
            'capture_rate': 0
        }
        all_trades.append(trade)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY FOR {ticker}")
    print(f"{'='*80}")
    print(f"Total Entry Opportunities: {len(entry_opportunities)}")
    print(f"Total Trades: {len(all_trades)}")
    
    if len(all_trades) > 0:
        wins = sum(1 for t in all_trades if t['pnl_pct'] > 0)
        losses = len(all_trades) - wins
        total_pnl_pct = sum(t['pnl_pct'] for t in all_trades)
        avg_pnl_pct = total_pnl_pct / len(all_trades)
        avg_hold_time = sum(t['hold_time_min'] for t in all_trades) / len(all_trades)
        
        print(f"Wins: {wins} ({wins/len(all_trades)*100:.1f}%)")
        print(f"Losses: {losses} ({losses/len(all_trades)*100:.1f}%)")
        print(f"Total P&L: {total_pnl_pct:.2f}%")
        print(f"Average P&L: {avg_pnl_pct:.2f}%")
        print(f"Average Hold Time: {avg_hold_time:.1f} minutes")
        
        print(f"\nPattern Performance:")
        for pattern, perf in pattern_performance.items():
            win_rate = (perf['wins'] / perf['trades'] * 100) if perf['trades'] > 0 else 0
            avg_pnl = (perf['total_pnl_pct'] / perf['trades']) if perf['trades'] > 0 else 0
            print(f"  {pattern}: {perf['trades']} trades, {perf['wins']} wins ({win_rate:.1f}%), Avg P&L: {avg_pnl:.2f}%")
    else:
        print("No trades executed")
    
    print(f"{'='*80}\n")
    
    return {
        'ticker': ticker,
        'trades': all_trades,
        'entry_opportunities': entry_opportunities,
        'pattern_performance': dict(pattern_performance)
    }

def main():
    """Run analysis for all 5 stocks"""
    
    # Date to analyze (2026-01-09 based on the CSV files)
    date_str = '2026-01-09'
    
    # List of stocks to analyze
    tickers = ['ANPA', 'INBS', 'GNPX', 'MLTX', 'VLN']
    
    print(f"\n{'='*80}")
    print(f"RUNNING FIXED IMPLEMENTATION ANALYSIS FOR ALL 5 STOCKS")
    print(f"Date: {date_str}")
    print(f"Stocks: {', '.join(tickers)}")
    print(f"{'='*80}\n")
    
    all_results = {}
    
    for ticker in tickers:
        try:
            result = analyze_trades_with_bot(ticker, date_str)
            if result:
                all_results[ticker] = result
                
                # Export to CSV
                if len(result['trades']) > 0:
                    trades_df = pd.DataFrame(result['trades'])
                    csv_filename = f"{ticker}_fixed_implementation_trades_{date_str}.csv"
                    csv_path = os.path.join(os.path.dirname(__file__), csv_filename)
                    trades_df.to_csv(csv_path, index=False)
                    print(f"Exported trades to {csv_filename}\n")
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # Overall summary
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY - ALL 5 STOCKS")
    print(f"{'='*80}")
    
    total_trades = 0
    total_wins = 0
    total_pnl = 0
    
    for ticker, result in all_results.items():
        trades = result['trades']
        if len(trades) > 0:
            total_trades += len(trades)
            total_wins += sum(1 for t in trades if t['pnl_pct'] > 0)
            total_pnl += sum(t['pnl_pct'] for t in trades)
            print(f"{ticker}: {len(trades)} trades, {sum(1 for t in trades if t['pnl_pct'] > 0)} wins, {sum(t['pnl_pct'] for t in trades):.2f}% total P&L")
        else:
            print(f"{ticker}: 0 trades")
    
    if total_trades > 0:
        print(f"\nTotal: {total_trades} trades, {total_wins} wins ({total_wins/total_trades*100:.1f}%), {total_pnl:.2f}% total P&L")
    
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
