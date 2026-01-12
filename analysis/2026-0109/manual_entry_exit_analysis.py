"""
Manual analysis of INBS and ANPA to identify optimal entry/exit points
based on price action, volume, and technical indicators
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

def identify_optimal_entries_exits(df, ticker):
    """Identify optimal entry and exit points based on technical analysis"""
    
    opportunities = []
    
    # Calculate additional indicators
    df['price_change_5'] = df['close'].pct_change(5) * 100
    df['price_change_10'] = df['close'].pct_change(10) * 100
    df['price_change_20'] = df['close'].pct_change(20) * 100
    
    # Volume indicators
    df['volume_ma_10'] = df['volume'].rolling(10).mean()
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_trend'] = df['volume'] / df['volume_ma_10']
    
    # Price position relative to range
    df['high_10'] = df['high'].rolling(10).max()
    df['low_10'] = df['low'].rolling(10).min()
    df['price_position'] = (df['close'] - df['low_10']) / (df['high_10'] - df['low_10']) * 100
    
    # Moving average crossovers
    df['sma5_above_sma10'] = df['sma_5'] > df['sma_10']
    df['sma10_above_sma20'] = df['sma_10'] > df['sma_20']
    df['price_above_all_ma'] = (df['close'] > df['sma_5']) & (df['close'] > df['sma_10']) & (df['close'] > df['sma_20'])
    
    # MACD indicators
    df['macd_bullish'] = df['macd'] > df['macd_signal']
    df['macd_hist_positive'] = df['macd_hist'] > 0
    df['macd_hist_increasing'] = df['macd_hist'] > df['macd_hist'].shift(1)
    
    # Entry criteria for slow movers
    for idx in range(30, len(df)):
        current = df.iloc[idx]
        current_time = pd.to_datetime(current['timestamp'])
        current_price = current.get('close', 0)
        
        # Entry criteria
        entry_score = 0
        entry_reasons = []
        
        # 1. Volume ratio in moderate-high range (1.8x - 3.5x)
        vol_ratio = current.get('volume_ratio', 0)
        if 1.8 <= vol_ratio < 3.5:
            entry_score += 1
            entry_reasons.append(f"Vol ratio {vol_ratio:.2f}x")
        
        # 2. Sustained momentum
        momentum_10 = current.get('price_change_10', 0)
        momentum_20 = current.get('price_change_20', 0)
        if momentum_10 >= 2.0 and momentum_20 >= 3.0:
            entry_score += 1
            entry_reasons.append(f"Momentum 10m={momentum_10:.1f}%, 20m={momentum_20:.1f}%")
        
        # 3. Volume building
        if current.get('volume_trend', 0) >= 1.3:
            entry_score += 1
            entry_reasons.append(f"Volume building ({current.get('volume_trend', 0):.2f}x)")
        
        # 4. MACD accelerating
        if current.get('macd_hist_positive', False) and current.get('macd_hist_increasing', False):
            entry_score += 1
            entry_reasons.append("MACD accelerating")
        
        # 5. Price breaking above consolidation
        if current.get('price_position', 0) >= 80:  # Top 20% of range
            entry_score += 1
            entry_reasons.append(f"Price breakout ({current.get('price_position', 0):.1f}% of range)")
        
        # 6. Higher highs pattern
        if idx >= 20:
            older_highs = df.iloc[idx-20:idx-10]['high'].values
            newer_highs = df.iloc[idx-10:idx+1]['high'].values
            if len(older_highs) > 0 and len(newer_highs) > 0:
                if max(newer_highs) > max(older_highs) * 1.02:
                    entry_score += 1
                    entry_reasons.append("Higher highs pattern")
        
        # 7. Technical setup
        if (current.get('price_above_all_ma', False) and 
            current.get('sma5_above_sma10', False) and 
            current.get('sma10_above_sma20', False) and
            current.get('macd_bullish', False)):
            entry_score += 1
            entry_reasons.append("Technical setup bullish")
        
        # 8. RSI in accumulation zone
        rsi = current.get('rsi', 50)
        if 50 <= rsi <= 65:
            entry_score += 1
            entry_reasons.append(f"RSI accumulation ({rsi:.1f})")
        
        # Entry if score >= 6 (similar to perfect setup score)
        if entry_score >= 6:
            # Calculate stop loss and target
            stop_loss = current_price * 0.85  # 15% stop loss
            target = current_price * 1.15  # 15% target
            
            opportunities.append({
                'type': 'ENTRY',
                'time': current_time,
                'price': current_price,
                'score': entry_score,
                'reasons': entry_reasons,
                'volume_ratio': vol_ratio,
                'momentum_10': momentum_10,
                'momentum_20': momentum_20,
                'stop_loss': stop_loss,
                'target': target,
                'idx': idx
            })
    
    # Identify exit points for each entry
    completed_trades = []
    for entry in opportunities:
        entry_idx = entry['idx']
        entry_price = entry['price']
        stop_loss = entry['stop_loss']
        target = entry['target']
        
        # Find exit point
        exit_found = False
        for idx in range(entry_idx + 1, len(df)):
            current = df.iloc[idx]
            current_time = pd.to_datetime(current['timestamp'])
            current_price = current.get('close', 0)
            current_high = current.get('high', 0)
            
            exit_reason = None
            exit_price = current_price
            
            # Check stop loss
            if current_price <= stop_loss:
                exit_reason = "Stop Loss"
                exit_price = stop_loss
            
            # Check profit target
            elif current_high >= target:
                exit_reason = "Profit Target"
                exit_price = target
            
            # Check trailing stop (3% from high)
            else:
                max_price = df.iloc[entry_idx:idx+1]['high'].max()
                trailing_stop = max_price * 0.97
                if current_price <= trailing_stop:
                    exit_reason = "Trailing Stop"
                    exit_price = trailing_stop
            
            # Check for reversal signals
            if exit_reason is None:
                # MACD bearish crossover
                if idx > 0:
                    prev_macd_bullish = df.iloc[idx-1].get('macd_bullish', False)
                    if prev_macd_bullish and not current.get('macd_bullish', False):
                        exit_reason = "MACD Bearish"
                        exit_price = current_price
                
                # Price below MAs
                if not current.get('price_above_all_ma', False):
                    exit_reason = "Price Below MAs"
                    exit_price = current_price
            
            if exit_reason:
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                hold_time = (current_time - entry['time']).total_seconds() / 60
                
                completed_trades.append({
                    'entry_time': entry['time'],
                    'entry_price': entry_price,
                    'exit_time': current_time,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl_pct': pnl_pct,
                    'hold_time_min': hold_time,
                    'entry_score': entry['score'],
                    'entry_reasons': entry['reasons']
                })
                exit_found = True
                break
        
        # If no exit found, use final price
        if not exit_found:
            final_price = df.iloc[-1].get('close', 0)
            pnl_pct = ((final_price - entry_price) / entry_price) * 100
            hold_time = (df.iloc[-1]['timestamp'] - entry['time']).total_seconds() / 60
            
            completed_trades.append({
                'entry_time': entry['time'],
                'entry_price': entry_price,
                'exit_time': df.iloc[-1]['timestamp'],
                'exit_price': final_price,
                'exit_reason': "End of Day",
                'pnl_pct': pnl_pct,
                'hold_time_min': hold_time,
                'entry_score': entry['score'],
                'entry_reasons': entry['reasons']
            })
    
    return opportunities, completed_trades

def analyze_stock(ticker, start_hour=4):
    """Analyze a stock for entry/exit opportunities"""
    
    print(f"\n{'='*80}")
    print(f"MANUAL ANALYSIS - {ticker}")
    print(f"{'='*80}\n")
    
    data_api = WebullDataAPI()
    pattern_detector = PatternDetector()
    
    # Fetch data
    print(f"Fetching {ticker} data from {start_hour}:00 AM...")
    df_1min = data_api.get_1min_data(ticker, minutes=800)
    
    if df_1min is None or len(df_1min) == 0:
        print(f"ERROR: No data available for {ticker}")
        return None
    
    print(f"Retrieved {len(df_1min)} minutes of data")
    
    # Filter data
    df_1min['timestamp'] = pd.to_datetime(df_1min['timestamp'])
    if df_1min['timestamp'].dt.tz is None:
        df_1min['timestamp'] = df_1min['timestamp'].dt.tz_localize('US/Eastern')
    
    et = pytz.timezone('US/Eastern')
    today = datetime.now(et).date()
    start_time = et.localize(datetime.combine(today, datetime.min.time().replace(hour=start_hour, minute=0)))
    
    df_filtered = df_1min[df_1min['timestamp'] >= start_time].copy()
    
    if len(df_filtered) == 0:
        df_filtered = df_1min.copy()
    
    print(f"Filtered to {len(df_filtered)} minutes from {start_hour}:00 AM onwards")
    
    # Calculate indicators
    print("Calculating indicators...")
    df_with_indicators = pattern_detector.calculate_indicators(df_filtered)
    
    if len(df_with_indicators) < 30:
        print(f"ERROR: Insufficient data")
        return None
    
    print(f"Indicators calculated\n")
    
    # Identify opportunities
    print("Identifying entry/exit opportunities...\n")
    opportunities, completed_trades = identify_optimal_entries_exits(df_with_indicators, ticker)
    
    # Print results
    print(f"{'='*80}")
    print(f"RESULTS - {ticker}")
    print(f"{'='*80}\n")
    
    print(f"Entry Opportunities Found: {len(opportunities)}")
    print(f"Completed Trades: {len(completed_trades)}\n")
    
    if len(opportunities) > 0:
        print("ENTRY OPPORTUNITIES:")
        for i, opp in enumerate(opportunities[:10], 1):  # Show first 10
            print(f"\n{i}. {opp['time'].strftime('%H:%M:%S')} @ ${opp['price']:.4f}")
            print(f"   Score: {opp['score']}/8")
            print(f"   Reasons: {', '.join(opp['reasons'])}")
            print(f"   Volume Ratio: {opp['volume_ratio']:.2f}x")
            print(f"   Momentum: 10m={opp['momentum_10']:.1f}%, 20m={opp['momentum_20']:.1f}%")
            print(f"   Target: ${opp['target']:.4f}, Stop: ${opp['stop_loss']:.4f}")
    
    if len(completed_trades) > 0:
        print(f"\n{'='*80}")
        print("COMPLETED TRADES:")
        print(f"{'='*80}\n")
        
        total_pnl = sum(t['pnl_pct'] for t in completed_trades)
        winning = [t for t in completed_trades if t['pnl_pct'] > 0]
        losing = [t for t in completed_trades if t['pnl_pct'] <= 0]
        
        for i, trade in enumerate(completed_trades, 1):
            print(f"{i}. Entry: {trade['entry_time'].strftime('%H:%M:%S')} @ ${trade['entry_price']:.4f}")
            print(f"   Exit: {trade['exit_time'].strftime('%H:%M:%S')} @ ${trade['exit_price']:.4f}")
            print(f"   Reason: {trade['exit_reason']}")
            print(f"   P&L: {trade['pnl_pct']:.2f}%")
            print(f"   Hold Time: {trade['hold_time_min']:.1f} minutes")
            print(f"   Entry Score: {trade['entry_score']}/8")
            print()
        
        print(f"SUMMARY:")
        print(f"  Total Trades: {len(completed_trades)}")
        print(f"  Winning: {len(winning)} ({len(winning)/len(completed_trades)*100:.1f}%)")
        print(f"  Losing: {len(losing)} ({len(losing)/len(completed_trades)*100:.1f}%)")
        print(f"  Average P&L: {total_pnl/len(completed_trades):.2f}%")
        print(f"  Total P&L: {total_pnl:.2f}%")
        print(f"  Best Trade: {max(completed_trades, key=lambda x: x['pnl_pct'])['pnl_pct']:.2f}%")
        print(f"  Worst Trade: {min(completed_trades, key=lambda x: x['pnl_pct'])['pnl_pct']:.2f}%")
    
    # Price analysis
    starting_price = df_with_indicators.iloc[0].get('close', 0)
    final_price = df_with_indicators.iloc[-1].get('close', 0)
    max_price = df_with_indicators['high'].max()
    
    max_gain = ((max_price - starting_price) / starting_price) * 100
    total_change = ((final_price - starting_price) / starting_price) * 100
    
    print(f"\n{'='*80}")
    print("PRICE ANALYSIS:")
    print(f"{'='*80}")
    print(f"Starting Price: ${starting_price:.4f}")
    print(f"Final Price: ${final_price:.4f}")
    print(f"Maximum Price: ${max_price:.4f}")
    print(f"Maximum Gain Available: {max_gain:.2f}%")
    print(f"Total Change: {total_change:.2f}%")
    
    if len(completed_trades) > 0:
        captured_pnl = sum(t['pnl_pct'] for t in completed_trades)
        print(f"Captured P&L: {captured_pnl:.2f}%")
        print(f"Capture Rate: {captured_pnl/max_gain*100:.1f}%")
    
    return {
        'ticker': ticker,
        'opportunities': opportunities,
        'completed_trades': completed_trades,
        'max_gain': max_gain,
        'total_change': total_change
    }

def main():
    """Main function"""
    print("\n" + "="*80)
    print("MANUAL ENTRY/EXIT OPPORTUNITY ANALYSIS")
    print("Identifying optimal entry/exit points for INBS and ANPA")
    print("="*80)
    
    results = {}
    
    # Analyze INBS
    results['INBS'] = analyze_stock('INBS', start_hour=4)
    
    # Analyze ANPA
    results['ANPA'] = analyze_stock('ANPA', start_hour=4)
    
    # Comparative summary
    print(f"\n{'='*80}")
    print("COMPARATIVE SUMMARY")
    print(f"{'='*80}\n")
    
    for ticker, result in results.items():
        if result:
            print(f"{ticker}:")
            print(f"  Max Gain Available: {result['max_gain']:.2f}%")
            print(f"  Entry Opportunities: {len(result['opportunities'])}")
            print(f"  Completed Trades: {len(result['completed_trades'])}")
            if len(result['completed_trades']) > 0:
                total_pnl = sum(t['pnl_pct'] for t in result['completed_trades'])
                print(f"  Total P&L: {total_pnl:.2f}%")
                print(f"  Capture Rate: {total_pnl/result['max_gain']*100:.1f}%")
            print()

if __name__ == "__main__":
    main()
