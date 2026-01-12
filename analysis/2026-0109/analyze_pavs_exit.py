"""
Analyze PAVS exit to understand why it exited early and missed the bull run
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from data.webull_data_api import WebullDataAPI
from analysis.pattern_detector import PatternDetector
from core.realtime_trader import RealtimeTrader

def analyze_pavs_exit():
    """Analyze PAVS exit and missed profit opportunity"""
    
    print(f"\n{'='*80}")
    print("PAVS EXIT ANALYSIS")
    print(f"{'='*80}\n")
    
    ticker = "PAVS"
    data_api = WebullDataAPI()
    pattern_detector = PatternDetector()
    trader = RealtimeTrader(min_confidence=0.72)
    
    # Entry details from logs
    entry_time = datetime(2026, 1, 9, 9, 6, 33, tzinfo=pytz.timezone('US/Eastern'))
    entry_price = 2.5400
    exit_time = datetime(2026, 1, 9, 9, 9, 42, tzinfo=pytz.timezone('US/Eastern'))
    exit_price = 2.5510
    
    print(f"Entry: {entry_time.strftime('%Y-%m-%d %H:%M:%S %Z')} @ ${entry_price:.4f}")
    print(f"Exit: {exit_time.strftime('%Y-%m-%d %H:%M:%S %Z')} @ ${exit_price:.4f}")
    print(f"Hold Time: {(exit_time - entry_time).total_seconds() / 60:.1f} minutes")
    print(f"P&L: {((exit_price - entry_price) / entry_price) * 100:.2f}% (${(exit_price - entry_price) * 2179:.2f})\n")
    
    # Fetch data from 4:00 AM to capture full day
    print("Fetching PAVS data from 4:00 AM...")
    df_1min = data_api.get_1min_data(ticker, minutes=800)
    
    if df_1min is None or len(df_1min) == 0:
        print(f"❌ ERROR: No data available for {ticker}")
        return
    
    print(f"✅ Retrieved {len(df_1min)} minutes of data")
    
    # Filter data from 4:00 AM ET onwards
    df_1min['timestamp'] = pd.to_datetime(df_1min['timestamp'])
    if df_1min['timestamp'].dt.tz is None:
        df_1min['timestamp'] = df_1min['timestamp'].dt.tz_localize('US/Eastern')
    
    # Find 4:00 AM on the trading day
    trading_day = entry_time.date()
    et = pytz.timezone('US/Eastern')
    start_time = et.localize(datetime.combine(trading_day, datetime.min.time().replace(hour=4, minute=0)))
    
    # Filter data from 4:00 AM
    df_filtered = df_1min[df_1min['timestamp'] >= start_time].copy()
    
    if len(df_filtered) == 0:
        print(f"⚠️  No data from 4:00 AM, using all available data")
        df_filtered = df_1min.copy()
    
    print(f"✅ Filtered to {len(df_filtered)} minutes from 4:00 AM onwards")
    print(f"   Data range: {df_filtered['timestamp'].min()} to {df_filtered['timestamp'].max()}\n")
    
    # Calculate indicators
    print("Calculating indicators...")
    df_with_indicators = pattern_detector.calculate_indicators(df_filtered)
    
    if len(df_with_indicators) < 30:
        print(f"❌ ERROR: Insufficient data ({len(df_with_indicators)} points, need 30+)")
        return
    
    print(f"✅ Indicators calculated for {len(df_with_indicators)} data points\n")
    
    # Find entry and exit indices
    entry_idx = None
    exit_idx = None
    
    for idx in range(len(df_with_indicators)):
        current = df_with_indicators.iloc[idx]
        current_time = pd.to_datetime(current['timestamp'])
        
        if entry_idx is None and current_time >= entry_time:
            entry_idx = idx
        if exit_idx is None and current_time >= exit_time:
            exit_idx = idx
            break
    
    if entry_idx is None or exit_idx is None:
        print(f"⚠️  Could not find exact entry/exit times in data")
        print(f"   Entry time: {entry_time}")
        print(f"   Exit time: {exit_time}")
        print(f"   Data range: {df_with_indicators['timestamp'].min()} to {df_with_indicators['timestamp'].max()}")
        # Use closest indices
        entry_idx = 0
        exit_idx = len(df_with_indicators) - 1
    
    print(f"Entry index: {entry_idx} ({df_with_indicators.iloc[entry_idx]['timestamp']})")
    print(f"Exit index: {exit_idx} ({df_with_indicators.iloc[exit_idx]['timestamp']})\n")
    
    # Analyze what happened at exit
    print(f"{'='*80}")
    print("EXIT ANALYSIS")
    print(f"{'='*80}\n")
    
    exit_data = df_with_indicators.iloc[exit_idx]
    exit_price_actual = exit_data['close']
    
    print(f"Exit Price (from data): ${exit_price_actual:.4f}")
    print(f"Exit Time: {exit_data['timestamp']}\n")
    
    # Check what exit signals would have been triggered
    print("Checking exit signals at exit time...")
    df_up_to_exit = df_with_indicators.iloc[:exit_idx+1]
    
    # Simulate position
    from core.realtime_trader import ActivePosition
    position = ActivePosition(
        ticker=ticker,
        entry_time=entry_time,
        entry_price=entry_price,
        shares=2179.0,
        entry_pattern="Strong_Bullish_Setup",
        target_price=2.92625,
        stop_loss=2.15372,
        confidence=0.85
    )
    
    # Check exit signals
    exit_signals = trader._check_exit_signals(df_up_to_exit, ticker, exit_price_actual)
    
    if exit_signals:
        print(f"\n✅ Exit signals found: {len(exit_signals)}")
        for sig in exit_signals:
            print(f"   - {sig.reason} @ ${sig.price:.4f}")
    else:
        print("\n⚠️  No exit signals found at exit time")
    
    # Analyze price movement after exit
    print(f"\n{'='*80}")
    print("PRICE MOVEMENT AFTER EXIT")
    print(f"{'='*80}\n")
    
    # Get data after exit
    df_after_exit = df_with_indicators.iloc[exit_idx+1:]
    
    if len(df_after_exit) > 0:
        max_price = df_after_exit['close'].max()
        max_price_time = df_after_exit.loc[df_after_exit['close'].idxmax(), 'timestamp']
        max_gain = ((max_price - entry_price) / entry_price) * 100
        max_profit = (max_price - entry_price) * 2179
        
        print(f"Maximum price after exit: ${max_price:.4f}")
        print(f"Time of maximum: {max_price_time}")
        print(f"Maximum gain if held: {max_gain:.2f}%")
        print(f"Maximum profit if held: ${max_profit:,.2f}")
        print(f"Missed profit: ${max_profit - (exit_price - entry_price) * 2179:,.2f}\n")
        
        # Show price progression
        print("Price progression after exit:")
        sample_indices = [0, len(df_after_exit)//4, len(df_after_exit)//2, len(df_after_exit)*3//4, len(df_after_exit)-1]
        for idx in sample_indices:
            if idx < len(df_after_exit):
                row = df_after_exit.iloc[idx]
                gain = ((row['close'] - entry_price) / entry_price) * 100
                profit = (row['close'] - entry_price) * 2179
                print(f"   {row['timestamp']}: ${row['close']:.4f} (+{gain:.2f}%, ${profit:,.2f})")
    
    # Analyze exit criteria
    print(f"\n{'='*80}")
    print("EXIT CRITERIA ANALYSIS")
    print(f"{'='*80}\n")
    
    # Check trailing stop
    print("Checking trailing stop logic...")
    if exit_idx > entry_idx:
        # Calculate what the trailing stop would have been
        df_for_trailing = df_with_indicators.iloc[entry_idx:exit_idx+1]
        
        max_price_seen = entry_price
        trailing_stop_price = entry_price
        
        for idx in range(len(df_for_trailing)):
            current_price = df_for_trailing.iloc[idx]['close']
            if current_price > max_price_seen:
                max_price_seen = current_price
            
            # Calculate ATR for trailing stop
            if idx >= 14:
                atr = calculate_atr(df_for_trailing.iloc[:idx+1], period=14)
                if len(atr) > 0 and not pd.isna(atr.iloc[-1]):
                    current_atr = atr.iloc[-1]
                    atr_pct = (current_atr / current_price) * 100 if current_price > 0 else 0
                    
                    # Trailing stop logic (from realtime_trader)
                    profit_pct = ((current_price - entry_price) / entry_price) * 100
                    
                    if profit_pct >= 3.0:  # Minimum profit threshold
                        if atr_pct > 0:
                            # Use ATR-based trailing stop (2x ATR)
                            trailing_stop_pct = min(2 * atr_pct, 5.0)  # Cap at 5%
                            trailing_stop_price = max(trailing_stop_price, current_price * (1 - trailing_stop_pct / 100))
                        else:
                            # Fallback to percentage-based
                            if profit_pct >= 15:
                                trailing_stop_pct = 5.0
                            elif profit_pct >= 10:
                                trailing_stop_pct = 4.0
                            elif profit_pct >= 5:
                                trailing_stop_pct = 3.0
                            else:
                                trailing_stop_pct = 2.5
                            
                            trailing_stop_price = max(trailing_stop_price, current_price * (1 - trailing_stop_pct / 100))
            
            if idx == len(df_for_trailing) - 1:
                print(f"   Final trailing stop: ${trailing_stop_price:.4f}")
                print(f"   Max price seen: ${max_price_seen:.4f}")
                print(f"   Exit price: ${exit_price_actual:.4f}")
                if exit_price_actual <= trailing_stop_price:
                    print(f"   ⚠️  Exit triggered by trailing stop!")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    if len(df_after_exit) > 0:
        max_price = df_after_exit['close'].max()
        max_gain = ((max_price - entry_price) / entry_price) * 100
        
        if max_gain > 10:
            print(f"⚠️  Significant missed opportunity: {max_gain:.2f}% gain was available")
            print("\nPotential improvements:")
            print("1. Consider wider trailing stops for premarket entries (premarket can be volatile)")
            print("2. Add minimum hold time for premarket entries (e.g., 15-30 minutes)")
            print("3. Consider not using trailing stops during premarket (only use stop loss)")
            print("4. Review trailing stop width - may be too tight for volatile stocks")
            print("5. Consider profit target instead of trailing stop for small gains")

if __name__ == "__main__":
    analyze_pavs_exit()
