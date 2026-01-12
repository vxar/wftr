"""Quick test to see which patterns are detected for ANPA"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from data.webull_data_api import WebullDataAPI
from analysis.pattern_detector import PatternDetector
from datetime import datetime

# Fetch ANPA data
api = WebullDataAPI()
df = api.get_1min_data('ANPA', minutes=1200)

if df is not None and len(df) >= 50:
    # Filter from 4 AM
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    
    import pytz
    et = pytz.timezone('US/Eastern')
    df['timestamp'] = df['timestamp'].dt.tz_convert(et)
    
    date_obj = datetime.strptime('2026-01-09', '%Y-%m-%d')
    start_time = et.localize(datetime(date_obj.year, date_obj.month, date_obj.day, 4, 0, 0))
    df = df[df['timestamp'] >= start_time].copy().reset_index(drop=True)
    
    if len(df) >= 50:
        # Calculate indicators
        pattern_detector = PatternDetector()
        df_with_indicators = pattern_detector.calculate_indicators(df)
        
        # Test pattern detection at several points
        print(f"\nTesting pattern detection for ANPA (from {df.iloc[50]['timestamp']} to {df.iloc[-1]['timestamp']})")
        print(f"Total bars: {len(df_with_indicators)}\n")
        
        # Test every 60 minutes (hourly)
        for idx in range(50, len(df_with_indicators), 60):
            if idx >= len(df_with_indicators):
                break
                
            current = df_with_indicators.iloc[idx]
            lookback = df_with_indicators.iloc[:idx + 1]
            
            signals = pattern_detector._detect_bullish_patterns(
                lookback, idx, current, 'ANPA', '2026-01-09'
            )
            
            if signals:
                print(f"\n[{current['timestamp']}] {len(signals)} patterns detected:")
                for sig in signals:
                    print(f"  - {sig.pattern_name}: {sig.confidence:.2%} confidence")
                    if sig.pattern_name in ['Volume_Breakout_Momentum', 'RSI_Accumulation_Entry']:
                        print(f"    *** BEST PATTERN ***")
                        print(f"    Indicators: {sig.indicators}")
