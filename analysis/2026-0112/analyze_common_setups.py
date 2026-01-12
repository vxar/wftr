"""
Deep analysis of common setups across all stocks
Identifies what characteristics are present at successful entry times
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from collections import Counter, defaultdict
import glob

# Expected entry/exit windows
STOCK_WINDOWS = {
    'OM': {'entry_after': '08:30', 'exit_after': '09:10'},
    'EVTV': {'entry_after': '04:30', 'exit_after': '10:40'},
    'SOGP': {'entry_after': '04:40', 'exit_after': '09:15'},
    'INBS': {'entry_after': '04:30', 'exit_after': '04:50'},
    'UP': {'entry_after': '07:30', 'exit_after': '11:00'},
    'BDSX': {'entry_after': '09:20', 'exit_after': '09:52'},
}

def parse_time(time_str):
    """Parse time string like '08:30' into hour and minute"""
    parts = time_str.split(':')
    return int(parts[0]), int(parts[1])

def analyze_common_setups():
    """Analyze common setups across all stocks"""
    
    et = pytz.timezone('US/Eastern')
    today = datetime.now(et).date()
    
    # Load all minute-by-minute data
    all_stock_data = {}
    
    for ticker in STOCK_WINDOWS.keys():
        csv_file = f"analysis/{ticker}_minute_by_minute_{today.strftime('%Y%m%d')}.csv"
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df['time'] = pd.to_datetime(df['time'])
            all_stock_data[ticker] = df
            print(f"Loaded {len(df)} minutes for {ticker}")
    
    # Analyze entry windows
    entry_window_analysis = []
    rejection_analysis = []
    
    for ticker, df in all_stock_data.items():
        entry_hour, entry_minute = parse_time(STOCK_WINDOWS[ticker]['entry_after'])
        
        # Get data in entry window (10 min before to 30 min after expected entry)
        for _, row in df.iterrows():
            current_time = row['time']
            hour = current_time.hour
            minute = current_time.minute
            
            # Check if in entry window range
            in_range = False
            if hour == entry_hour:
                if entry_minute - 10 <= minute <= entry_minute + 30:
                    in_range = True
            elif hour == entry_hour - 1 and minute >= 50:
                in_range = True
            elif hour == entry_hour + 1 and minute <= 30:
                in_range = True
            
            if in_range:
                is_entry_window = (hour > entry_hour or (hour == entry_hour and minute >= entry_minute))
                
                data_point = {
                    'ticker': ticker,
                    'time': current_time,
                    'price': row['price'],
                    'volume': row['volume'],
                    'volume_ratio': row['volume_ratio'],
                    'momentum_5': row['momentum_5'],
                    'momentum_10': row['momentum_10'],
                    'entry_signal': row['entry_signal'],
                    'pattern': row['pattern'],
                    'confidence': row['confidence'],
                    'in_entry_window': is_entry_window,
                    'rejection_reasons': row.get('rejection_reasons_str', '')
                }
                
                if is_entry_window:
                    if row['entry_signal']:
                        entry_window_analysis.append(data_point)
                    else:
                        rejection_analysis.append(data_point)
    
    # Analyze successful vs failed
    print(f"\n{'='*80}")
    print("ENTRY WINDOW ANALYSIS")
    print(f"{'='*80}")
    
    print(f"\nSuccessful Entries in Window: {len(entry_window_analysis)}")
    if entry_window_analysis:
        df_success = pd.DataFrame(entry_window_analysis)
        print(f"\nVolume Characteristics:")
        print(f"  Average Volume: {df_success['volume'].mean():,.0f}")
        print(f"  Average Volume Ratio: {df_success['volume_ratio'].mean():.2f}x")
        print(f"  Min Volume Ratio: {df_success['volume_ratio'].min():.2f}x")
        print(f"  Max Volume Ratio: {df_success['volume_ratio'].max():.2f}x")
        
        print(f"\nMomentum Characteristics:")
        print(f"  Average 5-min Momentum: {df_success['momentum_5'].mean():.2f}%")
        print(f"  Average 10-min Momentum: {df_success['momentum_10'].mean():.2f}%")
        print(f"  Min 5-min Momentum: {df_success['momentum_5'].min():.2f}%")
        print(f"  Max 5-min Momentum: {df_success['momentum_5'].max():.2f}%")
        
        print(f"\nPattern Distribution:")
        pattern_counts = df_success['pattern'].value_counts()
        for pattern, count in pattern_counts.items():
            print(f"  {pattern}: {count} times")
        
        print(f"\nConfidence Levels:")
        print(f"  Average: {df_success['confidence'].mean():.1f}%")
        print(f"  Min: {df_success['confidence'].min():.1f}%")
        print(f"  Max: {df_success['confidence'].max():.1f}%")
    
    print(f"\nRejections in Entry Window: {len(rejection_analysis)}")
    if rejection_analysis:
        df_reject = pd.DataFrame(rejection_analysis)
        print(f"\nVolume Characteristics:")
        print(f"  Average Volume: {df_reject['volume'].mean():,.0f}")
        print(f"  Average Volume Ratio: {df_reject['volume_ratio'].mean():.2f}x")
        
        print(f"\nMomentum Characteristics:")
        print(f"  Average 5-min Momentum: {df_reject['momentum_5'].mean():.2f}%")
        print(f"  Average 10-min Momentum: {df_reject['momentum_10'].mean():.2f}%")
        
        # Analyze rejection reasons
        all_reasons = []
        for reasons_str in df_reject['rejection_reasons']:
            if pd.notna(reasons_str) and reasons_str != 'None' and reasons_str:
                reasons = str(reasons_str).split('; ')
                all_reasons.extend(reasons)
        
        reason_counts = Counter(all_reasons)
        print(f"\nTop Rejection Reasons:")
        for reason, count in reason_counts.most_common(10):
            print(f"  {reason}: {count} times")
    
    # Compare successful vs failed
    if entry_window_analysis and rejection_analysis:
        df_success = pd.DataFrame(entry_window_analysis)
        df_reject = pd.DataFrame(rejection_analysis)
        
        print(f"\n{'='*80}")
        print("SUCCESSFUL vs FAILED COMPARISON")
        print(f"{'='*80}")
        
        print(f"\nVolume Ratio:")
        print(f"  Successful: {df_success['volume_ratio'].mean():.2f}x (min: {df_success['volume_ratio'].min():.2f}x, max: {df_success['volume_ratio'].max():.2f}x)")
        print(f"  Failed: {df_reject['volume_ratio'].mean():.2f}x (min: {df_reject['volume_ratio'].min():.2f}x, max: {df_reject['volume_ratio'].max():.2f}x)")
        
        print(f"\n5-min Momentum:")
        print(f"  Successful: {df_success['momentum_5'].mean():.2f}% (min: {df_success['momentum_5'].min():.2f}%, max: {df_success['momentum_5'].max():.2f}%)")
        print(f"  Failed: {df_reject['momentum_5'].mean():.2f}% (min: {df_reject['momentum_5'].min():.2f}%, max: {df_reject['momentum_5'].max():.2f}%)")
        
        print(f"\n10-min Momentum:")
        print(f"  Successful: {df_success['momentum_10'].mean():.2f}% (min: {df_success['momentum_10'].min():.2f}%, max: {df_success['momentum_10'].max():.2f}%)")
        print(f"  Failed: {df_reject['momentum_10'].mean():.2f}% (min: {df_reject['momentum_10'].min():.2f}%, max: {df_reject['momentum_10'].max():.2f}%)")
    
    # Generate detailed report
    generate_setup_report(entry_window_analysis, rejection_analysis, today)
    
    return entry_window_analysis, rejection_analysis

def generate_setup_report(entry_analysis, rejection_analysis, today):
    """Generate detailed setup analysis report"""
    
    report = []
    report.append("# Common Setup Analysis - Multi-Stock Bull Runs")
    report.append(f"Analysis Date: {today}")
    report.append("")
    report.append("## Overview")
    report.append("")
    report.append("This analysis examines 6 stocks that had significant bull runs:")
    report.append("- OM: Entry after 08:30, Exit after 09:10")
    report.append("- EVTV: Entry after 04:30, Exit after 10:40")
    report.append("- SOGP: Entry after 04:40, Exit after 09:15")
    report.append("- INBS: Entry after 04:30, Exit after 04:50")
    report.append("- UP: Entry after 07:30, Exit after 11:00")
    report.append("- BDSX: Entry after 09:20, Exit after 09:52")
    report.append("")
    
    if entry_analysis:
        df_success = pd.DataFrame(entry_analysis)
        report.append("## Successful Entry Characteristics")
        report.append("")
        report.append("### Volume Metrics")
        report.append(f"- Average Volume: {df_success['volume'].mean():,.0f}")
        report.append(f"- Average Volume Ratio: {df_success['volume_ratio'].mean():.2f}x")
        report.append(f"- Volume Ratio Range: {df_success['volume_ratio'].min():.2f}x - {df_success['volume_ratio'].max():.2f}x")
        report.append("")
        
        report.append("### Momentum Metrics")
        report.append(f"- Average 5-min Momentum: {df_success['momentum_5'].mean():.2f}%")
        report.append(f"- Average 10-min Momentum: {df_success['momentum_10'].mean():.2f}%")
        report.append(f"- 5-min Momentum Range: {df_success['momentum_5'].min():.2f}% - {df_success['momentum_5'].max():.2f}%")
        report.append("")
        
        report.append("### Pattern Distribution")
        pattern_counts = df_success['pattern'].value_counts()
        for pattern, count in pattern_counts.items():
            report.append(f"- {pattern}: {count} times ({count/len(df_success)*100:.1f}%)")
        report.append("")
        
        report.append("### Confidence Levels")
        report.append(f"- Average: {df_success['confidence'].mean():.1f}%")
        report.append(f"- Range: {df_success['confidence'].min():.1f}% - {df_success['confidence'].max():.1f}%")
        report.append("")
        
        report.append("### Successful Entries by Stock")
        for ticker in df_success['ticker'].unique():
            ticker_data = df_success[df_success['ticker'] == ticker]
            report.append(f"**{ticker}**: {len(ticker_data)} entries")
            for _, row in ticker_data.iterrows():
                report.append(f"- {row['time'].strftime('%H:%M:%S')} @ ${row['price']:.4f} - {row['pattern']} ({row['confidence']:.1f}%)")
            report.append("")
    
    if rejection_analysis:
        df_reject = pd.DataFrame(rejection_analysis)
        report.append("## Failed Entry Characteristics (Rejections)")
        report.append("")
        report.append(f"Total Rejections in Entry Windows: {len(df_reject)}")
        report.append("")
        
        report.append("### Volume Metrics")
        report.append(f"- Average Volume: {df_reject['volume'].mean():,.0f}")
        report.append(f"- Average Volume Ratio: {df_reject['volume_ratio'].mean():.2f}x")
        report.append("")
        
        report.append("### Momentum Metrics")
        report.append(f"- Average 5-min Momentum: {df_reject['momentum_5'].mean():.2f}%")
        report.append(f"- Average 10-min Momentum: {df_reject['momentum_10'].mean():.2f}%")
        report.append("")
        
        # Rejection reasons
        all_reasons = []
        for reasons_str in df_reject['rejection_reasons']:
            if pd.notna(reasons_str) and reasons_str != 'None' and reasons_str:
                reasons = str(reasons_str).split('; ')
                all_reasons.extend(reasons)
        
        reason_counts = Counter(all_reasons)
        report.append("### Top Rejection Reasons")
        for reason, count in reason_counts.most_common(15):
            pct = (count / len(df_reject)) * 100
            report.append(f"- {reason}: {count} times ({pct:.1f}%)")
        report.append("")
    
    # Comparison
    if entry_analysis and rejection_analysis:
        df_success = pd.DataFrame(entry_analysis)
        df_reject = pd.DataFrame(rejection_analysis)
        
        report.append("## Successful vs Failed Comparison")
        report.append("")
        report.append("### Key Differences")
        report.append("")
        
        vol_diff = df_success['volume_ratio'].mean() - df_reject['volume_ratio'].mean()
        mom5_diff = df_success['momentum_5'].mean() - df_reject['momentum_5'].mean()
        mom10_diff = df_success['momentum_10'].mean() - df_reject['momentum_10'].mean()
        
        report.append(f"**Volume Ratio Difference**: {vol_diff:+.2f}x (Successful: {df_success['volume_ratio'].mean():.2f}x vs Failed: {df_reject['volume_ratio'].mean():.2f}x)")
        report.append(f"**5-min Momentum Difference**: {mom5_diff:+.2f}% (Successful: {df_success['momentum_5'].mean():.2f}% vs Failed: {df_reject['momentum_5'].mean():.2f}%)")
        report.append(f"**10-min Momentum Difference**: {mom10_diff:+.2f}% (Successful: {df_success['momentum_10'].mean():.2f}% vs Failed: {df_reject['momentum_10'].mean():.2f}%)")
        report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    
    if entry_analysis and rejection_analysis:
        df_success = pd.DataFrame(entry_analysis)
        df_reject = pd.DataFrame(rejection_analysis)
        
        # Calculate thresholds
        min_vol_ratio_success = df_success['volume_ratio'].min()
        min_momentum5_success = df_success['momentum_5'].min()
        min_momentum10_success = df_success['momentum_10'].min()
        
        report.append("### Suggested Thresholds Based on Successful Entries")
        report.append("")
        report.append(f"- **Minimum Volume Ratio**: {min_vol_ratio_success:.2f}x (lowest successful entry)")
        report.append(f"- **Minimum 5-min Momentum**: {min_momentum5_success:.2f}% (lowest successful entry)")
        report.append(f"- **Minimum 10-min Momentum**: {min_momentum10_success:.2f}% (lowest successful entry)")
        report.append("")
        
        # Analyze rejection reasons to identify what to relax
        all_reasons = []
        for reasons_str in df_reject['rejection_reasons']:
            if pd.notna(reasons_str) and reasons_str != 'None' and reasons_str:
                reasons = str(reasons_str).split('; ')
                all_reasons.extend(reasons)
        
        reason_counts = Counter(all_reasons)
        top_reasons = reason_counts.most_common(5)
        
        report.append("### Top Blocking Issues to Address")
        report.append("")
        for reason, count in top_reasons:
            pct = (count / len(df_reject)) * 100
            report.append(f"1. **{reason}**: {count} times ({pct:.1f}% of rejections)")
            report.append("")
    
    # Write report
    report_text = '\n'.join(report)
    report_file = f"analysis/COMMON_SETUP_ANALYSIS_{today.strftime('%Y%m%d')}.md"
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(f"\nDetailed setup report saved to: {report_file}")

if __name__ == "__main__":
    entry_analysis, rejection_analysis = analyze_common_setups()
