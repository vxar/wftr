"""
Run analysis for ANPA only using the optimized exit logic
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))  # Add analysis directory to path

import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from data.webull_data_api import WebullDataAPI
from analysis.pattern_detector import PatternDetector
from comprehensive_stock_analysis import analyze_stock

def main():
    """Run analysis for ANPA only"""
    
    print("\n" + "="*80)
    print("ANPA ANALYSIS WITH OPTIMIZED EXIT LOGIC")
    print("="*80)
    
    # Analyze ANPA only
    result = analyze_stock('ANPA', start_hour=4, verbose=True)
    
    if result and len(result['trades']) > 0:
        # Export detailed CSV
        print(f"\n{'='*80}")
        print("EXPORTING DETAILED TRADES TO CSV")
        print(f"{'='*80}\n")
        
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
        filename = "ANPA_optimized_trades.csv"  # Save in current directory (analysis/)
        df_trades.to_csv(filename, index=False)
        print(f"Exported {len(trade_records)} trades for ANPA to {filename}")
        
        # Summary
        winning = [t for t in result['trades'] if t['pnl_pct'] > 0]
        losing = [t for t in result['trades'] if t['pnl_pct'] <= 0]
        total_pnl = sum(t['pnl_pct'] for t in result['trades'])
        
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}\n")
        print(f"Trades: {len(result['trades'])}")
        print(f"Winning: {len(winning)} ({result['win_rate']:.1f}%)")
        print(f"Losing: {len(losing)}")
        print(f"Total P&L: {total_pnl:.2f}%")
        print(f"Max Gain Available: {result['max_gain']:.2f}%")
        print(f"Capture Rate: {total_pnl/result['max_gain']*100:.1f}%" if result['max_gain'] > 0 else "N/A")
        print(f"Average P&L: {result['avg_pnl']:.2f}%")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
