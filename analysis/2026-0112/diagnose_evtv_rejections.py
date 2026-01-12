"""
Diagnostic script to capture minute-by-minute rejection reasons for EVTV
from 2 PM to 2:45 PM
"""
import sys
import os

# Add both root and src to path
root_dir = os.path.join(os.path.dirname(__file__), '..')
src_dir = os.path.join(root_dir, 'src')
sys.path.insert(0, root_dir)
sys.path.insert(0, src_dir)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from src.data.webull_data_api import WebullDataAPI
from src.core.realtime_trader import RealtimeTrader, TradeSignal, ActivePosition
from src.database.trading_database import TradingDatabase
import logging

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Stock to analyze
TICKER = 'EVTV'
START_HOUR = 14  # 2 PM
END_HOUR = 14  # 2 PM
END_MINUTE = 45  # 2:45 PM

def load_data_from_file(ticker, test_data_dir, api):
    """Load data from existing CSV file - need full day data for indicators"""
    
    et = pytz.timezone('US/Eastern')
    today = datetime.now(et).date()
    
    # Try to load from full day file first (need historical data for indicators)
    filename_full = f"{ticker}_1min_{today.strftime('%Y%m%d')}.csv"
    filepath_full = os.path.join(test_data_dir, filename_full)
    
    # Try to load from 2pm file for analysis window
    filename_2pm = f"{ticker}_1min_2pm_{today.strftime('%Y%m%d')}.csv"
    filepath_2pm = os.path.join(test_data_dir, filename_2pm)
    
    df_full = None
    df_2pm = None
    
    # Load full day file if available
    if os.path.exists(filepath_full):
        logger.info(f"Loading full day data from: {filepath_full}")
        df_full = pd.read_csv(filepath_full)
        df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
        if df_full['timestamp'].dt.tz is None:
            df_full['timestamp'] = df_full['timestamp'].dt.tz_localize('US/Eastern')
        df_full = df_full.sort_values('timestamp').reset_index(drop=True)
    
    # Load 2pm file for analysis window
    if os.path.exists(filepath_2pm):
        logger.info(f"Loading 2pm data from: {filepath_2pm}")
        df_2pm = pd.read_csv(filepath_2pm)
        df_2pm['timestamp'] = pd.to_datetime(df_2pm['timestamp'])
        if df_2pm['timestamp'].dt.tz is None:
            df_2pm['timestamp'] = df_2pm['timestamp'].dt.tz_localize('US/Eastern')
        df_2pm = df_2pm.sort_values('timestamp').reset_index(drop=True)
    
    # If we have both, combine them (remove duplicates)
    if df_full is not None and df_2pm is not None:
        # Combine and remove duplicates
        df_combined = pd.concat([df_full, df_2pm]).drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        df = df_combined
    elif df_full is not None:
        df = df_full
    elif df_2pm is not None:
        df = df_2pm
    else:
        logger.error(f"No data files found for {ticker}")
        return None
    
    # Filter to 2 PM to 2:45 PM for analysis window
    hour_mask = df['timestamp'].dt.hour == START_HOUR
    minute_mask = df['timestamp'].dt.minute <= END_MINUTE
    analysis_df = df[hour_mask & minute_mask].copy()
    
    logger.info(f"Loaded {len(df)} total minutes, analyzing {len(analysis_df)} minutes from 2:00 PM to 2:45 PM")
    if len(analysis_df) > 0:
        logger.info(f"Analysis window: {analysis_df['timestamp'].min()} to {analysis_df['timestamp'].max()}")
    else:
        logger.warning(f"No data found in analysis window (2:00 PM to 2:45 PM)")
    
    # Return full data for indicators, but track analysis window
    return df, analysis_df

def diagnose_rejections(ticker, df_full, df_analysis, trader):
    """Diagnose why trades were rejected minute-by-minute"""
    
    et = pytz.timezone('US/Eastern')
    today = datetime.now(et).date()
    
    logger.info(f"Diagnosing rejections for {ticker}...")
    
    # Prepare full DataFrame for indicators
    df = df_full[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Get analysis window indices
    analysis_start_time = df_analysis['timestamp'].min()
    analysis_end_time = df_analysis['timestamp'].max()
    
    # Track diagnostic data
    diagnostics = []
    
    # Process only the analysis window minute-by-minute
    for analysis_idx in range(len(df_analysis)):
        current_time = df_analysis.iloc[analysis_idx]['timestamp']
        current_price = df_analysis.iloc[analysis_idx]['close']
        current_volume = df_analysis.iloc[analysis_idx]['volume']
        
        # Find corresponding index in full dataframe
        full_idx = df[df['timestamp'] == current_time].index
        if len(full_idx) == 0:
            continue
        full_idx = full_idx[0]
        
        # Get all data up to current moment (from full dataframe for indicators)
        df_slice = df.iloc[:full_idx+1].copy()
        
        # Ensure minimum data for indicators
        if len(df_slice) < 50:
            diagnostics.append({
                'timestamp': current_time,
                'time': current_time.strftime('%H:%M:%S'),
                'price': current_price,
                'volume': current_volume,
                'patterns_detected': 'N/A',
                'pattern_confidence': None,
                'entry_signal': False,
                'rejection_reasons': 'Insufficient data (< 50 minutes)',
                'rejection_count': 1,
                'has_position': False
            })
            continue
        
        # Clear previous rejection reasons
        if ticker in trader.last_rejection_reasons:
            trader.last_rejection_reasons[ticker] = []
        
        # Analyze using bot's logic
        entry_signal, exit_signals = trader.analyze_data(df_slice, ticker, current_price)
        
        # Get rejection reasons
        rejection_reasons = trader.last_rejection_reasons.get(ticker, [])
        
        # Get pattern information if available
        patterns_detected = 'None'
        pattern_confidence = None
        
        if entry_signal:
            patterns_detected = entry_signal.pattern_name
            pattern_confidence = entry_signal.confidence * 100
            entry_signal_bool = True
        else:
            entry_signal_bool = False
            # Try to get what patterns were considered
            if rejection_reasons:
                # Check if patterns were detected but rejected
                if any('Confidence' in reason for reason in rejection_reasons):
                    patterns_detected = 'Pattern detected but rejected'
                elif any('Pattern' in reason for reason in rejection_reasons):
                    patterns_detected = 'Pattern issue'
        
        # Format rejection reasons
        if rejection_reasons:
            rejection_str = '; '.join(rejection_reasons)
        elif entry_signal:
            rejection_str = 'None - Entry signal generated'
        else:
            rejection_str = 'No pattern detected or pattern validation failed'
        
        diagnostics.append({
            'timestamp': current_time,
            'time': current_time.strftime('%H:%M:%S'),
            'price': current_price,
            'volume': current_volume,
            'patterns_detected': patterns_detected,
            'pattern_confidence': pattern_confidence,
            'entry_signal': entry_signal_bool,
            'rejection_reasons': rejection_str,
            'rejection_count': len(rejection_reasons),
            'has_position': ticker in trader.active_positions
        })
        
        # Log key moments
        if entry_signal:
            logger.info(f"[{current_time.strftime('%H:%M:%S')}] ENTRY SIGNAL: {patterns_detected} ({pattern_confidence:.1f}%)")
        elif rejection_reasons:
            logger.info(f"[{current_time.strftime('%H:%M:%S')}] REJECTED: {rejection_str}")
    
    return diagnostics

def main():
    """Main diagnostic function"""
    
    et = pytz.timezone('US/Eastern')
    today = datetime.now(et).date()
    
    # Initialize components
    api = WebullDataAPI()
    db = TradingDatabase()
    trader = RealtimeTrader(
        min_confidence=0.72,
        profit_target_pct=20.0,
        trailing_stop_pct=7.0
    )
    
    # Create test_data directory if it doesn't exist
    test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'test_data')
    os.makedirs(test_data_dir, exist_ok=True)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Diagnosing {TICKER} rejections from 2:00 PM to 2:45 PM ({today})")
    logger.info(f"{'='*80}")
    
    # Load data from file
    result = load_data_from_file(TICKER, test_data_dir, api)
    
    if result is None:
        logger.error(f"No data available for {TICKER}")
        return
    
    df_full, df_analysis = result
    
    if df_full is None or df_full.empty:
        logger.error(f"No data available for {TICKER}")
        return
    
    # Diagnose rejections
    diagnostics = diagnose_rejections(TICKER, df_full, df_analysis, trader)
    
    if diagnostics:
        # Create DataFrame
        df_diagnostics = pd.DataFrame(diagnostics)
        
        # Export to CSV
        csv_file = f"analysis/{TICKER}_REJECTION_DIAGNOSIS_2PM_245PM_{today.strftime('%Y%m%d')}.csv"
        df_diagnostics.to_csv(csv_file, index=False)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Diagnostics exported to: {csv_file}")
        logger.info(f"{'='*80}")
        
        # Print summary
        total_minutes = len(df_diagnostics)
        entry_signals = len(df_diagnostics[df_diagnostics['entry_signal'] == True])
        rejected = len(df_diagnostics[(df_diagnostics['rejection_reasons'].str.contains('REJECTED', case=False, na=False)) | 
                                      (df_diagnostics['rejection_reasons'].str.contains('Confidence', case=False, na=False)) |
                                      (df_diagnostics['rejection_reasons'].str.contains('not above', case=False, na=False)) |
                                      (df_diagnostics['rejection_reasons'].str.contains('not in bullish', case=False, na=False)) |
                                      (df_diagnostics['rejection_reasons'].str.contains('Too volatile', case=False, na=False)) |
                                      (df_diagnostics['rejection_reasons'].str.contains('lower lows', case=False, na=False))])
        no_pattern = len(df_diagnostics[df_diagnostics['rejection_reasons'].str.contains('No pattern', case=False, na=False)])
        
        logger.info(f"\nSummary:")
        logger.info(f"  Total Minutes Analyzed: {total_minutes}")
        logger.info(f"  Entry Signals Generated: {entry_signals}")
        logger.info(f"  Rejected (with reasons): {rejected}")
        logger.info(f"  No Pattern Detected: {no_pattern}")
        
        # Show most common rejection reasons
        if rejected > 0:
            logger.info(f"\nMost Common Rejection Reasons:")
            all_reasons = []
            for reasons in df_diagnostics['rejection_reasons']:
                if ';' in str(reasons):
                    all_reasons.extend([r.strip() for r in str(reasons).split(';')])
                else:
                    all_reasons.append(str(reasons))
            
            from collections import Counter
            reason_counts = Counter(all_reasons)
            for reason, count in reason_counts.most_common(10):
                if reason and reason != 'None - Entry signal generated' and 'Insufficient data' not in reason:
                    logger.info(f"  {reason}: {count} times")
    else:
        logger.error("No diagnostics generated")

if __name__ == "__main__":
    main()
