"""
FLYX Rejection Analysis (15:42 - 15:49)
Analyze why FLYX trade was rejected between 15:42 to 15:49
Rerun simulation with latest data to identify rejection reasons
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(Path(__file__).parent.parent, 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from data.webull_data_api import WebullDataAPI
from analysis.pattern_detector import PatternDetector
from core.realtime_trader import RealtimeTrader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_flyx_data():
    """Download latest FLYX data"""
    api = WebullDataAPI()
    
    logger.info("Downloading latest FLYX data from Webull API...")
    
    try:
        # Get 1-minute data
        data_1min = api.get_1min_data('FLYX', minutes=1200)  # Max available
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(data_1min['timestamp']):
            data_1min['timestamp'] = pd.to_datetime(data_1min['timestamp'])
        
        # Sort by timestamp
        data_1min = data_1min.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Downloaded {len(data_1min)} 1-minute bars")
        if len(data_1min) > 0:
            logger.info(f"Data range: {data_1min.iloc[0]['timestamp']} to {data_1min.iloc[-1]['timestamp']}")
        
        return data_1min
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def analyze_rejection_window(data_1min, start_time_str="15:42", end_time_str="15:49"):
    """Analyze why trade was rejected in specific time window"""
    logger.info(f"\n{'='*80}")
    logger.info(f"ANALYZING FLYX REJECTION WINDOW: {start_time_str} to {end_time_str}")
    logger.info(f"{'='*80}")
    
    # Initialize trader with same parameters as live bot
    trader = RealtimeTrader(
        min_confidence=0.72,
        min_entry_price_increase=5.5,
        trailing_stop_pct=2.5,
        profit_target_pct=8.0
    )
    
    detector = PatternDetector()
    
    # Calculate indicators
    logger.info("Calculating indicators...")
    data_1min = detector.calculate_indicators(data_1min)
    
    # Parse time window (assuming today's date)
    today = datetime.now().date()
    start_time = pd.to_datetime(f"{today} {start_time_str}")
    end_time = pd.to_datetime(f"{today} {end_time_str}")
    
    # Filter data to time window (with history before)
    window_data = data_1min[
        (data_1min['timestamp'] >= start_time - timedelta(hours=2)) &  # Get some history before
        (data_1min['timestamp'] <= end_time)
    ].copy()
    
    if len(window_data) == 0:
        logger.error("No data found for time window")
        return
    
    logger.info(f"Analyzing {len(window_data)} bars in time window")
    
    # Analyze each minute in the window
    rejection_reasons = []
    entry_attempts = []
    
    for idx in range(50, len(window_data)):  # Start after enough history
        df_window = window_data.iloc[:idx+1]
        current = df_window.iloc[-1]
        current_time = current['timestamp']
        
        # Only analyze the specific time window
        if current_time < start_time or current_time > end_time:
            continue
        
        ticker = 'FLYX'
        current_price = current['close']
        
        # Check for entry signals
        entry_signal, exit_signals = trader.analyze_data(df_window, ticker, current_price=current_price)
        
        if entry_signal:
            entry_attempts.append({
                'timestamp': current_time,
                'price': entry_signal.price,
                'pattern': entry_signal.pattern_name,
                'confidence': entry_signal.confidence,
                'reason': entry_signal.reason if hasattr(entry_signal, 'reason') else None
            })
            logger.info(f"✅ ENTRY SIGNAL @ {current_time}: ${entry_signal.price:.4f} - {entry_signal.pattern_name} ({entry_signal.confidence*100:.1f}%)")
        else:
            # Check rejection reasons
            if ticker in trader.last_rejection_reasons:
                reasons = trader.last_rejection_reasons[ticker]
                if reasons:
                    rejection_reasons.append({
                        'timestamp': current_time,
                        'price': current_price,
                        'reasons': reasons.copy()
                    })
                    logger.info(f"❌ REJECTED @ {current_time}: ${current_price:.4f}")
                    for reason in reasons:
                        logger.info(f"   - {reason}")
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"REJECTION ANALYSIS SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Time Window: {start_time_str} to {end_time_str}")
    logger.info(f"Entry Signals Found: {len(entry_attempts)}")
    logger.info(f"Rejection Events: {len(rejection_reasons)}")
    
    if entry_attempts:
        logger.info(f"\nEntry Signals:")
        for attempt in entry_attempts:
            logger.info(f"  {attempt['timestamp']}: ${attempt['price']:.4f} - {attempt['pattern']} ({attempt['confidence']*100:.1f}%)")
    
    if rejection_reasons:
        logger.info(f"\nRejection Reasons (Most Common):")
        all_reasons = []
        for rejection in rejection_reasons:
            all_reasons.extend(rejection['reasons'])
        
        from collections import Counter
        reason_counts = Counter(all_reasons)
        for reason, count in reason_counts.most_common(10):
            logger.info(f"  {reason}: {count} times")
    
    # Detailed analysis for each rejection
    if rejection_reasons:
        logger.info(f"\n{'='*80}")
        logger.info(f"DETAILED REJECTION ANALYSIS")
        logger.info(f"{'='*80}")
        
        for rejection in rejection_reasons:
            logger.info(f"\nRejection @ {rejection['timestamp']} (Price: ${rejection['price']:.4f}):")
            for reason in rejection['reasons']:
                logger.info(f"  - {reason}")
            
            # Get data at rejection time
            rejection_idx = window_data[window_data['timestamp'] == rejection['timestamp']].index
            if len(rejection_idx) > 0:
                idx = rejection_idx[0]
                df_window = window_data.iloc[:idx+1]
                current = df_window.iloc[-1]
                
                logger.info(f"  Indicators at rejection:")
                logger.info(f"    Price: ${current['close']:.4f}")
                logger.info(f"    High: ${current['high']:.4f}")
                logger.info(f"    Low: ${current['low']:.4f}")
                logger.info(f"    Volume: {current.get('volume', 'N/A'):,.0f}" if 'volume' in current else "    Volume: N/A")
                logger.info(f"    Volume Ratio: {current.get('volume_ratio', 'N/A'):.2f}" if 'volume_ratio' in current else "    Volume Ratio: N/A")
                logger.info(f"    RSI: {current.get('rsi', 'N/A'):.2f}" if 'rsi' in current else "    RSI: N/A")
                logger.info(f"    MACD: {current.get('macd', 'N/A'):.4f}" if 'macd' in current else "    MACD: N/A")
                logger.info(f"    MACD Signal: {current.get('macd_signal', 'N/A'):.4f}" if 'macd_signal' in current else "    MACD Signal: N/A")
                logger.info(f"    SMA 5: ${current.get('sma_5', 'N/A'):.4f}" if 'sma_5' in current else "    SMA 5: N/A")
                logger.info(f"    SMA 10: ${current.get('sma_10', 'N/A'):.4f}" if 'sma_10' in current else "    SMA 10: N/A")
                logger.info(f"    SMA 20: ${current.get('sma_20', 'N/A'):.4f}" if 'sma_20' in current else "    SMA 20: N/A")
    
    return {
        'entry_attempts': entry_attempts,
        'rejection_reasons': rejection_reasons,
        'window_data': window_data
    }


def main():
    """Main analysis function"""
    logger.info("="*80)
    logger.info("FLYX REJECTION ANALYSIS (15:42 - 15:49)")
    logger.info("="*80)
    
    try:
        # Download latest data
        data_1min = download_flyx_data()
        
        # Analyze rejection window
        results = analyze_rejection_window(data_1min, start_time_str="15:42", end_time_str="15:49")
        
        # Save report
        output_file = Path(__file__).parent / 'flyx_rejection_analysis.txt'
        with open(output_file, 'w') as f:
            f.write("FLYX Rejection Analysis (15:42 - 15:49)\n")
            f.write("="*80 + "\n\n")
            
            if results:
                f.write(f"Entry Signals Found: {len(results['entry_attempts'])}\n")
                if results['entry_attempts']:
                    for attempt in results['entry_attempts']:
                        f.write(f"  {attempt['timestamp']}: ${attempt['price']:.4f} - {attempt['pattern']}\n")
                
                f.write(f"\nRejection Events: {len(results['rejection_reasons'])}\n")
                if results['rejection_reasons']:
                    all_reasons = []
                    for rejection in results['rejection_reasons']:
                        all_reasons.extend(rejection['reasons'])
                    
                    from collections import Counter
                    reason_counts = Counter(all_reasons)
                    f.write("\nMost Common Rejection Reasons:\n")
                    for reason, count in reason_counts.most_common(10):
                        f.write(f"  {reason}: {count} times\n")
        
        logger.info(f"\nAnalysis complete! Report saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
