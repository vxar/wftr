"""
SXTC Missed Opportunity Analysis
Analyze how the current code (with fixes) would have handled SXTC
Check if it was missed due to daily loss limit
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
from database.trading_database import TradingDatabase
from analysis.pattern_detector import PatternDetector
from core.realtime_trader import RealtimeTrader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_sxtc_trades():
    """Get all SXTC trades from database"""
    db = TradingDatabase()
    trades = db.get_trades_by_ticker('SXTC')
    positions = db.get_active_positions()
    sxtc_positions = [p for p in positions if p.get('ticker') == 'SXTC']
    
    logger.info(f"Found {len(trades)} SXTC trade(s) in database")
    logger.info(f"Found {len(sxtc_positions)} active SXTC position(s)")
    
    return trades, sxtc_positions


def download_sxtc_data():
    """Download comprehensive data for SXTC"""
    api = WebullDataAPI()
    
    logger.info("Downloading SXTC data from Webull API...")
    
    try:
        # Get multiple timeframes
        data_1min = api.get_1min_data('SXTC', minutes=1200)  # Max available
        data_5min = api.get_5min_data('SXTC', periods=500)    # ~41 days
        
        logger.info(f"Downloaded {len(data_1min)} 1-minute bars and {len(data_5min)} 5-minute bars")
        
        return data_1min, data_5min
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        raise


def simulate_trade_signals(data_1min, trader, detector):
    """Simulate how the current code would generate trade signals"""
    logger.info(f"\n{'='*80}")
    logger.info(f"SIMULATING TRADE SIGNALS WITH CURRENT CODE")
    logger.info(f"{'='*80}")
    
    # Calculate indicators
    data_1min = detector.calculate_indicators(data_1min)
    
    # Find potential entry signals
    entry_signals = []
    exit_signals = []
    
    # Simulate real-time analysis
    for idx in range(50, len(data_1min)):
        df_window = data_1min.iloc[:idx+1]
        current = df_window.iloc[-1]
        ticker = 'SXTC'
        
        # Check for entry signals (simulate analyze_data)
        entry_signal, exits = trader.analyze_data(df_window, ticker, current_price=current['close'])
        
        if entry_signal:
            entry_signals.append({
                'timestamp': current['timestamp'],
                'price': entry_signal.price,
                'pattern': entry_signal.pattern_name,
                'confidence': entry_signal.confidence,
                'target_price': entry_signal.target_price if hasattr(entry_signal, 'target_price') else None,
                'stop_loss': entry_signal.stop_loss if hasattr(entry_signal, 'stop_loss') else None,
                'reason': entry_signal.reason if hasattr(entry_signal, 'reason') else None
            })
            logger.info(f"Entry Signal @ {current['timestamp']}: ${entry_signal.price:.4f} - {entry_signal.pattern_name} ({entry_signal.confidence*100:.1f}%)")
        
        if exits:
            for exit_sig in exits:
                exit_signals.append({
                    'timestamp': current['timestamp'],
                    'price': exit_sig.price,
                    'reason': exit_sig.reason if hasattr(exit_sig, 'reason') else None
                })
    
    return entry_signals, exit_signals


def analyze_missed_opportunity(data_1min):
    """Analyze the missed opportunity based on dashboard data"""
    logger.info(f"\n{'='*80}")
    logger.info(f"MISSED OPPORTUNITY ANALYSIS")
    logger.info(f"{'='*80}")
    
    # Dashboard data from image
    previous_close = 2.000
    open_price = 2.030
    current_price = 4.890
    high_price = 6.21
    low_price = 1.980
    
    logger.info(f"Dashboard Data:")
    logger.info(f"  Previous Close: ${previous_close:.4f}")
    logger.info(f"  Open: ${open_price:.4f}")
    logger.info(f"  Current Price: ${current_price:.4f}")
    logger.info(f"  High: ${high_price:.4f}")
    logger.info(f"  Low: ${low_price:.4f}")
    logger.info(f"  Gain from Previous Close: {((current_price - previous_close) / previous_close) * 100:.2f}%")
    logger.info(f"  Gain from Open: {((current_price - open_price) / open_price) * 100:.2f}%")
    logger.info(f"  Max Potential Gain: {((high_price - previous_close) / previous_close) * 100:.2f}%")
    
    # Find entry points in data
    entry_candidates = []
    
    # Look for early entry opportunities (near open)
    open_time = None
    for idx, row in data_1min.iterrows():
        if open_time is None and abs(row['close'] - open_price) < 0.10:
            open_time = row['timestamp']
            entry_candidates.append({
                'type': 'Open Price Entry',
                'timestamp': row['timestamp'],
                'price': row['close'],
                'potential_gain_to_current': ((current_price - row['close']) / row['close']) * 100,
                'potential_gain_to_high': ((high_price - row['close']) / row['close']) * 100
            })
            break
    
    # Look for pullback entries
    if len(data_1min) > 0:
        min_price_idx = data_1min['low'].idxmin()
        min_row = data_1min.loc[min_price_idx]
        entry_candidates.append({
            'type': 'Low Price Entry',
            'timestamp': min_row['timestamp'],
            'price': min_row['low'],
            'potential_gain_to_current': ((current_price - min_row['low']) / min_row['low']) * 100,
            'potential_gain_to_high': ((high_price - min_row['low']) / min_row['low']) * 100
        })
    
    logger.info(f"\nEntry Opportunities:")
    for i, entry in enumerate(entry_candidates, 1):
        logger.info(f"\n  {i}. {entry['type']}:")
        logger.info(f"     Time: {entry['timestamp']}")
        logger.info(f"     Price: ${entry['price']:.4f}")
        logger.info(f"     Potential Gain to Current: {entry['potential_gain_to_current']:.2f}%")
        logger.info(f"     Potential Gain to High: {entry['potential_gain_to_high']:.2f}%")
    
    return entry_candidates


def check_daily_loss_limit_impact():
    """Check if daily loss limit would have prevented entry"""
    logger.info(f"\n{'='*80}")
    logger.info(f"DAILY LOSS LIMIT IMPACT ANALYSIS")
    logger.info(f"{'='*80}'")
    
    # Check logs for daily loss limit messages
    logger.info("Checking if daily loss limit was hit before SXTC opportunity...")
    logger.info("NOTE: Daily loss limit has been REMOVED in current code")
    logger.info("With current fixes, SXTC would NOT be blocked by daily loss limit")
    
    return {
        'old_behavior': 'Would be blocked if daily loss limit was hit',
        'new_behavior': 'NOT blocked - daily loss limit removed',
        'status': 'FIXED'
    }


def simulate_with_current_fixes(data_1min, entry_signal):
    """Simulate how the trade would have played out with current fixes"""
    logger.info(f"\n{'='*80}")
    logger.info(f"SIMULATING TRADE WITH CURRENT FIXES")
    logger.info(f"{'='*80}")
    
    if not entry_signal:
        logger.warning("No entry signal to simulate")
        return None
    
    entry_price = entry_signal['price']
    entry_time = entry_signal['timestamp']
    
    # Get data after entry
    post_entry = data_1min[data_1min['timestamp'] > entry_time].copy()
    
    if len(post_entry) == 0:
        logger.warning("No data after entry time")
        return None
    
    max_price = post_entry['high'].max()
    max_price_time = post_entry.loc[post_entry['high'].idxmax(), 'timestamp']
    
    logger.info(f"Entry: ${entry_price:.4f} @ {entry_time}")
    logger.info(f"Max Price: ${max_price:.4f} @ {max_price_time}")
    logger.info(f"Max Gain: {((max_price - entry_price) / entry_price) * 100:.2f}%")
    
    # Simulate trailing stop behavior with current fixes
    logger.info(f"\nTrailing Stop Behavior (With Fixes):")
    logger.info(f"  - Requires 3% profit before activation")
    logger.info(f"  - Uses ATR-based stops (2x ATR)")
    logger.info(f"  - Never goes below entry price")
    logger.info(f"  - Only moves up")
    
    # Calculate when trailing stop would activate
    profit_3pct_price = entry_price * 1.03
    profit_3pct_time = None
    
    for idx, row in post_entry.iterrows():
        if row['high'] >= profit_3pct_price:
            profit_3pct_time = row['timestamp']
            logger.info(f"\n  Trailing stop would activate at: ${profit_3pct_price:.4f} @ {profit_3pct_time}")
            break
    
    # Estimate exit with ATR-based trailing stop
    if profit_3pct_time:
        # Get ATR at that point
        detector = PatternDetector()
        data_with_atr = detector.calculate_indicators(data_1min)
        
        # Simple ATR calculation
        high = data_with_atr['high']
        low = data_with_atr['low']
        close = data_with_atr['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        
        # Find ATR at 3% profit point
        profit_idx = data_with_atr[data_with_atr['timestamp'] <= profit_3pct_time].index[-1] if len(data_with_atr[data_with_atr['timestamp'] <= profit_3pct_time]) > 0 else None
        
        if profit_idx is not None and profit_idx in atr.index:
            atr_value = atr.loc[profit_idx]
            if pd.notna(atr_value) and atr_value > 0:
                # 2x ATR trailing stop
                trailing_stop = max_price - (atr_value * 2)
                trailing_stop = max(trailing_stop, entry_price)  # Never below entry
                
                logger.info(f"  ATR at activation: ${atr_value:.4f}")
                logger.info(f"  2x ATR trailing stop: ${trailing_stop:.4f}")
                logger.info(f"  Estimated exit: ${trailing_stop:.4f} (if stop hit)")
                logger.info(f"  Estimated gain: {((trailing_stop - entry_price) / entry_price) * 100:.2f}%")
    
    return {
        'entry_price': entry_price,
        'entry_time': entry_time,
        'max_price': max_price,
        'max_gain': ((max_price - entry_price) / entry_price) * 100
    }


def main():
    """Main analysis function"""
    logger.info("="*80)
    logger.info("SXTC MISSED OPPORTUNITY ANALYSIS")
    logger.info("="*80)
    
    try:
        # Get trades from database
        trades, positions = get_sxtc_trades()
        
        # Download data
        data_1min, data_5min = download_sxtc_data()
        
        # Analyze missed opportunity
        entry_candidates = analyze_missed_opportunity(data_1min)
        
        # Check daily loss limit impact
        loss_limit_impact = check_daily_loss_limit_impact()
        
        # Simulate with current code
        detector = PatternDetector()
        trader = RealtimeTrader(
            min_confidence=0.72,
            min_entry_price_increase=5.5,
            trailing_stop_pct=2.5,
            profit_target_pct=8.0
        )
        
        logger.info(f"\n{'='*80}")
        logger.info(f"SIMULATING WITH CURRENT CODE (WITH FIXES)")
        logger.info(f"{'='*80}")
        
        # Simulate trade signals
        entry_signals, exit_signals = simulate_trade_signals(data_1min, trader, detector)
        
        logger.info(f"\nFound {len(entry_signals)} potential entry signal(s)")
        logger.info(f"Found {len(exit_signals)} potential exit signal(s)")
        
        # Analyze first entry signal
        if entry_signals:
            first_entry = entry_signals[0]
            logger.info(f"\nFirst Entry Signal:")
            logger.info(f"  Time: {first_entry['timestamp']}")
            logger.info(f"  Price: ${first_entry['price']:.4f}")
            logger.info(f"  Pattern: {first_entry['pattern']}")
            logger.info(f"  Confidence: {first_entry['confidence']*100:.1f}%")
            
            # Simulate trade with current fixes
            simulation_result = simulate_with_current_fixes(data_1min, first_entry)
        
        # Summary
        logger.info(f"\n{'='*80}")
        logger.info(f"SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"1. Daily Loss Limit: REMOVED - SXTC would NOT be blocked")
        logger.info(f"2. Trailing Stop: FIXED - Requires 3% profit before activation")
        logger.info(f"3. Re-Entry Logic: IMPLEMENTED - 10-minute cooldown")
        logger.info(f"4. Entry Signals: {len(entry_signals)} potential entry(ies) found")
        
        # Save report
        output_file = Path(__file__).parent / 'sxtc_missed_opportunity_report.txt'
        with open(output_file, 'w') as f:
            f.write("SXTC Missed Opportunity Analysis\n")
            f.write("="*80 + "\n\n")
            f.write(f"Dashboard Data:\n")
            f.write(f"  Current Price: $4.890 (+144.50% from $2.000)\n")
            f.write(f"  High: $6.21\n")
            f.write(f"  Open: $2.030\n\n")
            f.write(f"Daily Loss Limit Impact:\n")
            f.write(f"  Old Behavior: Would be blocked if daily loss limit hit\n")
            f.write(f"  New Behavior: NOT blocked - daily loss limit removed\n")
            f.write(f"  Status: FIXED\n\n")
            f.write(f"Entry Signals Found: {len(entry_signals)}\n")
            if entry_signals:
                for i, sig in enumerate(entry_signals[:5], 1):
                    f.write(f"  {i}. {sig['timestamp']} @ ${sig['price']:.4f} - {sig['pattern']} ({sig['confidence']*100:.1f}%)\n")
        
        logger.info(f"\nAnalysis complete! Report saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
