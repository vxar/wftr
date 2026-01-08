"""
FLYX Full Day Simulation
Simulate trading bot from 4:00 AM with current code (all fixes applied)
Identify all trade opportunities and how they would have played out
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


def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def download_flyx_data():
    """Download comprehensive data for FLYX from 4:00 AM"""
    api = WebullDataAPI()
    
    logger.info("Downloading FLYX data from Webull API...")
    
    try:
        # Get 1-minute data (should cover from 4:00 AM)
        data_1min = api.get_1min_data('FLYX', minutes=1200)  # Max available
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(data_1min['timestamp']):
            data_1min['timestamp'] = pd.to_datetime(data_1min['timestamp'])
        
        # Filter to trading hours (4:00 AM - 8:00 PM ET)
        data_1min['hour'] = pd.to_datetime(data_1min['timestamp']).dt.hour
        data_1min = data_1min[
            (data_1min['hour'] >= 4) & (data_1min['hour'] < 20)
        ].copy()
        
        # Sort by timestamp
        data_1min = data_1min.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Downloaded {len(data_1min)} 1-minute bars (4:00 AM - 8:00 PM)")
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
    data_1min['atr'] = calculate_atr(data_1min, period=14)
    data_1min['atr_pct'] = (data_1min['atr'] / data_1min['close']) * 100
    
    # Parse time window (assuming today's date)
    # Check if timestamps are timezone-aware
    sample_timestamp = data_1min['timestamp'].iloc[0] if len(data_1min) > 0 else None
    is_tz_aware = sample_timestamp is not None and pd.notna(sample_timestamp) and hasattr(sample_timestamp, 'tz') and sample_timestamp.tz is not None
    
    if is_tz_aware:
        # Get timezone from data
        tz = sample_timestamp.tz
        # Create timezone-aware start/end times
        today = datetime.now(tz).date()
        start_time = pd.Timestamp(f"{today} {start_time_str}", tz=tz)
        end_time = pd.Timestamp(f"{today} {end_time_str}", tz=tz)
    else:
        # Timezone-naive
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
        return None
    
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


def simulate_full_day_trading(data_1min):
    """Simulate full day trading from 4:00 AM"""
    logger.info(f"\n{'='*80}")
    logger.info(f"FULL DAY SIMULATION - FLYX")
    logger.info(f"{'='*80}")
    
    # Initialize trader with current settings
    trader = RealtimeTrader(
        min_confidence=0.72,
        min_entry_price_increase=5.5,
        trailing_stop_pct=2.5,
        profit_target_pct=8.0
    )
    
    detector = PatternDetector()
    
    # Calculate indicators for entire dataset
    logger.info("Calculating indicators...")
    data_1min = detector.calculate_indicators(data_1min)
    data_1min['atr'] = calculate_atr(data_1min, period=14)
    data_1min['atr_pct'] = (data_1min['atr'] / data_1min['close']) * 100
    
    # Track all opportunities
    entry_signals = []
    simulated_trades = []
    active_positions = {}  # ticker -> entry info
    
    # Simulate minute-by-minute from 4:00 AM
    logger.info("Simulating minute-by-minute trading...")
    logger.info(f"Total bars to process: {len(data_1min)}")
    
    # Start after enough history for indicators (50 bars = ~50 minutes)
    start_idx = max(50, len(data_1min) // 20)  # Start after 5% of data or 50 bars, whichever is larger
    logger.info(f"Starting simulation from index {start_idx}")
    
    for idx in range(start_idx, len(data_1min)):  # Start after enough history
        df_window = data_1min.iloc[:idx+1]
        current = df_window.iloc[-1]
        ticker = 'FLYX'
        current_time = current['timestamp']
        current_price = current['close']
        
        # Check for entry signals (only if no active position)
        if ticker not in active_positions:
            entry_signal, exit_signals = trader.analyze_data(df_window, ticker, current_price=current_price)
            
            if entry_signal:
                entry_info = {
                    'timestamp': current_time,
                    'price': entry_signal.price,
                    'pattern': entry_signal.pattern_name,
                    'confidence': entry_signal.confidence,
                    'target_price': entry_signal.target_price if hasattr(entry_signal, 'target_price') else None,
                    'stop_loss': entry_signal.stop_loss if hasattr(entry_signal, 'stop_loss') else None,
                    'reason': entry_signal.reason if hasattr(entry_signal, 'reason') else None,
                    'entry_idx': idx
                }
                entry_signals.append(entry_info)
                
                # Simulate entering position
                active_positions[ticker] = {
                    'entry_time': current_time,
                    'entry_price': entry_signal.price,
                    'entry_idx': idx,
                    'max_price_reached': entry_signal.price,
                    'trailing_stop_price': None,
                    'trailing_stop_active': False,
                    'stop_loss': entry_signal.stop_loss if hasattr(entry_signal, 'stop_loss') else None,
                    'target_price': entry_signal.target_price if hasattr(entry_signal, 'target_price') else None,
                    'atr': current.get('atr', 0),
                    'pattern': entry_signal.pattern_name
                }
                
                logger.info(f"ENTRY SIGNAL @ {current_time}: ${entry_signal.price:.4f} - {entry_signal.pattern_name} ({entry_signal.confidence*100:.1f}%)")
        
        # Check for exit signals on active positions
        if ticker in active_positions:
            position = active_positions[ticker]
            
            # Update max price reached
            if current_price > position['max_price_reached']:
                position['max_price_reached'] = current_price
            
            # Calculate unrealized P&L
            unrealized_pnl_pct = ((current_price - position['entry_price']) / position['entry_price']) * 100
            
            # Check exit conditions (simulate current fixed logic)
            exit_reason = None
            exit_price = current_price
            
            # 1. Stop loss hit
            if position.get('stop_loss') and current_price <= position['stop_loss']:
                exit_reason = f"Stop loss hit at ${position['stop_loss']:.4f}"
                exit_price = position['stop_loss']
            
            # 2. Profit target reached
            elif position.get('target_price') and current_price >= position['target_price']:
                exit_reason = f"Profit target reached at ${position['target_price']:.4f}"
                exit_price = position['target_price']
            
            # 3. Trailing stop (WITH FIXES: requires 3% profit, ATR-based)
            elif unrealized_pnl_pct >= 3.0:  # NEW: Requires 3% profit
                if not position['trailing_stop_active']:
                    position['trailing_stop_active'] = True
                    # Use ATR-based stop if available
                    atr = current.get('atr', 0)
                    if pd.notna(atr) and atr > 0:
                        trailing_stop = position['max_price_reached'] - (atr * 2)  # 2x ATR
                    else:
                        # Fallback to percentage
                        trailing_stop_pct = 2.5
                        trailing_stop = position['max_price_reached'] * (1 - trailing_stop_pct / 100)
                    
                    # Never below entry price
                    trailing_stop = max(trailing_stop, position['entry_price'])
                    position['trailing_stop_price'] = trailing_stop
                    logger.info(f"  ✅ Trailing stop activated @ {current_time}: ${trailing_stop:.4f} (3% profit reached)")
                
                # Update trailing stop (only move up)
                atr = current.get('atr', 0)
                if pd.notna(atr) and atr > 0:
                    new_stop = position['max_price_reached'] - (atr * 2)
                else:
                    trailing_stop_pct = 2.5
                    new_stop = position['max_price_reached'] * (1 - trailing_stop_pct / 100)
                
                new_stop = max(new_stop, position['entry_price'])  # Never below entry
                
                # Only move stop up
                if new_stop > position['trailing_stop_price']:
                    position['trailing_stop_price'] = new_stop
                    logger.debug(f"  Trailing stop moved up to ${new_stop:.4f}")
                
                # Check if stop hit (use low price to check)
                if current.get('low', current_price) <= position['trailing_stop_price']:
                    exit_reason = f"Trailing stop hit at ${position['trailing_stop_price']:.4f} (ATR-based)"
                    exit_price = position['trailing_stop_price']
            
            # Execute exit if condition met
            if exit_reason:
                # Calculate profit on $1000 position
                position_value = 1000
                shares = position_value / position['entry_price']
                profit = shares * (exit_price - position['entry_price'])
                
                trade_result = {
                    'entry_time': position['entry_time'],
                    'entry_price': position['entry_price'],
                    'exit_time': current_time,
                    'exit_price': exit_price,
                    'exit_idx': idx,
                    'max_price_reached': position['max_price_reached'],
                    'pnl_pct': unrealized_pnl_pct,
                    'exit_reason': exit_reason,
                    'pattern': position['pattern'],
                    'hold_time_minutes': (current_time - position['entry_time']).total_seconds() / 60,
                    'shares': shares,
                    'profit_dollars': profit
                }
                simulated_trades.append(trade_result)
                
                logger.info(f"EXIT @ {current_time}: ${exit_price:.4f} - {exit_reason}")
                logger.info(f"  Entry: ${position['entry_price']:.4f} @ {position['entry_time']}")
                logger.info(f"  P&L: {unrealized_pnl_pct:+.2f}% (${profit:+.2f} on $1000)")
                logger.info(f"  Max Price: ${position['max_price_reached']:.4f}")
                logger.info(f"  Hold Time: {trade_result['hold_time_minutes']:.1f} minutes ({trade_result['hold_time_minutes']/60:.2f} hours)")
                
                # Remove position
                del active_positions[ticker]
    
    # Close any remaining positions at end of day
    for ticker, position in active_positions.items():
        final_price = data_1min.iloc[-1]['close']
        final_time = data_1min.iloc[-1]['timestamp']
        final_pnl = ((final_price - position['entry_price']) / position['entry_price']) * 100
        
        position_value = 1000
        shares = position_value / position['entry_price']
        profit = shares * (final_price - position['entry_price'])
        
        trade_result = {
            'entry_time': position['entry_time'],
            'entry_price': position['entry_price'],
            'exit_time': final_time,
            'exit_price': final_price,
            'exit_idx': len(data_1min) - 1,
            'max_price_reached': position['max_price_reached'],
            'pnl_pct': final_pnl,
            'exit_reason': 'End of day',
            'pattern': position['pattern'],
            'hold_time_minutes': (final_time - position['entry_time']).total_seconds() / 60,
            'shares': shares,
            'profit_dollars': profit
        }
        simulated_trades.append(trade_result)
        
        logger.info(f"END OF DAY EXIT @ {final_time}: ${final_price:.4f}")
        logger.info(f"  Entry: ${position['entry_price']:.4f} @ {position['entry_time']}")
        logger.info(f"  P&L: {final_pnl:+.2f}% (${profit:+.2f} on $1000)")
    
    return entry_signals, simulated_trades, data_1min


def analyze_opportunities(entry_signals, simulated_trades, data_1min):
    """Analyze all opportunities and create detailed report"""
    logger.info(f"\n{'='*80}")
    logger.info(f"OPPORTUNITY ANALYSIS")
    logger.info(f"{'='*80}")
    
    logger.info(f"\nEntry Signals Found: {len(entry_signals)}")
    for i, signal in enumerate(entry_signals, 1):
        logger.info(f"\n  {i}. Entry Signal:")
        logger.info(f"     Time: {signal['timestamp']}")
        logger.info(f"     Price: ${signal['price']:.4f}")
        logger.info(f"     Pattern: {signal['pattern']}")
        logger.info(f"     Confidence: {signal['confidence']*100:.1f}%")
        if signal['target_price']:
            expected_gain = ((signal['target_price'] - signal['price']) / signal['price']) * 100
            logger.info(f"     Target: ${signal['target_price']:.4f} (+{expected_gain:.1f}%)")
    
    logger.info(f"\nSimulated Trades: {len(simulated_trades)}")
    total_profit = 0
    for i, trade in enumerate(simulated_trades, 1):
        logger.info(f"\n  {i}. Trade:")
        logger.info(f"     Entry: ${trade['entry_price']:.4f} @ {trade['entry_time']}")
        logger.info(f"     Exit: ${trade['exit_price']:.4f} @ {trade['exit_time']}")
        logger.info(f"     P&L: {trade['pnl_pct']:+.2f}%")
        logger.info(f"     Max Price: ${trade['max_price_reached']:.4f}")
        logger.info(f"     Hold Time: {trade['hold_time_minutes']:.1f} minutes ({trade['hold_time_minutes']/60:.2f} hours)")
        logger.info(f"     Exit Reason: {trade['exit_reason']}")
        logger.info(f"     Profit (on $1000): ${trade.get('profit_dollars', 0):+.2f}")
        total_profit += trade.get('profit_dollars', 0)
    
    logger.info(f"\nTotal Profit (on $1000 per trade): ${total_profit:+.2f}")
    
    # Calculate statistics
    if simulated_trades:
        wins = [t for t in simulated_trades if t['pnl_pct'] > 0]
        losses = [t for t in simulated_trades if t['pnl_pct'] <= 0]
        win_rate = (len(wins) / len(simulated_trades)) * 100 if simulated_trades else 0
        avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
        total_return = sum([t['pnl_pct'] for t in simulated_trades])
        
        logger.info(f"\nTrade Statistics:")
        logger.info(f"  Total Trades: {len(simulated_trades)}")
        logger.info(f"  Wins: {len(wins)}")
        logger.info(f"  Losses: {len(losses)}")
        logger.info(f"  Win Rate: {win_rate:.1f}%")
        logger.info(f"  Average Win: {avg_win:+.2f}%")
        logger.info(f"  Average Loss: {avg_loss:+.2f}%")
        logger.info(f"  Total Return: {total_return:+.2f}%")
        if simulated_trades:
            best = max(simulated_trades, key=lambda x: x['pnl_pct'])
            worst = min(simulated_trades, key=lambda x: x['pnl_pct'])
            logger.info(f"  Best Trade: {best['pnl_pct']:+.2f}% @ {best['entry_time']}")
            logger.info(f"  Worst Trade: {worst['pnl_pct']:+.2f}% @ {worst['entry_time']}")
    
    return {
        'entry_signals': entry_signals,
        'simulated_trades': simulated_trades,
        'total_profit': total_profit
    }


def main():
    """Main simulation function"""
    import sys
    
    # Check if analyzing rejection window
    if len(sys.argv) > 1 and sys.argv[1] == '--rejection':
        # Analyze rejection window (15:42 - 15:49)
        logger.info("="*80)
        logger.info("FLYX REJECTION ANALYSIS (15:42 - 15:49)")
        logger.info("="*80)
        
        try:
            # Download data
            data_1min = download_flyx_data()
            
            if len(data_1min) == 0:
                logger.error("No data downloaded")
                return
            
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
        return
    
    # Full day simulation
    logger.info("="*80)
    logger.info("FLYX FULL DAY SIMULATION (4:00 AM - 8:00 PM)")
    logger.info("="*80)
    logger.info("Simulating with CURRENT CODE (all fixes applied):")
    logger.info("  - Daily loss limit: REMOVED")
    logger.info("  - Trailing stop: Requires 3% profit, ATR-based")
    logger.info("  - Re-entry: 10-minute cooldown")
    logger.info("="*80)
    
    try:
        # Download data
        data_1min = download_flyx_data()
        
        if len(data_1min) == 0:
            logger.error("No data downloaded")
            return
        
        logger.info(f"Data range: {data_1min.iloc[0]['timestamp']} to {data_1min.iloc[-1]['timestamp']}")
        
        # Simulate full day trading
        entry_signals, simulated_trades, data_1min = simulate_full_day_trading(data_1min)
        
        # Analyze opportunities
        results = analyze_opportunities(entry_signals, simulated_trades, data_1min)
        
        # Save detailed report
        output_file = Path(__file__).parent / 'flyx_full_day_simulation_report.txt'
        with open(output_file, 'w') as f:
            f.write("FLYX Full Day Simulation Report\n")
            f.write("="*80 + "\n\n")
            f.write("Simulation Settings:\n")
            f.write("  - Start Time: 4:00 AM ET\n")
            f.write("  - End Time: 8:00 PM ET\n")
            f.write("  - Daily Loss Limit: REMOVED\n")
            f.write("  - Trailing Stop: 3% activation, ATR-based\n")
            f.write("  - Re-entry: 10-minute cooldown\n\n")
            
            f.write(f"Entry Signals Found: {len(entry_signals)}\n\n")
            for i, signal in enumerate(entry_signals, 1):
                f.write(f"{i}. Entry Signal:\n")
                f.write(f"   Time: {signal['timestamp']}\n")
                f.write(f"   Price: ${signal['price']:.4f}\n")
                f.write(f"   Pattern: {signal['pattern']}\n")
                f.write(f"   Confidence: {signal['confidence']*100:.1f}%\n")
                if signal['target_price']:
                    expected_gain = ((signal['target_price'] - signal['price']) / signal['price']) * 100
                    f.write(f"   Target: ${signal['target_price']:.4f} (+{expected_gain:.1f}%)\n")
                f.write("\n")
            
            f.write(f"Simulated Trades: {len(simulated_trades)}\n\n")
            for i, trade in enumerate(simulated_trades, 1):
                f.write(f"{i}. Trade:\n")
                f.write(f"   Entry: ${trade['entry_price']:.4f} @ {trade['entry_time']}\n")
                f.write(f"   Exit: ${trade['exit_price']:.4f} @ {trade['exit_time']}\n")
                f.write(f"   P&L: {trade['pnl_pct']:+.2f}%\n")
                f.write(f"   Max Price: ${trade['max_price_reached']:.4f}\n")
                f.write(f"   Hold Time: {trade['hold_time_minutes']:.1f} minutes ({trade['hold_time_minutes']/60:.2f} hours)\n")
                f.write(f"   Exit Reason: {trade['exit_reason']}\n")
                f.write(f"   Profit (on $1000): ${trade.get('profit_dollars', 0):+.2f}\n\n")
            
            f.write(f"Total Profit (on $1000 per trade): ${results['total_profit']:+.2f}\n\n")
            
            # Write statistics
            if simulated_trades:
                wins = [t for t in simulated_trades if t['pnl_pct'] > 0]
                losses = [t for t in simulated_trades if t['pnl_pct'] <= 0]
                win_rate = (len(wins) / len(simulated_trades)) * 100 if simulated_trades else 0
                avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
                avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
                total_return = sum([t['pnl_pct'] for t in simulated_trades])
                
                f.write("Trade Statistics:\n")
                f.write(f"  Total Trades: {len(simulated_trades)}\n")
                f.write(f"  Wins: {len(wins)}\n")
                f.write(f"  Losses: {len(losses)}\n")
                f.write(f"  Win Rate: {win_rate:.1f}%\n")
                f.write(f"  Average Win: {avg_win:+.2f}%\n")
                f.write(f"  Average Loss: {avg_loss:+.2f}%\n")
                f.write(f"  Total Return: {total_return:+.2f}%\n")
                if simulated_trades:
                    best = max(simulated_trades, key=lambda x: x['pnl_pct'])
                    worst = min(simulated_trades, key=lambda x: x['pnl_pct'])
                    f.write(f"  Best Trade: {best['pnl_pct']:+.2f}% @ {best['entry_time']}\n")
                    f.write(f"  Worst Trade: {worst['pnl_pct']:+.2f}% @ {worst['entry_time']}\n")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Simulation complete! Report saved to: {output_file}")
        logger.info(f"{'='*80}")
        
    except Exception as e:
        logger.error(f"Error during simulation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
