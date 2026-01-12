"""
Run SOGP simulation from 9 AM using bot logic
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from src.data.webull_data_api import WebullDataAPI
from src.core.realtime_trader import RealtimeTrader, TradeSignal, ActivePosition
from src.database.trading_database import TradingDatabase
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_sogp_simulation():
    """Run SOGP simulation from 9 AM"""
    
    ticker = "SOGP"
    et = pytz.timezone('US/Eastern')
    today = datetime.now(et).date()
    
    # Check database for actual trade
    db = TradingDatabase()
    trades = db.get_all_trades()
    
    sogp_trades = []
    for trade in trades:
        if trade['ticker'] == ticker:
            entry_time = pd.to_datetime(trade['entry_time'])
            if entry_time.tz is None:
                entry_time = et.localize(entry_time)
            else:
                entry_time = entry_time.astimezone(et)
            
            if entry_time.date() == today:
                sogp_trades.append(trade)
    
    print(f"\n{'='*80}")
    print(f"SOGP SIMULATION FROM 9 AM")
    print(f"{'='*80}")
    
    if sogp_trades:
        actual_trade = sogp_trades[0]
        print(f"\nActual Trade from Database:")
        print(f"  Entry: {pd.to_datetime(actual_trade['entry_time']).strftime('%Y-%m-%d %H:%M:%S')} @ ${actual_trade['entry_price']:.4f}")
        print(f"  Exit:  {pd.to_datetime(actual_trade['exit_time']).strftime('%Y-%m-%d %H:%M:%S')} @ ${actual_trade['exit_price']:.4f}")
        print(f"  P&L: ${actual_trade['pnl_dollars']:+.2f} ({actual_trade['pnl_pct']:+.2f}%)")
        print(f"  Exit Reason: {actual_trade['exit_reason']}")
        print(f"  Pattern: {actual_trade['entry_pattern']}")
    
    # Initialize bot
    trader = RealtimeTrader(
        min_confidence=0.72,
        profit_target_pct=20.0,
        trailing_stop_pct=7.0
    )
    
    api = WebullDataAPI()
    
    # Fetch data
    logger.info(f"Fetching 1-minute data for {ticker}...")
    
    try:
        df = api.get_1min_data(ticker, minutes=1000)
        
        if df is None or df.empty:
            logger.error(f"No data returned for {ticker}")
            return None
        
        # Convert timestamp
        if 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            df['timestamp'] = pd.to_datetime(df.index)
        
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        df['timestamp'] = df['timestamp'].dt.tz_convert(et)
        df['date'] = df['timestamp'].dt.date
        
        # Filter to today
        df_today = df[df['date'] == today].copy()
        
        if df_today.empty:
            logger.warning(f"No data for {ticker} today. Using all available data.")
            df_today = df.copy()
        
        # For indicator calculations, we need historical data
        # Include data from before 9 AM for proper indicator calculation
        # But only analyze/process from 9 AM onwards
        df_all_for_indicators = df.copy()  # Use all available data for indicators
        
        logger.info(f"Total data points for today: {len(df_today)}")
        logger.info(f"Time range: {df_today['timestamp'].min()} to {df_today['timestamp'].max()}")
        
        # Filter from 9 AM to 10 AM
        df_today['hour'] = df_today['timestamp'].dt.hour
        df_today['minute'] = df_today['timestamp'].dt.minute
        # Keep data from 9:00 to 10:00
        df_9am_10am = df_today[((df_today['hour'] == 9) | (df_today['hour'] == 10))].copy()
        # For 10 AM, only keep 10:00
        df_9am_10am = df_9am_10am[~((df_9am_10am['hour'] == 10) & (df_9am_10am['minute'] > 0))].copy()
        
        logger.info(f"Data points from 9 AM to 10 AM: {len(df_9am_10am)}")
        if len(df_9am_10am) > 0:
            logger.info(f"Time range (9-10 AM): {df_9am_10am['timestamp'].min()} to {df_9am_10am['timestamp'].max()}")
            logger.info(f"First few timestamps: {df_9am_10am['timestamp'].head(10).tolist()}")
        
        df_today = df_9am_10am
        
        # Prepare DataFrame - use all data for indicators, but only process from 9 AM
        df_all_for_indicators = df_all_for_indicators[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        df_all_for_indicators = df_all_for_indicators.sort_values('timestamp').reset_index(drop=True)
        
        # Filter to 9 AM to 10 AM for processing
        df_all_for_indicators['hour'] = df_all_for_indicators['timestamp'].dt.hour
        df_all_for_indicators['minute'] = df_all_for_indicators['timestamp'].dt.minute
        df_9am_10am = df_all_for_indicators[((df_all_for_indicators['hour'] == 9) | (df_all_for_indicators['hour'] == 10))].copy()
        df_9am_10am = df_9am_10am[~((df_9am_10am['hour'] == 10) & (df_9am_10am['minute'] > 0))].copy()
        
        # Find the index where 9 AM starts in the full dataset
        nine_am_idx = df_all_for_indicators[df_all_for_indicators['timestamp'] >= pd.Timestamp(f"{today} 09:00:00", tz=et)].index[0] if len(df_all_for_indicators[df_all_for_indicators['timestamp'] >= pd.Timestamp(f"{today} 09:00:00", tz=et)]) > 0 else 0
        
        df_for_analysis = df_all_for_indicators  # Use full dataset for analysis
        
        # Track trades
        trades = []
        max_price_reached = 0
        max_price_time = None
        
        # Track rejection details for 9 AM to 10 AM
        rejection_log = []
        
        # Process data minute-by-minute from 9 AM onwards
        # Start from index where we have enough data (50 periods) or from 9 AM, whichever is later
        start_idx = max(50, nine_am_idx) if 'nine_am_idx' in locals() else 50
        
        for idx in range(start_idx, len(df_for_analysis)):
            current_time = df_for_analysis.iloc[idx]['timestamp']
            current_price = df_for_analysis.iloc[idx]['close']
            
            # Only process 9 AM to 10 AM
            if current_time.hour > 9 or (current_time.hour == 10 and current_time.minute > 0):
                break
            
            # Track max price
            if current_price > max_price_reached:
                max_price_reached = current_price
                max_price_time = current_time
            
            # Get all data up to current moment
            df_slice = df_for_analysis.iloc[:idx+1].copy()
            
            # Ensure minimum data
            if len(df_slice) < 50:
                continue
            
            # Analyze using bot's logic
            entry_signal, exit_signals = trader.analyze_data(df_slice, ticker, current_price)
            
            # Log detailed rejection reasons for 9 AM to 10 AM
            if current_time.hour == 9 or (current_time.hour == 10 and current_time.minute == 0):
                current_row = df_slice.iloc[-1] if len(df_slice) > 0 else None
                volume = current_row.get('volume', 0) if current_row is not None else 0
                volume_ratio = current_row.get('volume_ratio', 0) if current_row is not None else 0
                
                if entry_signal:
                    rejection_log.append({
                        'time': current_time,
                        'price': current_price,
                        'volume': volume,
                        'volume_ratio': volume_ratio,
                        'status': 'ENTRY SIGNAL',
                        'pattern': entry_signal.pattern_name,
                        'confidence': entry_signal.confidence * 100,
                        'rejection_reasons': []
                    })
                    logger.info(f"[{current_time.strftime('%H:%M:%S')}] ✅ ENTRY SIGNAL: {entry_signal.pattern_name} ({entry_signal.confidence*100:.1f}%) @ ${current_price:.4f}")
                elif ticker in trader.last_rejection_reasons:
                    rejection_reasons = trader.last_rejection_reasons[ticker].copy()
                    rejection_log.append({
                        'time': current_time,
                        'price': current_price,
                        'volume': volume,
                        'volume_ratio': volume_ratio,
                        'status': 'REJECTED',
                        'pattern': 'N/A',
                        'confidence': 0,
                        'rejection_reasons': rejection_reasons
                    })
                    logger.info(f"[{current_time.strftime('%H:%M:%S')}] ❌ REJECTED: {', '.join(rejection_reasons)}")
                    logger.info(f"  Price: ${current_price:.4f}, Volume: {volume:,.0f}, Vol Ratio: {volume_ratio:.2f}x")
                else:
                    # No pattern detected or no rejection reasons logged
                    rejection_log.append({
                        'time': current_time,
                        'price': current_price,
                        'volume': volume,
                        'volume_ratio': volume_ratio,
                        'status': 'NO PATTERN',
                        'pattern': 'N/A',
                        'confidence': 0,
                        'rejection_reasons': ['No pattern detected']
                    })
                    logger.info(f"[{current_time.strftime('%H:%M:%S')}] ⚠️  NO PATTERN: Price ${current_price:.4f}, Volume {volume:,.0f}, Vol Ratio {volume_ratio:.2f}x")
            
            # Process exit signals first
            for exit_signal in exit_signals:
                if exit_signal.signal_type == 'partial_exit':
                    if ticker in trader.active_positions:
                        position = trader.active_positions[ticker]
                        exit_price = exit_signal.price
                        exit_shares = position.shares * 0.5
                        exit_value = exit_shares * exit_price
                        pnl = exit_value - (exit_shares * position.entry_price)
                        pnl_pct = (pnl / (exit_shares * position.entry_price)) * 100
                        
                        position.shares -= exit_shares
                        position.entry_value = position.shares * position.entry_price
                        
                        logger.info(f"[PARTIAL EXIT] {ticker} @ ${exit_price:.4f} - {exit_signal.reason}")
                else:
                    if ticker in trader.active_positions:
                        position = trader.active_positions[ticker]
                        exit_price = exit_signal.price
                        exit_time = current_time
                        exit_reason = exit_signal.reason
                        exit_value = position.shares * exit_price
                        pnl = exit_value - position.entry_value
                        pnl_pct = (pnl / position.entry_value) * 100
                        hold_time = (exit_time - position.entry_time).total_seconds() / 60
                        
                        # Calculate potential profit
                        potential_profit = ((max_price_reached - position.entry_price) / position.entry_price) * 100
                        missed_profit = potential_profit - pnl_pct
                        
                        trades.append({
                            'ticker': ticker,
                            'entry_time': position.entry_time,
                            'exit_time': exit_time,
                            'entry_price': position.entry_price,
                            'exit_price': exit_price,
                            'shares': position.shares,
                            'entry_value': position.entry_value,
                            'exit_value': exit_value,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'entry_pattern': position.entry_pattern,
                            'exit_reason': exit_reason,
                            'confidence': position.entry_confidence,
                            'hold_time_minutes': hold_time,
                            'max_price': max_price_reached,
                            'max_price_time': max_price_time,
                            'potential_profit_pct': potential_profit,
                            'missed_profit_pct': missed_profit
                        })
                        
                        logger.info(f"[EXIT] {ticker} @ ${exit_price:.4f} - {exit_reason}")
                        logger.info(f"  P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%), Hold: {hold_time:.1f} min")
                        logger.info(f"  Max Price: ${max_price_reached:.4f}, Potential: {potential_profit:+.2f}%, Missed: {missed_profit:+.2f}%")
                        
                        trader.active_positions.pop(ticker)
                        max_price_reached = 0  # Reset for next trade
                        max_price_time = None
                        break
            
            # Process entry signal
            if entry_signal and ticker not in trader.active_positions:
                entry_price = entry_signal.price
                entry_time = current_time
                shares = 1000
                entry_value = shares * entry_price
                
                position = ActivePosition(
                    ticker=ticker,
                    entry_time=entry_time,
                    entry_price=entry_price,
                    entry_pattern=entry_signal.pattern_name,
                    entry_confidence=entry_signal.confidence,
                    target_price=entry_signal.target_price or entry_price * 1.20,
                    stop_loss=entry_signal.stop_loss or entry_price * 0.85,
                    current_price=entry_price,
                    shares=shares,
                    entry_value=entry_value
                )
                
                trader.active_positions[ticker] = position
                
                logger.info(f"[ENTRY] {ticker} @ ${entry_price:.4f} - {entry_signal.pattern_name} ({entry_signal.confidence*100:.1f}%)")
                logger.info(f"  Entry time: {entry_time.strftime('%H:%M:%S')}")
                max_price_reached = entry_price  # Reset max price tracking
                max_price_time = entry_time
        
        # Close any remaining position
        if ticker in trader.active_positions:
            position = trader.active_positions[ticker]
            exit_price = df_for_analysis.iloc[-1]['close']
            exit_time = df_for_analysis.iloc[-1]['timestamp']
            exit_reason = 'End of day'
            exit_value = position.shares * exit_price
            pnl = exit_value - position.entry_value
            pnl_pct = (pnl / position.entry_value) * 100
            hold_time = (exit_time - position.entry_time).total_seconds() / 60
            potential_profit = ((max_price_reached - position.entry_price) / position.entry_price) * 100
            missed_profit = potential_profit - pnl_pct
            
            trades.append({
                'ticker': ticker,
                'entry_time': position.entry_time,
                'exit_time': exit_time,
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'shares': position.shares,
                'entry_value': position.entry_value,
                'exit_value': exit_value,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'entry_pattern': position.entry_pattern,
                'exit_reason': exit_reason,
                'confidence': position.entry_confidence,
                'hold_time_minutes': hold_time,
                'max_price': max_price_reached,
                'max_price_time': max_price_time,
                'potential_profit_pct': potential_profit,
                'missed_profit_pct': missed_profit
            })
        
        # Summary
        print(f"\n{'='*80}")
        print("SIMULATION RESULTS")
        print(f"{'='*80}")
        
        print(f"\nTrades Generated: {len(trades)}")
        if trades:
            total_pnl = 0
            total_potential = 0
            total_missed = 0
            
            for i, trade in enumerate(trades, 1):
                print(f"\n  Trade {i}:")
                print(f"    Entry: {trade['entry_time'].strftime('%H:%M:%S')} @ ${trade['entry_price']:.4f}")
                print(f"    Exit:  {trade['exit_time'].strftime('%H:%M:%S')} @ ${trade['exit_price']:.4f}")
                print(f"    Hold: {trade['hold_time_minutes']:.1f} min")
                print(f"    P&L: ${trade['pnl']:+.2f} ({trade['pnl_pct']:+.2f}%)")
                print(f"    Exit Reason: {trade['exit_reason']}")
                print(f"    Pattern: {trade['entry_pattern']} ({trade['confidence']*100:.1f}%)")
                print(f"    Max Price: ${trade['max_price']:.4f} at {trade.get('max_price_time', 'N/A')}")
                print(f"    Potential Profit: {trade['potential_profit_pct']:+.2f}%")
                print(f"    Missed Profit: {trade['missed_profit_pct']:+.2f}%")
                
                total_pnl += trade['pnl']
                total_potential += trade['potential_profit_pct']
                total_missed += trade['missed_profit_pct']
            
            print(f"\n  Summary:")
            print(f"    Total P&L: ${total_pnl:+.2f}")
            print(f"    Average P&L: ${total_pnl/len(trades):+.2f}")
            print(f"    Total Potential: {total_potential:+.2f}%")
            print(f"    Total Missed: {total_missed:+.2f}%")
            if total_potential > 0:
                capture_rate = (total_pnl/len(trades)) / (total_potential/len(trades)) * 100 if total_potential > 0 else 0
                print(f"    Capture Rate: {capture_rate:.1f}%")
        
        # Export to CSV
        if trades:
            df_trades = pd.DataFrame(trades)
            csv_file = f"analysis/{ticker}_simulation_9am_{today.strftime('%Y%m%d')}.csv"
            df_trades.to_csv(csv_file, index=False)
            logger.info(f"\nTrades exported to: {csv_file}")
        
        # Export rejection log
        if rejection_log:
            df_rejections = pd.DataFrame(rejection_log)
            # Expand rejection_reasons list into a string
            df_rejections['rejection_reasons_str'] = df_rejections['rejection_reasons'].apply(lambda x: '; '.join(x) if isinstance(x, list) else str(x))
            csv_file = f"analysis/{ticker}_rejections_9am_10am_{today.strftime('%Y%m%d')}.csv"
            df_rejections[['time', 'price', 'volume', 'volume_ratio', 'status', 'pattern', 'confidence', 'rejection_reasons_str']].to_csv(csv_file, index=False)
            logger.info(f"Rejection log exported to: {csv_file}")
            
            # Print summary of rejections
            print(f"\n{'='*80}")
            print("REJECTION ANALYSIS (9 AM - 10 AM)")
            print(f"{'='*80}")
            
            total_minutes = len(rejection_log)
            entry_signals = len([r for r in rejection_log if r['status'] == 'ENTRY SIGNAL'])
            rejected = len([r for r in rejection_log if r['status'] == 'REJECTED'])
            no_pattern = len([r for r in rejection_log if r['status'] == 'NO PATTERN'])
            
            print(f"\nTotal Minutes Analyzed: {total_minutes}")
            print(f"Entry Signals: {entry_signals}")
            print(f"Rejected: {rejected}")
            print(f"No Pattern: {no_pattern}")
            
            # Count rejection reasons
            all_reasons = []
            for r in rejection_log:
                if r['rejection_reasons']:
                    all_reasons.extend(r['rejection_reasons'])
            
            if all_reasons:
                from collections import Counter
                reason_counts = Counter(all_reasons)
                print(f"\nTop Rejection Reasons:")
                for reason, count in reason_counts.most_common(10):
                    print(f"  {reason}: {count} times")
        
        return trades
        
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    result = run_sogp_simulation()
