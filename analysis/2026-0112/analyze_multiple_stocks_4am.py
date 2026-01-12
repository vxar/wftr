"""
Comprehensive analysis of multiple stocks from 4 AM
Identifies common patterns and setups that trigger bull runs
"""
import sys
import os
# Add src to path so we can import packages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from data.webull_data_api import WebullDataAPI
from core.realtime_trader import RealtimeTrader, TradeSignal, ActivePosition
from database.trading_database import TradingDatabase
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Stock analysis configuration
STOCKS_TO_ANALYZE = {
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

def analyze_stock(ticker, entry_after, exit_after, api, trader):
    """Analyze a single stock from 4 AM"""
    
    et = pytz.timezone('US/Eastern')
    today = datetime.now(et).date()
    
    entry_hour, entry_minute = parse_time(entry_after)
    exit_hour, exit_minute = parse_time(exit_after)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Analyzing {ticker}")
    logger.info(f"Expected Entry Window: After {entry_after}")
    logger.info(f"Expected Exit Window: After {exit_after}")
    logger.info(f"{'='*80}")
    
    try:
        # Fetch data from 4 AM
        df = api.get_1min_data(ticker, minutes=1000)
        
        if df is None or df.empty:
            logger.warning(f"No data returned for {ticker}")
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
        
        # Filter from 4 AM
        df_today['hour'] = df_today['timestamp'].dt.hour
        df_today = df_today[df_today['hour'] >= 4].copy()
        
        if df_today.empty:
            logger.error(f"No data for {ticker} from 4 AM today")
            return None
        
        logger.info(f"Analyzing {len(df_today)} minutes of data for {ticker}")
        logger.info(f"Time range: {df_today['timestamp'].min()} to {df_today['timestamp'].max()}")
        
        # Prepare DataFrame - use all data for indicators
        df_all = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        df_all = df_all.sort_values('timestamp').reset_index(drop=True)
        
        # Find index where 4 AM starts
        four_am_idx = df_all[df_all['timestamp'] >= pd.Timestamp(f"{today} 04:00:00", tz=et)].index[0] if len(df_all[df_all['timestamp'] >= pd.Timestamp(f"{today} 04:00:00", tz=et)]) > 0 else 50
        
        # Track analysis results
        analysis_results = {
            'ticker': ticker,
            'entry_window': entry_after,
            'exit_window': exit_after,
            'minute_data': [],
            'entry_opportunities': [],
            'rejections': [],
            'bot_trades': []
        }
        
        # Process data minute-by-minute from 4 AM
        start_idx = max(50, four_am_idx)
        
        for idx in range(start_idx, len(df_all)):
            current_time = df_all.iloc[idx]['timestamp']
            current_price = df_all.iloc[idx]['close']
            
            # Only process up to exit window
            if current_time.hour > exit_hour or (current_time.hour == exit_hour and current_time.minute > exit_minute):
                break
            
            # Get all data up to current moment
            df_slice = df_all.iloc[:idx+1].copy()
            
            # Analyze using bot's logic
            entry_signal, exit_signals = trader.analyze_data(df_slice, ticker, current_price)
            
            # Get current indicators
            current_row = df_slice.iloc[-1] if len(df_slice) > 0 else None
            volume = current_row.get('volume', 0) if current_row is not None else 0
            volume_ratio = current_row.get('volume_ratio', 0) if current_row is not None else 0
            
            # Calculate momentum
            momentum_5 = 0
            momentum_10 = 0
            if len(df_slice) >= 10:
                momentum_10 = ((current_price - df_slice.iloc[-10]['close']) / df_slice.iloc[-10]['close']) * 100
            if len(df_slice) >= 5:
                momentum_5 = ((current_price - df_slice.iloc[-5]['close']) / df_slice.iloc[-5]['close']) * 100
            
            # Check if we're in entry window
            in_entry_window = (current_time.hour > entry_hour or 
                             (current_time.hour == entry_hour and current_time.minute >= entry_minute))
            
            # Check if we're in exit window
            in_exit_window = (current_time.hour > exit_hour or 
                            (current_time.hour == exit_hour and current_time.minute >= exit_minute))
            
            # Store minute data
            minute_data = {
                'time': current_time,
                'price': current_price,
                'volume': volume,
                'volume_ratio': volume_ratio,
                'momentum_5': momentum_5,
                'momentum_10': momentum_10,
                'in_entry_window': in_entry_window,
                'in_exit_window': in_exit_window,
                'entry_signal': entry_signal is not None,
                'pattern': entry_signal.pattern_name if entry_signal else None,
                'confidence': entry_signal.confidence * 100 if entry_signal else 0,
                'rejection_reasons': trader.last_rejection_reasons.get(ticker, []).copy() if entry_signal is None else []
            }
            
            analysis_results['minute_data'].append(minute_data)
            
            # Track entry opportunities
            if entry_signal:
                analysis_results['entry_opportunities'].append({
                    'time': current_time,
                    'price': entry_signal.price,
                    'pattern': entry_signal.pattern_name,
                    'confidence': entry_signal.confidence * 100,
                    'in_entry_window': in_entry_window
                })
            
            # Track rejections in entry window
            if entry_signal is None and in_entry_window:
                rejection_reasons = trader.last_rejection_reasons.get(ticker, [])
                if rejection_reasons:
                    analysis_results['rejections'].append({
                        'time': current_time,
                        'price': current_price,
                        'volume': volume,
                        'volume_ratio': volume_ratio,
                        'momentum_5': momentum_5,
                        'momentum_10': momentum_10,
                        'rejection_reasons': rejection_reasons.copy()
                    })
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {e}", exc_info=True)
        return None

def find_common_patterns(all_results):
    """Analyze all stock results to find common patterns"""
    
    logger.info(f"\n{'='*80}")
    logger.info("COMMON PATTERN ANALYSIS")
    logger.info(f"{'='*80}")
    
    # Analyze successful entry windows
    entry_window_characteristics = []
    rejection_characteristics = []
    
    for result in all_results:
        if result is None:
            continue
        
        ticker = result['ticker']
        
        # Find entries in the expected entry window
        for opp in result['entry_opportunities']:
            if opp['in_entry_window']:
                entry_window_characteristics.append({
                    'ticker': ticker,
                    'time': opp['time'],
                    'price': opp['price'],
                    'pattern': opp['pattern'],
                    'confidence': opp['confidence']
                })
        
        # Find rejections in entry window
        for rej in result['rejections']:
            rejection_characteristics.append({
                'ticker': ticker,
                'time': rej['time'],
                'price': rej['price'],
                'volume': rej['volume'],
                'volume_ratio': rej['volume_ratio'],
                'momentum_5': rej['momentum_5'],
                'momentum_10': rej['momentum_10'],
                'rejection_reasons': rej['rejection_reasons']
            })
    
    # Analyze minute data around entry windows
    entry_window_data = []
    for result in all_results:
        if result is None:
            continue
        
        ticker = result['ticker']
        entry_hour, entry_minute = parse_time(result['entry_window'])
        
        # Get data from 10 minutes before to 10 minutes after entry window
        for minute_data in result['minute_data']:
            current_time = minute_data['time']
            if (current_time.hour == entry_hour and entry_minute - 10 <= current_time.minute <= entry_minute + 10) or \
               (current_time.hour == entry_hour - 1 and current_time.minute >= 50) or \
               (current_time.hour == entry_hour + 1 and current_time.minute <= 10):
                entry_window_data.append({
                    'ticker': ticker,
                    'time': current_time,
                    'price': minute_data['price'],
                    'volume': minute_data['volume'],
                    'volume_ratio': minute_data['volume_ratio'],
                    'momentum_5': minute_data['momentum_5'],
                    'momentum_10': minute_data['momentum_10'],
                    'entry_signal': minute_data['entry_signal'],
                    'pattern': minute_data['pattern'],
                    'confidence': minute_data['confidence'],
                    'in_entry_window': minute_data['in_entry_window']
                })
    
    return {
        'entry_opportunities': entry_window_characteristics,
        'rejections': rejection_characteristics,
        'entry_window_data': entry_window_data
    }

def main():
    """Main analysis function"""
    
    et = pytz.timezone('US/Eastern')
    today = datetime.now(et).date()
    
    # Initialize components
    api = WebullDataAPI()
    trader = RealtimeTrader(
        min_confidence=0.72,
        profit_target_pct=20.0,
        trailing_stop_pct=7.0
    )
    
    # Analyze all stocks
    all_results = []
    
    for ticker, config in STOCKS_TO_ANALYZE.items():
        result = analyze_stock(ticker, config['entry_after'], config['exit_after'], api, trader)
        if result:
            all_results.append(result)
    
    # Find common patterns
    common_patterns = find_common_patterns(all_results)
    
    # Generate reports
    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    for result in all_results:
        if result is None:
            continue
        
        ticker = result['ticker']
        entry_opps = len(result['entry_opportunities'])
        rejections = len(result['rejections'])
        entry_opps_in_window = len([o for o in result['entry_opportunities'] if o['in_entry_window']])
        
        print(f"\n{ticker}:")
        print(f"  Entry Opportunities: {entry_opps} (in window: {entry_opps_in_window})")
        print(f"  Rejections in Entry Window: {rejections}")
    
    # Export detailed data
    for result in all_results:
        if result is None:
            continue
        
        ticker = result['ticker']
        
        # Export minute data
        if result['minute_data']:
            df_minutes = pd.DataFrame(result['minute_data'])
            # Expand rejection_reasons list
            df_minutes['rejection_reasons_str'] = df_minutes['rejection_reasons'].apply(
                lambda x: '; '.join(x) if isinstance(x, list) and x else 'None'
            )
            csv_file = f"analysis/{ticker}_minute_by_minute_{today.strftime('%Y%m%d')}.csv"
            df_minutes[['time', 'price', 'volume', 'volume_ratio', 'momentum_5', 'momentum_10', 
                       'in_entry_window', 'in_exit_window', 'entry_signal', 'pattern', 'confidence', 
                       'rejection_reasons_str']].to_csv(csv_file, index=False)
            logger.info(f"Exported minute data to: {csv_file}")
    
    # Export common patterns
    if common_patterns['entry_window_data']:
        df_common = pd.DataFrame(common_patterns['entry_window_data'])
        csv_file = f"analysis/COMMON_PATTERNS_ENTRY_WINDOWS_{today.strftime('%Y%m%d')}.csv"
        df_common.to_csv(csv_file, index=False)
        logger.info(f"Exported common patterns to: {csv_file}")
    
    # Generate comprehensive report
    generate_comprehensive_report(all_results, common_patterns, today)
    
    return all_results, common_patterns

def generate_comprehensive_report(all_results, common_patterns, today):
    """Generate comprehensive analysis report"""
    
    report = []
    report.append("# Comprehensive Multi-Stock Analysis (4 AM Start)")
    report.append(f"Analysis Date: {today}")
    report.append("")
    report.append("## Stocks Analyzed")
    report.append("")
    
    for ticker, config in STOCKS_TO_ANALYZE.items():
        report.append(f"- **{ticker}**: Entry after {config['entry_after']}, Exit after {config['exit_after']}")
    
    report.append("")
    report.append("## Individual Stock Analysis")
    report.append("")
    
    for result in all_results:
        if result is None:
            continue
        
        ticker = result['ticker']
        entry_opps = result['entry_opportunities']
        rejections = result['rejections']
        entry_opps_in_window = [o for o in entry_opps if o['in_entry_window']]
        
        report.append(f"### {ticker}")
        report.append("")
        report.append(f"- **Expected Entry Window**: After {result['entry_window']}")
        report.append(f"- **Expected Exit Window**: After {result['exit_window']}")
        report.append(f"- **Total Entry Opportunities**: {len(entry_opps)}")
        report.append(f"- **Entry Opportunities in Window**: {len(entry_opps_in_window)}")
        report.append(f"- **Rejections in Entry Window**: {len(rejections)}")
        report.append("")
        
        if entry_opps_in_window:
            report.append("**Entry Signals in Window:**")
            for opp in entry_opps_in_window[:5]:  # Show first 5
                report.append(f"- {opp['time'].strftime('%H:%M:%S')} @ ${opp['price']:.4f} - {opp['pattern']} ({opp['confidence']:.1f}%)")
            report.append("")
        
        if rejections:
            # Count rejection reasons
            all_reasons = []
            for rej in rejections:
                all_reasons.extend(rej['rejection_reasons'])
            
            from collections import Counter
            reason_counts = Counter(all_reasons)
            report.append("**Top Rejection Reasons in Entry Window:**")
            for reason, count in reason_counts.most_common(5):
                report.append(f"- {reason}: {count} times")
            report.append("")
    
    report.append("## Common Patterns Analysis")
    report.append("")
    
    # Analyze entry window characteristics
    entry_data = common_patterns['entry_window_data']
    if entry_data:
        df_entry = pd.DataFrame(entry_data)
        
        # Filter to entries that actually happened
        successful_entries = df_entry[df_entry['entry_signal'] == True]
        failed_entries = df_entry[(df_entry['entry_signal'] == False) & (df_entry['in_entry_window'] == True)]
        
        report.append("### Successful Entry Characteristics")
        report.append("")
        
        if len(successful_entries) > 0:
            report.append("**Volume Characteristics:**")
            report.append(f"- Average Volume: {successful_entries['volume'].mean():,.0f}")
            report.append(f"- Average Volume Ratio: {successful_entries['volume_ratio'].mean():.2f}x")
            report.append(f"- Min Volume Ratio: {successful_entries['volume_ratio'].min():.2f}x")
            report.append(f"- Max Volume Ratio: {successful_entries['volume_ratio'].max():.2f}x")
            report.append("")
            
            report.append("**Momentum Characteristics:**")
            report.append(f"- Average 5-min Momentum: {successful_entries['momentum_5'].mean():.2f}%")
            report.append(f"- Average 10-min Momentum: {successful_entries['momentum_10'].mean():.2f}%")
            report.append("")
            
            report.append("**Pattern Distribution:**")
            pattern_counts = successful_entries['pattern'].value_counts()
            for pattern, count in pattern_counts.items():
                report.append(f"- {pattern}: {count} times")
            report.append("")
            
            report.append("**Confidence Levels:**")
            report.append(f"- Average Confidence: {successful_entries['confidence'].mean():.1f}%")
            report.append(f"- Min Confidence: {successful_entries['confidence'].min():.1f}%")
            report.append(f"- Max Confidence: {successful_entries['confidence'].max():.1f}%")
            report.append("")
        else:
            report.append("**No successful entries detected by bot**")
            report.append("")
        
        report.append("### Failed Entry Characteristics (Rejections in Window)")
        report.append("")
        
        if len(failed_entries) > 0:
            report.append("**Volume Characteristics:**")
            report.append(f"- Average Volume: {failed_entries['volume'].mean():,.0f}")
            report.append(f"- Average Volume Ratio: {failed_entries['volume_ratio'].mean():.2f}x")
            report.append("")
            
            report.append("**Momentum Characteristics:**")
            report.append(f"- Average 5-min Momentum: {failed_entries['momentum_5'].mean():.2f}%")
            report.append(f"- Average 10-min Momentum: {failed_entries['momentum_10'].mean():.2f}%")
            report.append("")
            
            # Count rejection reasons
            all_reasons = []
            for _, row in failed_entries.iterrows():
                # Get rejection reasons from original data
                for result in all_results:
                    if result and result['ticker'] == row['ticker']:
                        for minute_data in result['minute_data']:
                            if minute_data['time'] == row['time']:
                                all_reasons.extend(minute_data.get('rejection_reasons', []))
                                break
            
            from collections import Counter
            reason_counts = Counter(all_reasons)
            report.append("**Top Rejection Reasons:**")
            for reason, count in reason_counts.most_common(10):
                report.append(f"- {reason}: {count} times")
            report.append("")
    
    # Write report
    report_text = '\n'.join(report)
    report_file = f"analysis/MULTI_STOCK_ANALYSIS_REPORT_{today.strftime('%Y%m%d')}.md"
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    logger.info(f"\nComprehensive report saved to: {report_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("COMMON PATTERNS SUMMARY")
    print(f"{'='*80}")
    
    if entry_data:
        df_entry = pd.DataFrame(entry_data)
        successful = df_entry[df_entry['entry_signal'] == True]
        failed = df_entry[(df_entry['entry_signal'] == False) & (df_entry['in_entry_window'] == True)]
        
        print(f"\nSuccessful Entries: {len(successful)}")
        if len(successful) > 0:
            print(f"  Avg Volume Ratio: {successful['volume_ratio'].mean():.2f}x")
            print(f"  Avg 5-min Momentum: {successful['momentum_5'].mean():.2f}%")
            print(f"  Avg Confidence: {successful['confidence'].mean():.1f}%")
        
        print(f"\nFailed Entries (Rejections): {len(failed)}")
        if len(failed) > 0:
            print(f"  Avg Volume Ratio: {failed['volume_ratio'].mean():.2f}x")
            print(f"  Avg 5-min Momentum: {failed['momentum_5'].mean():.2f}%")

if __name__ == "__main__":
    all_results, common_patterns = main()
