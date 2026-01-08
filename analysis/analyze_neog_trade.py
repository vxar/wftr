"""
NEOG Trade Analysis
Analyze why the NEOG trade was exited prematurely or missed opportunity
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_neog_trades():
    """Get all NEOG trades from database"""
    db = TradingDatabase()
    trades = db.get_trades_by_ticker('NEOG')
    positions = db.get_active_positions()
    neog_positions = [p for p in positions if p.get('ticker') == 'NEOG']
    
    logger.info(f"Found {len(trades)} NEOG trade(s) in database")
    logger.info(f"Found {len(neog_positions)} active NEOG position(s)")
    
    return trades, neog_positions


def download_neog_data():
    """Download comprehensive data for NEOG"""
    api = WebullDataAPI()
    
    logger.info("Downloading NEOG data from Webull API...")
    
    try:
        # Get multiple timeframes
        data_1min = api.get_1min_data('NEOG', minutes=1200)  # Max available
        data_5min = api.get_5min_data('NEOG', periods=500)    # ~41 days
        
        logger.info(f"Downloaded {len(data_1min)} 1-minute bars and {len(data_5min)} 5-minute bars")
        
        return data_1min, data_5min
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        raise


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


def analyze_trade_exit(data_1min, data_5min, trade):
    """Analyze why the trade was exited prematurely"""
    logger.info(f"\n{'='*80}")
    logger.info(f"ANALYZING NEOG TRADE")
    logger.info(f"{'='*80}")
    logger.info(f"Entry Time: {trade['entry_time']}")
    logger.info(f"Exit Time: {trade['exit_time']}")
    logger.info(f"Entry Price: ${trade['entry_price']:.4f}")
    logger.info(f"Exit Price: ${trade['exit_price']:.4f}")
    logger.info(f"P&L: ${trade['pnl_dollars']:.2f} ({trade['pnl_pct']:.2f}%)")
    logger.info(f"Exit Reason: {trade['exit_reason']}")
    logger.info(f"Entry Pattern: {trade['entry_pattern']}")
    
    # Parse times
    entry_time = pd.to_datetime(trade['entry_time'])
    exit_time = pd.to_datetime(trade['exit_time'])
    
    # Filter data to trade period and beyond
    trade_1min = data_1min[
        (data_1min['timestamp'] >= entry_time - timedelta(hours=1)) & 
        (data_1min['timestamp'] <= exit_time + timedelta(hours=6))
    ].copy()
    
    if len(trade_1min) == 0:
        logger.error("No 1-minute data found for trade period")
        return None
    
    # Calculate indicators
    detector = PatternDetector()
    trade_1min = detector.calculate_indicators(trade_1min)
    
    # Add ATR
    trade_1min['atr'] = calculate_atr(trade_1min, period=14)
    trade_1min['atr_pct'] = (trade_1min['atr'] / trade_1min['close']) * 100
    
    # Find entry and exit points
    entry_idx = None
    exit_idx = None
    
    for idx, row in trade_1min.iterrows():
        if entry_idx is None and row['timestamp'] >= entry_time:
            entry_idx = idx
        if exit_idx is None and row['timestamp'] >= exit_time:
            exit_idx = idx
            break
    
    if entry_idx is None or exit_idx is None:
        logger.error("Could not find entry or exit point in data")
        return None
    
    entry_price = trade['entry_price']
    exit_price = trade['exit_price']
    
    # Get price data during and after trade
    trade_period = trade_1min[(trade_1min.index >= entry_idx) & (trade_1min.index <= exit_idx)]
    post_exit = trade_1min[trade_1min.index > exit_idx]
    
    max_price_during = trade_period['high'].max() if len(trade_period) > 0 else entry_price
    max_price_after = post_exit['high'].max() if len(post_exit) > 0 else exit_price
    max_price_overall = max(max_price_during, max_price_after)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"PRICE ANALYSIS")
    logger.info(f"{'='*80}")
    logger.info(f"Entry Price: ${entry_price:.4f}")
    logger.info(f"Exit Price: ${exit_price:.4f}")
    logger.info(f"Max Price During Trade: ${max_price_during:.4f}")
    logger.info(f"Max Price After Exit: ${max_price_after:.4f}")
    logger.info(f"Max Price Overall: ${max_price_overall:.4f}")
    
    max_potential_gain = ((max_price_overall - entry_price) / entry_price) * 100
    actual_gain = trade['pnl_pct']
    lost_potential = max_potential_gain - actual_gain
    
    logger.info(f"\nGain Analysis:")
    logger.info(f"  Max Potential Gain: {max_potential_gain:.2f}%")
    logger.info(f"  Actual Gain: {actual_gain:.2f}%")
    logger.info(f"  Lost Potential: {lost_potential:.2f}%")
    logger.info(f"  Lost Dollars: ${(max_price_overall - exit_price) * trade['shares']:.2f}")
    
    # Analyze exit conditions
    exit_row = trade_1min.loc[exit_idx] if exit_idx in trade_1min.index else None
    
    logger.info(f"\n{'='*80}")
    logger.info(f"EXIT ANALYSIS")
    logger.info(f"{'='*80}")
    logger.info(f"Exit Reason: {trade['exit_reason']}")
    
    if exit_row is not None:
        logger.info(f"\nIndicators at Exit Time ({exit_row['timestamp']}):")
        logger.info(f"  Price: ${exit_row['close']:.4f}")
        logger.info(f"  High: ${exit_row['high']:.4f}")
        logger.info(f"  Low: ${exit_row['low']:.4f}")
        logger.info(f"  RSI: {exit_row.get('rsi', 'N/A'):.2f}" if 'rsi' in exit_row else "  RSI: N/A")
        logger.info(f"  MACD: {exit_row.get('macd', 'N/A'):.4f}" if 'macd' in exit_row else "  MACD: N/A")
        logger.info(f"  MACD Signal: {exit_row.get('macd_signal', 'N/A'):.4f}" if 'macd_signal' in exit_row else "  MACD Signal: N/A")
        logger.info(f"  Volume Ratio: {exit_row.get('volume_ratio', 'N/A'):.2f}" if 'volume_ratio' in exit_row else "  Volume Ratio: N/A")
        logger.info(f"  ATR: ${exit_row.get('atr', 'N/A'):.4f}" if 'atr' in exit_row else "  ATR: N/A")
        logger.info(f"  ATR %: {exit_row.get('atr_pct', 'N/A'):.2f}%" if 'atr_pct' in exit_row else "  ATR %: N/A")
    
    # Analyze trailing stop issue
    if 'trailing stop' in trade['exit_reason'].lower():
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAILING STOP ANALYSIS")
        logger.info(f"{'='*80}")
        
        # Simulate trailing stop logic
        max_price_reached = trade_period['high'].max() if len(trade_period) > 0 else entry_price
        trailing_stop_pct = 2.5  # Current setting (before fix)
        trailing_stop_price = max_price_reached * (1 - trailing_stop_pct / 100)
        
        logger.info(f"Max Price Reached: ${max_price_reached:.4f}")
        logger.info(f"Trailing Stop %: {trailing_stop_pct}%")
        logger.info(f"Trailing Stop Price: ${trailing_stop_price:.4f}")
        logger.info(f"Entry Price: ${entry_price:.4f}")
        logger.info(f"Exit Price: ${exit_price:.4f}")
        
        if trailing_stop_price < entry_price:
            logger.warning(f"⚠️  ISSUE: Trailing stop price (${trailing_stop_price:.4f}) is BELOW entry price (${entry_price:.4f})!")
            logger.warning(f"   This means the trailing stop was set before meaningful profit was achieved.")
        
        # Calculate what ATR-based stop would be
        if exit_row is not None and 'atr' in exit_row and not pd.isna(exit_row['atr']):
            atr = exit_row['atr']
            atr_stop_price = max_price_reached - (atr * 2)  # 2x ATR stop
            logger.info(f"\nATR-Based Stop Analysis:")
            logger.info(f"  ATR: ${atr:.4f}")
            logger.info(f"  ATR %: {exit_row.get('atr_pct', 0):.2f}%")
            logger.info(f"  2x ATR Stop Price: ${atr_stop_price:.4f}")
            logger.info(f"  2x ATR Stop %: {((max_price_reached - atr_stop_price) / max_price_reached) * 100:.2f}%")
    
    return {
        'entry_time': entry_time,
        'exit_time': exit_time,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'max_price_during': max_price_during,
        'max_price_after': max_price_after,
        'max_price_overall': max_price_overall,
        'max_potential_gain': max_potential_gain,
        'actual_gain': actual_gain,
        'lost_potential': lost_potential,
        'exit_reason': trade['exit_reason'],
        'trade_data_1min': trade_1min,
        'entry_idx': entry_idx,
        'exit_idx': exit_idx
    }


def identify_issues(analysis_result):
    """Identify issues that caused premature exit"""
    issues = []
    recommendations = []
    
    if analysis_result is None:
        return issues, recommendations
    
    exit_reason = analysis_result['exit_reason']
    max_potential = analysis_result['max_potential_gain']
    actual_gain = analysis_result['actual_gain']
    lost_potential = analysis_result['lost_potential']
    
    logger.info(f"\n{'='*80}")
    logger.info(f"ISSUE IDENTIFICATION")
    logger.info(f"{'='*80}")
    
    # Check exit reason
    if 'trailing stop' in exit_reason.lower():
        issues.append("Trailing stop hit too early - 2.5% is too tight for volatile stocks")
        recommendations.append("Use ATR-based trailing stops instead of fixed percentage")
        recommendations.append("Don't activate trailing stop until profit >= 3-5%")
        recommendations.append("Trailing stop should only move UP, never down")
        recommendations.append("For volatile stocks, use wider stops (4-5% or 2x ATR)")
    
    if 'stop loss' in exit_reason.lower():
        issues.append("Stop loss triggered")
        recommendations.append("Review stop loss placement - may be too tight")
        recommendations.append("Use ATR-based stop loss instead of fixed percentage")
    
    if lost_potential > 10:
        issues.append(f"Premature exit - missed {lost_potential:.2f}% potential gain")
        recommendations.append("Implement profit target scaling (take partial profits, let winners run)")
        recommendations.append("Use wider trailing stops for strong trends")
        recommendations.append("Consider holding through minor pullbacks if trend is intact")
        recommendations.append("Add confirmation before exiting (e.g., multiple bearish signals)")
    
    if max_potential > 20 and actual_gain < 5:
        issues.append(f"Massive missed opportunity - {max_potential:.2f}% potential vs {actual_gain:.2f}% actual")
        recommendations.append("Trailing stop logic needs major revision")
        recommendations.append("Consider position scaling: take 50% at +5%, let rest run with wider stop")
        recommendations.append("Use trend strength indicators to determine stop width")
    
    logger.info(f"\nIssues Found:")
    for i, issue in enumerate(issues, 1):
        logger.info(f"  {i}. {issue}")
    
    logger.info(f"\nRecommendations:")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"  {i}. {rec}")
    
    return issues, recommendations


def rerun_trade_analysis(data_1min, trade):
    """Rerun trade analysis to find optimal entry and exit points"""
    logger.info(f"\n{'='*80}")
    logger.info(f"RERUNNING TRADE ANALYSIS - FINDING OPTIMAL ENTRIES/EXITS")
    logger.info(f"{'='*80}")
    
    detector = PatternDetector()
    data_1min = detector.calculate_indicators(data_1min)
    data_1min['atr'] = calculate_atr(data_1min, period=14)
    data_1min['atr_pct'] = (data_1min['atr'] / data_1min['close']) * 100
    
    entry_time = pd.to_datetime(trade['entry_time'])
    exit_time = pd.to_datetime(trade['exit_time'])
    
    # Get data after entry
    post_entry = data_1min[data_1min['timestamp'] > entry_time].copy()
    
    if len(post_entry) == 0:
        logger.warning("No data after entry time")
        return []
    
    entry_price = trade['entry_price']
    max_price = post_entry['high'].max()
    max_price_time = post_entry.loc[post_entry['high'].idxmax(), 'timestamp']
    
    logger.info(f"\nPrice Movement After Entry:")
    logger.info(f"  Entry: ${entry_price:.4f} @ {entry_time}")
    logger.info(f"  Max Price: ${max_price:.4f} @ {max_price_time}")
    logger.info(f"  Max Gain: {((max_price - entry_price) / entry_price) * 100:.2f}%")
    
    # Find optimal exits
    optimal_exits = []
    
    # Scenario 1: Profit target exit (8%)
    profit_target = entry_price * 1.08
    profit_exit = post_entry[post_entry['high'] >= profit_target]
    if len(profit_exit) > 0:
        profit_exit_time = profit_exit.iloc[0]['timestamp']
        optimal_exits.append({
            'type': 'Profit Target (8%)',
            'time': profit_exit_time,
            'price': profit_target,
            'gain': 8.0,
            'description': f"Take profit at 8% target"
        })
    
    # Scenario 2: ATR-based trailing stop
    trailing_stop_active = False
    trailing_stop_price = None
    max_price_seen = entry_price
    
    for idx, row in post_entry.iterrows():
        # Update max price seen
        if row['high'] > max_price_seen:
            max_price_seen = row['high']
            # Activate trailing stop only after 3% profit
            if not trailing_stop_active and ((max_price_seen - entry_price) / entry_price) * 100 >= 3.0:
                trailing_stop_active = True
                atr = row.get('atr', 0)
                if pd.notna(atr) and atr > 0:
                    trailing_stop_price = max_price_seen - (atr * 2)  # 2x ATR
                else:
                    trailing_stop_price = max_price_seen * 0.95  # 5% fallback
                logger.info(f"  Trailing stop activated at ${max_price_seen:.4f}, stop at ${trailing_stop_price:.4f}")
        
        # Update trailing stop (only move up)
        if trailing_stop_active and row['high'] > max_price_seen:
            max_price_seen = row['high']
            atr = row.get('atr', 0)
            if pd.notna(atr) and atr > 0:
                new_stop = max_price_seen - (atr * 2)
            else:
                new_stop = max_price_seen * 0.95
            # Only move stop up
            if new_stop > trailing_stop_price:
                trailing_stop_price = new_stop
        
        # Check if stop hit
        if trailing_stop_active and row['low'] <= trailing_stop_price:
            optimal_exits.append({
                'type': 'ATR-Based Trailing Stop',
                'time': row['timestamp'],
                'price': trailing_stop_price,
                'gain': ((trailing_stop_price - entry_price) / entry_price) * 100,
                'description': f"ATR trailing stop hit (2x ATR from high of ${max_price_seen:.4f})"
            })
            break
    
    # Scenario 3: Max price exit (ideal)
    optimal_exits.append({
        'type': 'Max Price (Ideal)',
        'time': max_price_time,
        'price': max_price,
        'gain': ((max_price - entry_price) / entry_price) * 100,
        'description': f"Exit at maximum price reached"
    })
    
    # Scenario 4: Partial profit strategy (50% at +5%, rest with wider stop)
    partial_exit_price = entry_price * 1.05
    partial_exit = post_entry[post_entry['high'] >= partial_exit_price]
    if len(partial_exit) > 0:
        partial_exit_time = partial_exit.iloc[0]['timestamp']
        # Calculate remaining position gain
        remaining_data = post_entry[post_entry['timestamp'] > partial_exit_time]
        if len(remaining_data) > 0:
            remaining_max = remaining_data['high'].max()
            # Weighted gain: 50% at 5% + 50% at remaining max
            weighted_gain = (0.5 * 5.0) + (0.5 * ((remaining_max - entry_price) / entry_price) * 100)
            optimal_exits.append({
                'type': 'Partial Profit Strategy',
                'time': partial_exit_time,
                'price': f"50% @ ${partial_exit_price:.4f}, 50% @ ${remaining_max:.4f}",
                'gain': weighted_gain,
                'description': f"Take 50% profit at +5%, let rest run to ${remaining_max:.4f}"
            })
    
    return optimal_exits


def main():
    """Main analysis function"""
    logger.info("="*80)
    logger.info("NEOG TRADE ANALYSIS")
    logger.info("="*80)
    
    try:
        # Get trades from database
        trades, positions = get_neog_trades()
        
        if not trades:
            logger.warning("No NEOG trades found in database")
            if positions:
                logger.info(f"Found {len(positions)} active NEOG position(s)")
            # Still analyze the opportunity based on dashboard data
            logger.info("\nAnalyzing missed opportunity based on dashboard data...")
            logger.info("Current Price: $9.51 (+28.80%)")
            logger.info("Open: $9.25 @ 13:29 EST")
            logger.info("High: $10.24")
            logger.info("Previous Close: $7.38")
            return
        
        # Download data
        data_1min, data_5min = download_neog_data()
        
        # Analyze each trade
        for trade in trades:
            analysis = analyze_trade_exit(data_1min, data_5min, trade)
            if analysis:
                issues, recommendations = identify_issues(analysis)
                
                # Rerun trade analysis
                optimal_exits = rerun_trade_analysis(data_1min, trade)
                
                # Print optimal exits
                logger.info(f"\n{'='*80}")
                logger.info(f"OPTIMAL EXIT SCENARIOS")
                logger.info(f"{'='*80}")
                
                for i, exit_scenario in enumerate(optimal_exits, 1):
                    logger.info(f"\nScenario {i}: {exit_scenario['type']}")
                    logger.info(f"  Time: {exit_scenario['time']}")
                    logger.info(f"  Price: ${exit_scenario['price']:.4f}" if isinstance(exit_scenario['price'], (int, float)) else f"  Price: {exit_scenario['price']}")
                    logger.info(f"  Gain: {exit_scenario['gain']:.2f}%")
                    logger.info(f"  Description: {exit_scenario['description']}")
        
        # Save summary report
        output_file = Path(__file__).parent / 'neog_analysis_report.txt'
        with open(output_file, 'w') as f:
            f.write("NEOG Trade Analysis Report\n")
            f.write("="*80 + "\n\n")
            if trades:
                for trade in trades:
                    f.write(f"Trade Analysis:\n")
                    f.write(f"  Entry: ${trade['entry_price']:.4f} @ {trade['entry_time']}\n")
                    f.write(f"  Exit: ${trade['exit_price']:.4f} @ {trade['exit_time']}\n")
                    f.write(f"  P&L: ${trade['pnl_dollars']:.2f} ({trade['pnl_pct']:.2f}%)\n")
                    f.write(f"  Exit Reason: {trade['exit_reason']}\n\n")
            else:
                f.write("No trades found in database.\n")
                f.write("Dashboard shows: $9.51 (+28.80%) from $7.38 previous close\n")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Analysis complete! Report saved to: {output_file}")
        logger.info(f"{'='*80}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
