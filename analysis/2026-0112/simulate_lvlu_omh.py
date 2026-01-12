"""
Simulate bot trading for LVLU and OMH from 4 AM
Downloads data, saves to test_data, and runs bot simulation
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Stocks to simulate with their entry windows
STOCKS_TO_SIMULATE = {
    'LVLU': {'entry_after': '10:00'},  # potential trade entry after 10am
    'OMH': {'entry_after': '09:36'}    # potential trade entry after 9:36am
}

def parse_time(time_str):
    """Parse time string like '10:00' into hour and minute"""
    parts = time_str.split(':')
    return int(parts[0]), int(parts[1])

def download_and_save_data(ticker, api, test_data_dir):
    """Download 1-minute data and save to test_data folder"""
    
    et = pytz.timezone('US/Eastern')
    today = datetime.now(et).date()
    
    logger.info(f"Downloading data for {ticker}...")
    
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
        
        # Filter to today and from 4 AM
        df_today = df[df['date'] == today].copy()
        df_today['hour'] = df_today['timestamp'].dt.hour
        df_today = df_today[df_today['hour'] >= 4].copy()
        
        if df_today.empty:
            logger.warning(f"No data for {ticker} from 4 AM today")
            return None
        
        # Prepare data for saving
        df_save = df_today[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        df_save = df_save.sort_values('timestamp').reset_index(drop=True)
        
        # Save to test_data folder
        filename = f"{ticker}_1min_{today.strftime('%Y%m%d')}.csv"
        filepath = os.path.join(test_data_dir, filename)
        df_save.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df_save)} minutes of data to {filepath}")
        
        return df_save
        
    except Exception as e:
        logger.error(f"Error downloading data for {ticker}: {e}", exc_info=True)
        return None

def simulate_bot_trading(ticker, df_data, trader, entry_after_time):
    """Simulate bot trading on the data"""
    
    et = pytz.timezone('US/Eastern')
    today = datetime.now(et).date()
    
    entry_hour, entry_minute = parse_time(entry_after_time)
    
    logger.info(f"Simulating bot trading for {ticker}...")
    logger.info(f"Entry window: After {entry_after_time}")
    
    # Prepare DataFrame
    df = df_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Track trades
    trades = []
    
    # Process data minute-by-minute
    for idx in range(len(df)):
        current_time = df.iloc[idx]['timestamp']
        current_price = df.iloc[idx]['close']
        
        # Check if we're in the entry window
        in_entry_window = (current_time.hour > entry_hour or 
                          (current_time.hour == entry_hour and current_time.minute >= entry_minute))
        
        # Get all data up to current moment
        df_slice = df.iloc[:idx+1].copy()
        
        # Ensure minimum data for indicators
        if len(df_slice) < 50:
            continue
        
        # Analyze using bot's logic
        entry_signal, exit_signals = trader.analyze_data(df_slice, ticker, current_price)
        
        # Process exit signals first
        for exit_signal in exit_signals:
            if exit_signal.signal_type == 'partial_exit':
                # Handle partial exit
                if ticker in trader.active_positions:
                    position = trader.active_positions[ticker]
                    exit_price = exit_signal.price
                    exit_shares = position.shares * 0.5  # 50% exit
                    exit_value = exit_shares * exit_price
                    pnl = exit_value - (exit_shares * position.entry_price)
                    pnl_pct = (pnl / (exit_shares * position.entry_price)) * 100
                    
                    # Update position
                    position.shares -= exit_shares
                    position.entry_value = position.shares * position.entry_price
                    
                    logger.info(f"[PARTIAL EXIT] {ticker} @ ${exit_price:.4f} - {exit_signal.reason}")
                    logger.info(f"  Exited {exit_shares:.0f} shares, P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
            else:
                # Handle full exit
                if ticker in trader.active_positions:
                    position = trader.active_positions[ticker]
                    exit_price = exit_signal.price
                    exit_time = current_time
                    exit_reason = exit_signal.reason
                    exit_value = position.shares * exit_price
                    pnl = exit_value - position.entry_value
                    pnl_pct = (pnl / position.entry_value) * 100
                    
                    # Calculate hold time
                    hold_time_minutes = (exit_time - position.entry_time).total_seconds() / 60
                    
                    # Track max price during hold
                    entry_idx = df[df['timestamp'] <= position.entry_time].index[-1] if len(df[df['timestamp'] <= position.entry_time]) > 0 else idx
                    exit_idx = idx
                    if entry_idx < exit_idx:
                        max_price = df.iloc[entry_idx:exit_idx+1]['high'].max()
                        max_price_pct = ((max_price - position.entry_price) / position.entry_price) * 100
                    else:
                        max_price = exit_price
                        max_price_pct = pnl_pct
                    
                    trades.append({
                        'ticker': ticker,
                        'entry_time': position.entry_time,
                        'exit_time': exit_time,
                        'entry_price': position.entry_price,
                        'exit_price': exit_price,
                        'shares': position.shares,
                        'entry_value': position.entry_value,
                        'exit_value': exit_value,
                        'pnl_dollars': pnl,
                        'pnl_pct': pnl_pct,
                        'entry_pattern': position.entry_pattern,
                        'exit_reason': exit_reason,
                        'confidence': position.entry_confidence * 100,
                        'hold_time_minutes': hold_time_minutes,
                        'max_price': max_price,
                        'max_price_pct': max_price_pct,
                        'capture_rate': (pnl_pct / max_price_pct * 100) if max_price_pct > 0 else 0
                    })
                    
                    logger.info(f"[EXIT] {ticker} @ ${exit_price:.4f} - {exit_reason}")
                    logger.info(f"  P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%), Hold: {hold_time_minutes:.1f} min")
                    trader.active_positions.pop(ticker)
        
        # Process entry signal - only if in entry window
        if entry_signal and ticker not in trader.active_positions and in_entry_window:
            # Simulate entry
            shares = 1000  # Fixed shares for simulation
            entry_price = entry_signal.price
            entry_time = entry_signal.timestamp
            
            # Use default stop loss and profit target percentages
            stop_loss_pct = 15.0  # 15% stop loss
            profit_target_pct = 20.0  # 20% profit target
            
            # Calculate stop loss (should be below entry price)
            calculated_stop_loss = entry_price * (1 - stop_loss_pct / 100)
            
            position = ActivePosition(
                ticker=ticker,
                entry_time=entry_time,
                entry_price=entry_price,
                entry_pattern=entry_signal.pattern_name,
                entry_confidence=entry_signal.confidence,
                target_price=entry_price * (1 + profit_target_pct / 100),
                stop_loss=calculated_stop_loss,
                current_price=entry_price,
                unrealized_pnl_pct=0.0,
                max_price_reached=entry_price,
                shares=shares,
                entry_value=shares * entry_price
            )
            
            logger.debug(f"[{ticker}] Entry: ${entry_price:.4f}, Stop Loss: ${calculated_stop_loss:.4f} ({stop_loss_pct}%), Target: ${position.target_price:.4f} ({profit_target_pct}%)")
            trader.active_positions[ticker] = position
            logger.info(f"[ENTRY] {ticker} @ ${entry_price:.4f} - {entry_signal.pattern_name} ({entry_signal.confidence*100:.1f}%)")
            logger.info(f"  Entry time: {entry_time.strftime('%H:%M:%S')}")
        elif entry_signal and ticker not in trader.active_positions and not in_entry_window:
            logger.debug(f"[{ticker}] Entry signal detected at {current_time.strftime('%H:%M:%S')} but before entry window ({entry_after_time})")
    
    # Handle any remaining open positions (mark as open)
    for ticker_pos, position in trader.active_positions.items():
        if ticker_pos == ticker:
            current_price = df.iloc[-1]['close']
            current_time = df.iloc[-1]['timestamp']
            exit_value = position.shares * current_price
            pnl = exit_value - position.entry_value
            pnl_pct = (pnl / position.entry_value) * 100
            hold_time_minutes = (current_time - position.entry_time).total_seconds() / 60
            
            trades.append({
                'ticker': ticker_pos,
                'entry_time': position.entry_time,
                'exit_time': current_time,
                'entry_price': position.entry_price,
                'exit_price': current_price,
                'shares': position.shares,
                'entry_value': position.entry_value,
                'exit_value': exit_value,
                'pnl_dollars': pnl,
                'pnl_pct': pnl_pct,
                'entry_pattern': position.entry_pattern,
                'exit_reason': 'OPEN POSITION',
                'confidence': position.entry_confidence * 100,
                'hold_time_minutes': hold_time_minutes,
                'max_price': position.max_price_reached,
                'max_price_pct': ((position.max_price_reached - position.entry_price) / position.entry_price) * 100,
                'capture_rate': 0
            })
    
    return trades

def main():
    """Main simulation function"""
    
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
    
    # Process each stock
    all_trades = []
    
    for ticker, config in STOCKS_TO_SIMULATE.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {ticker}")
        logger.info(f"Entry Window: After {config['entry_after']}")
        logger.info(f"{'='*80}")
        
        # Download and save data
        df_data = download_and_save_data(ticker, api, test_data_dir)
        
        if df_data is None or df_data.empty:
            logger.warning(f"Skipping {ticker} - no data available")
            continue
        
        # Simulate bot trading
        trades = simulate_bot_trading(ticker, df_data, trader, config['entry_after'])
        
        if trades:
            all_trades.extend(trades)
            logger.info(f"\n{ticker} Summary:")
            logger.info(f"  Total Trades: {len(trades)}")
            total_pnl = sum(t['pnl_dollars'] for t in trades)
            total_pnl_pct = sum(t['pnl_pct'] for t in trades)
            logger.info(f"  Total P&L: ${total_pnl:+.2f} ({total_pnl_pct:+.2f}%)")
            
            # Count wins/losses
            wins = len([t for t in trades if t['pnl_dollars'] > 0])
            losses = len([t for t in trades if t['pnl_dollars'] < 0])
            logger.info(f"  Wins: {wins}, Losses: {losses}")
        else:
            logger.info(f"No trades generated for {ticker}")
    
    # Export all trades to CSV
    if all_trades:
        df_trades = pd.DataFrame(all_trades)
        csv_file = f"analysis/BOT_SIMULATION_TRADES_LVLU_OMH_{today.strftime('%Y%m%d')}.csv"
        df_trades.to_csv(csv_file, index=False)
        logger.info(f"\n{'='*80}")
        logger.info(f"All trades exported to: {csv_file}")
        logger.info(f"{'='*80}")
        
        # Print summary
        logger.info(f"\nOverall Summary:")
        logger.info(f"  Total Trades: {len(all_trades)}")
        total_pnl = df_trades['pnl_dollars'].sum()
        total_pnl_pct = df_trades['pnl_pct'].sum()
        logger.info(f"  Total P&L: ${total_pnl:+.2f} ({total_pnl_pct:+.2f}%)")
        
        wins = len(df_trades[df_trades['pnl_dollars'] > 0])
        losses = len(df_trades[df_trades['pnl_dollars'] < 0])
        win_rate = (wins / len(all_trades) * 100) if all_trades else 0
        logger.info(f"  Wins: {wins}, Losses: {losses}, Win Rate: {win_rate:.1f}%")
        
        avg_pnl = df_trades['pnl_dollars'].mean()
        avg_pnl_pct = df_trades['pnl_pct'].mean()
        logger.info(f"  Average P&L per Trade: ${avg_pnl:+.2f} ({avg_pnl_pct:+.2f}%)")
        
        avg_hold_time = df_trades['hold_time_minutes'].mean()
        logger.info(f"  Average Hold Time: {avg_hold_time:.1f} minutes")
        
        avg_capture_rate = df_trades[df_trades['capture_rate'] > 0]['capture_rate'].mean()
        if pd.notna(avg_capture_rate):
            logger.info(f"  Average Capture Rate: {avg_capture_rate:.1f}%")
    else:
        logger.warning("No trades generated for any stock")

if __name__ == "__main__":
    main()
