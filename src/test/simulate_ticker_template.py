"""
Universal Trading Bot Simulation Template
==========================================

This template can be used to simulate any ticker from a specific detection time.
Simply configure the parameters at the top of the file and run.

Usage:
    python src/test/simulate_ticker_template.py

Configuration:
    - TICKER: Stock symbol to simulate
    - DETECTION_DATE: Date when ticker was detected (YYYY-MM-DD)
    - DETECTION_TIME: Time when ticker was detected (HH:MM format, ET)
    - START_HOUR: Hour to start data collection from (default: 4 AM)
    - LIVE_BOT_HAD_TRADES: True if live bot had trades, False otherwise
    - DETAILED_ANALYSIS_MINUTES: Number of minutes to analyze in detail after detection (default: 20)
"""
import sys
import os

# Add both root and src to path
# From src/test/, go up two levels to project root, then src is one level up
root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
src_dir = os.path.join(os.path.dirname(__file__), '..')
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

# ============================================================================
# CONFIGURATION - MODIFY THESE VALUES FOR EACH SIMULATION
# ============================================================================

TICKER = 'PRFX'  # Stock symbol to simulate
DETECTION_DATE = '2026-01-15'  # Date when ticker was detected (YYYY-MM-DD)
DETECTION_TIME = '16:05'  # Time when ticker was detected (HH:MM format, ET)
START_HOUR = 4  # Hour to start data collection from (default: 4 AM)
LIVE_BOT_HAD_TRADES = True  # True if live bot had trades, False otherwise
DETAILED_ANALYSIS_MINUTES = 20  # Number of minutes to analyze in detail after detection

# Optional: Override trader settings
MIN_CONFIDENCE = 0.72  # Minimum pattern confidence
PROFIT_TARGET_PCT = 20.0  # Profit target percentage
TRAILING_STOP_PCT = 7.0  # Trailing stop percentage

# Capital settings (matching live bot)
INITIAL_CAPITAL = 2000.0  # Starting capital in USD (matching realtime logic)
POSITION_SIZE_PCT = 1.0  # Percentage of capital to use per trade (100% - use full $2000)

# ============================================================================
# END CONFIGURATION
# ============================================================================

# Configure logging
et_tz = pytz.timezone('US/Eastern')
detection_date_obj = datetime.strptime(DETECTION_DATE, '%Y-%m-%d').date()

# Parse detection time
detection_hour, detection_minute = map(int, DETECTION_TIME.split(':'))
detection_datetime = datetime.combine(detection_date_obj, datetime.min.time().replace(hour=detection_hour, minute=detection_minute))

# Create absolute path for log file (in analysis directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
log_filename = os.path.join(script_dir, f"{TICKER}_SIMULATION_LOG_{detection_date_obj.strftime('%Y%m%d')}.log")

# Ensure the directory exists
os.makedirs(script_dir, exist_ok=True)

# Clear any existing handlers to avoid conflicts
logging.root.handlers = []

# Set up logging with both file and console handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True  # Force reconfiguration
)
logger = logging.getLogger(__name__)
logger.info(f"Logging initialized - Log file: {log_filename}")

def download_and_save_data(ticker, api, test_data_dir, start_hour=4, target_date=None):
    """Download 1-minute data from start_hour and save to test_data folder"""
    
    et = pytz.timezone('US/Eastern')
    if target_date is None:
        target_date = datetime.now(et).date()
    elif isinstance(target_date, str):
        target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
    
    logger.info(f"Downloading full day data for {ticker} from {start_hour}:00 AM on {target_date}...")
    
    try:
        # Fetch data (get enough minutes to cover full day + after hours)
        df = api.get_1min_data(ticker, minutes=1200)
        
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
        
        # Filter to target date and from start_hour
        df_target = df[df['date'] == target_date].copy()
        df_target['hour'] = df_target['timestamp'].dt.hour
        df_target = df_target[df_target['hour'] >= start_hour].copy()
        
        # Keep ALL data from start_hour (don't filter to detection time yet)
        # We need historical data for indicator calculations
        logger.info(f"Downloaded data from {start_hour}:00 AM (keeping all data for indicator calculations)")
        
        if df_target.empty:
            logger.warning(f"No data for {ticker} from {start_hour}:00 on {target_date}")
            return None
        
        # Prepare data for saving
        df_save = df_target[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        df_save = df_save.sort_values('timestamp').reset_index(drop=True)
        
        # Save to test_data folder
        filename = f"{ticker}_1min_{target_date.strftime('%Y%m%d')}.csv"
        filepath = os.path.join(test_data_dir, filename)
        df_save.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df_save)} minutes of data to {filepath}")
        logger.info(f"Time range: {df_save['timestamp'].min()} to {df_save['timestamp'].max()}")
        
        return df_save
        
    except Exception as e:
        logger.error(f"Error downloading data for {ticker}: {e}", exc_info=True)
        return None

def load_existing_data(ticker, test_data_dir, target_date=None):
    """Load existing data from test_data folder if available"""
    
    et = pytz.timezone('US/Eastern')
    if target_date is None:
        target_date = datetime.now(et).date()
    elif isinstance(target_date, str):
        target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
    
    filename = f"{ticker}_1min_{target_date.strftime('%Y%m%d')}.csv"
    filepath = os.path.join(test_data_dir, filename)
    
    if os.path.exists(filepath):
        logger.info(f"Loading existing data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('US/Eastern')
        
        # Filter to start_hour
        df['hour'] = df['timestamp'].dt.hour
        df = df[df['hour'] >= START_HOUR].copy()
        
        # Keep ALL data from start_hour
        logger.info(f"Loaded data from {START_HOUR}:00 AM (keeping all data for indicator calculations)")
        
        if df.empty:
            logger.warning(f"No data for {ticker} from {START_HOUR}:00 in existing file")
            return None
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} minutes of data from {filepath}")
        logger.info(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    return None

def simulate_bot_trading(ticker, df_data, trader, detection_time, detailed_minutes=20):
    """Simulate bot trading on the data
    
    Args:
        ticker: Stock symbol
        df_data: DataFrame with price data
        trader: RealtimeTrader instance
        detection_time: Datetime when ticker was detected
        detailed_minutes: Number of minutes to analyze in detail after detection
    
    Returns:
        tuple: (trades list, rejected_entries list)
    """
    
    et = pytz.timezone('US/Eastern')
    
    # Ensure entry_time is timezone-aware when creating positions
    def normalize_datetime(dt):
        """Normalize datetime to timezone-aware"""
        if dt is None:
            return None
        if dt.tzinfo is None:
            return et.localize(dt)
        return dt
    
    logger.info(f"Simulating bot trading for {ticker}...")
    
    # Initialize capital tracking (matching live bot logic)
    current_capital = INITIAL_CAPITAL
    position_size_pct = POSITION_SIZE_PCT
    logger.info(f"Initial Capital: ${current_capital:,.2f}")
    logger.info(f"Position Size: {position_size_pct*100:.0f}% per trade")
    
    # Prepare DataFrame - keep ALL data for indicator calculations
    df_all = df_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df_all = df_all.sort_values('timestamp').reset_index(drop=True)
    
    # Filter to start simulation loop from detection_time
    # But keep all earlier data for indicator calculations
    if detection_time.tzinfo is None:
        detection_time_et = et.localize(detection_time)
    else:
        detection_time_et = detection_time.astimezone(et)
    
    df_simulation = df_all[df_all['timestamp'] >= detection_time_et].copy()
    
    logger.info(f"Total data points: {len(df_all)} (from {df_all['timestamp'].min()} to {df_all['timestamp'].max()})")
    logger.info(f"Simulation will start from: {detection_time_et.strftime('%H:%M:%S')} ({len(df_simulation)} data points to simulate)")
    logger.info(f"This ensures we have {len(df_all[df_all['timestamp'] < detection_time_et])} bars of historical data for indicator calculations")
    
    # Track trades
    trades = []
    
    # Track rejected entries with timestamps
    rejected_entries = []
    current_time_ref = [None]  # Use list to allow modification in closure
    position_entry_times = {}  # Track actual entry times for positions (ticker -> entry_time)
    
    def rejection_callback(ticker, price, reason):
        """Callback to track rejected entries"""
        if current_time_ref[0]:
            rejected_entries.append({
                'ticker': ticker,
                'timestamp': current_time_ref[0],
                'price': price,
                'reason': reason
            })
    
    # Override rejection callback
    trader.rejection_callback = rejection_callback
    
    # Process data minute by minute starting from detection_time
    start_idx = None
    
    # Find the index where simulation starts (detection_time)
    for i in range(len(df_all)):
        if df_all.iloc[i]['timestamp'] >= detection_time_et:
            start_idx = i
            break
    
    if start_idx is None:
        logger.warning(f"Could not find start time {detection_time_et} in data, starting from beginning")
        start_idx = 0
    
    # Check how many bars we have before the start time
    bars_before_start = start_idx
    min_bars_for_patterns = 30  # Original entry logic needs 30+ bars
    min_bars_for_surge = 4  # Surge detection needs only 4 bars
    
    if bars_before_start < min_bars_for_surge:
        logger.warning(f"Only {bars_before_start} bars before start time {detection_time_et}. Minimum {min_bars_for_surge} needed for surge detection.")
        logger.warning(f"Starting from index {min_bars_for_surge} instead of {start_idx}")
        start_idx = min_bars_for_surge
    elif bars_before_start < min_bars_for_patterns:
        logger.info(f"Starting from {detection_time_et} with {bars_before_start} bars of historical data")
        logger.info(f"Pattern detection may be limited, but surge detection will work")
    else:
        logger.info(f"Starting from {detection_time_et} with {bars_before_start} bars of historical data (sufficient for full analysis)")
    
    logger.info(f"Starting simulation from index {start_idx} (timestamp: {df_all.iloc[start_idx]['timestamp']})")
    
    # Detailed logging for detection time period
    logger.info(f"\n{'='*80}")
    logger.info(f"CHECKING {detection_time_et.strftime('%H:%M')} DETECTION FOR {ticker}")
    if LIVE_BOT_HAD_TRADES:
        logger.info(f"NOTE: Live bot had trades - comparing simulation vs live results")
    else:
        logger.info(f"NOTE: Live bot had NO trades - checking why entry was missed")
    logger.info(f"{'='*80}")
    
    # Process first N minutes in detail
    detailed_end_idx = min(start_idx + detailed_minutes, len(df_all))
    
    for i in range(start_idx, detailed_end_idx):
        current_time_ref[0] = df_all.iloc[i]['timestamp']
        current_price = df_all.iloc[i]['close']
        current_volume = df_all.iloc[i]['volume']
        
        time_str = current_time_ref[0].strftime('%H:%M:%S')
        logger.info(f"\n[{time_str}] ===== ANALYSIS =====")
        logger.info(f"[{time_str}] Price: ${current_price:.4f}, Volume: {current_volume:,.0f}")
        
        # Get all data up to current point (for indicator calculations)
        df_up_to_now = df_all.iloc[:i+1].copy()
        
        # Check if we have an active position
        has_position = ticker in trader.active_positions
        if has_position:
            position = trader.active_positions[ticker]
            logger.info(f"[{time_str}] Active Position: Entry @ ${position.entry_price:.4f}, Current P&L: {position.unrealized_pnl_pct:+.2f}%")
        else:
            logger.info(f"[{time_str}] No active position - checking for entry signals...")
        
        # Analyze for entry/exit signals
        entry_signal, exit_signals = trader.analyze_data(df_up_to_now, ticker, current_price=current_price)
        
        # Process exit signals first
        if exit_signals:
            for exit_signal in exit_signals:
                if ticker in trader.active_positions:
                    position = trader.active_positions[ticker]
                    entry_price = position.entry_price
                    entry_time = position.entry_time
                    exit_price = exit_signal.price
                    exit_time = current_time_ref[0]
                    
                    # Use tracked entry time if available (more accurate than position.entry_time)
                    if ticker in position_entry_times:
                        entry_time = position_entry_times[ticker]
                    else:
                        entry_time = position.entry_time
                    
                    # Normalize timezones - ensure both are timezone-aware
                    if exit_time.tzinfo is None:
                        exit_time = et.localize(exit_time)
                    elif entry_time.tzinfo is None:
                        entry_time = et.localize(entry_time)
                    elif exit_time.tzinfo != entry_time.tzinfo:
                        # Convert both to same timezone
                        entry_time = entry_time.astimezone(exit_time.tzinfo)
                    
                    # Calculate P&L and update capital (matching live bot logic)
                    shares = position.shares if hasattr(position, 'shares') and position.shares > 0 else 0
                    entry_value_stored = position.entry_value if hasattr(position, 'entry_value') and position.entry_value > 0 else 0
                    
                    if shares == 0 or entry_value_stored == 0:
                        # Calculate shares from current capital if not set
                        position_value = current_capital * position_size_pct
                        shares = round(position_value / entry_price) if entry_price > 0 else 0
                        entry_value_stored = shares * entry_price
                    
                    entry_value = entry_value_stored  # Use stored entry value
                    exit_value = shares * exit_price
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                    pnl_dollars = exit_value - entry_value
                    hold_minutes = (exit_time - entry_time).total_seconds() / 60
                    
                    # Update capital (matching live bot logic)
                    # When exiting: add exit_value back to capital (we already deducted entry_value when entering)
                    current_capital += exit_value
                    
                    trades.append({
                        'ticker': ticker,
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'shares': shares,
                        'entry_value': entry_value,
                        'exit_value': exit_value,
                        'pnl_pct': pnl_pct,
                        'pnl_dollars': pnl_dollars,
                        'hold_minutes': hold_minutes,
                        'entry_pattern': position.entry_pattern,
                        'exit_reason': exit_signal.reason,
                        'entry_confidence': position.entry_confidence * 100 if position.entry_confidence else 0
                    })
                    
                    logger.info(f"\n{'='*60}")
                    logger.info(f"EXIT SIGNAL at {exit_time}")
                    logger.info(f"  Entry: ${entry_price:.4f} @ {entry_time} ({shares} shares, ${entry_value:,.2f})")
                    logger.info(f"  Exit: ${exit_price:.4f} @ {exit_time} (${exit_value:,.2f})")
                    logger.info(f"  P&L: {pnl_pct:+.2f}% (${pnl_dollars:+,.2f})")
                    logger.info(f"  Hold Time: {hold_minutes:.1f} minutes")
                    logger.info(f"  Capital: ${current_capital:,.2f}")
                    logger.info(f"  Pattern: {position.entry_pattern}")
                    logger.info(f"  Exit Reason: {exit_signal.reason}")
                    logger.info(f"{'='*60}\n")
                    
                    trader.exit_position(exit_signal)
                    # Remove from tracked entry times since position is closed
                    if ticker in position_entry_times:
                        del position_entry_times[ticker]
        
        # Process entry signals
        if entry_signal:
            logger.info(f"\n{'='*60}")
            logger.info(f"ENTRY SIGNAL DETECTED at {current_time_ref[0]}")
            logger.info(f"  Price: ${current_price:.4f}")
            logger.info(f"  Pattern: {entry_signal.pattern_name}")
            logger.info(f"  Confidence: {entry_signal.confidence*100:.1f}%")
            logger.info(f"  Target: ${entry_signal.target_price:.4f}" if entry_signal.target_price else "  Target: N/A")
            logger.info(f"  Stop Loss: ${entry_signal.stop_loss:.4f}" if entry_signal.stop_loss else "  Stop Loss: N/A")
            logger.info(f"{'='*60}\n")
            
            if ticker not in trader.active_positions:
                # Check capital and calculate position size (matching live bot logic)
                position_value = current_capital * position_size_pct
                
                if position_value < 100:  # Minimum $100 to trade
                    logger.warning(f"[REJECTED] Entry: {ticker} @ ${current_price:.4f} - Insufficient capital (${current_capital:.2f} < $100)")
                    continue
                
                # Calculate shares (round to whole number)
                shares = round(position_value / current_price)
                entry_value = shares * current_price
                
                position = trader.enter_position(entry_signal, df=df_up_to_now)
                if position:
                    # Set shares and entry_value in position (matching live bot)
                    position.shares = shares
                    position.entry_value = entry_value
                    
                    # Track the actual entry time from simulation (not from position.entry_time which might be wrong)
                    position_entry_times[ticker] = current_time_ref[0]
                    
                    # Update capital (deduct entry value)
                    current_capital -= entry_value
                    
                    logger.info(f"Position entered: {ticker} @ ${current_price:.4f} ({shares} shares, ${entry_value:,.2f})")
                    logger.info(f"  Capital remaining: ${current_capital:,.2f}")
        else:
            logger.info(f"[{time_str}] No entry signal detected")
    
    # Continue with rest of simulation (less detailed logging)
    for i in range(detailed_end_idx, len(df_all)):
        current_time_ref[0] = df_all.iloc[i]['timestamp']
        current_price = df_all.iloc[i]['close']
        
        # Get all data up to current point (for indicator calculations)
        df_up_to_now = df_all.iloc[:i+1].copy()
        
        # Analyze for entry/exit signals
        entry_signal, exit_signals = trader.analyze_data(df_up_to_now, ticker, current_price=current_price)
        
        # Process exit signals first
        if exit_signals:
            for exit_signal in exit_signals:
                if ticker in trader.active_positions:
                    position = trader.active_positions[ticker]
                    entry_price = position.entry_price
                    # Use tracked entry time if available (more accurate than position.entry_time)
                    if ticker in position_entry_times:
                        entry_time = position_entry_times[ticker]
                    else:
                        entry_time = position.entry_time
                    exit_price = exit_signal.price
                    exit_time = current_time_ref[0]
                    
                    # Normalize timezones - ensure both are timezone-aware
                    if exit_time.tzinfo is None:
                        exit_time = et.localize(exit_time)
                    elif entry_time.tzinfo is None:
                        entry_time = et.localize(entry_time)
                    elif exit_time.tzinfo != entry_time.tzinfo:
                        # Convert both to same timezone
                        entry_time = entry_time.astimezone(exit_time.tzinfo)
                    
                    # Calculate P&L and update capital (matching live bot logic)
                    shares = position.shares if hasattr(position, 'shares') and position.shares > 0 else 0
                    entry_value_stored = position.entry_value if hasattr(position, 'entry_value') and position.entry_value > 0 else 0
                    
                    if shares == 0 or entry_value_stored == 0:
                        # Calculate shares from current capital if not set
                        position_value = current_capital * position_size_pct
                        shares = round(position_value / entry_price) if entry_price > 0 else 0
                        entry_value_stored = shares * entry_price
                    
                    entry_value = entry_value_stored  # Use stored entry value
                    exit_value = shares * exit_price
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                    pnl_dollars = exit_value - entry_value
                    hold_minutes = (exit_time - entry_time).total_seconds() / 60
                    
                    # Update capital (matching live bot logic)
                    # When exiting: add exit_value back to capital (we already deducted entry_value when entering)
                    current_capital += exit_value
                    
                    trades.append({
                        'ticker': ticker,
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'shares': shares,
                        'entry_value': entry_value,
                        'exit_value': exit_value,
                        'pnl_pct': pnl_pct,
                        'pnl_dollars': pnl_dollars,
                        'hold_minutes': hold_minutes,
                        'entry_pattern': position.entry_pattern,
                        'exit_reason': exit_signal.reason,
                        'entry_confidence': position.entry_confidence * 100 if position.entry_confidence else 0
                    })
                    
                    logger.info(f"\n{'='*60}")
                    logger.info(f"EXIT SIGNAL at {exit_time}")
                    logger.info(f"  Entry: ${entry_price:.4f} @ {entry_time} ({shares} shares, ${entry_value:,.2f})")
                    logger.info(f"  Exit: ${exit_price:.4f} @ {exit_time} (${exit_value:,.2f})")
                    logger.info(f"  P&L: {pnl_pct:+.2f}% (${pnl_dollars:+,.2f})")
                    logger.info(f"  Hold Time: {hold_minutes:.1f} minutes")
                    logger.info(f"  Capital: ${current_capital:,.2f}")
                    logger.info(f"  Pattern: {position.entry_pattern}")
                    logger.info(f"  Exit Reason: {exit_signal.reason}")
                    logger.info(f"{'='*60}\n")
                    
                    trader.exit_position(exit_signal)
                    # Remove from tracked entry times since position is closed
                    if ticker in position_entry_times:
                        del position_entry_times[ticker]
        
        # Process entry signals
        if entry_signal and ticker not in trader.active_positions:
            logger.info(f"\n{'='*60}")
            logger.info(f"ENTRY SIGNAL DETECTED at {current_time_ref[0]}")
            logger.info(f"  Price: ${current_price:.4f}")
            logger.info(f"  Pattern: {entry_signal.pattern_name}")
            logger.info(f"  Confidence: {entry_signal.confidence*100:.1f}%")
            logger.info(f"  Target: ${entry_signal.target_price:.4f}" if entry_signal.target_price else "  Target: N/A")
            logger.info(f"  Stop Loss: ${entry_signal.stop_loss:.4f}" if entry_signal.stop_loss else "  Stop Loss: N/A")
            logger.info(f"{'='*60}\n")
            
            # Check capital and calculate position size (matching live bot logic)
            position_value = current_capital * position_size_pct
            
            if position_value < 100:  # Minimum $100 to trade
                logger.warning(f"[REJECTED] Entry: {ticker} @ ${current_price:.4f} - Insufficient capital (${current_capital:.2f} < $100)")
            else:
                # Calculate shares (round to whole number)
                shares = round(position_value / current_price)
                entry_value = shares * current_price
                
                position = trader.enter_position(entry_signal, df=df_up_to_now)
                if position:
                    # Set shares and entry_value in position (matching live bot)
                    position.shares = shares
                    position.entry_value = entry_value
                    
                    # Track the actual entry time from simulation (not from position.entry_time which might be wrong)
                    position_entry_times[ticker] = current_time_ref[0]
                    
                    # Update capital (deduct entry value)
                    current_capital -= entry_value
                    
                    logger.info(f"Position entered: {ticker} @ ${current_price:.4f} ({shares} shares, ${entry_value:,.2f})")
                    logger.info(f"  Capital remaining: ${current_capital:,.2f}")
    
    # Close any remaining positions at end of simulation
    if ticker in trader.active_positions:
        position = trader.active_positions[ticker]
        current_price = df_all.iloc[-1]['close']
        current_time = df_all.iloc[-1]['timestamp']
        entry_price = position.entry_price
        
        # Use tracked entry time if available (more accurate than position.entry_time)
        if ticker in position_entry_times:
            entry_time = position_entry_times[ticker]
            logger.debug(f"Using tracked entry time {entry_time} for {ticker}")
        else:
            entry_time = position.entry_time
            # If entry_time looks wrong (has microseconds suggesting it's system time),
            # use a fallback
            if entry_time and entry_time.microsecond > 0 and entry_time.hour >= 20:
                logger.warning(f"Entry time {entry_time} looks incorrect (system time?), using current_time - 1 hour as estimate")
                entry_time = current_time - timedelta(hours=1)
        
        # Normalize timezones - ensure both are timezone-aware
        if current_time.tzinfo is None:
            current_time = et.localize(current_time)
        elif entry_time.tzinfo is None:
            entry_time = et.localize(entry_time)
        elif current_time.tzinfo != entry_time.tzinfo:
            # Convert both to same timezone
            entry_time = entry_time.astimezone(current_time.tzinfo)
        
        # Final check: ensure entry_time is before current_time
        if entry_time > current_time:
            logger.warning(f"Entry time {entry_time} is after exit time {current_time}, adjusting entry_time")
            # Set entry_time to be 1 hour before current_time as fallback
            entry_time = current_time - timedelta(hours=1)
        
        # Calculate P&L and update capital (matching live bot logic)
        shares = position.shares if hasattr(position, 'shares') and position.shares > 0 else 0
        entry_value_stored = position.entry_value if hasattr(position, 'entry_value') and position.entry_value > 0 else 0
        
        if shares == 0 or entry_value_stored == 0:
            # Calculate shares from current capital if not set
            position_value = current_capital * position_size_pct
            shares = round(position_value / entry_price) if entry_price > 0 else 0
            entry_value_stored = shares * entry_price
        
        entry_value = entry_value_stored  # Use stored entry value
        exit_value = shares * current_price
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        pnl_dollars = exit_value - entry_value
        hold_minutes = (current_time - entry_time).total_seconds() / 60
        
        # Update capital (matching live bot logic)
        # When exiting: add exit_value back to capital (we already deducted entry_value when entering)
        current_capital += exit_value
        
        trades.append({
            'ticker': ticker,
            'entry_time': entry_time,
            'exit_time': current_time,
            'entry_price': entry_price,
            'exit_price': current_price,
            'shares': shares,
            'entry_value': entry_value,
            'exit_value': exit_value,
            'pnl_pct': pnl_pct,
            'pnl_dollars': pnl_dollars,
            'hold_minutes': hold_minutes,
            'entry_pattern': position.entry_pattern,
            'exit_reason': 'Position closed at end of simulation',
            'entry_confidence': position.entry_confidence * 100 if position.entry_confidence else 0
        })
        
        logger.info(f"\n{'='*60}")
        logger.info(f"POSITION CLOSED AT END OF SIMULATION")
        logger.info(f"  Entry: ${entry_price:.4f} @ {entry_time} ({shares} shares, ${entry_value:,.2f})")
        logger.info(f"  Exit: ${current_price:.4f} @ {current_time} (${exit_value:,.2f})")
        logger.info(f"  P&L: {pnl_pct:+.2f}% (${pnl_dollars:+,.2f})")
        logger.info(f"  Hold Time: {hold_minutes:.1f} minutes")
        logger.info(f"  Final Capital: ${current_capital:,.2f}")
        logger.info(f"{'='*60}\n")
    
    return trades, rejected_entries

def main():
    """Main simulation function"""
    
    et = pytz.timezone('US/Eastern')
    
    logger.info(f"Starting simulation for {TICKER} - Log file: {log_filename}")
    logger.info(f"Configuration:")
    logger.info(f"  Ticker: {TICKER}")
    logger.info(f"  Detection Date: {DETECTION_DATE}")
    logger.info(f"  Detection Time: {DETECTION_TIME} ET")
    logger.info(f"  Start Hour: {START_HOUR}:00 AM")
    logger.info(f"  Live Bot Had Trades: {LIVE_BOT_HAD_TRADES}")
    logger.info(f"  Detailed Analysis: First {DETAILED_ANALYSIS_MINUTES} minutes after detection")
    
    # Initialize components
    api = WebullDataAPI()
    db = TradingDatabase()
    
    # Create test_data directory if it doesn't exist (from src/test/ go up to project root)
    test_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'test_data')
    os.makedirs(test_data_dir, exist_ok=True)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Simulating {TICKER} from {DETECTION_TIME} ET on {DETECTION_DATE}")
    logger.info(f"Note: Starting from {DETECTION_TIME} when ticker was identified")
    logger.info(f"{'='*80}")
    
    # Try loading existing data first
    df_data = load_existing_data(TICKER, test_data_dir, DETECTION_DATE)
    
    # If no existing data, download fresh
    if df_data is None or df_data.empty:
        logger.info(f"Downloading fresh data for {TICKER}...")
        df_data = download_and_save_data(TICKER, api, test_data_dir, START_HOUR, DETECTION_DATE)
    
    if df_data is None or df_data.empty:
        logger.warning(f"Skipping {TICKER} - no data available")
        return
    
    # Create new trader instance to reset state
    trader = RealtimeTrader(
        min_confidence=MIN_CONFIDENCE,
        profit_target_pct=PROFIT_TARGET_PCT,
        trailing_stop_pct=TRAILING_STOP_PCT
    )
    
    # Use module-level detection_datetime (may be overridden by external scripts)
    sim_detection_datetime = globals().get('detection_datetime', detection_datetime)
    
    # Simulate bot trading
    trades, rejected_entries = simulate_bot_trading(
        TICKER, 
        df_data, 
        trader, 
        sim_detection_datetime,
        DETAILED_ANALYSIS_MINUTES
    )
    
    # Print rejected entries summary
    logger.info(f"\n{'='*80}")
    logger.info(f"REJECTED ENTRIES: {len(rejected_entries)}")
    logger.info(f"{'='*80}")
    if rejected_entries:
        # Group by reason
        reasons_count = {}
        for rej in rejected_entries:
            reason = rej['reason']
            if reason not in reasons_count:
                reasons_count[reason] = []
            reasons_count[reason].append(rej)
        
        for reason, rej_list in reasons_count.items():
            logger.info(f"\n{reason}: {len(rej_list)} occurrences")
            # Show first few examples
            for rej in rej_list[:3]:
                logger.info(f"  {rej['timestamp']} @ ${rej['price']:.4f}")
            if len(rej_list) > 3:
                logger.info(f"  ... and {len(rej_list) - 3} more")
    
    # Print results
    logger.info(f"\n{'='*80}")
    logger.info(f"SIMULATION RESULTS: {TICKER} from {DETECTION_TIME} on {DETECTION_DATE}")
    logger.info(f"{'='*80}\n")
    
    if trades:
        logger.info(f"TRADES PLACED: {len(trades)}")
        
        df_trades = pd.DataFrame(trades)
        
        total_trades = len(df_trades)
        wins = len(df_trades[df_trades['pnl_pct'] > 0])
        losses = len(df_trades[df_trades['pnl_pct'] < 0])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = df_trades['pnl_pct'].sum()
        avg_pnl = df_trades['pnl_pct'].mean()
        
        # Calculate total P&L in dollars
        total_pnl_dollars = df_trades['pnl_dollars'].sum() if 'pnl_dollars' in df_trades.columns else 0
        final_capital = INITIAL_CAPITAL + total_pnl_dollars
        total_return_pct = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        
        logger.info(f"\nTRADE SUMMARY:")
        logger.info(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
        logger.info(f"  Final Capital: ${final_capital:,.2f}")
        logger.info(f"  Total Return: {total_return_pct:+.2f}%")
        logger.info(f"  Total Trades: {total_trades}")
        logger.info(f"  Wins: {wins} ({wins/total_trades*100:.1f}%)")
        logger.info(f"  Losses: {losses} ({losses/total_trades*100:.1f}%)")
        logger.info(f"  Win Rate: {win_rate:.1f}%")
        logger.info(f"  Total P&L: {total_pnl:+.2f}% (${total_pnl_dollars:+,.2f})")
        logger.info(f"  Average P&L: {avg_pnl:+.2f}%")
        
        # Export to CSV (from src/test/ go up to project root, then to analysis/)
        time_str = DETECTION_TIME.replace(':', '')
        analysis_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        csv_file = os.path.join(analysis_dir, f"BOT_SIMULATION_{TICKER}_{time_str}_{detection_date_obj.strftime('%Y%m%d')}.csv")
        df_trades.to_csv(csv_file, index=False)
        logger.info(f"\nTrades exported to: {csv_file}")
        
        # Print individual trades
        logger.info(f"\n{'='*80}")
        logger.info("INDIVIDUAL TRADES:")
        logger.info(f"{'='*80}")
        for i, trade in enumerate(trades, 1):
            logger.info(f"\nTrade #{i}:")
            logger.info(f"  Entry: {trade['entry_time']} @ ${trade['entry_price']:.4f}")
            logger.info(f"  Exit: {trade['exit_time']} @ ${trade['exit_price']:.4f}")
            if 'shares' in trade:
                logger.info(f"  Shares: {trade['shares']}")
                logger.info(f"  Entry Value: ${trade.get('entry_value', 0):,.2f}")
                logger.info(f"  Exit Value: ${trade.get('exit_value', 0):,.2f}")
            if 'pnl_dollars' in trade:
                logger.info(f"  P&L: {trade['pnl_pct']:+.2f}% (${trade['pnl_dollars']:+,.2f})")
            else:
                logger.info(f"  P&L: {trade['pnl_pct']:+.2f}%")
            logger.info(f"  Hold Time: {trade['hold_minutes']:.1f} minutes")
            logger.info(f"  Pattern: {trade['entry_pattern']}")
            logger.info(f"  Exit Reason: {trade['exit_reason']}")
    else:
        logger.info(f"NO TRADES PLACED FOR {TICKER}")
        logger.info(f"Rejected entries: {len(rejected_entries)}")
        
        if rejected_entries:
            logger.info(f"\nEntry signals were detected but all were rejected.")
            logger.info(f"Check rejected entries above to see why.")
        else:
            logger.info(f"\nNo entry signals were detected for {TICKER} after {DETECTION_TIME}.")
        
        if LIVE_BOT_HAD_TRADES:
            logger.warning(f"\nWARNING: Live bot had trades but simulation found none!")
            logger.warning(f"This suggests a discrepancy between live and simulation logic.")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Simulation complete. Log saved to: {log_filename}")
    logger.info(f"{'='*80}\n")

if __name__ == '__main__':
    main()
