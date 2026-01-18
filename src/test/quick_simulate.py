"""
Quick Simulation Runner
======================

This script allows you to run simulations quickly by passing parameters
as command-line arguments instead of editing the template file.

Usage:
    python src/test/quick_simulate.py PRFX 2026-01-15 16:05 True
    python src/test/quick_simulate.py VERO 2026-01-15 16:31 True
    python src/test/quick_simulate.py SPHL 2026-01-15 04:00 False

Arguments:
    ticker: Stock symbol (required)
    date: Detection date in YYYY-MM-DD format (required)
    time: Detection time in HH:MM format, ET (required)
    had_trades: True/False - whether live bot had trades (optional, default: True)
    detailed_minutes: Minutes to analyze in detail (optional, default: 20)
"""
import sys
import os
import argparse

# Add both root and src to path
# From src/test/, go up two levels to project root, then src is one level up
root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
src_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, root_dir)
sys.path.insert(0, src_dir)

# Import the template module to modify config (both files are in same directory)
import simulate_ticker_template as template_module

def run_quick_simulation(ticker, date, time, had_trades=True, detailed_minutes=20, 
                        start_hour=4, min_confidence=0.72, profit_target=20.0, trailing_stop=7.0):
    """Run a quick simulation with provided parameters"""
    
    # Update template module configuration
    template_module.TICKER = ticker
    template_module.DETECTION_DATE = date
    template_module.DETECTION_TIME = time
    template_module.START_HOUR = start_hour
    template_module.LIVE_BOT_HAD_TRADES = had_trades
    template_module.DETAILED_ANALYSIS_MINUTES = detailed_minutes
    template_module.MIN_CONFIDENCE = min_confidence
    template_module.PROFIT_TARGET_PCT = profit_target
    template_module.TRAILING_STOP_PCT = trailing_stop
    
    # Parse detection time and update module variables
    from datetime import datetime
    import pytz
    
    detection_hour, detection_minute = map(int, time.split(':'))
    detection_date_obj = datetime.strptime(date, '%Y-%m-%d').date()
    detection_datetime = datetime.combine(
        detection_date_obj, 
        datetime.min.time().replace(hour=detection_hour, minute=detection_minute)
    )
    
    # Update module-level variables
    template_module.detection_datetime = detection_datetime
    template_module.detection_date_obj = detection_date_obj
    template_module.et_tz = pytz.timezone('US/Eastern')
    
    # Update log filename
    import pytz
    et_tz = pytz.timezone('US/Eastern')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_filename = os.path.join(script_dir, f"{ticker}_SIMULATION_LOG_{detection_date_obj.strftime('%Y%m%d')}.log")
    template_module.log_filename = log_filename
    
    # Run the simulation using the template's main function
    template_module.main()

def main():
    """Parse command-line arguments and run simulation"""
    parser = argparse.ArgumentParser(
        description='Quick Trading Bot Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Required Arguments (in order):
  1. TICKER    - Stock symbol (e.g., PRFX, VERO, SPHL)
  2. DATE      - Detection date in YYYY-MM-DD format (e.g., 2026-01-15)
  3. TIME      - Detection time in HH:MM format, ET (e.g., 16:05)

Optional Arguments:
  4. HAD_TRADES - Whether live bot had trades (True/False, default: True)

Examples:
  python quick_simulate.py PRFX 2026-01-15 16:05 True
  python quick_simulate.py VERO 2026-01-15 16:31 True
  python quick_simulate.py SPHL 2026-01-15 04:00 False
  python quick_simulate.py PRFX 2026-01-15 16:05 True --detailed-minutes 30
        """
    )
    
    parser.add_argument('ticker', type=str, help='Stock symbol (e.g., PRFX)')
    parser.add_argument('date', type=str, help='Detection date in YYYY-MM-DD format')
    parser.add_argument('time', type=str, help='Detection time in HH:MM format, ET (e.g., 16:05)')
    parser.add_argument('had_trades', type=str, nargs='?', default='True',
                       help='Whether live bot had trades (True/False, default: True)')
    parser.add_argument('--detailed-minutes', type=int, default=20,
                       help='Minutes to analyze in detail (default: 20)')
    parser.add_argument('--start-hour', type=int, default=4,
                       help='Hour to start data collection from (default: 4)')
    parser.add_argument('--min-confidence', type=float, default=0.72,
                       help='Minimum pattern confidence (default: 0.72)')
    parser.add_argument('--profit-target', type=float, default=20.0,
                       help='Profit target percentage (default: 20.0)')
    parser.add_argument('--trailing-stop', type=float, default=7.0,
                       help='Trailing stop percentage (default: 7.0)')
    
    args = parser.parse_args()
    
    # Check if ticker looks like it might be a date (common mistake - missing ticker)
    if args.ticker and '-' in args.ticker and len(args.ticker.split('-')) == 3:
        # Check if it's a valid date format
        try:
            from datetime import datetime
            datetime.strptime(args.ticker, '%Y-%m-%d')
            # If we get here, ticker is actually a date - user forgot ticker!
            print(f"\n{'='*80}")
            print(f"ERROR: Missing TICKER argument!")
            print(f"{'='*80}")
            print(f"\nYou provided: {args.ticker} {args.date} {args.time}")
            print(f"\nCorrect usage: python quick_simulate.py TICKER DATE TIME [HAD_TRADES]")
            print(f"\nExample:")
            print(f"  python quick_simulate.py PRFX 2026-01-15 16:05 True")
            print(f"\nIt looks like you forgot the ticker. Did you mean:")
            print(f"  python quick_simulate.py <TICKER> {args.ticker} {args.date} {args.time}")
            print(f"\nWhere <TICKER> is the stock symbol (e.g., PRFX, VERO, SPHL)")
            print(f"{'='*80}\n")
            sys.exit(1)
        except ValueError:
            # Ticker is not a valid date format, so it's probably fine
            pass
    
    # Convert had_trades string to boolean
    had_trades = args.had_trades.lower() in ('true', '1', 'yes', 'y') if args.had_trades else True
    
    # Validate date format
    try:
        from datetime import datetime
        datetime.strptime(args.date, '%Y-%m-%d')
    except ValueError:
        print(f"\nError: Invalid date format '{args.date}'. Use YYYY-MM-DD format.")
        print(f"Example: 2026-01-15")
        print(f"\nUsage: python quick_simulate.py TICKER DATE TIME [HAD_TRADES]")
        print(f"Example: python quick_simulate.py PRFX 2026-01-15 16:05 True")
        sys.exit(1)
    
    # Validate time format
    try:
        hour, minute = map(int, args.time.split(':'))
        if not (0 <= hour < 24 and 0 <= minute < 60):
            raise ValueError
    except (ValueError, AttributeError):
        print(f"\nError: Invalid time format '{args.time}'. Use HH:MM format (e.g., 16:05).")
        print(f"\nUsage: python quick_simulate.py TICKER DATE TIME [HAD_TRADES]")
        print(f"Example: python quick_simulate.py PRFX 2026-01-15 16:05 True")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Quick Simulation Configuration")
    print(f"{'='*80}")
    print(f"Ticker: {args.ticker}")
    print(f"Date: {args.date}")
    print(f"Time: {args.time} ET")
    print(f"Live Bot Had Trades: {had_trades}")
    print(f"Detailed Analysis: {args.detailed_minutes} minutes")
    print(f"Start Hour: {args.start_hour}:00 AM")
    print(f"{'='*80}\n")
    
    # Run simulation
    run_quick_simulation(
        ticker=args.ticker,
        date=args.date,
        time=args.time,
        had_trades=had_trades,
        detailed_minutes=args.detailed_minutes,
        start_hour=args.start_hour,
        min_confidence=args.min_confidence,
        profit_target=args.profit_target,
        trailing_stop=args.trailing_stop
    )

if __name__ == '__main__':
    main()
