"""
Run Live Trading Bot
Production-ready entry point for live trading
"""
import sys
import logging
import threading
from core.live_trading_bot import LiveTradingBot
from data.api_interface import CSVDataAPI  # Replace with your live API
from web.trading_web_interface import set_trading_bot, run_web_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point"""
    
    # ============================================
    # CONFIGURATION - ADJUST THESE VALUES
    # ============================================
    
    INITIAL_CAPITAL = 10000.0
    TARGET_CAPITAL = 100000.0
    
    # Daily profit targets
    DAILY_PROFIT_TARGET_MIN = 500.0  # Daily profit target: $500 (5% of $10,000)
    DAILY_PROFIT_TARGET_MAX = 50000.0  # Fixed target: $500 per day
    
    # Trading window (Eastern Time) - 4 AM to 8 PM
    TRADING_START_TIME = "04:00"  # 4:00 AM ET (start of trading day)
    TRADING_END_TIME = "20:00"    # 8:00 PM ET (end of trading day)
    
    # Trading parameters (BALANCED for autonomous operation - quality with opportunities)
    MIN_CONFIDENCE = 0.72  # 72% - high-quality trades with reasonable opportunities
    MIN_ENTRY_PRICE_INCREASE = 5.5  # 5.5% expected gain - good quality setups
    TRAILING_STOP_PCT = 2.5  # 2.5% trailing stop - tighter stops, cut losses faster
    PROFIT_TARGET_PCT = 8.0  # 8% profit target - realistic profit target
    POSITION_SIZE_PCT = 0.50  # 50% of capital per trade - conservative sizing
    MAX_POSITIONS = 3  # Up to 3 positions at once
    MAX_LOSS_PER_TRADE_PCT = 2.5  # Max 2.5% loss per trade - cut losses faster
    
    # Autonomous safety limits
    MAX_TRADES_PER_DAY = 8  # Limit trades per day for quality
    MAX_DAILY_LOSS = -300.0  # Stop trading if daily loss exceeds this
    CONSECUTIVE_LOSS_LIMIT = 3  # Pause after N consecutive losses
    
    # ============================================
    # API SETUP - Webull API for Live Trading
    # ============================================
    
    # Use Webull API for live trading
    from data.webull_data_api import WebullDataAPI
    api = WebullDataAPI()
    
    # For testing with CSV files, uncomment:
    # from data.api_interface import CSVDataAPI
    # api = CSVDataAPI(data_dir="test_data")
    
    # ============================================
    # TICKER SELECTION
    # ============================================
    
    # Options:
    # 1. Pass as command line arguments: python run_live_bot.py TICKER1 TICKER2 ...
    # 2. Use 'auto' to fetch from top gainers (DEFAULT)
    # 3. Use 'swing' to fetch from swing screener
    # 4. Use 'manual TICKER1 TICKER2 ...' for specific tickers
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == 'auto':
            # Auto-fetch from multiple sources using StockDiscovery
            logger.info("Auto-discovering tickers from multiple sources (gainers, news, most active, etc.)...")
            try:
                from analysis.stock_discovery import StockDiscovery
                discovery = StockDiscovery(api)
                TICKERS = discovery.discover_stocks(
                    include_gainers=True,
                    include_news=True,
                    include_most_active=True,
                    include_unusual_volume=True,
                    include_breakouts=True,
                    include_reversals=False,
                    max_total=30
                )
                if not TICKERS:
                    logger.warning("No tickers discovered, trying top gainers only...")
                    TICKERS = api.get_stock_list_from_gainers(count=20)
                    if not TICKERS:
                        TICKERS = api.get_stock_list_from_swing_screener(count=20, min_price=5.0, max_price=100.0)
                logger.info(f"Discovered {len(TICKERS)} tickers from multiple sources: {TICKERS[:10]}...")
            except ImportError:
                logger.warning("StockDiscovery not available, using top gainers only...")
                try:
                    TICKERS = api.get_stock_list_from_gainers(count=20)
                    if not TICKERS:
                        TICKERS = api.get_stock_list_from_swing_screener(count=20, min_price=5.0, max_price=100.0)
                except:
                    TICKERS = ["YIBO", "TVGN", "NAOV"]
            except Exception as e:
                logger.error(f"Error discovering stocks: {e}")
                logger.info("Falling back to top gainers only...")
                try:
                    TICKERS = api.get_stock_list_from_gainers(count=20)
                    if not TICKERS:
                        TICKERS = ["YIBO", "TVGN", "NAOV"]
                except:
                    TICKERS = ["YIBO", "TVGN", "NAOV"]
        elif sys.argv[1].lower() == 'swing':
            # Auto-fetch from swing screener
            logger.info("Auto-fetching tickers from Webull swing screener...")
            try:
                TICKERS = api.get_stock_list_from_swing_screener(count=20, min_price=5.0, max_price=100.0)
                if not TICKERS:
                    logger.warning("No tickers returned from swing screener, trying top gainers...")
                    TICKERS = api.get_stock_list_from_gainers(count=20)
                logger.info(f"Fetched {len(TICKERS)} tickers from Webull API: {TICKERS[:10]}...")
            except Exception as e:
                logger.error(f"Error fetching from Webull API: {e}")
                logger.info("Falling back to default tickers...")
                TICKERS = ["YIBO", "TVGN", "NAOV"]
        elif sys.argv[1].lower() == 'manual':
            # Use provided tickers after 'manual' keyword
            if len(sys.argv) > 2:
                TICKERS = sys.argv[2:]
                logger.info(f"Using manually specified tickers: {TICKERS}")
            else:
                logger.error("No tickers provided after 'manual' keyword")
                logger.info("Usage: python run_live_bot.py manual TICKER1 TICKER2 ...")
                sys.exit(1)
        else:
            # Use provided tickers (backward compatibility)
            TICKERS = sys.argv[1:]
            logger.info(f"Using provided tickers: {TICKERS}")
    else:
        # DEFAULT: Auto-discover from multiple sources
        logger.info("No tickers specified - discovering from multiple sources (default behavior)...")
        try:
            from stock_discovery import StockDiscovery
            discovery = StockDiscovery(api)
            TICKERS = discovery.discover_stocks(
                include_gainers=True,
                include_news=True,
                include_most_active=True,
                include_unusual_volume=True,
                include_breakouts=True,
                include_reversals=False,
                max_total=30
            )
            if not TICKERS:
                logger.warning("No tickers discovered, trying top gainers only...")
                TICKERS = api.get_stock_list_from_gainers(count=20)
                if not TICKERS:
                    TICKERS = api.get_stock_list_from_swing_screener(count=20, min_price=5.0, max_price=100.0)
            if TICKERS:
                logger.info(f"Successfully fetched {len(TICKERS)} tickers from Webull API")
                logger.info(f"Tickers: {TICKERS[:10]}...")
            else:
                logger.warning("No tickers available from Webull API, using fallback list")
                TICKERS = ["YIBO", "TVGN", "NAOV"]
        except Exception as e:
            logger.error(f"Error fetching from Webull API: {e}")
            logger.info("Falling back to default tickers...")
            TICKERS = ["YIBO", "TVGN", "NAOV"]
        
        logger.info("")
        logger.info("Usage options:")
        logger.info("  python run_live_bot.py                    # Auto-fetch from top gainers (DEFAULT)")
        logger.info("  python run_live_bot.py auto                # Auto-fetch from top gainers")
        logger.info("  python run_live_bot.py swing               # Auto-fetch from swing screener")
        logger.info("  python run_live_bot.py manual TICKER1 ...  # Use specific tickers")
        logger.info("  python run_live_bot.py TICKER1 TICKER2 ... # Use specific tickers (legacy)")
    
    # ============================================
    # CREATE AND RUN BOT
    # ============================================
    
    logger.info("="*80)
    logger.info("LIVE TRADING BOT - STARTING")
    logger.info("="*80)
    logger.info(f"Data Source: Webull API (Live Market Data)")
    logger.info(f"Goal: Grow ${INITIAL_CAPITAL:,.2f} to ${TARGET_CAPITAL:,.2f} (10x return)")
    logger.info(f"Daily Profit Target: ${DAILY_PROFIT_TARGET_MIN:,.2f} (5% daily return)")
    logger.info(f"Trading Window: {TRADING_START_TIME} - {TRADING_END_TIME} ET (4 AM - 8 PM)")
    logger.info(f"Monitoring {len(TICKERS)} tickers from Webull API: {', '.join(TICKERS[:10])}{'...' if len(TICKERS) > 10 else ''}")
    logger.info("")
    logger.info("OPTIMIZED PARAMETERS (from simulation testing):")
    logger.info(f"  Min Confidence: {MIN_CONFIDENCE*100:.0f}% - balanced quality and opportunities")
    logger.info(f"  Min Entry Price Increase: {MIN_ENTRY_PRICE_INCREASE:.1f}% - capture more opportunities")
    logger.info(f"  Trailing Stop: {TRAILING_STOP_PCT:.1f}% - balanced, let winners run")
    logger.info(f"  Profit Target: {PROFIT_TARGET_PCT:.1f}% - capture more gains")
    logger.info(f"  Position Size: {POSITION_SIZE_PCT*100:.0f}% - maximize returns")
    logger.info(f"  Max Positions: {MAX_POSITIONS}")
    logger.info("="*80)
    
    # Create bot
    bot = LiveTradingBot(
        data_api=api,
        initial_capital=INITIAL_CAPITAL,
        target_capital=TARGET_CAPITAL,
        min_confidence=MIN_CONFIDENCE,
        min_entry_price_increase=MIN_ENTRY_PRICE_INCREASE,
        trailing_stop_pct=TRAILING_STOP_PCT,
        profit_target_pct=PROFIT_TARGET_PCT,
        position_size_pct=POSITION_SIZE_PCT,
        max_positions=MAX_POSITIONS,
        max_loss_per_trade_pct=MAX_LOSS_PER_TRADE_PCT,
        daily_profit_target_min=DAILY_PROFIT_TARGET_MIN,
        daily_profit_target_max=DAILY_PROFIT_TARGET_MAX,
        trading_start_time=TRADING_START_TIME,
        trading_end_time=TRADING_END_TIME,
        max_trades_per_day=MAX_TRADES_PER_DAY,
        max_daily_loss=MAX_DAILY_LOSS,
        consecutive_loss_limit=CONSECUTIVE_LOSS_LIMIT
    )
    
    # Add tickers
    for ticker in TICKERS:
        bot.add_ticker(ticker)
    
    # Store check interval in bot for web interface
    bot._check_interval = 20
    
    # Set trading bot in web interface BEFORE starting web server
    # This must happen before the web server starts
    set_trading_bot(bot)
    logger.info("Trading bot registered with web interface")
    
    # Import and verify immediately
    from web import trading_web_interface
    if trading_web_interface.trading_bot is None:
        logger.error("ERROR: Failed to set trading bot in web interface!")
        # Try again
        trading_web_interface.trading_bot = bot
        logger.info("Manually set trading bot in web interface module")
    
    # Give a moment for the bot to be set
    import time
    time.sleep(0.5)
    
    # Start web interface in background thread
    web_thread = threading.Thread(target=run_web_server, args=('0.0.0.0', 5000, False), daemon=True)
    web_thread.start()
    logger.info("Web interface started at http://127.0.0.1:5000")
    
    # Verify bot is accessible after web server starts
    time.sleep(0.5)
    if trading_web_interface.trading_bot is None:
        logger.warning("WARNING: Trading bot may not be accessible from web interface")
    else:
        logger.info("Trading bot is accessible from web interface")
    
    # Run bot
    try:
        # Run continuously until stopped
        logger.info("\n" + "="*80)
        logger.info("RUNNING IN LIVE MODE - Continuous Trading")
        logger.info("="*80)
        logger.info("Bot will run continuously until stopped")
        logger.info("Use the web dashboard at http://127.0.0.1:5000 to start/stop trading")
        logger.info("Or press Ctrl+C to stop")
        logger.info("Checking for trading opportunities every 15 seconds")
        logger.info("="*80 + "\n")
        
        # Verify bot is set in web interface before starting
        from web.trading_web_interface import trading_bot as check_bot
        if check_bot is None:
            logger.error("ERROR: Trading bot not set in web interface! Re-setting...")
            set_trading_bot(bot)
            import time
            time.sleep(0.5)
        
        # Start bot automatically (runs in main thread)
        # Note: The Start/Stop buttons in web interface control a separate thread if needed
        bot.run_continuous(check_on_second=5)  # Check on 5th second of every minute
        
    except KeyboardInterrupt:
        logger.info("\nTrading bot stopped by user (Ctrl+C)")
        bot.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        bot.stop()


if __name__ == "__main__":
    main()

