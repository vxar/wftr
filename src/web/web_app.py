"""
Web Application Entry Point
Primary entry point for the trading bot - web interface controls everything
"""
import sys
import logging
import threading
from core.live_trading_bot import LiveTradingBot
from web.trading_web_interface import set_trading_bot, run_web_server
from data.webull_data_api import WebullDataAPI

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


def create_trading_bot():
    """Create and configure the trading bot"""
    
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
    api = WebullDataAPI()
    
    # ============================================
    # TICKER SELECTION
    # ============================================
    
    # Use StockDiscovery to get stocks from multiple sources
    logger.info("Discovering stocks from multiple sources (gainers, news, most active, etc.)...")
    try:
        from analysis.stock_discovery import StockDiscovery
        
        discovery = StockDiscovery(api)
        TICKERS = discovery.discover_stocks(
            include_gainers=True,  # Always include top gainers
            include_news=True,  # Include news-driven stocks
            include_most_active=True,  # Include most active stocks
            include_unusual_volume=True,  # Include unusual volume stocks
            include_breakouts=True,  # Include breakout candidates
            include_reversals=False,  # Don't include reversals (more risky)
            max_total=30  # Get up to 30 unique stocks
        )
        
        if TICKERS:
            logger.info(f"Successfully discovered {len(TICKERS)} tickers from multiple sources")
            logger.info(f"Tickers: {TICKERS[:10]}...")
        else:
            logger.warning("No tickers discovered, falling back to top gainers only...")
            TICKERS = api.get_stock_list_from_gainers(count=20)
            if not TICKERS:
                TICKERS = ["YIBO", "TVGN", "NAOV"]
    except ImportError as e:
        logger.warning(f"StockDiscovery not available: {e}, using top gainers only...")
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
    
    # ============================================
    # CREATE BOT (but don't start it yet)
    # ============================================
    
    logger.info("="*80)
    logger.info("TRADING BOT - INITIALIZED (Not Started)")
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
    
    # Store check timing in bot for web interface
    # Bot will check on the 5th second of every minute (to avoid premature exits from intra-minute price moves)
    bot._check_on_second = 5
    
    return bot


def main():
    """Main entry point - starts web interface and auto-starts trading"""
    
    logger.info("="*80)
    logger.info("TRADING BOT WEB APPLICATION")
    logger.info("="*80)
    logger.info("Starting web interface...")
    logger.info("Trading bot will be created and started AUTOMATICALLY")
    logger.info("="*80 + "\n")
    
    # Create the trading bot
    bot = create_trading_bot()
    
    # Set trading bot in web interface
    set_trading_bot(bot)
    logger.info("Trading bot registered with web interface")
    
    # Verify bot is set
    from web import trading_web_interface
    if trading_web_interface.trading_bot is None:
        logger.error("ERROR: Failed to set trading bot in web interface!")
        trading_web_interface.trading_bot = bot
        logger.info("Manually set trading bot in web interface module")
    
    # AUTO-START TRADING
    logger.info("\n" + "="*80)
    logger.info("AUTO-STARTING TRADING BOT")
    logger.info("="*80)
    try:
        import threading
        import time
        
        def auto_start_bot():
            """Auto-start the bot in a separate thread"""
            # Wait a moment for web server to start
            time.sleep(2)
            
            # Get check second from bot or default to 5 (5th second of every minute)
            check_on_second = getattr(bot, '_check_on_second', 5)
            logger.info(f"Auto-starting trading bot (checking on {check_on_second}th second of every minute)")
            
            try:
                bot.run_continuous(check_on_second=check_on_second)
            except Exception as e:
                logger.error(f"Error in trading bot thread: {e}")
                import traceback
                logger.error(traceback.format_exc())
                if hasattr(bot, 'running'):
                    bot.running = False
        
        # Create and start bot thread
        bot_thread = threading.Thread(target=auto_start_bot, daemon=True, name="TradingBotThread")
        bot_thread.start()
        bot._bot_thread = bot_thread
        
        logger.info("Trading bot thread started - will begin trading automatically")
        logger.info("Bot will run continuously from 4 AM to 8 PM ET daily")
        
    except Exception as e:
        logger.error(f"Error auto-starting trading bot: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Start web interface (this will block and run the Flask server)
    logger.info("\n" + "="*80)
    logger.info("Web Interface Starting...")
    logger.info("Open http://127.0.0.1:5000 in your browser")
    logger.info("Trading is already started automatically")
    logger.info("Use the dashboard to monitor and control trading")
    logger.info("="*80 + "\n")
    
    # Run web server (this blocks until server stops)
    run_web_server(host='0.0.0.0', port=5000, debug=False)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nApplication stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())

