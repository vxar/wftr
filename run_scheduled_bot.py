"""
Trading Bot Launcher with Automatic Scheduler
Starts the trading bot with automatic scheduling (4 AM - 8 PM ET, weekdays only)
"""
import logging
import sys
import signal
import time
from pathlib import Path
import pytz
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.autonomous_trading_bot import AutonomousTradingBot
from src.config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.logging.level),
    format=settings.logging.format,
    handlers=[
        logging.FileHandler(settings.logging.file_name),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class TradingBotLauncher:
    """Main launcher for the trading bot with scheduler integration"""
    
    def __init__(self):
        self.bot = None
        self.et_timezone = pytz.timezone('America/New_York')
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown()
    
    def start(self):
        """Start the trading bot with scheduler"""
        try:
            logger.info("=" * 60)
            logger.info("TRADING BOT LAUNCHER STARTING")
            logger.info("=" * 60)
            
            current_time = datetime.now(self.et_timezone)
            logger.info(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S ET')}")
            logger.info(f"Trading window: {settings.trading_window.start_time} - {settings.trading_window.end_time} ET")
            logger.info(f"Timezone: {settings.trading_window.timezone}")
            
            # Initialize the bot
            logger.info("Initializing trading bot...")
            self.bot = AutonomousTradingBot()
            
            # Start the bot (this will automatically start the scheduler)
            logger.info("Starting trading bot with automatic scheduler...")
            self.bot.start()
            
            self.running = True
            logger.info("Trading bot started successfully!")
            logger.info("The bot will automatically:")
            logger.info("  - Start trading at 4:00 AM ET on weekdays")
            logger.info("  - Stop trading at 8:00 PM ET on weekdays")
            logger.info("  - Remain in sleep mode on weekends")
            logger.info("  - Handle market volatility pauses automatically")
            
            # Main monitoring loop
            self._monitoring_loop()
            
        except Exception as e:
            logger.error(f"Error starting trading bot: {e}")
            self.shutdown()
            sys.exit(1)
    
    def _monitoring_loop(self):
        """Main monitoring loop for the launcher"""
        logger.info("Entering monitoring loop...")
        
        while self.running:
            try:
                if self.bot:
                    # Get bot status every 5 minutes
                    status = self.bot.get_bot_status()
                    scheduler_status = self.bot.get_scheduler_status()
                    
                    # Log status every 30 minutes
                    current_time = datetime.now(self.et_timezone)
                    if current_time.minute % 30 == 0 and current_time.second < 10:
                        logger.info(f"Status Check - Bot Running: {status.get('running', False)}, "
                                  f"Paused: {status.get('paused', False)}, "
                                  f"Scheduler Active: {scheduler_status.get('scheduler_running', False)}")
                        
                        # Log position summary if there are active positions
                        positions = status.get('positions', {})
                        if positions.get('active_positions'):
                            active_count = len(positions['active_positions'])
                            total_pnl = positions.get('total_unrealized_pnl', 0)
                            logger.info(f"Active Positions: {active_count}, Total P&L: ${total_pnl:.2f}")
                
                # Sleep for 1 minute before next check
                time.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt...")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait before retrying
        
        logger.info("Monitoring loop ended")
    
    def shutdown(self):
        """Gracefully shutdown the trading bot"""
        logger.info("Initiating graceful shutdown...")
        
        self.running = False
        
        if self.bot:
            try:
                logger.info("Stopping trading bot...")
                self.bot.stop()
                logger.info("Trading bot stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping bot: {e}")
        
        logger.info("Trading bot launcher shutdown complete")
    
    def get_status(self):
        """Get current status of the bot and scheduler"""
        if not self.bot:
            return {'error': 'Bot not initialized'}
        
        try:
            bot_status = self.bot.get_bot_status()
            scheduler_status = self.bot.get_scheduler_status()
            
            return {
                'launcher_running': self.running,
                'bot_status': bot_status,
                'scheduler_status': scheduler_status,
                'current_time_et': datetime.now(self.et_timezone).strftime('%Y-%m-%d %H:%M:%S ET')
            }
        except Exception as e:
            return {'error': str(e)}

def main():
    """Main entry point"""
    launcher = TradingBotLauncher()
    
    try:
        launcher.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        launcher.shutdown()

if __name__ == "__main__":
    main()
