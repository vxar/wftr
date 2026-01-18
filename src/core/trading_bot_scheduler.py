"""
Trading Bot Scheduler
Automatically starts and stops the trading bot based on configured trading hours
"""
import schedule
import time
import logging
from datetime import datetime, timedelta
import pytz
from typing import Optional, Callable
from threading import Thread, Event
import asyncio

logger = logging.getLogger(__name__)

class TradingBotScheduler:
    """
    Scheduler for automatic trading bot start/stop based on trading hours
    """
    
    def __init__(self, bot, trading_window_config):
        """
        Initialize the scheduler
        
        Args:
            bot: Trading bot instance
            trading_window_config: Trading window configuration
        """
        self.bot = bot
        self.trading_window_config = trading_window_config
        self.et_timezone = pytz.timezone('America/New_York')
        
        # Scheduler state
        self.scheduler_running = False
        self.scheduler_thread = None
        self.stop_event = Event()
        
        # Bot state tracking
        self.bot_auto_started = False
        self.last_check_time = None
        
        # Parse trading hours
        self.start_time = self._parse_time(trading_window_config.start_time)
        self.end_time = self._parse_time(trading_window_config.end_time)
        
        logger.info(f"Scheduler initialized - Trading window: {trading_window_config.start_time} to {trading_window_config.end_time} ET")
    
    def _parse_time(self, time_str: str) -> datetime.time:
        """Parse time string to datetime.time object"""
        try:
            return datetime.strptime(time_str, "%H:%M").time()
        except ValueError as e:
            logger.error(f"Invalid time format {time_str}: {e}")
            raise
    
    def _is_weekday(self, current_time: datetime) -> bool:
        """Check if current time is a weekday (Monday-Friday)"""
        return current_time.weekday() < 5  # 0-4 are Monday-Friday
    
    def _is_within_trading_window(self, current_time: datetime) -> bool:
        """Check if current time is within the trading window"""
        current_time_et = current_time.astimezone(self.et_timezone)
        current_time_only = current_time_et.time()
        
        # Handle case where end_time is after midnight (not applicable for 4 AM - 8 PM)
        if self.start_time <= self.end_time:
            return self.start_time <= current_time_only <= self.end_time
        else:
            # For windows that cross midnight (not needed for current config)
            return current_time_only >= self.start_time or current_time_only <= self.end_time
    
    def _should_bot_be_running(self, current_time: datetime) -> bool:
        """Determine if bot should be running based on time and day"""
        return self._is_weekday(current_time) and self._is_within_trading_window(current_time)
    
    def start_scheduler(self):
        """Start the scheduler thread"""
        if self.scheduler_running:
            logger.warning("Scheduler is already running")
            return
        
        self.scheduler_running = True
        self.stop_event.clear()
        
        self.scheduler_thread = Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Trading bot scheduler started")
    
    def stop_scheduler(self):
        """Stop the scheduler thread"""
        if not self.scheduler_running:
            return
        
        self.scheduler_running = False
        self.stop_event.set()
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Trading bot scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        logger.info("Scheduler loop started")
        
        while self.scheduler_running and not self.stop_event.is_set():
            try:
                current_time = datetime.now(self.et_timezone)
                self.last_check_time = current_time
                
                should_run = self._should_bot_be_running(current_time)
                
                # Log current status
                is_weekday = self._is_weekday(current_time)
                is_in_window = self._is_within_trading_window(current_time)
                
                logger.debug(f"Scheduler check - Time: {current_time.strftime('%Y-%m-%d %H:%M:%S ET')} | "
                           f"Weekday: {is_weekday} | In Window: {is_in_window} | Should Run: {should_run}")
                
                # Handle bot start/stop
                if should_run and not self.bot.running:
                    logger.info(f"Starting trading bot at {current_time.strftime('%H:%M:%S ET')} - "
                               f"Weekday: {is_weekday}, In trading window: {is_in_window}")
                    self.bot.start()
                    self.bot_auto_started = True
                    
                elif not should_run and self.bot.running:
                    logger.info(f"Stopping trading bot at {current_time.strftime('%H:%M:%S ET')} - "
                               f"Weekday: {is_weekday}, In trading window: {is_in_window}")
                    self.bot.stop()
                    self.bot_auto_started = False
                
                # Check for state transitions and log them
                if self.bot_auto_started and not should_run:
                    logger.info("Trading window ended - Bot entering sleep mode")
                elif not self.bot_auto_started and should_run and not self.bot.running:
                    logger.info("Trading window started - Bot waking up")
                
                # Sleep for 30 seconds before next check
                if self.stop_event.wait(timeout=30):
                    break
                    
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(30)  # Wait before retrying
        
        logger.info("Scheduler loop ended")
    
    def get_scheduler_status(self) -> dict:
        """Get current scheduler status"""
        current_time = datetime.now(self.et_timezone)
        
        return {
            'scheduler_running': self.scheduler_running,
            'current_time_et': current_time.strftime('%Y-%m-%d %H:%M:%S ET'),
            'is_weekday': self._is_weekday(current_time),
            'is_within_trading_window': self._is_within_trading_window(current_time),
            'should_bot_be_running': self._should_bot_be_running(current_time),
            'trading_window_start': self.trading_window_config.start_time,
            'trading_window_end': self.trading_window_config.end_time,
            'bot_running': self.bot.running if self.bot else False,
            'bot_auto_started': self.bot_auto_started,
            'last_check_time': self.last_check_time.strftime('%Y-%m-%d %H:%M:%S ET') if self.last_check_time else None
        }
    
    def force_check(self):
        """Force an immediate check of trading conditions"""
        try:
            current_time = datetime.now(self.et_timezone)
            should_run = self._should_bot_be_running(current_time)
            
            logger.info(f"Force check - Time: {current_time.strftime('%H:%M:%S ET')} | "
                       f"Should run: {should_run} | Bot running: {self.bot.running}")
            
            if should_run and not self.bot.running:
                logger.info("Force starting trading bot")
                self.bot.start()
                self.bot_auto_started = True
            elif not should_run and self.bot.running:
                logger.info("Force stopping trading bot")
                self.bot.stop()
                self.bot_auto_started = False
                
            return self.get_scheduler_status()
            
        except Exception as e:
            logger.error(f"Error in force check: {e}")
            return {'error': str(e)}
