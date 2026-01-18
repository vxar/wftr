"""
Test script for the trading bot scheduler
Verifies that the scheduler correctly identifies trading hours
"""
import sys
from pathlib import Path
import pytz
from datetime import datetime, time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.trading_bot_scheduler import TradingBotScheduler
from src.config.settings import settings

class MockBot:
    """Mock bot for testing scheduler"""
    def __init__(self):
        self.running = False
    
    def start(self):
        print("MockBot: Starting...")
        self.running = True
    
    def stop(self):
        print("MockBot: Stopping...")
        self.running = False

def test_scheduler():
    """Test the scheduler logic"""
    print("=" * 60)
    print("TRADING BOT SCHEDULER TEST")
    print("=" * 60)
    
    # Create mock bot and scheduler
    mock_bot = MockBot()
    scheduler = TradingBotScheduler(mock_bot, settings.trading_window)
    
    et_timezone = pytz.timezone('America/New_York')
    
    # Test different times
    test_times = [
        # Weekday tests
        datetime(2024, 1, 15, 3, 59, 0),  # Monday 3:59 AM (before trading)
        datetime(2024, 1, 15, 4, 0, 0),   # Monday 4:00 AM (trading starts)
        datetime(2024, 1, 15, 12, 0, 0),  # Monday 12:00 PM (trading)
        datetime(2024, 1, 15, 19, 59, 0),  # Monday 7:59 PM (trading)
        datetime(2024, 1, 15, 20, 0, 0),  # Monday 8:00 PM (trading ends)
        datetime(2024, 1, 15, 21, 0, 0),  # Monday 9:00 PM (after trading)
        
        # Weekend tests
        datetime(2024, 1, 13, 10, 0, 0),  # Saturday 10:00 AM (weekend)
        datetime(2024, 1, 14, 15, 0, 0),  # Sunday 3:00 PM (weekend)
    ]
    
    print(f"Trading Window: {settings.trading_window.start_time} - {settings.trading_window.end_time} ET")
    print(f"Timezone: {settings.trading_window.timezone}")
    print()
    
    for test_time in test_times:
        # Convert to ET
        test_time_et = test_time.astimezone(et_timezone)
        
        # Test scheduler logic
        is_weekday = scheduler._is_weekday(test_time)
        is_in_window = scheduler._is_within_trading_window(test_time)
        should_run = scheduler._should_bot_be_running(test_time)
        
        print(f"Time: {test_time_et.strftime('%A %Y-%m-%d %H:%M:%S ET')}")
        print(f"  Weekday: {is_weekday}")
        print(f"  In Trading Window: {is_in_window}")
        print(f"  Should Bot Run: {should_run}")
        print()
    
    # Test force check functionality
    print("Testing force check functionality...")
    status = scheduler.force_check()
    print(f"Scheduler status: {status}")
    
    print("=" * 60)
    print("SCHEDULER TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    test_scheduler()
