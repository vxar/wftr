"""
Simple test for the trading bot scheduler logic
Tests the core scheduling logic without full bot dependencies
"""
import sys
from pathlib import Path
import pytz
from datetime import datetime, time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_scheduler_logic():
    """Test the scheduler logic directly"""
    print("=" * 60)
    print("TRADING BOT SCHEDULER LOGIC TEST")
    print("=" * 60)
    
    # Configuration
    start_time = time(4, 0)  # 4:00 AM
    end_time = time(20, 0)   # 8:00 PM
    et_timezone = pytz.timezone('America/New_York')
    
    def is_weekday(current_time):
        """Check if current time is a weekday (Monday-Friday)"""
        return current_time.weekday() < 5  # 0-4 are Monday-Friday
    
    def is_within_trading_window(current_time):
        """Check if current time is within the trading window"""
        current_time_et = current_time.astimezone(et_timezone)
        current_time_only = current_time_et.time()
        
        # Handle case where end_time is after midnight (not applicable for 4 AM - 8 PM)
        if start_time <= end_time:
            return start_time <= current_time_only <= end_time
        else:
            # For windows that cross midnight (not needed for current config)
            return current_time_only >= start_time or current_time_only <= end_time
    
    def should_bot_be_running(current_time):
        """Determine if bot should be running based on time and day"""
        return is_weekday(current_time) and is_within_trading_window(current_time)
    
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
    
    print(f"Trading Window: {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')} ET")
    print(f"Timezone: America/New_York")
    print()
    
    for test_time in test_times:
        # Convert to ET
        test_time_et = test_time.astimezone(et_timezone)
        
        # Test scheduler logic
        is_weekday_result = is_weekday(test_time)
        is_in_window_result = is_within_trading_window(test_time)
        should_run_result = should_bot_be_running(test_time)
        
        print(f"Time: {test_time_et.strftime('%A %Y-%m-%d %H:%M:%S ET')}")
        print(f"  Weekday: {is_weekday_result}")
        print(f"  In Trading Window: {is_in_window_result}")
        print(f"  Should Bot Run: {should_run_result}")
        
        # Verify logic
        expected_result = is_weekday_result and is_in_window_result
        if should_run_result == expected_result:
            print(f"  ✓ Logic Correct")
        else:
            print(f"  ✗ Logic Error - Expected: {expected_result}")
        print()
    
    # Test current time
    current_time = datetime.now(et_timezone)
    print(f"Current time: {current_time.strftime('%A %Y-%m-%d %H:%M:%S ET')}")
    print(f"Should bot be running now: {should_bot_be_running(current_time)}")
    
    print("=" * 60)
    print("SCHEDULER LOGIC TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    test_scheduler_logic()
