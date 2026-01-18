#!/usr/bin/env python3
"""Simple bot runner without complex imports"""
import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "src"))

# Set the current working directory
os.chdir(current_dir)

def main():
    """Main function to start the enhanced bot"""
    try:
        print("🚀 Starting Simple Autonomous Trading Bot...")
        print("=" * 50)
        
        # Import bot directly
        from src.core.autonomous_trading_bot import AutonomousTradingBot
        
        # Initialize bot with default configuration
        config = {
            'initial_capital': 10000.0,
            'max_positions': 3,
            'position_size_pct': 0.33,
            'dashboard_enabled': False,  # Disable dashboard
            'learning_mode': 'OFF'  # Disable learning
        }
        
        # Initialize bot
        bot = AutonomousTradingBot(config)
        
        print("✅ Bot initialized successfully")
        print("🚀 Starting bot...")
        
        # Start bot
        bot.start()
        
        print("✅ Bot started successfully")
        
        # Keep running for a few seconds then stop
        import time
        time.sleep(5)
        
        # Stop bot
        bot.stop()
        
        print("✅ Bot stopped successfully")
        print("🎯 Simple bot test completed")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
