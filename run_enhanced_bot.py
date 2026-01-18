#!/usr/bin/env python3
"""
Enhanced Autonomous Trading Bot - Startup Script
Run this script to start the enhanced autonomous trading bot
"""

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
        print("🚀 Starting Enhanced Autonomous Trading Bot...")
        print("=" * 50)
        
        # Import bot with proper path handling
        try:
            from simple_bot import SimpleAutonomousBot as AutonomousTradingBot
        except ImportError:
            # Try alternative import path
            print("❌ Could not import bot. Please check dependencies.")
            sys.exit(1)
        
        # Initialize bot with default configuration
        config = {
            'initial_capital': 10000.0,
            'max_positions': 3,
            'position_size_pct': 0.33,
            'scanner_update_interval': 60,
            'dashboard_enabled': True,
            'dashboard_port': 5000
        }
        
        bot = AutonomousTradingBot(config)
        
        print(f"✅ Bot initialized with ${config['initial_capital']:,.2f} initial capital")
        print(f"🔄 Max positions: {config['max_positions']}")
        print(f"⏱️ Scanner update interval: {config['scanner_update_interval']} seconds")
        print(f"🌐 Dashboard: http://localhost:{config['dashboard_port']}")
        print("=" * 50)
        
        # Start the bot
        bot.start()
        
        # Keep the main thread alive
        try:
            print("\n🎯 Bot is running autonomously...")
            print("📊 Trading will begin automatically...")
            print("Press Ctrl+C to stop the bot\n")
            
            import time
            while True:
                time.sleep(60)
                
                # Print status every 60 seconds
                status = bot.get_bot_status()
                if status.get('running', False):
                    print(f"💰 Initial Capital: ${config['initial_capital']:,.2f} | "
                          f"Active Positions: {status['active_positions']}/{config['max_positions']} | "
                          f"Total Trades: {status['performance_metrics']['total_trades']} | "
                          f"Total P&L: ${status['performance_metrics']['total_pnl']:+.2f}")
                
        except KeyboardInterrupt:
            print("\n🛑 Shutting down bot...")
            bot.stop()
            print("✅ Bot stopped safely")
            
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
