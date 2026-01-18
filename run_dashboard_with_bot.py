#!/usr/bin/env python3
"""
Run Dashboard with Real Bot
Connects actual trading bot to the enhanced dashboard
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import dashboard
from simple_dashboard import set_bot_instance, run_dashboard

# Try to import and create bot instance
try:
    # Try to import the main bot
    from src.core.autonomous_trading_bot import AutonomousTradingBot
    
    # Create bot instance (you may need to adjust parameters)
    bot = AutonomousTradingBot(
        config={
            'paper_trading': True,  # Set to False for live trading
            'initial_capital': 10000
        }
    )
    
    print("🤖 Real trading bot connected to dashboard")
    print("📊 Dashboard will show live trading data")
    
except ImportError as e:
    print(f"⚠️ Could not import trading bot: {e}")
    print("📊 Dashboard will run with demo data")
    bot = None
except Exception as e:
    print(f"❌ Error creating bot: {e}")
    bot = None

# Set bot instance for dashboard
set_bot_instance(bot)

if __name__ == '__main__':
    try:
        print("🌐 Starting enhanced dashboard...")
        print("📱 Open http://localhost:5000 to view")
        print("🔄 Real-time updates every 5 seconds")
        print("⚡ Features:")
        print("   • Live position tracking")
        print("   • Real P&L calculations") 
        print("   • Individual position controls")
        print("   • Modern glass-morphism UI")
        
        # Run dashboard
        run_dashboard(port=5000)
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
