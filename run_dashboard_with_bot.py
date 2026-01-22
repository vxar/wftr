#!/usr/bin/env python3
"""
Run Dashboard with Real Bot
Connects actual trading bot to the enhanced dashboard
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import enhanced dashboard
from src.web.enhanced_dashboard import EnhancedDashboard

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
    
    print("Real trading bot connected to dashboard")
    print("Dashboard will show live trading data")
    
    # Create enhanced dashboard instance with bot
    dashboard = EnhancedDashboard(trading_bot=bot)
    
    print("Real trading bot connected to enhanced dashboard")
    print("Dashboard will show live trading data")
    
except ImportError as e:
    print(f"WARNING: Could not import trading bot: {e}")
    bot = None
    dashboard = None
except Exception as e:
    print(f"ERROR: Error creating bot: {e}")
    bot = None
    dashboard = None

# Run the enhanced dashboard
if __name__ == "__main__":
    if dashboard:
        print("Starting enhanced dashboard...")
        print("Open http://localhost:5000 to view")
        print("Real-time updates every 5 seconds")
        print("Features:")
        print("   - Live position tracking")
        print("   - Real P&L calculations") 
        print("   - Individual position controls")
        print("   - Modern glass-morphism UI")
        
        # Run the enhanced dashboard
        dashboard.run()
        
    else:
        print("\nNo dashboard to run")
