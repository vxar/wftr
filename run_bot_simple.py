#!/usr/bin/env python3
"""
Simple startup script for the Enhanced Autonomous Trading Bot
"""

import sys
import os
from pathlib import Path

def main():
    """Start the enhanced trading bot"""
    try:
        # Change to the script's directory
        script_dir = Path(__file__).parent
        os.chdir(script_dir)
        
        # Add current directory to path (not src)
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        
        print("🚀 Starting Enhanced Autonomous Trading Bot...")
        print("=" * 50)
        
        # Import and run the bot using absolute imports
        from src.core.autonomous_trading_bot import main as bot_main
        bot_main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nPlease check:")
        print("1. All dependencies are installed: pip install -r requirements.txt")
        print("2. Virtual environment is activated")
        print("3. You're in the correct directory")
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
        sys.exit(0)
    
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
