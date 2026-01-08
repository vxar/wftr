"""
Run Live Trading Bot
Run this script to start the trading bot
"""
import sys
import os

# Add src to path so we can import packages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from scripts.run_live_bot import main

if __name__ == "__main__":
    main()
