#!/usr/bin/env python3
"""
Daily Trade Analysis Startup Script
Initializes the daily trade analyzer and ensures it runs automatically at 8pm ET
"""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.analysis.daily_trade_analyzer import DailyTradeAnalyzer

def main():
    """Initialize and start daily trade analyzer"""
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger = logging.getLogger(__name__)
        logger.info("Starting Daily Trade Analyzer...")
        
        # Initialize the analyzer (this will automatically schedule the 8pm analysis)
        analyzer = DailyTradeAnalyzer()
        
        logger.info("Daily Trade Analyzer initialized successfully!")
        logger.info("Automatic analysis scheduled for 8:00 PM ET daily.")
        logger.info("Press Ctrl+C to stop the scheduler.")
        
        # Keep the script running to maintain the scheduler
        import time
        try:
            while True:
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Daily Trade Analyzer stopped by user.")
            
    except Exception as e:
        logger.error(f"Error starting Daily Trade Analyzer: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
