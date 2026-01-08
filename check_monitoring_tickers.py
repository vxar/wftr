"""
Check Current Monitoring Ticker List
Uses Webull API to fetch and display the current tickers that would be monitored
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(Path(__file__).parent, 'src'))

from data.webull_data_api import WebullDataAPI
from analysis.stock_discovery import StockDiscovery
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Fetch and display current monitoring tickers"""
    print("="*80)
    print("CURRENT MONITORING TICKER LIST")
    print("="*80)
    
    try:
        # Initialize APIs
        api = WebullDataAPI()
        discovery = StockDiscovery(api)
        
        # Fetch tickers from multiple sources (same as live bot)
        print("\nFetching tickers from multiple sources...")
        tickers = discovery.discover_stocks(
            include_gainers=True,
            include_news=True,
            include_most_active=True,
            include_unusual_volume=True,
            include_breakouts=True,
            include_reversals=False,
            max_total=30
        )
        
        print(f"\n{'='*80}")
        print(f"MONITORING TICKER LIST ({len(tickers)} tickers)")
        print(f"{'='*80}")
        
        if tickers:
            # Print in columns for better readability
            tickers_sorted = sorted(tickers)
            cols = 5
            for i in range(0, len(tickers_sorted), cols):
                row = tickers_sorted[i:i+cols]
                print("  " + "  ".join(f"{ticker:8s}" for ticker in row))
            
            print(f"\nTotal: {len(tickers)} tickers")
        else:
            print("No tickers found!")
        
        # Also show breakdown by source
        print(f"\n{'='*80}")
        print("TICKER SOURCES BREAKDOWN")
        print(f"{'='*80}")
        
        # Top gainers
        try:
            gainers = api.fetch_top_gainers(count=20)
            gainer_symbols = [g.get('symbol', '') for g in gainers if g.get('symbol')]
            print(f"\nTop Gainers: {len(gainer_symbols)} tickers")
            if gainer_symbols:
                print(f"  {', '.join(gainer_symbols[:20])}")
        except Exception as e:
            print(f"Error fetching top gainers: {e}")
        
        print(f"\n{'='*80}")
        print("COMPLETE")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
