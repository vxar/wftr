"""
Test script to verify all imports work correctly
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing imports...")

try:
    print("1. Testing core imports...")
    from core.live_trading_bot import LiveTradingBot
    from core.realtime_trader import RealtimeTrader, TradeSignal, ActivePosition
    print("   ✓ Core imports successful")
except Exception as e:
    print(f"   ✗ Core imports failed: {e}")
    sys.exit(1)

try:
    print("2. Testing analysis imports...")
    from analysis.pattern_detector import PatternDetector, PatternSignal
    from analysis.premarket_analyzer import PreMarketAnalyzer
    from analysis.stock_discovery import StockDiscovery
    print("   ✓ Analysis imports successful")
except Exception as e:
    print(f"   ✗ Analysis imports failed: {e}")
    sys.exit(1)

try:
    print("3. Testing data imports...")
    from data.api_interface import DataAPI, CSVDataAPI
    from data.webull_data_api import WebullDataAPI
    print("   ✓ Data imports successful")
except Exception as e:
    print(f"   ✗ Data imports failed: {e}")
    sys.exit(1)

try:
    print("4. Testing database imports...")
    from database.trading_database import TradingDatabase, TradeRecord, PositionRecord
    print("   ✓ Database imports successful")
except Exception as e:
    print(f"   ✗ Database imports failed: {e}")
    sys.exit(1)

try:
    print("5. Testing web imports...")
    from web.trading_web_interface import set_trading_bot, run_web_server, app
    print("   ✓ Web imports successful")
except Exception as e:
    print(f"   ✗ Web imports failed: {e}")
    sys.exit(1)

try:
    print("6. Testing utils imports...")
    from utils.utils import logger
    print("   ✓ Utils imports successful")
except Exception as e:
    print(f"   ✗ Utils imports failed: {e}")
    sys.exit(1)

print("\n✅ All imports successful! The package structure is working correctly.")
