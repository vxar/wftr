#!/usr/bin/env python3
"""
Simple test to verify daily analysis components work
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test if all required modules can be imported"""
    try:
        print("Testing imports...")
        
        # Test basic imports
        import logging
        from datetime import datetime
        print("✓ Basic imports successful")
        
        # Test schedule import
        try:
            import schedule
            print("✓ Schedule module available")
        except ImportError as e:
            print(f"✗ Schedule module missing: {e}")
            return False
        
        # Test pandas import
        try:
            import pandas as pd
            import numpy as np
            print("✓ Pandas and NumPy available")
        except ImportError as e:
            print(f"✗ Pandas/NumPy missing: {e}")
            return False
        
        # Test database import
        try:
            from src.database.trading_database import TradingDatabase
            print("✓ Trading database available")
        except ImportError as e:
            print(f"✗ Trading database import error: {e}")
            return False
        
        # Test daily analyzer import
        try:
            from src.analysis.daily_trade_analyzer import DailyTradeAnalyzer
            print("✓ Daily trade analyzer available")
        except ImportError as e:
            print(f"✗ Daily analyzer import error: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_database():
    """Test database connection"""
    try:
        print("\nTesting database...")
        from src.database.trading_database import TradingDatabase
        
        db = TradingDatabase()
        print("✓ Database connection successful")
        
        # Test getting trades
        trades = db.get_all_trades(limit=5)
        print(f"✓ Found {len(trades)} trades in database")
        
        return True
        
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Daily Analysis System Test")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Install missing dependencies:")
        print("   pip install schedule pandas numpy")
        return 1
    
    # Test database
    if not test_database():
        print("\n❌ Database tests failed.")
        return 1
    
    print("\n✅ All tests passed! Daily analysis system is ready.")
    print("\nTo start the automatic 8pm analysis:")
    print("   python start_daily_analysis.py")
    print("\nTo run manual analysis:")
    print("   python -c \"from src.analysis.daily_trade_analyzer import DailyTradeAnalyzer; print(DailyTradeAnalyzer().run_daily_analysis())\"")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
