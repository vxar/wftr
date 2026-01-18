#!/usr/bin/env python3
"""Simple test to isolate import issues"""
print("Testing direct import...")
try:
    from src.core.autonomous_trading_bot import AutonomousTradingBot
    print("✅ SUCCESS: AutonomousTradingBot imported directly")
except ImportError as e:
    print(f"❌ FAILED: {e}")

print("Testing from src package...")
try:
    from src import AutonomousTradingBot
    print("✅ SUCCESS: AutonomousTradingBot imported from src")
except ImportError as e:
    print(f"❌ FAILED: {e}")
