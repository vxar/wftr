#!/usr/bin/env python3
"""
Test imports for dashboard
"""
try:
    print("Testing imports...")
    
    # Test flask imports
    from flask import Flask, render_template, jsonify, request
    print("✅ Flask imports successful")
    
    from flask_cors import CORS
    print("✅ Flask-CORS import successful")
    
    # Test dashboard imports
    from simple_dashboard import set_bot_instance, run_dashboard
    print("✅ Dashboard imports successful")
    
    # Test bot import
    from src.core.autonomous_trading_bot import AutonomousTradingBot
    print("✅ Bot import successful")
    
    print("\n🎉 All imports successful!")
    print("The dashboard should work properly now.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
