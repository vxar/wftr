#!/usr/bin/env python3
"""
Simple Dashboard Test
Test the dashboard with a mock bot to verify button functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_dashboard import app, set_bot_instance

# Create a mock bot for testing
class MockBot:
    def __init__(self):
        self.running = False
        self.current_capital = 10000
        self.daily_profit = 0
        self.active_positions = {}
        self.trade_history = []
    
    def start(self):
        self.running = True
        print("✅ Mock bot started")
        return True
    
    def stop(self):
        self.running = False
        print("✅ Mock bot stopped")
        return True
    

def test_dashboard():
    """Test the dashboard with mock bot"""
    print("🔧 Setting up mock bot...")
    mock_bot = MockBot()
    set_bot_instance(mock_bot)
    
    print("🌐 Starting dashboard test server...")
    print("📱 Open http://localhost:5001 to test button functionality")
    print("🧪 This is a test server with mock data")
    print("⚠️  Press Ctrl+C to stop the server")
    
    try:
        # Run on a different port to avoid conflicts
        app.run(host='0.0.0.0', port=5001, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Test server stopped")

if __name__ == '__main__':
    test_dashboard()
