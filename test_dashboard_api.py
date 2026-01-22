#!/usr/bin/env python3
"""
Test Dashboard API
Simple test to verify the dashboard API endpoints work correctly
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify
import requests
import time

# Import dashboard
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
        print("Mock bot started")
    
    def stop(self):
        self.running = False
        print("Mock bot stopped")
    

# Set mock bot instance
mock_bot = MockBot()
set_bot_instance(mock_bot)

def test_api_endpoints():
    """Test all dashboard API endpoints"""
    base_url = "http://localhost:5000"
    
    # Test endpoints
    endpoints = [
        ('/api/status', 'GET'),
        ('/api/positions', 'GET'),
        ('/api/start', 'POST'),
        ('/api/stop', 'POST'),
        ('/api/pause', 'POST'),
        ('/api/resume', 'POST'),
    ]
    
    print("Testing dashboard API endpoints...")
    
    with app.test_client() as client:
        for endpoint, method in endpoints:
            try:
                if method == 'GET':
                    response = client.get(endpoint)
                else:
                    response = client.post(endpoint)
                
                print(f"{method} {endpoint}: {response.status_code} - {response.get_json()}")
            except Exception as e:
                print(f"{method} {endpoint}: ERROR - {e}")

if __name__ == '__main__':
    test_api_endpoints()
