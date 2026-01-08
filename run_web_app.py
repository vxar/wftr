"""
Web Application Entry Point
Run this script to start the trading bot web interface
"""
import sys
import os

# Add src to path so we can import packages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from web.web_app import main

if __name__ == "__main__":
    main()
