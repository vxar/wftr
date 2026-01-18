"""
Trading Bot Package
Main package for AI trading bot system
"""
# Export main components for easy access
from .core.autonomous_trading_bot import AutonomousTradingBot
from .core.live_trading_bot import LiveTradingBot, Trade, RealtimeTrader, TradeSignal, ActivePosition
from .analysis import PatternDetector, PatternSignal, PreMarketAnalyzer, StockDiscovery
from .data import DataAPI, CSVDataAPI, WebullDataAPI
from .database import TradingDatabase, TradeRecord, PositionRecord
from .web import set_trading_bot, run_web_server, app

__all__ = [
    # Core
    'AutonomousTradingBot',
    'LiveTradingBot',
    'Trade',
    'RealtimeTrader',
    'TradeSignal',
    'ActivePosition',
    # Analysis
    'PatternDetector',
    'PatternSignal',
    'PreMarketAnalyzer',
    'StockDiscovery',
    # Data
    'DataAPI',
    'CSVDataAPI',
    'WebullDataAPI',
    # Database
    'TradingDatabase',
    'TradeRecord',
    'PositionRecord',
    # Web
    'set_trading_bot',
    'run_web_server',
    'app'
]