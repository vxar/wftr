"""
Core Trading Logic Package
Contains main trading bot and real-time trader components
"""
# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == 'LiveTradingBot':
        from .live_trading_bot import LiveTradingBot
        return LiveTradingBot
    elif name == 'AutonomousTradingBot':
        from .autonomous_trading_bot import AutonomousTradingBot
        return AutonomousTradingBot
    elif name == 'Trade':
        from .live_trading_bot import Trade
        return Trade
    elif name == 'RealtimeTrader':
        from .realtime_trader import RealtimeTrader
        return RealtimeTrader
    elif name == 'TradeSignal':
        from .realtime_trader import TradeSignal
        return TradeSignal
    elif name == 'ActivePosition':
        from .realtime_trader import ActivePosition
        return ActivePosition
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'LiveTradingBot',
    'Trade',
    'RealtimeTrader',
    'TradeSignal',
    'ActivePosition',
    'AutonomousTradingBot'
]
