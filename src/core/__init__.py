"""
Core Trading Logic Package
Contains the main trading bot and real-time trader components
"""
# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == 'LiveTradingBot':
        from core.live_trading_bot import LiveTradingBot
        return LiveTradingBot
    elif name == 'Trade':
        from core.live_trading_bot import Trade
        return Trade
    elif name == 'RealtimeTrader':
        from core.realtime_trader import RealtimeTrader
        return RealtimeTrader
    elif name == 'TradeSignal':
        from core.realtime_trader import TradeSignal
        return TradeSignal
    elif name == 'ActivePosition':
        from core.realtime_trader import ActivePosition
        return ActivePosition
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'LiveTradingBot',
    'Trade',
    'RealtimeTrader',
    'TradeSignal',
    'ActivePosition'
]
