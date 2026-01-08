"""
Web Interface Package
Contains Flask web application and API endpoints
"""
from web.trading_web_interface import (
    set_trading_bot,
    run_web_server,
    app
)

__all__ = [
    'set_trading_bot',
    'run_web_server',
    'app'
]
