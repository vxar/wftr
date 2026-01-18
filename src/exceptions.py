"""
Custom exceptions for the trading bot
Provides specific exception types for better error handling
"""

class TradingBotException(Exception):
    """Base exception for all trading bot errors"""
    pass


class ConfigurationError(TradingBotException):
    """Raised when there's a configuration issue"""
    pass


class DataAPIError(TradingBotException):
    """Raised when there's an issue with data API calls"""
    pass


class DataValidationError(TradingBotException):
    """Raised when data validation fails"""
    pass


class InsufficientDataError(DataAPIError):
    """Raised when there's not enough data for analysis"""
    pass


class PatternDetectionError(TradingBotException):
    """Raised when pattern detection fails"""
    pass


class PositionError(TradingBotException):
    """Raised when there's an issue with position management"""
    pass


class RiskManagementError(TradingBotException):
    """Raised when risk management rules are violated"""
    pass


class DatabaseError(TradingBotException):
    """Raised when there's a database operation error"""
    pass


class WebInterfaceError(TradingBotException):
    """Raised when there's an issue with the web interface"""
    pass


class AuthenticationError(TradingBotException):
    """Raised when API authentication fails"""
    pass


class RateLimitError(DataAPIError):
    """Raised when API rate limits are exceeded"""
    pass


class NetworkError(DataAPIError):
    """Raised when there's a network connectivity issue"""
    pass


class MarketDataError(DataAPIError):
    """Raised when market data is invalid or unavailable"""
    pass


class TradingWindowError(TradingBotException):
    """Raised when trading is attempted outside allowed window"""
    pass


class CapitalError(TradingBotException):
    """Raised when there's insufficient capital for trading"""
    pass


class SurgeDetectionError(PatternDetectionError):
    """Raised when surge detection encounters an error"""
    pass
