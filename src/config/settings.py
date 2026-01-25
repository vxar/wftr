"""
Centralized Configuration Settings
Manages all trading parameters and system settings
"""
from dataclasses import dataclass
from typing import Dict, Any
import os
from pathlib import Path


@dataclass
class TradingConfig:
    """Trading strategy configuration"""
    # Entry requirements
    min_confidence: float = 0.72  # 72% - high-quality trades with reasonable opportunities
    min_entry_price_increase: float = 5.5  # 5.5% expected gain - good quality setups
    min_price_filter: float = 0.50  # Minimum stock price $0.50
    
    # Exit management (optimized from simulator)
    trailing_stop_pct: float = 3.5  # 3.5% trailing stop - increased from 3.0%
    profit_target_pct: float = 8.0  # 8% profit target - unchanged
    max_loss_per_trade_pct: float = 4.0  # Max 4% loss per trade - reduced from 6.0% (widened stops)
    
    # Position sizing
    position_size_pct: float = 0.50  # 50% of capital per trade
    max_positions: int = 5  # Maximum concurrent positions
    
    # Risk management
    max_trades_per_day: int = 1000  # Limit trades per day for quality
    max_daily_loss: float = -3000.0  # Stop trading if daily loss exceeds this
    consecutive_loss_limit: int = 300  # Pause after N consecutive losses


@dataclass
class SurgeDetectionConfig:
    """Surge detection configuration (optimized from simulator)"""
    enabled: bool = True
    min_volume: int = 30000  # Reduced from 50000 (-40%)
    min_volume_ratio: float = 100.0  # Reduced from 150.0 (-33%)
    min_price_increase: float = 30.0
    continuation_min_volume: int = 300000  # Reduced from 500000 (-40%)
    exit_min_hold_minutes: int = 10  # Increased from 5 to 10 (minimum hold time)
    exit_max_hold_minutes: int = 30
    exit_trailing_stop_pct: float = 10.0
    exit_hard_stop_pct: float = 12.0


@dataclass
class CapitalConfig:
    """Capital management configuration"""
    initial_capital: float = 20000.0
    target_capital: float = 100000.0
    daily_profit_target_min: float = 500.0
    daily_profit_target_max: float = 50000.0


@dataclass
class TradingWindowConfig:
    """Trading window configuration"""
    start_time: str = "04:00"  # 4:00 AM ET
    end_time: str = "20:00"    # 8:00 PM ET
    timezone: str = "US/Eastern"


@dataclass
class APIConfig:
    """API configuration"""
    webull_timeout: int = 30
    max_retries: int = 3
    rate_limit_delay: float = 0.1
    cache_ttl_minutes: int = 5


@dataclass
class DatabaseConfig:
    """Database configuration"""
    path: str = "trading_data.db"
    timeout: float = 30.0
    connection_pool_size: int = 5
    enable_wal_mode: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_name: str = "trading_bot.log"
    max_file_size_mb: int = 10
    backup_count: int = 5


@dataclass
class BrokerConfig:
    """Broker integration configuration"""
    enabled: bool = False  # Enable real broker trading
    paper_trading: bool = True  # Use paper trading account
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497  # 7497 for TWS, 7496 for Gateway
    client_id: int = 1
    max_order_size_usd: float = 10000.0
    min_order_size_usd: float = 100.0
    order_timeout_seconds: int = 30  # Timeout for order completion
    max_retries: int = 3  # Maximum retry attempts for failed orders
    retry_delay_seconds: int = 5  # Delay between retries


@dataclass
class WebConfig:
    """Web interface configuration"""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    auto_start_trading: bool = True
    check_on_second: int = 5  # 5th second of every minute


class Settings:
    """Main settings manager"""
    
    def __init__(self, env_file: str = None):
        """Initialize settings with optional environment file"""
        self.trading = TradingConfig()
        self.surge_detection = SurgeDetectionConfig()
        self.capital = CapitalConfig()
        self.trading_window = TradingWindowConfig()
        self.api = APIConfig()
        self.database = DatabaseConfig()
        self.logging = LoggingConfig()
        self.broker = BrokerConfig()
        self.web = WebConfig()
        
        # Load environment overrides if provided
        if env_file and Path(env_file).exists():
            self._load_from_env_file(env_file)
        
        # Load environment variable overrides
        self._load_from_env()
    
    def _load_from_env_file(self, env_file: str):
        """Load settings from .env file"""
        # Simple .env file parser
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    self._set_env_value(key.strip(), value.strip())
    
    def _load_from_env(self):
        """Load settings from environment variables"""
        env_mappings = {
            'TRADING_MIN_CONFIDENCE': ('trading', 'min_confidence', float),
            'TRADING_MIN_ENTRY_PRICE_INCREASE': ('trading', 'min_entry_price_increase', float),
            'TRADING_MIN_PRICE_FILTER': ('trading', 'min_price_filter', float),
            'TRADING_TRAILING_STOP_PCT': ('trading', 'trailing_stop_pct', float),
            'TRADING_PROFIT_TARGET_PCT': ('trading', 'profit_target_pct', float),
            'TRADING_MAX_LOSS_PER_TRADE_PCT': ('trading', 'max_loss_per_trade_pct', float),
            'TRADING_POSITION_SIZE_PCT': ('trading', 'position_size_pct', float),
            'TRADING_MAX_POSITIONS': ('trading', 'max_positions', int),
            'TRADING_MAX_TRADES_PER_DAY': ('trading', 'max_trades_per_day', int),
            'TRADING_MAX_DAILY_LOSS': ('trading', 'max_daily_loss', float),
            'TRADING_CONSECUTIVE_LOSS_LIMIT': ('trading', 'consecutive_loss_limit', int),
            
            'CAPITAL_INITIAL': ('capital', 'initial_capital', float),
            'CAPITAL_TARGET': ('capital', 'target_capital', float),
            'CAPITAL_DAILY_PROFIT_TARGET_MIN': ('capital', 'daily_profit_target_min', float),
            'CAPITAL_DAILY_PROFIT_TARGET_MAX': ('capital', 'daily_profit_target_max', float),
            
            'WEB_HOST': ('web', 'host', str),
            'WEB_PORT': ('web', 'port', int),
            'WEB_DEBUG': ('web', 'debug', lambda x: x.lower() == 'true'),
            
            'DATABASE_PATH': ('database', 'path', str),
            'DATABASE_TIMEOUT': ('database', 'timeout', float),
            
            'LOGGING_LEVEL': ('logging', 'level', str),
            'LOGGING_FILE': ('logging', 'file_name', str),
            
            'BROKER_ENABLED': ('broker', 'enabled', bool),
            'BROKER_PAPER_TRADING': ('broker', 'paper_trading', bool),
            'BROKER_HOST': ('broker', 'ib_host', str),
            'BROKER_PORT': ('broker', 'ib_port', int),
            'BROKER_CLIENT_ID': ('broker', 'client_id', int),
            'BROKER_MAX_ORDER_SIZE': ('broker', 'max_order_size_usd', float),
            'BROKER_MIN_ORDER_SIZE': ('broker', 'min_order_size_usd', float),
            'BROKER_ORDER_TIMEOUT': ('broker', 'order_timeout_seconds', int),
            'BROKER_MAX_RETRIES': ('broker', 'max_retries', int),
            'BROKER_RETRY_DELAY': ('broker', 'retry_delay_seconds', int),
        }
        
        for env_var, (section, attr, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = converter(value)
                    setattr(getattr(self, section), attr, converted_value)
                except (ValueError, AttributeError) as e:
                    print(f"Warning: Invalid environment variable {env_var}={value}: {e}")
    
    def _set_env_value(self, key: str, value: str):
        """Set a configuration value from env file"""
        # Map env keys to config attributes
        env_mappings = {
            'TRADING_MIN_CONFIDENCE': ('trading', 'min_confidence', float),
            'TRADING_MIN_ENTRY_PRICE_INCREASE': ('trading', 'min_entry_price_increase', float),
            # Add more mappings as needed
        }
        
        if key in env_mappings:
            section, attr, converter = env_mappings[key]
            try:
                converted_value = converter(value)
                setattr(getattr(self, section), attr, converted_value)
            except (ValueError, AttributeError):
                pass  # Ignore invalid values
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all settings to dictionary"""
        return {
            'trading': self.trading.__dict__,
            'surge_detection': self.surge_detection.__dict__,
            'capital': self.capital.__dict__,
            'trading_window': self.trading_window.__dict__,
            'api': self.api.__dict__,
            'database': self.database.__dict__,
            'logging': self.logging.__dict__,
            'broker': self.broker.__dict__,
            'web': self.web.__dict__,
        }


# Global settings instance
settings = Settings()
