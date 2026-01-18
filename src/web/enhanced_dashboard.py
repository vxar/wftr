"""
Enhanced Web Dashboard
Comprehensive monitoring and configuration interface for the trading bot
"""
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional
import pytz
from pathlib import Path
import plotly.graph_objs as go
import plotly.utils
from dataclasses import asdict

# Import daily trade analyzer
from ..analysis.daily_trade_analyzer import DailyTradeAnalyzer

logger = logging.getLogger(__name__)

class EnhancedDashboard:
    """
    Enhanced web dashboard with real-time monitoring and configuration
    """
    
    def __init__(self, 
                 trading_bot=None,
                 port: int = 5000,
                 host: str = '0.0.0.0',
                 debug: bool = False):
        """
        Args:
            trading_bot: Reference to the trading bot instance
            port: Port to run the dashboard on
            host: Host address
            debug: Debug mode
        """
        self.trading_bot = trading_bot
        self.port = port
        self.host = host
        self.debug = debug
        self.et_timezone = pytz.timezone('America/New_York')
        
        # Initialize daily trade analyzer
        self.daily_analyzer = DailyTradeAnalyzer()
        
        # Initialize Flask app with correct template folder
        template_folder = Path(__file__).parent.parent.parent / 'templates'
        self.app = Flask(__name__, template_folder=str(template_folder))
        self.app.secret_key = 'trading_bot_secret_key'
        
        # Initialize SocketIO for real-time updates
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Dashboard state
        self.dashboard_state = {
            'last_update': None,
            'connected_clients': 0,
            'alerts': [],
            'notifications': []
        }
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio_events()
        
        # Template data
        self.template_data = {
            'bot_status': 'stopped',
            'current_positions': {},
            'performance_metrics': {},
            'market_conditions': {},
            'recent_trades': [],
            'system_health': {}
        }
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template('enhanced_dashboard.html', **self._get_template_data())
        
        @self.app.route('/positions')
        def positions():
            """Positions page"""
            return render_template('positions.html', **self._get_template_data())
        
        @self.app.route('/trades')
        def trades():
            """Trades history page"""
            return render_template('trades.html', **self._get_template_data())
        
        @self.app.route('/analytics')
        def analytics():
            """Analytics page"""
            return render_template('analytics.html', **self._get_template_data())
        
        @self.app.route('/settings')
        def settings():
            """Settings page"""
            return render_template('settings.html', **self._get_template_data())
        
        @self.app.route('/backtest')
        def backtest():
            """Backtesting page"""
            return render_template('backtest.html', **self._get_template_data())
        
        # API Routes
        
        @self.app.route('/api/status')
        def api_status():
            """Get bot status"""
            return jsonify(self._get_bot_status())
        
        @self.app.route('/api/positions')
        def api_positions():
            """Get current positions"""
            return jsonify(self._get_positions_data())
        
        @self.app.route('/api/trades')
        def api_trades():
            """Get recent trades"""
            limit = request.args.get('limit', 50, type=int)
            return jsonify(self._get_trades_data(limit))
        
        @self.app.route('/api/performance')
        def api_performance():
            """Get performance metrics"""
            return jsonify(self._get_performance_data())
        
        @self.app.route('/api/market')
        def api_market():
            """Get market conditions"""
            return jsonify(self._get_market_data())
        
        @self.app.route('/api/alerts')
        def api_alerts():
            """Get alerts"""
            return jsonify(self._get_alerts())
        
        @self.app.route('/api/chart/equity')
        def api_equity_chart():
            """Get equity chart data"""
            return jsonify(self._get_equity_chart_data())
        
        @self.app.route('/api/chart/performance')
        def api_performance_chart():
            """Get performance chart data"""
            return jsonify(self._get_performance_chart_data())
        
        @self.app.route('/api/daily-analysis')
        def api_daily_analysis():
            """Get daily trade analysis report"""
            try:
                report = self.daily_analyzer.get_latest_report()
                if report:
                    return jsonify(asdict(report))
                else:
                    # Run analysis for today if no report exists
                    today = datetime.now(self.et_timezone).strftime('%Y-%m-%d')
                    report = self.daily_analyzer.run_daily_analysis(today)
                    return jsonify(asdict(report))
            except Exception as e:
                logger.error(f"Error getting daily analysis: {e}")
                return jsonify({'error': str(e)})
        
        @self.app.route('/api/daily-analysis/run', methods=['POST'])
        def api_run_daily_analysis():
            """Run daily analysis manually"""
            try:
                date = request.json.get('date') if request.json else None
                report = self.daily_analyzer.run_daily_analysis(date)
                return jsonify(asdict(report))
            except Exception as e:
                logger.error(f"Error running daily analysis: {e}")
                return jsonify({'error': str(e)})
        
        # Control Routes
        
        @self.app.route('/api/start', methods=['POST'])
        def api_start_bot():
            """Start the trading bot"""
            try:
                if self.trading_bot:
                    self.trading_bot.start()
                    flash('Trading bot started successfully', 'success')
                    return jsonify({'status': 'success', 'message': 'Bot started'})
                else:
                    return jsonify({'status': 'error', 'message': 'Bot not available'})
            except Exception as e:
                logger.error(f"Error starting bot: {e}")
                return jsonify({'status': 'error', 'message': str(e)})
        
        @self.app.route('/api/stop', methods=['POST'])
        def api_stop_bot():
            """Stop the trading bot"""
            try:
                if self.trading_bot:
                    self.trading_bot.stop()
                    flash('Trading bot stopped', 'info')
                    return jsonify({'status': 'success', 'message': 'Bot stopped'})
                else:
                    return jsonify({'status': 'error', 'message': 'Bot not available'})
            except Exception as e:
                logger.error(f"Error stopping bot: {e}")
                return jsonify({'status': 'error', 'message': str(e)})
        
        @self.app.route('/api/settings', methods=['GET', 'POST'])
        def api_settings():
            """Get or update settings"""
            if request.method == 'GET':
                return jsonify(self._get_settings())
            else:
                try:
                    settings_data = request.get_json()
                    self._update_settings(settings_data)
                    flash('Settings updated successfully', 'success')
                    return jsonify({'status': 'success', 'message': 'Settings updated'})
                except Exception as e:
                    logger.error(f"Error updating settings: {e}")
                    return jsonify({'status': 'error', 'message': str(e)})
        
        @self.app.route('/api/position/close', methods=['POST'])
        def api_close_position():
            """Close a specific position"""
            try:
                ticker = request.json.get('ticker')
                if self.trading_bot and ticker:
                    self.trading_bot.close_position(ticker)
                    flash(f'Position {ticker} closed', 'info')
                    return jsonify({'status': 'success', 'message': f'Position {ticker} closed'})
                else:
                    return jsonify({'status': 'error', 'message': 'Invalid request'})
            except Exception as e:
                logger.error(f"Error closing position: {e}")
                return jsonify({'status': 'error', 'message': str(e)})
    
    def _setup_socketio_events(self):
        """Setup SocketIO events for real-time updates"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            self.dashboard_state['connected_clients'] += 1
            logger.info(f"Client connected. Total clients: {self.dashboard_state['connected_clients']}")
            emit('status', self._get_bot_status())
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            self.dashboard_state['connected_clients'] -= 1
            logger.info(f"Client disconnected. Total clients: {self.dashboard_state['connected_clients']}")
        
        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            """Handle subscription to updates"""
            logger.info(f"Client subscribed to: {data}")
    
    def _get_template_data(self) -> Dict:
        """Get template data for rendering"""
        return {
            'bot_status': self._get_bot_status(),
            'positions': self._get_positions_data(),
            'trades': self._get_trades_data(20),
            'performance': self._get_performance_data(),
            'market': self._get_market_data(),
            'settings': self._get_settings(),
            'alerts': self._get_alerts(),
            'last_update': datetime.now(self.et_timezone).strftime('%Y-%m-%d %H:%M:%S ET')
        }
    
    def _get_bot_status(self) -> Dict:
        """Get current bot status"""
        if not self.trading_bot:
            return {
                'status': 'offline',
                'running': False,
                'paused': False,
                'message': 'Bot not available'
            }
        
        try:
            return {
                'status': 'running' if self.trading_bot.running else 'stopped',
                'running': self.trading_bot.running,
                'paused': getattr(self.trading_bot, 'trading_paused', False),
                'current_capital': getattr(self.trading_bot, 'current_capital', 0),
                'daily_profit': getattr(self.trading_bot, 'daily_profit', 0),
                'active_positions': len(getattr(self.trading_bot, 'active_positions', {})),
                'daily_trades': getattr(self.trading_bot, 'daily_trade_count', 0),
                'last_update': datetime.now(self.et_timezone).isoformat(),
                'message': 'Bot operating normally'
            }
        except Exception as e:
            logger.error(f"Error getting bot status: {e}")
            return {
                'status': 'error',
                'running': False,
                'paused': False,
                'message': f'Error: {str(e)}'
            }
    
    def _get_positions_data(self) -> Dict:
        """Get current positions data"""
        if not self.trading_bot:
            return {'positions': [], 'total_value': 0, 'total_unrealized': 0}
        
        try:
            positions = getattr(self.trading_bot, 'active_positions', {})
            positions_data = []
            total_value = 0
            total_unrealized = 0
            
            for ticker, position in positions.items():
                position_data = {
                    'ticker': ticker,
                    'shares': getattr(position, 'shares', 0),
                    'entry_price': getattr(position, 'entry_price', 0),
                    'current_price': getattr(position, 'current_price', 0),
                    'entry_value': getattr(position, 'entry_value', 0),
                    'current_value': getattr(position, 'current_value', 0),
                    'unrealized_pnl': getattr(position, 'unrealized_pnl_dollars', 0),
                    'unrealized_pnl_pct': getattr(position, 'unrealized_pnl_pct', 0),
                    'entry_time': getattr(position, 'entry_time', datetime.now()).isoformat(),
                    'position_type': getattr(position, 'position_type', 'unknown'),
                    'entry_pattern': getattr(position, 'entry_pattern', ''),
                    'max_unrealized_pct': getattr(position, 'max_unrealized_pct', 0)
                }
                positions_data.append(position_data)
                total_value += position_data['current_value']
                total_unrealized += position_data['unrealized_pnl']
            
            return {
                'positions': positions_data,
                'total_value': total_value,
                'total_unrealized': total_unrealized,
                'count': len(positions_data)
            }
            
        except Exception as e:
            logger.error(f"Error getting positions data: {e}")
            return {'positions': [], 'total_value': 0, 'total_unrealized': 0}
    
    def _get_trades_data(self, limit: int = 50) -> Dict:
        """Get recent trades data"""
        if not self.trading_bot:
            return {'trades': [], 'total_count': 0}
        
        try:
            trade_history = getattr(self.trading_bot, 'trade_history', [])
            recent_trades = trade_history[-limit:] if len(trade_history) > limit else trade_history
            
            trades_data = []
            for trade in recent_trades:
                trade_data = {
                    'ticker': getattr(trade, 'ticker', ''),
                    'entry_time': getattr(trade, 'entry_time', datetime.now()).isoformat(),
                    'exit_time': getattr(trade, 'exit_time', datetime.now()).isoformat(),
                    'entry_price': getattr(trade, 'entry_price', 0),
                    'exit_price': getattr(trade, 'exit_price', 0),
                    'shares': getattr(trade, 'shares', 0),
                    'pnl': getattr(trade, 'pnl_dollars', 0),
                    'pnl_pct': getattr(trade, 'pnl_pct', 0),
                    'entry_pattern': getattr(trade, 'entry_pattern', ''),
                    'exit_reason': getattr(trade, 'exit_reason', ''),
                    'confidence': getattr(trade, 'confidence', 0)
                }
                trades_data.append(trade_data)
            
            return {
                'trades': trades_data,
                'total_count': len(trade_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting trades data: {e}")
            return {'trades': [], 'total_count': 0}
    
    def _get_performance_data(self) -> Dict:
        """Get performance metrics"""
        if not self.trading_bot:
            return {}
        
        try:
            # Get performance from various sources
            performance = {}
            
            # Basic metrics
            performance['current_capital'] = getattr(self.trading_bot, 'current_capital', 0)
            performance['initial_capital'] = getattr(self.trading_bot, 'initial_capital', 0)
            performance['daily_profit'] = getattr(self.trading_bot, 'daily_profit', 0)
            
            # Trade statistics
            trade_history = getattr(self.trading_bot, 'trade_history', [])
            if trade_history:
                wins = [t for t in trade_history if getattr(t, 'pnl_dollars', 0) > 0]
                losses = [t for t in trade_history if getattr(t, 'pnl_dollars', 0) < 0]
                
                performance['total_trades'] = len(trade_history)
                performance['winning_trades'] = len(wins)
                performance['losing_trades'] = len(losses)
                performance['win_rate'] = len(wins) / len(trade_history) if trade_history else 0
                performance['avg_win'] = np.mean([getattr(t, 'pnl_dollars', 0) for t in wins]) if wins else 0
                performance['avg_loss'] = np.mean([getattr(t, 'pnl_dollars', 0) for t in losses]) if losses else 0
                
                total_wins = sum(getattr(t, 'pnl_dollars', 0) for t in wins)
                total_losses = abs(sum(getattr(t, 'pnl_dollars', 0) for t in losses))
                performance['profit_factor'] = total_wins / total_losses if total_losses > 0 else float('inf')
            else:
                performance['total_trades'] = 0
                performance['win_rate'] = 0
                performance['profit_factor'] = 0
            
            # Learning system performance
            if hasattr(self.trading_bot, 'learning_system'):
                learning_summary = self.trading_bot.learning_system.get_learning_summary()
                performance['learning_metrics'] = learning_summary
            
            return performance
            
        except Exception as e:
            logger.error(f"Error getting performance data: {e}")
            return {}
    
    def _get_market_data(self) -> Dict:
        """Get market conditions data"""
        if not self.trading_bot:
            return {}
        
        try:
            market_data = {}
            
            # Volatility manager data
            if hasattr(self.trading_bot, 'volatility_manager'):
                market_summary = self.trading_bot.volatility_manager.get_market_summary()
                market_data['volatility'] = market_summary
            
            # Top gainers
            if hasattr(self.trading_bot, 'top_gainers_data'):
                market_data['top_gainers'] = self.trading_bot.top_gainers_data[:10]
            
            # Market time
            current_time = datetime.now(self.et_timezone)
            market_data['current_time'] = current_time.strftime('%Y-%m-%d %H:%M:%S ET')
            market_data['is_trading_hours'] = self._is_trading_hours(current_time)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {}
    
    def _get_settings(self) -> Dict:
        """Get current settings"""
        if not self.trading_bot:
            return {}
        
        try:
            # Get settings from various sources
            settings = {}
            
            # Trading settings
            if hasattr(self.trading_bot, 'min_confidence'):
                settings['trading'] = {
                    'min_confidence': self.trading_bot.min_confidence,
                    'min_entry_price_increase': self.trading_bot.min_entry_price_increase,
                    'trailing_stop_pct': self.trading_bot.trailing_stop_pct,
                    'profit_target_pct': self.trading_bot.profit_target_pct,
                    'position_size_pct': self.trading_bot.position_size_pct,
                    'max_positions': self.trading_bot.max_positions
                }
            
            # Learning system settings
            if hasattr(self.trading_bot, 'learning_system'):
                settings['learning'] = {
                    'mode': self.trading_bot.learning_system.learning_mode.value,
                    'adaptive_parameters': self.trading_bot.learning_system.get_adaptive_parameters()
                }
            
            return settings
            
        except Exception as e:
            logger.error(f"Error getting settings: {e}")
            return {}
    
    def _update_settings(self, settings_data: Dict):
        """Update bot settings"""
        if not self.trading_bot:
            return
        
        try:
            # Update trading settings
            if 'trading' in settings_data:
                trading_settings = settings_data['trading']
                for key, value in trading_settings.items():
                    if hasattr(self.trading_bot, key):
                        setattr(self.trading_bot, key, value)
            
            # Update learning system settings
            if 'learning' in settings_data and hasattr(self.trading_bot, 'learning_system'):
                learning_settings = settings_data['learning']
                if 'mode' in learning_settings:
                    from ..learning.adaptive_learning_system import LearningMode
                    mode = LearningMode(learning_settings['mode'])
                    self.trading_bot.learning_system.set_learning_mode(mode)
                
                if 'adaptive_parameters' in learning_settings:
                    params = learning_settings['adaptive_parameters']
                    for key, value in params.items():
                        self.trading_bot.learning_system.update_adaptive_parameter(key, value)
            
            logger.info("Settings updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            raise
    
    def _get_alerts(self) -> List[Dict]:
        """Get current alerts"""
        alerts = []
        
        if not self.trading_bot:
            return alerts
        
        try:
            # Check for various alert conditions
            
            # High drawdown alert
            performance = self._get_performance_data()
            if 'current_capital' in performance and 'initial_capital' in performance and performance['initial_capital'] > 0:
                drawdown = (performance['initial_capital'] - performance['current_capital']) / performance['initial_capital']
                if drawdown > 0.1:  # 10% drawdown
                    alerts.append({
                        'type': 'warning',
                        'message': f'High drawdown detected: {drawdown:.1%}',
                        'timestamp': datetime.now(self.et_timezone).isoformat()
                    })
            
            # Low win rate alert
            if 'win_rate' in performance and performance['win_rate'] < 0.3 and performance['total_trades'] > 10:
                alerts.append({
                    'type': 'warning',
                    'message': f'Low win rate: {performance["win_rate"]:.1%}',
                    'timestamp': datetime.now(self.et_timezone).isoformat()
                })
            
            # Trading paused alert
            bot_status = self._get_bot_status()
            if bot_status.get('paused', False):
                alerts.append({
                    'type': 'info',
                    'message': 'Trading is currently paused',
                    'timestamp': datetime.now(self.et_timezone).isoformat()
                })
            
            # Market volatility alert
            market_data = self._get_market_data()
            if 'volatility' in market_data and market_data['volatility'].get('current_condition') == 'extreme':
                alerts.append({
                    'type': 'warning',
                    'message': 'Extreme market volatility detected',
                    'timestamp': datetime.now(self.et_timezone).isoformat()
                })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
    
    def _get_equity_chart_data(self) -> Dict:
        """Get equity curve chart data"""
        # This would need to be implemented based on your equity tracking
        return {
            'dates': [],
            'equity_values': [],
            'returns': []
        }
    
    def _get_performance_chart_data(self) -> Dict:
        """Get performance chart data"""
        # This would need to be implemented based on your performance tracking
        return {
            'daily_returns': [],
            'cumulative_returns': [],
            'drawdowns': []
        }
    
    def _is_trading_hours(self, current_time: datetime) -> bool:
        """Check if current time is during trading hours"""
        # Simplified trading hours check
        weekday = current_time.weekday()
        hour = current_time.hour
        minute = current_time.minute
        
        # Monday-Friday, 9:30 AM - 4:00 PM ET
        if weekday < 5 and (9 < hour < 16 or (hour == 9 and minute >= 30) or (hour == 16 and minute == 0)):
            return True
        
        return False
    
    def broadcast_update(self, event_type: str, data: Dict):
        """Broadcast update to all connected clients"""
        try:
            self.socketio.emit(event_type, data)
            self.dashboard_state['last_update'] = datetime.now(self.et_timezone)
        except Exception as e:
            logger.error(f"Error broadcasting update: {e}")
    
    def run(self):
        """Run the dashboard"""
        logger.info(f"Starting enhanced dashboard on {self.host}:{self.port}")
        self.socketio.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=self.debug,
            use_reloader=False
        )
    
    def update_dashboard_data(self):
        """Update dashboard data (call this periodically)"""
        try:
            # Broadcast status update
            self.broadcast_update('status_update', self._get_bot_status())
            
            # Broadcast positions update
            self.broadcast_update('positions_update', self._get_positions_data())
            
            # Broadcast performance update
            self.broadcast_update('performance_update', self._get_performance_data())
            
            # Broadcast alerts
            alerts = self._get_alerts()
            if alerts:
                self.broadcast_update('alerts', alerts)
            
        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")


# Template HTML files would need to be created separately
# Here's a basic structure for the main dashboard template:

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Bot Dashboard</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .metric-card {
            border-left: 4px solid #007bff;
            margin-bottom: 1rem;
        }
        .positive { color: #28a745; }
        .negative { color: #dc3545; }
        .status-running { border-left-color: #28a745; }
        .status-stopped { border-left-color: #dc3545; }
        .status-paused { border-left-color: #ffc107; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container-fluid">
            <a class="navbar-brand" href="/">Trading Bot Dashboard</a>
            <div class="navbar-nav">
                <a class="nav-link" href="/">Dashboard</a>
                <a class="nav-link" href="/positions">Positions</a>
                <a class="nav-link" href="/trades">Trades</a>
                <a class="nav-link" href="/analytics">Analytics</a>
                <a class="nav-link" href="/settings">Settings</a>
            </div>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <div class="row">
            <div class="col-md-12">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h2>Trading Bot Status</h2>
                    <div>
                        <button id="startBtn" class="btn btn-success">Start</button>
                        <button id="stopBtn" class="btn btn-danger">Stop</button>
                        <button id="pauseBtn" class="btn btn-warning">Pause</button>
                        <button id="resumeBtn" class="btn btn-info">Resume</button>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-md-3">
                <div class="card metric-card status-{{ bot_status.status }}">
                    <div class="card-body">
                        <h5 class="card-title">Bot Status</h5>
                        <p class="card-text">{{ bot_status.status|title }}</p>
                        <small class="text-muted">{{ bot_status.message }}</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body">
                        <h5 class="card-title">Current Capital</h5>
                        <p class="card-text">${{ "%.2f"|format(bot_status.current_capital) }}</p>
                        <small class="text-muted">Daily: ${{ "%.2f"|format(bot_status.daily_profit) }}</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body">
                        <h5 class="card-title">Active Positions</h5>
                        <p class="card-text">{{ bot_status.active_positions }}</p>
                        <small class="text-muted">{{ bot_status.daily_trades }} trades today</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body">
                        <h5 class="card-title">Win Rate</h5>
                        <p class="card-text">{{ "%.1%"|format(performance.win_rate) }}</p>
                        <small class="text-muted">{{ performance.total_trades }} total trades</small>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <h5>Equity Curve</h5>
                    </div>
                    <div class="card-body">
                        <div id="equityChart"></div>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h5>Recent Trades</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>Ticker</th>
                                        <th>P&L</th>
                                        <th>Pattern</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for trade in trades.trades[:10] %}
                                    <tr>
                                        <td>{{ trade.ticker }}</td>
                                        <td class="{{ 'positive' if trade.pnl > 0 else 'negative' }}">
                                            ${{ "%.2f"|format(trade.pnl) }}
                                        </td>
                                        <td>{{ trade.entry_pattern }}</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h5>Current Positions</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table">
                                <thead>
                                    <tr>
                                        <th>Ticker</th>
                                        <th>Shares</th>
                                        <th>Entry Price</th>
                                        <th>Current Price</th>
                                        <th>P&L</th>
                                        <th>P&L %</th>
                                        <th>Type</th>
                                        <th>Action</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for position in positions.positions %}
                                    <tr>
                                        <td>{{ position.ticker }}</td>
                                        <td>{{ "%.2f"|format(position.shares) }}</td>
                                        <td>${{ "%.4f"|format(position.entry_price) }}</td>
                                        <td>${{ "%.4f"|format(position.current_price) }}</td>
                                        <td class="{{ 'positive' if position.unrealized_pnl > 0 else 'negative' }}">
                                            ${{ "%.2f"|format(position.unrealized_pnl) }}
                                        </td>
                                        <td class="{{ 'positive' if position.unrealized_pnl_pct > 0 else 'negative' }}">
                                            {{ "%.2f%"|format(position.unrealized_pnl_pct) }}
                                        </td>
                                        <td>{{ position.position_type }}</td>
                                        <td>
                                            <button class="btn btn-sm btn-danger close-position" 
                                                    data-ticker="{{ position.ticker }}">Close</button>
                                        </td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Socket.IO connection
        const socket = io();
        
        // Real-time updates
        socket.on('status_update', function(data) {
            updateBotStatus(data);
        });
        
        socket.on('positions_update', function(data) {
            updatePositions(data);
        });
        
        socket.on('performance_update', function(data) {
            updatePerformance(data);
        });
        
        socket.on('alerts', function(data) {
            showAlerts(data);
        });
        
        // Control buttons
        document.getElementById('startBtn').addEventListener('click', function() {
            fetch('/api/start', {method: 'POST'})
                .then(response => response.json())
                .then(data => console.log(data));
        });
        
        document.getElementById('stopBtn').addEventListener('click', function() {
            fetch('/api/stop', {method: 'POST'})
                .then(response => response.json())
                .then(data => console.log(data));
        });
        
        document.getElementById('pauseBtn').addEventListener('click', function() {
            fetch('/api/pause', {method: 'POST'})
                .then(response => response.json())
                .then(data => console.log(data));
        });
        
        document.getElementById('resumeBtn').addEventListener('click', function() {
            fetch('/api/resume', {method: 'POST'})
                .then(response => response.json())
                .then(data => console.log(data));
        });
        
        // Close position buttons
        document.querySelectorAll('.close-position').forEach(button => {
            button.addEventListener('click', function() {
                const ticker = this.dataset.ticker;
                fetch('/api/position/close', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ticker: ticker})
                })
                .then(response => response.json())
                .then(data => console.log(data));
            });
        });
        
        // Update functions
        function updateBotStatus(data) {
            // Update status display
        }
        
        function updatePositions(data) {
            // Update positions table
        }
        
        function updatePerformance(data) {
            // Update performance metrics
        }
        
        function showAlerts(data) {
            // Show alerts
        }
        
        // Initialize charts
        function initCharts() {
            // Initialize equity chart
        }
        
        // Page load
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
        });
    </script>
</body>
</html>
"""
