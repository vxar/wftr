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
from ..database.trading_database import TradingDatabase
from pathlib import Path

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
        
        # Initialize database
        db_path = Path("trading_data.db")
        self.db = TradingDatabase(str(db_path))
        
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
        try:
            self._setup_routes()
            logger.info("All routes registered successfully")
        except Exception as e:
            logger.error(f"Error setting up routes: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
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
        
        @self.app.route('/dashboard')
        def dashboard():
            """Alternative dashboard page (dashboard.html)"""
            return render_template('dashboard.html')
        
        @self.app.route('/positions')
        def positions():
            """Positions page"""
            return render_template('positions.html', **self._get_template_data())
        
        @self.app.route('/trades')
        def trades():
            """Trades history page"""
            template_data = self._get_template_data()
            # Add performance stats directly for trades template
            performance = template_data.get('performance', {})
            win_rate = performance.get('win_rate', 0.0)
            # Convert win_rate from percentage (0-100) to decimal (0-1) if needed
            if win_rate > 1.0:
                win_rate = win_rate / 100.0
            # Get all trades (no limit) for the trades page
            all_trades_data = self._get_trades_data(limit=None)
            # Extract trades list from trades dict (which has 'trades' and 'total_count' keys)
            if isinstance(all_trades_data, dict):
                trades_list = all_trades_data.get('trades', [])
            elif isinstance(all_trades_data, list):
                trades_list = all_trades_data  # Already a list
            else:
                trades_list = []  # Fallback to empty list
            template_data.update({
                'total_pnl': performance.get('total_pnl', 0.0),
                'win_rate': win_rate,
                'winning_trades': performance.get('winning_trades', 0),
                'losing_trades': performance.get('losing_trades', 0),
                'trades': trades_list  # Replace dict with list for template
            })
            return render_template('trades.html', **template_data)
        
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
            positions_data = self._get_positions_data()
            # Check if request wants array format (for dashboard.html)
            wants_array = request.args.get('format') == 'array'
            if wants_array and isinstance(positions_data, dict) and 'positions' in positions_data:
                return jsonify(positions_data['positions'])
            # Return full object for enhanced_dashboard.html
            return jsonify(positions_data)
        
        @self.app.route('/api/trades')
        def api_trades():
            """Get recent trades"""
            limit = request.args.get('limit', 50, type=int)
            trades_data = self._get_trades_data(limit)
            # Check if request wants array format (for dashboard.html)
            wants_array = request.args.get('format') == 'array'
            if wants_array and isinstance(trades_data, dict) and 'trades' in trades_data:
                return jsonify(trades_data['trades'])
            # Return full object by default
            return jsonify(trades_data)
        
        @self.app.route('/api/performance')
        def api_performance():
            """Get performance metrics"""
            return jsonify(self._get_performance_data())
        
        @self.app.route('/api/statistics')
        def api_statistics():
            """Get trading statistics (alias for dashboard.html compatibility)"""
            try:
                stats = self.db.get_statistics()
                return jsonify(stats)
            except Exception as e:
                logger.error(f"Error getting statistics: {e}")
                return jsonify({'error': str(e)})
        
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
        
        @self.app.route('/api/test-route')
        def api_test_route():
            """Test route to verify routing works"""
            return jsonify({'status': 'ok', 'message': 'Route registration works'})
        
        @self.app.route('/api/rejected-entries')
        def api_rejected_entries():
            """Get rejected entry signals"""
            try:
                date = request.args.get('date', None)
                limit = request.args.get('limit', 200, type=int)
                rejected_entries = self.db.get_rejected_entries(date=date, limit=limit)
                return jsonify({
                    'rejected_entries': rejected_entries,
                    'count': len(rejected_entries),
                    'date': date or datetime.now(self.et_timezone).strftime('%Y-%m-%d')
                })
            except Exception as e:
                logger.error(f"Error getting rejected entries: {e}")
                return jsonify({'error': str(e), 'rejected_entries': [], 'count': 0})
        
        @self.app.route('/api/tickers')
        def get_tickers():
            """Get monitored tickers (top gainers)"""
            try:
                # Get tickers from the bot's top gainers data
                if hasattr(self, 'trading_bot') and self.trading_bot:
                    # Check for top_gainers_data (LiveTradingBot)
                    if hasattr(self.trading_bot, 'top_gainers_data') and self.trading_bot.top_gainers_data:
                        # Convert top_gainers_data to format expected by dashboard
                        tickers = []
                        for idx, gainer in enumerate(self.trading_bot.top_gainers_data[:30], 1):
                            ticker_data = {
                                'symbol': gainer.get('symbol', ''),
                                'ticker': gainer.get('symbol', ''),
                                'price': gainer.get('price', 0.0),
                                'change_pct': gainer.get('change_ratio', 0.0) or gainer.get('changeRatio', 0.0),
                                'change': gainer.get('change', 0.0),
                                'volume': gainer.get('volume', 0),
                                'rank': idx,
                                'rank_type': gainer.get('rank_type', 'unknown')
                            }
                            tickers.append(ticker_data)
                        
                        return jsonify({
                            'tickers': tickers,
                            'count': len(tickers),
                            'last_update': datetime.now(self.et_timezone).strftime('%Y-%m-%d %H:%M:%S')
                        })
                    # Fallback to monitored_tickers (AutonomousTradingBot)
                    elif hasattr(self.trading_bot, 'monitored_tickers'):
                        monitored_tickers = self.trading_bot.monitored_tickers
                        last_update = getattr(self.trading_bot, 'last_ticker_update', None)
                        
                        return jsonify({
                            'tickers': monitored_tickers,
                            'count': len(monitored_tickers),
                            'last_update': last_update.strftime('%Y-%m-%d %H:%M:%S') if last_update else None
                        })
                    else:
                        return jsonify({'tickers': [], 'count': 0, 'error': 'Bot tickers not available'})
                else:
                    return jsonify({'tickers': [], 'count': 0, 'error': 'Bot not available'})
            except Exception as e:
                logger.error(f"Error getting tickers: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return jsonify({'error': str(e), 'tickers': []})
        
        @self.app.route('/api/daily-analysis')
        def api_daily_analysis():
            """Get daily trade analysis report"""
            try:
                logger.info("Daily analysis endpoint called")
                if not hasattr(self, 'daily_analyzer'):
                    logger.error("daily_analyzer not initialized")
                    return jsonify({'error': 'Daily analyzer not initialized'})
                
                report = self.daily_analyzer.get_latest_report()
                if report:
                    logger.info(f"Returning existing report: {report}")
                    return jsonify(asdict(report))
                else:
                    # Don't run analysis automatically on page load - just return status
                    logger.info("No existing report found, returning status (not running analysis on page load)")
                    return jsonify({
                        'status': 'no_report',
                        'message': 'No daily analysis report available. Click "Run Analysis" to generate one.',
                        'date': datetime.now(self.et_timezone).strftime('%Y-%m-%d'),
                        'total_trades': 0,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'win_rate': 0.0,
                        'total_pnl': 0.0,
                        'net_profit': 0.0,
                        'largest_win': 0.0,
                        'largest_loss': 0.0,
                        'avg_win': 0.0,
                        'avg_loss': 0.0,
                        'sharpe_ratio': 0.0,
                        'max_drawdown': 0.0
                    })
            except Exception as e:
                logger.error(f"Error getting daily analysis: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return jsonify({'error': str(e)})
        
        @self.app.route('/api/daily-analysis/test')
        def api_daily_analysis_test():
            """Test endpoint for daily analysis"""
            return jsonify({'status': 'ok', 'message': 'Daily analysis routing works'})
        
        @self.app.route('/api/daily-analysis/run', methods=['POST'])
        def api_run_daily_analysis():
            """Run daily analysis manually"""
            try:
                # Handle request with or without JSON body
                date = None
                if request.is_json and request.json:
                    date = request.json.get('date')
                
                # For manual runs, default to today if no date specified
                report = self.daily_analyzer.run_daily_analysis(date, default_to_today=True)
                return jsonify(asdict(report))
            except Exception as e:
                logger.error(f"Error running daily analysis: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
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
                if not self.trading_bot or not ticker:
                    return jsonify({'status': 'error', 'message': 'Invalid request'})
                
                # Handle different bot types
                success = False
                if hasattr(self.trading_bot, 'position_manager'):
                    # AutonomousTradingBot - use position manager
                    from ..core.intelligent_position_manager import ExitReason
                    
                    # Get position details from database first (most reliable source)
                    position_info = None
                    db_positions = self.db.get_active_positions()
                    for db_pos in db_positions:
                        if db_pos.get('ticker') == ticker:
                            entry_price = float(db_pos.get('entry_price', 0))
                            shares = float(db_pos.get('shares', 0))
                            entry_time_str = db_pos.get('entry_time')
                            entry_pattern = db_pos.get('entry_pattern', 'Unknown')
                            confidence = float(db_pos.get('confidence', 0.0))
                            
                            # Parse entry time
                            if entry_time_str:
                                try:
                                    entry_time = pd.to_datetime(entry_time_str)
                                    if entry_time.tzinfo is None:
                                        entry_time = self.et_timezone.localize(entry_time)
                                except:
                                    entry_time = datetime.now(self.et_timezone)
                            else:
                                entry_time = datetime.now(self.et_timezone)
                            
                            # Get current price
                            current_price = entry_price
                            if self.trading_bot and hasattr(self.trading_bot, 'data_api'):
                                try:
                                    df = self.trading_bot.data_api.get_1min_data(ticker, minutes=1)
                                    if df is not None and len(df) > 0:
                                        current_price = float(df.iloc[-1]['close'])
                                    else:
                                        current_price = self.trading_bot.data_api.get_current_price(ticker)
                                        if current_price is None or current_price <= 0:
                                            current_price = entry_price
                                except:
                                    current_price = entry_price
                            
                            position_info = {
                                'entry_price': entry_price,
                                'shares': shares,
                                'entry_time': entry_time,
                                'entry_pattern': entry_pattern,
                                'confidence': confidence,
                                'current_price': current_price
                            }
                            break
                    
                    # If not in database, try memory as fallback
                    if not position_info and ticker in self.trading_bot.position_manager.active_positions:
                        pos = self.trading_bot.position_manager.active_positions[ticker]
                        position_info = {
                            'entry_price': pos.entry_price,
                            'shares': pos.original_shares if hasattr(pos, 'original_shares') else pos.shares,
                            'entry_time': pos.entry_time,
                            'entry_pattern': getattr(pos, 'entry_pattern', 'Unknown'),
                            'confidence': getattr(pos, 'entry_confidence', 0.0),
                            'current_price': pos.current_price
                        }
                    
                    # Exit the position from memory if it exists there
                    if ticker in self.trading_bot.position_manager.active_positions:
                        success = self.trading_bot.position_manager.exit_position(ticker, ExitReason.MANUAL_EXIT)
                    else:
                        # Position not in memory, just mark as success for database cleanup
                        success = True
                    
                    # Always save trade to database if we have position info
                    if position_info:
                        try:
                            from ..database.trading_database import TradeRecord
                            exit_time = datetime.now(self.et_timezone)
                            entry_value = position_info['entry_price'] * position_info['shares']
                            exit_value = position_info['current_price'] * position_info['shares']
                            pnl = exit_value - entry_value
                            pnl_pct = ((position_info['current_price'] - position_info['entry_price']) / position_info['entry_price'] * 100) if position_info['entry_price'] > 0 else 0
                            
                            # Ensure confidence is between 0.0-1.0
                            confidence = position_info['confidence']
                            if confidence > 1.0:
                                confidence = confidence / 100.0
                            elif confidence < 0.0:
                                confidence = 0.0
                            
                            trade_record = TradeRecord(
                                ticker=ticker,
                                entry_time=position_info['entry_time'],
                                exit_time=exit_time,
                                entry_price=position_info['entry_price'],
                                exit_price=position_info['current_price'],
                                shares=position_info['shares'],
                                entry_value=entry_value,
                                exit_value=exit_value,
                                pnl_pct=pnl_pct,
                                pnl_dollars=pnl,
                                entry_pattern=position_info['entry_pattern'],
                                exit_reason='Manual exit',
                                confidence=confidence
                            )
                            trade_id = self.db.add_trade(trade_record)
                            logger.info(f"Manual exit trade saved to database: {ticker} (trade_id: {trade_id})")
                        except Exception as trade_error:
                            logger.error(f"Error saving manual exit trade to database: {trade_error}")
                            import traceback
                            logger.error(traceback.format_exc())
                    else:
                        logger.warning(f"Could not get position info for {ticker} - trade not saved")
                    
                    # Close position in database
                    try:
                        self.db.close_position(ticker)
                    except Exception as db_error:
                        logger.warning(f"Error closing position in database: {db_error}")
                elif hasattr(self.trading_bot, 'trader') and hasattr(self.trading_bot.trader, 'active_positions'):
                    # LiveTradingBot - use trader's exit method
                    if ticker in self.trading_bot.trader.active_positions:
                        position = self.trading_bot.trader.active_positions[ticker]
                        # Create exit signal
                        from ..core.realtime_trader import TradeSignal
                        from datetime import datetime
                        exit_signal = TradeSignal(
                            signal_type='exit',
                            ticker=ticker,
                            price=position.current_price,
                            timestamp=datetime.now(self.et_timezone),
                            reason='Manual exit'
                        )
                        # Execute exit
                        trade = self.trading_bot._execute_exit(exit_signal)
                        success = trade is not None
                elif hasattr(self.trading_bot, 'close_position'):
                    # Fallback: try direct method if it exists
                    self.trading_bot.close_position(ticker)
                    success = True
                else:
                    # Last resort: close directly in database
                    self.db.close_position(ticker)
                    success = True
                
                if success:
                    flash(f'Position {ticker} closed', 'info')
                    return jsonify({'status': 'success', 'message': f'Position {ticker} closed'})
                else:
                    return jsonify({'status': 'error', 'message': f'Failed to close position {ticker}'})
                    
            except Exception as e:
                logger.error(f"Error closing position: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return jsonify({'status': 'error', 'message': str(e)})
        
        @self.app.route('/api/positions/close', methods=['POST'])
        def api_positions_close():
            """Close a specific position (alias for dashboard.html compatibility)"""
            try:
                ticker = request.json.get('ticker')
                if not self.trading_bot or not ticker:
                    return jsonify({'error': 'Invalid request'})
                
                # Handle different bot types
                success = False
                if hasattr(self.trading_bot, 'position_manager'):
                    # AutonomousTradingBot - use position manager
                    from ..core.intelligent_position_manager import ExitReason
                    
                    # Get position details from database first (most reliable source)
                    position_info = None
                    db_positions = self.db.get_active_positions()
                    for db_pos in db_positions:
                        if db_pos.get('ticker') == ticker:
                            entry_price = float(db_pos.get('entry_price', 0))
                            shares = float(db_pos.get('shares', 0))
                            entry_time_str = db_pos.get('entry_time')
                            entry_pattern = db_pos.get('entry_pattern', 'Unknown')
                            confidence = float(db_pos.get('confidence', 0.0))
                            
                            # Parse entry time
                            if entry_time_str:
                                try:
                                    entry_time = pd.to_datetime(entry_time_str)
                                    if entry_time.tzinfo is None:
                                        entry_time = self.et_timezone.localize(entry_time)
                                except:
                                    entry_time = datetime.now(self.et_timezone)
                            else:
                                entry_time = datetime.now(self.et_timezone)
                            
                            # Get current price
                            current_price = entry_price
                            if self.trading_bot and hasattr(self.trading_bot, 'data_api'):
                                try:
                                    df = self.trading_bot.data_api.get_1min_data(ticker, minutes=1)
                                    if df is not None and len(df) > 0:
                                        current_price = float(df.iloc[-1]['close'])
                                    else:
                                        current_price = self.trading_bot.data_api.get_current_price(ticker)
                                        if current_price is None or current_price <= 0:
                                            current_price = entry_price
                                except:
                                    current_price = entry_price
                            
                            position_info = {
                                'entry_price': entry_price,
                                'shares': shares,
                                'entry_time': entry_time,
                                'entry_pattern': entry_pattern,
                                'confidence': confidence,
                                'current_price': current_price
                            }
                            break
                    
                    # If not in database, try memory as fallback
                    if not position_info and ticker in self.trading_bot.position_manager.active_positions:
                        pos = self.trading_bot.position_manager.active_positions[ticker]
                        position_info = {
                            'entry_price': pos.entry_price,
                            'shares': pos.original_shares if hasattr(pos, 'original_shares') else pos.shares,
                            'entry_time': pos.entry_time,
                            'entry_pattern': getattr(pos, 'entry_pattern', 'Unknown'),
                            'confidence': getattr(pos, 'entry_confidence', 0.0),
                            'current_price': pos.current_price
                        }
                    
                    # Exit the position from memory if it exists there
                    if ticker in self.trading_bot.position_manager.active_positions:
                        success = self.trading_bot.position_manager.exit_position(ticker, ExitReason.MANUAL_EXIT)
                    else:
                        # Position not in memory, just mark as success for database cleanup
                        success = True
                    
                    # Always save trade to database if we have position info
                    if position_info:
                        try:
                            from ..database.trading_database import TradeRecord
                            exit_time = datetime.now(self.et_timezone)
                            entry_value = position_info['entry_price'] * position_info['shares']
                            exit_value = position_info['current_price'] * position_info['shares']
                            pnl = exit_value - entry_value
                            pnl_pct = ((position_info['current_price'] - position_info['entry_price']) / position_info['entry_price'] * 100) if position_info['entry_price'] > 0 else 0
                            
                            # Ensure confidence is between 0.0-1.0
                            confidence = position_info['confidence']
                            if confidence > 1.0:
                                confidence = confidence / 100.0
                            elif confidence < 0.0:
                                confidence = 0.0
                            
                            trade_record = TradeRecord(
                                ticker=ticker,
                                entry_time=position_info['entry_time'],
                                exit_time=exit_time,
                                entry_price=position_info['entry_price'],
                                exit_price=position_info['current_price'],
                                shares=position_info['shares'],
                                entry_value=entry_value,
                                exit_value=exit_value,
                                pnl_pct=pnl_pct,
                                pnl_dollars=pnl,
                                entry_pattern=position_info['entry_pattern'],
                                exit_reason='Manual exit',
                                confidence=confidence
                            )
                            trade_id = self.db.add_trade(trade_record)
                            logger.info(f"Manual exit trade saved to database: {ticker} (trade_id: {trade_id})")
                        except Exception as trade_error:
                            logger.error(f"Error saving manual exit trade to database: {trade_error}")
                            import traceback
                            logger.error(traceback.format_exc())
                    else:
                        logger.warning(f"Could not get position info for {ticker} - trade not saved")
                    
                    # Close position in database
                    try:
                        self.db.close_position(ticker)
                    except Exception as db_error:
                        logger.warning(f"Error closing position in database: {db_error}")
                elif hasattr(self.trading_bot, 'trader') and hasattr(self.trading_bot.trader, 'active_positions'):
                    # LiveTradingBot - use trader's exit method
                    if ticker in self.trading_bot.trader.active_positions:
                        position = self.trading_bot.trader.active_positions[ticker]
                        # Create exit signal
                        from ..core.realtime_trader import TradeSignal
                        from datetime import datetime
                        exit_signal = TradeSignal(
                            signal_type='exit',
                            ticker=ticker,
                            price=position.current_price,
                            timestamp=datetime.now(self.et_timezone),
                            reason='Manual exit'
                        )
                        # Execute exit
                        trade = self.trading_bot._execute_exit(exit_signal)
                        success = trade is not None
                elif hasattr(self.trading_bot, 'close_position'):
                    # Fallback: try direct method if it exists
                    self.trading_bot.close_position(ticker)
                    success = True
                else:
                    # Last resort: close directly in database
                    self.db.close_position(ticker)
                    success = True
                
                if success:
                    return jsonify({'success': True, 'message': f'Position {ticker} closed'})
                else:
                    return jsonify({'error': f'Failed to close position {ticker}'})
                    
            except Exception as e:
                logger.error(f"Error closing position: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return jsonify({'error': str(e)})
        
        @self.app.route('/api/positions/update', methods=['POST'])
        def api_positions_update():
            """Update position target price and stop loss"""
            try:
                data = request.json
                ticker = data.get('ticker')
                target_price = data.get('target_price')
                stop_loss = data.get('stop_loss')
                
                if not ticker:
                    return jsonify({'error': 'Ticker is required'})
                
                # Update position in database
                success = self.db.update_position(ticker, target_price=target_price, stop_loss=stop_loss)
                
                if success:
                    return jsonify({'success': True, 'message': f'Position {ticker} updated'})
                else:
                    return jsonify({'error': f'Failed to update position {ticker}'})
            except Exception as e:
                logger.error(f"Error updating position: {e}")
                return jsonify({'error': str(e)})
        
        @self.app.route('/api/monitoring')
        def api_monitoring():
            """Get monitoring data (top gainers)"""
            try:
                # Get top gainers from bot
                top_gainers = []
                last_refresh = None
                
                if self.trading_bot:
                    # Check for monitored_tickers (AutonomousTradingBot)
                    if hasattr(self.trading_bot, 'monitored_tickers') and self.trading_bot.monitored_tickers:
                        for item in self.trading_bot.monitored_tickers:
                            if isinstance(item, dict):
                                ticker = item.get('ticker') or item.get('symbol', '')
                                if ticker:
                                    top_gainers.append({
                                        'ticker': ticker,
                                        'current_price': item.get('current_price', item.get('price', 0)),
                                        'change_pct': item.get('change_pct', item.get('change_percent', 0)),
                                        'status': item.get('status', 'monitoring')
                                    })
                        last_refresh = getattr(self.trading_bot, 'last_ticker_update', None)
                    # Check for top_gainers_data (LiveTradingBot)
                    elif hasattr(self.trading_bot, 'top_gainers_data') and self.trading_bot.top_gainers_data:
                        for gainer in self.trading_bot.top_gainers_data[:30]:
                            ticker = gainer.get('symbol', '')
                            if ticker:
                                # Determine status
                                status = 'monitoring'
                                if self.trading_bot and hasattr(self.trading_bot, 'position_manager'):
                                    position_summary = self.trading_bot.position_manager.get_position_summary()
                                    if ticker in position_summary.get('active_positions', {}):
                                        status = 'active_position'
                                
                                top_gainers.append({
                                    'ticker': ticker,
                                    'current_price': gainer.get('price', 0),
                                    'change_pct': gainer.get('change_ratio', 0) or gainer.get('changeRatio', 0),
                                    'status': status
                                })
                        last_refresh = datetime.now(self.et_timezone)
                
                return jsonify({
                    'top_gainers': top_gainers,
                    'last_refresh': last_refresh.isoformat() if last_refresh else None
                })
            except Exception as e:
                logger.error(f"Error getting monitoring data: {e}")
                return jsonify({'error': str(e), 'top_gainers': [], 'last_refresh': None})
    
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
            'monitored_tickers': self._get_monitored_tickers(),
            'last_update': datetime.now(self.et_timezone).strftime('%Y-%m-%d %H:%M:%S ET')
        }
    
    def _get_bot_status(self) -> Dict:
        """Get current bot status from database and bot instance"""
        try:
            # Get bot running status
            running = False
            if self.trading_bot:
                running = getattr(self.trading_bot, 'running', False)
            
            # Get initial capital from bot config
            initial_capital = 10000.0
            max_positions = 5
            if self.trading_bot:
                initial_capital = getattr(self.trading_bot, 'initial_capital', 10000.0)
                if not initial_capital:
                    initial_capital = getattr(self.trading_bot, 'config', {}).get('initial_capital', 10000.0)
                max_positions = getattr(self.trading_bot, 'config', {}).get('max_positions', 5)
            
            # Get current capital from database (cash available)
            current_capital = self.db.get_current_capital_from_db(initial_capital)
            
            # Get daily profit from database
            daily_data = self.db.get_daily_profit_from_db(initial_capital)
            daily_profit = daily_data.get('daily_profit', 0.0)
            
            # Get active positions from database
            active_positions_db = self.db.get_active_positions()
            active_positions_count = len(active_positions_db)
            
            # Calculate portfolio value (cash + current value of active positions)
            portfolio_value = current_capital
            for pos in active_positions_db:
                # Try to get current price from bot if available
                ticker = pos.get('ticker')
                if ticker and self.trading_bot and hasattr(self.trading_bot, 'data_api'):
                    try:
                        current_price = self.trading_bot.data_api.get_current_price(ticker)
                        if current_price:
                            shares = pos.get('shares', 0)
                            portfolio_value += current_price * shares
                    except:
                        # Fallback to entry price if current price unavailable
                        entry_price = pos.get('entry_price', 0)
                        shares = pos.get('shares', 0)
                        if entry_price and shares:
                            portfolio_value += entry_price * shares
            
            # Get total trades from database statistics
            stats = self.db.get_statistics()
            total_trades = stats.get('total_trades', 0)
            
            # Get monitored tickers count from bot
            tickers_monitored = 0
            if self.trading_bot:
                if hasattr(self.trading_bot, 'monitored_tickers'):
                    tickers_monitored = len(self.trading_bot.monitored_tickers) if self.trading_bot.monitored_tickers else 0
                elif hasattr(self.trading_bot, 'top_gainers_data'):
                    tickers_monitored = len(self.trading_bot.top_gainers_data) if self.trading_bot.top_gainers_data else 0
            
            # Calculate total return percentage
            total_return_pct = ((portfolio_value - initial_capital) / initial_capital * 100) if initial_capital > 0 else 0.0
            
            return {
                'status': 'running' if running else 'stopped',
                'running': running,
                'current_capital': current_capital,
                'initial_capital': initial_capital,
                'portfolio_value': portfolio_value,
                'total_return_pct': total_return_pct,
                'daily_profit': daily_profit,
                'active_positions_count': active_positions_count,
                'active_positions': active_positions_count,  # Keep for backward compatibility
                'max_positions': max_positions,
                'total_trades': total_trades,
                'tickers_monitored': tickers_monitored,
                'daily_trades': daily_data.get('daily_trades_count', 0),
                'last_update': datetime.now(self.et_timezone).isoformat(),
                'message': 'Bot operating normally' if running else 'Bot stopped'
            }
        except Exception as e:
            logger.error(f"Error getting bot status: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'running': False,
                'message': f'Error: {str(e)}'
            }
    
    def _get_positions_data(self) -> Dict:
        """Get current positions data from database"""
        try:
            # Get active positions from database
            db_positions = self.db.get_active_positions()
            positions_data = []
            total_value = 0
            total_unrealized = 0
            
            for db_pos in db_positions:
                ticker = db_pos.get('ticker')
                if not ticker:
                    continue
                
                # Get position details from database
                entry_price = float(db_pos.get('entry_price', 0))
                shares_raw = float(db_pos.get('shares', 0))
                # Don't round to 0 - keep fractional shares for positions with partial profits
                # Only round for display purposes, but keep original for calculations
                shares = shares_raw if shares_raw > 0 else 0
                entry_value = float(db_pos.get('entry_value', 0))
                
                # Skip positions with no shares (fully closed)
                if shares <= 0:
                    logger.debug(f"Skipping position {ticker} with zero or negative shares: {shares}")
                    continue
                entry_time_str = db_pos.get('entry_time')
                entry_pattern = db_pos.get('entry_pattern', 'Unknown')
                confidence = float(db_pos.get('confidence', 0.0))
                target_price = db_pos.get('target_price')
                stop_loss = db_pos.get('stop_loss')
                
                # Parse entry time
                if entry_time_str:
                    try:
                        entry_time = pd.to_datetime(entry_time_str)
                        if entry_time.tzinfo is None:
                            entry_time = self.et_timezone.localize(entry_time)
                    except:
                        entry_time = datetime.now(self.et_timezone)
                else:
                    entry_time = datetime.now(self.et_timezone)
                
                # Get current price from bot's data API if available
                current_price = entry_price
                if self.trading_bot and hasattr(self.trading_bot, 'data_api'):
                    try:
                        # Try to get current price from 1-minute data first (more accurate)
                        df = self.trading_bot.data_api.get_1min_data(ticker, minutes=1)
                        if df is not None and len(df) > 0:
                            current_price = float(df.iloc[-1]['close'])
                        else:
                            # Fallback to get_current_price
                            current_price = self.trading_bot.data_api.get_current_price(ticker)
                            if current_price is None or current_price <= 0:
                                current_price = entry_price
                    except Exception as e:
                        logger.debug(f"Error getting current price for {ticker}: {e}")
                        current_price = entry_price
                
                # Calculate current value and P&L
                current_value = current_price * shares
                unrealized_pnl = current_value - entry_value
                # Calculate P&L % as decimal (e.g., -0.0211 for -2.11%), template will multiply by 100
                unrealized_pnl_pct = ((current_price - entry_price) / entry_price) if entry_price > 0 else 0
                
                # Log for debugging
                logger.debug(f"Position {ticker}: entry=${entry_price:.4f}, current=${current_price:.4f}, shares={shares:.2f}, pnl=${unrealized_pnl:.2f}")
                
                # Calculate time in position
                time_in_position = (datetime.now(self.et_timezone) - entry_time).total_seconds() / 60
                
                position_data = {
                    'ticker': ticker,
                    'shares': round(shares) if shares >= 1 else shares,  # Round whole shares, keep fractional
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'entry_value': entry_value,
                    'current_value': current_value,
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_pct': unrealized_pnl_pct,
                    'entry_time': entry_time.isoformat(),
                    'time_in_minutes': time_in_position,
                    'entry_pattern': entry_pattern,
                    'confidence': confidence,
                    'target_price': float(target_price) if target_price else None,
                    'stop_loss': float(stop_loss) if stop_loss else None,
                    'partial_profit_taken': bool(db_pos.get('partial_profit_taken', False))  # Indicate if partial profit was taken
                }
                positions_data.append(position_data)
                total_value += current_value
                total_unrealized += unrealized_pnl
            
            return {
                'positions': positions_data,
                'total_value': total_value,
                'total_unrealized': total_unrealized,
                'count': len(positions_data)
            }
            
        except Exception as e:
            logger.error(f"Error getting positions data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'positions': [], 'total_value': 0, 'total_unrealized': 0}
    
    def _get_trades_data(self, limit: int = 50) -> Dict:
        """Get recent trades data from database"""
        try:
            # Get all trades from database
            all_trades = self.db.get_all_trades(limit=limit)
            
            trades_data = []
            for trade in all_trades:
                try:
                    ticker = trade.get('ticker', '')
                    if not ticker:
                        continue
                    
                    # Parse timestamps
                    entry_time_str = trade.get('entry_time')
                    exit_time_str = trade.get('exit_time')
                    
                    if entry_time_str:
                        try:
                            entry_time = pd.to_datetime(entry_time_str)
                            if entry_time.tzinfo is None:
                                entry_time = self.et_timezone.localize(entry_time)
                        except:
                            entry_time = datetime.now(self.et_timezone)
                    else:
                        entry_time = datetime.now(self.et_timezone)
                    
                    if exit_time_str:
                        try:
                            exit_time = pd.to_datetime(exit_time_str)
                            if exit_time.tzinfo is None:
                                exit_time = self.et_timezone.localize(exit_time)
                        except:
                            exit_time = datetime.now(self.et_timezone)
                    else:
                        exit_time = datetime.now(self.et_timezone)
                    
                    # Get trade values
                    entry_price = float(trade.get('entry_price', 0))
                    exit_price = float(trade.get('exit_price', 0))
                    shares = round(float(trade.get('shares', 0)))  # Round to whole number
                    entry_value = float(trade.get('entry_value', 0))
                    exit_value = float(trade.get('exit_value', 0))
                    pnl_dollars = float(trade.get('pnl_dollars', 0))
                    pnl_pct = float(trade.get('pnl_pct', 0))
                    entry_pattern = trade.get('entry_pattern', 'Unknown')
                    exit_reason = trade.get('exit_reason', 'Unknown')
                    confidence = float(trade.get('confidence', 0.0))
                    
                    # Calculate hold time
                    hold_time_minutes = (exit_time - entry_time).total_seconds() / 60
                    
                    # Format duration for display
                    if hold_time_minutes < 60:
                        duration = f"{int(hold_time_minutes)}m"
                    elif hold_time_minutes < 1440:  # Less than 24 hours
                        hours = int(hold_time_minutes / 60)
                        minutes = int(hold_time_minutes % 60)
                        duration = f"{hours}h {minutes}m"
                    else:
                        days = int(hold_time_minutes / 1440)
                        hours = int((hold_time_minutes % 1440) / 60)
                        duration = f"{days}d {hours}h"
                    
                    # Determine win/loss status
                    status = 'WIN' if pnl_dollars > 0 else 'LOSS' if pnl_dollars < 0 else 'BREAKEVEN'
                    
                    trade_data = {
                        'id': trade.get('id'),
                        'ticker': ticker,
                        'entry_time': entry_time.isoformat(),
                        'exit_time': exit_time.isoformat(),
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'shares': int(shares),  # Store as integer
                        'entry_value': entry_value,
                        'exit_value': exit_value,
                        'pnl': pnl_dollars,
                        'pnl_pct': pnl_pct,
                        'entry_pattern': entry_pattern,
                        'exit_reason': exit_reason,
                        'confidence': confidence,
                        'hold_time_minutes': hold_time_minutes,
                        'duration': duration,  # Formatted duration for display
                        'status': status
                    }
                    trades_data.append(trade_data)
                except Exception as e:
                    logger.warning(f"Error processing trade: {e}")
                    continue
            
            # Get total count (without limit)
            total_trades = self.db.get_all_trades(limit=None)
            total_count = len(total_trades) if total_trades else 0
            
            return {
                'trades': trades_data,
                'total_count': total_count
            }
            
        except Exception as e:
            logger.error(f"Error getting trades data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'trades': [], 'total_count': 0}
    
    def _get_performance_data(self) -> Dict:
        """Get performance metrics from database"""
        try:
            # Get statistics from database
            stats = self.db.get_statistics()
            
            # Get initial capital from bot config
            initial_capital = 10000.0
            if self.trading_bot:
                initial_capital = getattr(self.trading_bot, 'initial_capital', 10000.0)
                if not initial_capital:
                    initial_capital = getattr(self.trading_bot, 'config', {}).get('initial_capital', 10000.0)
            
            # Get current capital and daily profit from database
            current_capital = self.db.get_current_capital_from_db(initial_capital)
            daily_data = self.db.get_daily_profit_from_db(initial_capital)
            daily_profit = daily_data.get('daily_profit', 0.0)
            portfolio_value = daily_data.get('portfolio_value', initial_capital)
            
            # Calculate total return
            total_return = portfolio_value - initial_capital
            total_return_pct = (total_return / initial_capital * 100) if initial_capital > 0 else 0.0
            
            # Build performance dictionary
            performance = {
                'current_capital': current_capital,
                'initial_capital': initial_capital,
                'portfolio_value': portfolio_value,
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'daily_profit': daily_profit,
                'total_trades': stats.get('total_trades', 0),
                'winning_trades': stats.get('winning_trades', 0),
                'losing_trades': stats.get('losing_trades', 0),
                'win_rate': stats.get('win_rate', 0.0),
                'total_pnl': stats.get('total_pnl', 0.0),
                'avg_pnl': stats.get('avg_pnl', 0.0),
                'best_trade': stats.get('best_trade'),
                'worst_trade': stats.get('worst_trade')
            }
            
            # Calculate additional metrics if we have trades
            if performance['total_trades'] > 0:
                # Calculate profit factor from stats
                if performance['losing_trades'] > 0:
                    avg_win = performance['winning_trades'] > 0 and (performance['total_pnl'] / performance['winning_trades']) or 0
                    avg_loss = abs(performance['total_pnl'] / performance['losing_trades']) if performance['losing_trades'] > 0 else 0
                    performance['profit_factor'] = avg_win / avg_loss if avg_loss > 0 else float('inf')
                else:
                    performance['profit_factor'] = float('inf')
            else:
                performance['profit_factor'] = 0.0
            
            return performance
            
        except Exception as e:
            logger.error(f"Error getting performance data: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
    
    def _get_monitored_tickers(self) -> List:
        """Get monitored tickers for display"""
        try:
            if hasattr(self, 'trading_bot') and self.trading_bot:
                # Get tickers from bot's monitored list (populated by trading loop)
                if hasattr(self.trading_bot, 'monitored_tickers'):
                    return self.trading_bot.monitored_tickers
                else:
                    return []
            else:
                return []
        except Exception as e:
            logger.error(f"Error getting monitored tickers: {e}")
            return []
    
    def _get_ticker_recommendation(self, ticker_data: Dict) -> str:
        """Generate recommendation based on ticker analysis"""
        try:
            score = 0
            reasons = []
            
            # Quality and surge analysis
            if ticker_data['quality_score'] > 0.7:
                score += 2
                reasons.append("high quality")
            elif ticker_data['quality_score'] > 0.5:
                score += 1
                reasons.append("medium quality")
            
            if ticker_data['surge_score'] > 0.8:
                score += 2
                reasons.append("strong surge")
            elif ticker_data['surge_score'] > 0.6:
                score += 1
                reasons.append("moderate surge")
            
            # Manipulation check
            if ticker_data['manipulation_score'] < 0.3:
                score += 1
                reasons.append("low manipulation risk")
            elif ticker_data['manipulation_score'] > 0.7:
                score -= 2
                reasons.append("high manipulation risk")
            
            # Risk/reward analysis
            if ticker_data['risk_reward_ratio'] > 2.0:
                score += 2
                reasons.append("excellent risk/reward")
            elif ticker_data['risk_reward_ratio'] > 1.5:
                score += 1
                reasons.append("good risk/reward")
            elif ticker_data['risk_reward_ratio'] < 1.0:
                score -= 1
                reasons.append("poor risk/reward")
            
            # Technical indicators
            if ticker_data['rsi'] < 30:
                score += 1
                reasons.append("oversold")
            elif ticker_data['rsi'] > 70:
                score -= 1
                reasons.append("overbought")
            
            # Momentum
            if ticker_data['momentum'] > 0.5:
                score += 1
                reasons.append("positive momentum")
            elif ticker_data['momentum'] < -0.5:
                score -= 1
                reasons.append("negative momentum")
            
            # Generate recommendation
            if score >= 4:
                return "STRONG BUY"
            elif score >= 2:
                return "BUY"
            elif score >= 0:
                return "HOLD"
            elif score >= -2:
                return "SELL"
            else:
                return "STRONG SELL"
                
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return "HOLD"
    
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
        """Get equity curve chart data from trades"""
        try:
            # Get initial capital
            initial_capital = 10000.0
            if self.trading_bot:
                initial_capital = getattr(self.trading_bot, 'initial_capital', 10000.0)
                if not initial_capital:
                    initial_capital = getattr(self.trading_bot, 'config', {}).get('initial_capital', 10000.0)
            
            # Get today's date
            today = datetime.now(self.et_timezone).strftime('%Y-%m-%d')
            
            # Get all trades for today
            all_trades = self.db.get_all_trades()
            today_trades = []
            
            for trade in all_trades:
                exit_time = trade.get('exit_time', '')
                if exit_time and exit_time.startswith(today):
                    today_trades.append(trade)
            
            # Sort by exit time
            today_trades.sort(key=lambda x: x.get('exit_time', ''))
            
            # Calculate starting capital for today (initial + cumulative P&L before today)
            daily_data = self.db.get_daily_profit_from_db(initial_capital, today)
            starting_capital = daily_data.get('daily_start_capital', initial_capital)
            
            # Build equity curve
            timestamps = []
            equity_values = []
            
            # Start with opening balance
            market_open = datetime.now(self.et_timezone).replace(hour=9, minute=30, second=0, microsecond=0)
            timestamps.append(market_open.strftime('%H:%M'))
            equity_values.append(starting_capital)
            
            # Add each trade's impact
            cumulative_pnl = 0
            for trade in today_trades:
                exit_time_str = trade.get('exit_time', '')
                pnl = trade.get('pnl_dollars', 0)
                
                try:
                    exit_dt = pd.to_datetime(exit_time_str)
                    if exit_dt.tzinfo is None:
                        exit_dt = self.et_timezone.localize(exit_dt)
                    
                    cumulative_pnl += pnl
                    timestamps.append(exit_dt.strftime('%H:%M'))
                    equity_values.append(starting_capital + cumulative_pnl)
                except:
                    continue
            
            # Add current time with current portfolio value
            current_time = datetime.now(self.et_timezone)
            current_positions = self.db.get_active_positions()
            
            # Calculate unrealized P&L
            unrealized_pnl = 0
            for pos in current_positions:
                ticker = pos.get('ticker')
                entry_price = float(pos.get('entry_price', 0))
                shares = float(pos.get('shares', 0))
                entry_value = float(pos.get('entry_value', 0))
                
                if ticker and self.trading_bot and hasattr(self.trading_bot, 'data_api'):
                    try:
                        current_price = self.trading_bot.data_api.get_current_price(ticker)
                        if current_price and current_price > 0:
                            current_value = current_price * shares
                            unrealized_pnl += (current_value - entry_value)
                    except:
                        pass
            
            timestamps.append(current_time.strftime('%H:%M'))
            equity_values.append(starting_capital + cumulative_pnl + unrealized_pnl)
            
            return {
                'timestamps': timestamps,
                'equity_values': equity_values,
                'starting_capital': starting_capital,
                'current_value': equity_values[-1] if equity_values else starting_capital
            }
            
        except Exception as e:
            logger.error(f"Error getting equity chart data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'timestamps': [],
                'equity_values': [],
                'starting_capital': 10000.0,
                'current_value': 10000.0
            }
    
    def _get_performance_chart_data(self) -> Dict:
        """Get performance chart data (alias for equity chart)"""
        return self._get_equity_chart_data()
    
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
        
        # Auto-start trading bot if configured
        if hasattr(self, 'trading_bot') and self.trading_bot:
            try:
                # Always start bot for testing (remove trading hours restriction)
                if not self.trading_bot.running:
                    logger.info("Auto-starting trading bot...")
                    self.trading_bot.start()
                else:
                    logger.info("Bot already running")
            except Exception as e:
                logger.error(f"Error auto-starting bot: {e}")
        
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
                                        <td>{{ position.shares|int }}</td>
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
