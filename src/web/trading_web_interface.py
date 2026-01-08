"""
Trading Bot Web Interface
Provides a web dashboard to view current and completed trades
"""
from flask import Flask, render_template, jsonify
from typing import Dict, List, Optional
from datetime import datetime
import threading
import logging
from pathlib import Path
import pytz
import pandas as pd

logger = logging.getLogger(__name__)

# Disable Flask/Werkzeug request logging
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# Set template folder path
# Path structure: src/web/trading_web_interface.py -> src/web/ -> src/ -> root/ -> templates/
# From src/web/, we need to go up 2 levels to get to root
template_dir = Path(__file__).parent.parent.parent / 'templates'
app = Flask(__name__, template_folder=str(template_dir))

# Global reference to the trading bot
trading_bot = None


def set_trading_bot(bot):
    """Set the trading bot instance for the web interface"""
    global trading_bot
    trading_bot = bot
    logger.info(f"Trading bot set in web interface: {bot is not None}")
    if bot:
        logger.info(f"Bot has {len(bot.tickers)} tickers, running={getattr(bot, 'running', False)}")


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/debug')
def debug_info():
    """Debug endpoint to check bot status"""
    global trading_bot
    return jsonify({
        'bot_is_none': trading_bot is None,
        'bot_type': str(type(trading_bot)) if trading_bot else 'None',
        'bot_running': getattr(trading_bot, 'running', False) if trading_bot else False,
        'bot_tickers': len(getattr(trading_bot, 'tickers', [])) if trading_bot else 0,
        'bot_has_db': hasattr(trading_bot, 'db') if trading_bot else False,
        'bot_thread_alive': getattr(trading_bot, '_bot_thread', None) is not None and getattr(trading_bot._bot_thread, 'is_alive', lambda: False)() if trading_bot else False
    })


@app.route('/api/status')
def get_status():
    """Get overall bot status"""
    if not trading_bot:
        return jsonify({'error': 'Trading bot not initialized'}), 503
    
    try:
        portfolio_value = trading_bot.get_portfolio_value()
        total_return = ((portfolio_value - trading_bot.initial_capital) / trading_bot.initial_capital) * 100
        
        # Count active positions from both memory and database
        active_positions_count = len(trading_bot.trader.active_positions)
        if hasattr(trading_bot, 'db'):
            db_positions = trading_bot.db.get_active_positions()
            # Count unique tickers from database that aren't already in memory
            memory_tickers = set(trading_bot.trader.active_positions.keys())
            db_tickers = set(pos['ticker'] for pos in db_positions)
            # Total count = memory positions + database positions not in memory
            active_positions_count = len(memory_tickers | db_tickers)
        
        status = {
            'initial_capital': trading_bot.initial_capital,
            'current_capital': trading_bot.current_capital,
            'portfolio_value': portfolio_value,
            'total_return_pct': total_return,
            'daily_profit': trading_bot.daily_profit,
            'daily_profit_target_min': trading_bot.daily_profit_target_min,
            'daily_profit_target_max': trading_bot.daily_profit_target_max,
            'active_positions_count': active_positions_count,
            'total_trades': trading_bot.db.get_statistics().get('total_trades', len(trading_bot.trade_history)) if hasattr(trading_bot, 'db') else len(trading_bot.trade_history),
            'max_positions': trading_bot.max_positions,
            'tickers_monitored': len(trading_bot.tickers),
            'running': trading_bot.running if hasattr(trading_bot, 'running') else False
        }
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/positions')
def get_positions():
    """Get current active positions"""
    if not trading_bot:
        return jsonify({'error': 'Trading bot not initialized'}), 503
    
    try:
        # Get positions from both memory (active) and database (for persistence)
        positions = []
        
        # First, get active positions from memory (current session)
        # Create a copy of items to avoid "dictionary changed size during iteration" error
        active_tickers = set()
        active_positions_copy = dict(trading_bot.trader.active_positions)  # Create a snapshot
        for ticker, position in active_positions_copy.items():
            # Skip positions with null/empty tickers
            if not ticker or not str(ticker).strip():
                logger.warning(f"Skipping position with invalid ticker: {ticker}")
                continue
            
            ticker = str(ticker).strip()
            active_tickers.add(ticker)
            try:
                # Get current price from latest 1-minute data (most recent close)
                try:
                    df = trading_bot.data_api.get_1min_data(ticker, minutes=1)
                    if len(df) > 0:
                        current_price = df.iloc[-1]['close']
                    else:
                        # Fallback to API current price
                        current_price = trading_bot.data_api.get_current_price(ticker)
                except:
                    # Fallback to API current price if 1-min data fails
                    current_price = trading_bot.data_api.get_current_price(ticker)
                
                shares = position.shares if hasattr(position, 'shares') else 0
                position_value = shares * current_price
                entry_value = position.entry_value if hasattr(position, 'entry_value') else 0
                unrealized_pnl = position_value - entry_value
                # Prevent divide by zero
                entry_price = position.entry_price if hasattr(position, 'entry_price') else 0
                unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                
                positions.append({
                    'ticker': ticker,
                    'entry_price': position.entry_price,
                    'current_price': current_price,
                    'shares': shares,
                    'entry_value': entry_value,
                    'current_value': position_value,
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_pct': unrealized_pnl_pct,
                    'entry_time': position.entry_time.isoformat() if hasattr(position.entry_time, 'isoformat') else str(position.entry_time),
                    'entry_pattern': position.entry_pattern if hasattr(position, 'entry_pattern') else 'N/A',
                    'confidence': position.entry_confidence if hasattr(position, 'entry_confidence') else 0,
                    'target_price': position.target_price if hasattr(position, 'target_price') else None,
                    'stop_loss': position.stop_loss if hasattr(position, 'stop_loss') else None
                })
            except Exception as e:
                logger.error(f"Error getting position data for {ticker}: {e}")
                positions.append({
                    'ticker': ticker,
                    'error': str(e)
                })
        
        # Also check database for any positions that might not be in memory (after restart)
        if hasattr(trading_bot, 'db'):
            db_positions = trading_bot.db.get_active_positions()
            for db_pos in db_positions:
                # Skip positions with null/empty tickers
                db_ticker = db_pos.get('ticker')
                if not db_ticker or not str(db_ticker).strip():
                    logger.warning(f"Skipping database position with invalid ticker: {db_pos.get('id', 'unknown')}")
                    continue
                
                db_ticker = str(db_ticker).strip()
                if db_ticker not in active_tickers:
                    # Position exists in DB but not in memory (bot was restarted)
                    try:
                        # Get current price from latest 1-minute data (most recent close)
                        try:
                            df = trading_bot.data_api.get_1min_data(db_ticker, minutes=1)
                            if len(df) > 0:
                                current_price = df.iloc[-1]['close']
                            else:
                                # Fallback to API current price
                                current_price = trading_bot.data_api.get_current_price(db_ticker)
                        except:
                            # Fallback to API current price if 1-min data fails
                            current_price = trading_bot.data_api.get_current_price(db_ticker)
                        
                        position_value = db_pos['shares'] * current_price
                        unrealized_pnl = position_value - db_pos['entry_value']
                        # Prevent divide by zero
                        entry_price = db_pos.get('entry_price', 0) or 0
                        unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                        
                        positions.append({
                            'ticker': db_ticker,
                            'entry_price': db_pos['entry_price'],
                            'current_price': current_price,
                            'shares': db_pos['shares'],
                            'entry_value': db_pos['entry_value'],
                            'current_value': position_value,
                            'unrealized_pnl': unrealized_pnl,
                            'unrealized_pnl_pct': unrealized_pnl_pct,
                            'entry_time': db_pos['entry_time'],
                            'entry_pattern': db_pos['entry_pattern'],
                            'confidence': db_pos['confidence'],
                            'target_price': db_pos['target_price'],
                            'stop_loss': db_pos['stop_loss'],
                            'from_database': True  # Flag to indicate this came from DB
                        })
                    except Exception as e:
                        logger.error(f"Error getting position data for {db_ticker} from database: {e}")
        
        return jsonify(positions)
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/positions/update', methods=['POST'])
def update_position():
    """Update target price and stop loss for a position"""
    global trading_bot
    if not trading_bot:
        return jsonify({'error': 'Trading bot not initialized'}), 503
    
    try:
        from flask import request
        data = request.get_json()
        
        ticker = data.get('ticker')
        target_price = data.get('target_price')
        stop_loss = data.get('stop_loss')
        
        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400
        
        if target_price is None or stop_loss is None:
            return jsonify({'error': 'Both target_price and stop_loss are required'}), 400
        
        # Update position in memory if it exists
        if ticker in trading_bot.trader.active_positions:
            position = trading_bot.trader.active_positions[ticker]
            position.target_price = float(target_price)
            position.stop_loss = float(stop_loss)
            logger.info(f"Updated position {ticker}: target=${target_price}, stop=${stop_loss}")
        
        # Update position in database
        if hasattr(trading_bot, 'db'):
            try:
                trading_bot.db.update_position(ticker, target_price=target_price, stop_loss=stop_loss)
            except Exception as e:
                logger.error(f"Error updating position in database: {e}")
        
        return jsonify({'message': f'Position {ticker} updated successfully', 'ticker': ticker}), 200
    except Exception as e:
        logger.error(f"Error updating position: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/positions/close', methods=['POST'])
def close_position():
    """Manually close a position"""
    global trading_bot
    if not trading_bot:
        return jsonify({'error': 'Trading bot not initialized'}), 503
    
    try:
        from flask import request
        from datetime import datetime
        from core.realtime_trader import TradeSignal
        
        data = request.get_json()
        ticker = data.get('ticker')
        
        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400
        
        # Check if position exists in memory
        position_in_memory = ticker in trading_bot.trader.active_positions
        
        # If not in memory, check database and restore it
        if not position_in_memory:
            if hasattr(trading_bot, 'db'):
                db_positions = trading_bot.db.get_active_positions()
                db_position = next((p for p in db_positions if p['ticker'] == ticker), None)
                
                if not db_position:
                    return jsonify({'error': f'Position {ticker} not found'}), 404
                
                # Restore position from database to memory
                try:
                    from core.realtime_trader import ActivePosition
                    import pandas as pd
                    
                    restored_position = ActivePosition(
                        ticker=db_position['ticker'],
                        entry_time=pd.to_datetime(db_position['entry_time']),
                        entry_price=db_position['entry_price'],
                        entry_pattern=db_position.get('entry_pattern', 'Unknown'),
                        entry_confidence=db_position.get('confidence', 0.0),
                        target_price=db_position.get('target_price', db_position['entry_price'] * 1.25),
                        stop_loss=db_position.get('stop_loss', db_position['entry_price'] * 0.97),
                        current_price=db_position['entry_price'],
                        shares=db_position.get('shares', 0),
                        entry_value=db_position.get('entry_value', 0)
                    )
                    trading_bot.trader.active_positions[ticker] = restored_position
                    logger.info(f"Restored position {ticker} from database for manual close")
                except Exception as e:
                    logger.error(f"Error restoring position {ticker} from database: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return jsonify({'error': f'Failed to restore position {ticker} from database: {str(e)}'}), 500
            else:
                return jsonify({'error': f'Position {ticker} not found'}), 404
        
        # Get current price
        try:
            current_price = trading_bot.data_api.get_current_price(ticker)
        except Exception as e:
            logger.error(f"Error getting current price for {ticker}: {e}")
            return jsonify({'error': f'Could not get current price for {ticker}'}), 500
        
        # Create exit signal
        exit_signal = TradeSignal(
            signal_type='exit',
            ticker=ticker,
            timestamp=datetime.now(),
            price=current_price,
            reason='Manually closed from dashboard',
            confidence=1.0
        )
        
        # Execute exit
        trade = trading_bot._execute_exit(exit_signal)
        
        if trade:
            return jsonify({
                'message': f'Position {ticker} closed successfully',
                'ticker': ticker,
                'exit_price': current_price,
                'pnl': trade.pnl_dollars
            }), 200
        else:
            return jsonify({'error': f'Failed to close position {ticker}'}), 500
            
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/monitoring')
def get_monitoring_status():
    """Get monitoring status for all tickers, separated by source"""
    global trading_bot
    if not trading_bot:
        return jsonify({'error': 'Trading bot not initialized'}), 503
    
    try:
        # Get top gainers list and data (separate from other sources)
        top_gainers = getattr(trading_bot, 'top_gainers', [])
        top_gainers_data_full = getattr(trading_bot, 'top_gainers_data', [])
        top_gainers_set = set(top_gainers)
        
        # Create a map of ticker -> change % from gainer data
        gainer_change_map = {}
        for gainer in top_gainers_data_full:
            symbol = gainer.get('symbol', '')
            if symbol:
                change_pct = gainer.get('change_ratio', 0) or gainer.get('changeRatio', 0) or 0
                gainer_change_map[symbol] = change_pct
        
        top_gainers_data = []
        other_tickers_data = []
        
        for ticker in trading_bot.tickers:
            # Skip null/empty tickers
            if not ticker or not str(ticker).strip():
                continue
            
            ticker = str(ticker).strip()
            
            status_info = trading_bot.monitoring_status.get(ticker, {
                'status': 'monitoring',
                'last_check': None,
                'rejection_reasons': [],
                'has_position': False,
                'current_price': None
            })
            
            # Check if ticker has active position
            has_position = ticker in trading_bot.trader.active_positions
            
            # For top gainers, always get current price from minute data
            current_price = None
            change_pct = gainer_change_map.get(ticker, None)
            
            if ticker in top_gainers_set:
                try:
                    # Get minute data to get current price and calculate change percentage
                    df_1min = trading_bot.data_api.get_1min_data(ticker, minutes=390)  # Get ~6.5 hours of data
                    if df_1min is not None and not df_1min.empty and len(df_1min) > 0:
                        # Always use the latest close price from minute data as current price
                        current_price = float(df_1min.iloc[-1]['close'])
                        
                        # Check if we're in after hours (after 4:00 PM ET)
                        from datetime import datetime
                        import pytz
                        et = pytz.timezone('US/Eastern')
                        now_et = datetime.now(et)
                        current_hour = now_et.hour
                        is_after_hours = current_hour >= 16  # 4:00 PM or later
                        
                        if is_after_hours:
                            # For after hours: use day's close price (last regular market hour close, typically 4:00 PM)
                            # Find the last close price before 4:00 PM using pandas
                            day_close = None
                            
                            # Ensure timestamp column is timezone-aware (ET)
                            df_copy = df_1min.copy()
                            ts_series = pd.to_datetime(df_copy['timestamp'])
                            if ts_series.dt.tz is None:
                                # Timezone-naive, localize to ET
                                ts_series = ts_series.dt.tz_localize('US/Eastern')
                            else:
                                # Already timezone-aware, convert to ET
                                ts_series = ts_series.dt.tz_convert('US/Eastern')
                            
                            # Find the last row before 4:00 PM (16:00)
                            df_copy['hour'] = ts_series.dt.hour
                            df_before_4pm = df_copy[df_copy['hour'] < 16]
                            if not df_before_4pm.empty:
                                # Get the last close price before 4 PM
                                day_close = float(df_before_4pm.iloc[-1]['close'])
                            else:
                                # If no pre-4PM price found, use the first price of the day as fallback
                                day_close = float(df_1min.iloc[0]['close'])
                            
                            # Calculate change percentage from day's close to current price
                            if day_close > 0 and current_price:
                                change_pct = ((current_price - day_close) / day_close) * 100
                        else:
                            # Regular market hours: use first (opening) price
                            first_close = float(df_1min.iloc[0]['close'])
                            if first_close > 0 and current_price:
                                # Calculate percentage change from opening price to current price
                                change_pct = ((current_price - first_close) / first_close) * 100
                except Exception as e:
                    logger.debug(f"Could not get minute data for {ticker}: {e}")
                    # Fallback to API current price
                    try:
                        current_price = trading_bot.data_api.get_current_price(ticker)
                    except:
                        current_price = status_info.get('current_price')
            else:
                # For non-top-gainers, use regular API current price
                try:
                    current_price = trading_bot.data_api.get_current_price(ticker)
                except:
                    current_price = status_info.get('current_price')
            
            # Ensure change_pct is a number (not None) for sorting
            if change_pct is None:
                change_pct = 0.0
            else:
                try:
                    change_pct = float(change_pct)
                except (ValueError, TypeError):
                    change_pct = 0.0
            
            ticker_data = {
                'ticker': ticker,
                'status': 'active_position' if has_position else status_info.get('status', 'monitoring'),
                'current_price': current_price,
                'rejection_reasons': status_info.get('rejection_reasons', []),
                'last_check': status_info.get('last_check').isoformat() if status_info.get('last_check') else None,
                'has_position': has_position,
                'change_pct': change_pct  # Calculated change % from minute data (always a number for sorting)
            }
            
            # Separate top gainers from other sources
            if ticker in top_gainers_set:
                top_gainers_data.append(ticker_data)
            else:
                other_tickers_data.append(ticker_data)
        
        # Sort top gainers by calculated change percentage (descending - highest first)
        # Use the calculated change_pct value for sorting
        top_gainers_data.sort(
            key=lambda x: float(x.get('change_pct', 0) or 0), 
            reverse=True
        )
        
        return jsonify({
            'top_gainers': top_gainers_data,
            'other_tickers': other_tickers_data,
            'last_refresh': trading_bot.last_stock_discovery.isoformat() if trading_bot.last_stock_discovery else None
        })
    except Exception as e:
        logger.error(f"Error getting monitoring status: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/rejected-entries')
def get_rejected_entries():
    """Get list of rejected entry signals"""
    global trading_bot
    if not trading_bot:
        return jsonify({'rejected_entries': []}), 200
    
    try:
        rejected_entries = getattr(trading_bot, 'rejected_entries', [])
        # Convert datetime objects to ISO format strings
        entries_data = []
        for entry in rejected_entries:
            timestamp = entry.get('timestamp')
            if timestamp:
                # Handle both datetime objects and strings
                if hasattr(timestamp, 'isoformat'):
                    timestamp_str = timestamp.isoformat()
                else:
                    timestamp_str = str(timestamp)
            else:
                timestamp_str = None
            
            entries_data.append({
                'ticker': entry.get('ticker', ''),
                'price': float(entry.get('price', 0)),
                'reason': entry.get('reason', ''),
                'timestamp': timestamp_str
            })
        # Return in reverse order (most recent first)
        return jsonify({'rejected_entries': list(reversed(entries_data))})
    except Exception as e:
        logger.error(f"Error getting rejected entries: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'rejected_entries': [], 'error': str(e)}), 200


@app.route('/api/trades')
def get_trades():
    """Get completed trades history"""
    if not trading_bot:
        return jsonify({'error': 'Trading bot not initialized'}), 503
    
    try:
        # Get trades from database (persistent storage)
        # Completed trades are frozen at the time they were written to the database
        # Do NOT update them with current prices - use stored values only
        if hasattr(trading_bot, 'db'):
            trades = trading_bot.db.get_all_trades()
        else:
            # Fallback to in-memory if database not available
            trades = []
            for trade in trading_bot.trade_history:
                trades.append({
                    'ticker': trade.ticker,
                    'entry_time': trade.entry_time.isoformat() if hasattr(trade.entry_time, 'isoformat') else str(trade.entry_time),
                    'exit_time': trade.exit_time.isoformat() if hasattr(trade.exit_time, 'isoformat') else str(trade.exit_time),
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'shares': trade.shares,
                    'entry_value': trade.entry_value,
                    'exit_value': trade.exit_value,
                    'pnl_pct': trade.pnl_pct,
                    'pnl_dollars': trade.pnl_dollars,
                    'entry_pattern': trade.entry_pattern,
                    'exit_reason': trade.exit_reason,
                    'confidence': trade.confidence,
                    'status': 'WIN' if trade.pnl_dollars > 0 else 'LOSS'
                })
        
        # Filter out trades with null/empty tickers and ensure all values are correct
        valid_trades = []
        for trade in trades:
            # Skip trades with null/empty/whitespace-only tickers
            ticker = trade.get('ticker')
            if not ticker or not str(ticker).strip():
                logger.warning(f"Skipping trade with invalid ticker: {trade.get('id', 'unknown')}")
                continue
            
            # Normalize ticker to string
            trade['ticker'] = str(ticker).strip()
            
            # Ensure numeric values are properly typed
            if 'shares' in trade:
                trade['shares'] = float(trade['shares']) if trade['shares'] is not None else 0
            if 'entry_price' in trade:
                trade['entry_price'] = float(trade['entry_price']) if trade['entry_price'] is not None else 0
            if 'exit_price' in trade:
                trade['exit_price'] = float(trade['exit_price']) if trade['exit_price'] is not None else 0
            if 'entry_value' in trade:
                trade['entry_value'] = float(trade['entry_value']) if trade['entry_value'] is not None else 0
            if 'exit_value' in trade:
                trade['exit_value'] = float(trade['exit_value']) if trade['exit_value'] is not None else 0
            
            valid_trades.append(trade)
        
        trades = valid_trades
        
        # Continue processing valid trades
        for trade in trades:
            
            # Recalculate exit_value if it seems incorrect
            if 'shares' in trade and 'exit_price' in trade and trade['shares'] > 0 and trade['exit_price'] > 0:
                expected_exit_value = trade['shares'] * trade['exit_price']
                if 'exit_value' not in trade or abs(trade.get('exit_value', 0) - expected_exit_value) > 0.01:
                    trade['exit_value'] = expected_exit_value
                    # Recalculate P&L
                    if 'entry_value' in trade:
                        trade['pnl_dollars'] = trade['exit_value'] - trade['entry_value']
                        entry_price = trade.get('entry_price', 0) or 0
                        if entry_price > 0:
                            trade['pnl_pct'] = ((trade['exit_price'] - entry_price) / entry_price) * 100
                        else:
                            trade['pnl_pct'] = 0
            
            # Ensure entry_value is correct
            if 'shares' in trade and 'entry_price' in trade and trade['shares'] > 0 and trade['entry_price'] > 0:
                expected_entry_value = trade['shares'] * trade['entry_price']
                if 'entry_value' not in trade or abs(trade.get('entry_value', 0) - expected_entry_value) > 0.01:
                    trade['entry_value'] = expected_entry_value
                    # Recalculate P&L if exit_value exists
                    if 'exit_value' in trade:
                        trade['pnl_dollars'] = trade['exit_value'] - trade['entry_value']
                        entry_price = trade.get('entry_price', 0) or 0
                        if entry_price > 0:
                            trade['pnl_pct'] = ((trade['exit_price'] - entry_price) / entry_price) * 100
                        else:
                            trade['pnl_pct'] = 0
            
            # Determine WIN/LOSS status based on recalculated P&L
            pnl_dollars = trade.get('pnl_dollars', 0)
            if isinstance(pnl_dollars, (str, int)):
                pnl_dollars = float(pnl_dollars)
            
            # Verify status matches actual P&L
            exit_price = float(trade.get('exit_price', 0))
            entry_price = float(trade.get('entry_price', 0))
            
            # Recalculate P&L if prices suggest different status
            if exit_price > 0 and entry_price > 0:
                if exit_price > entry_price and pnl_dollars <= 0:
                    # Exit price is higher than entry, should be WIN - recalculate
                    if 'shares' in trade and trade['shares'] > 0:
                        trade['exit_value'] = trade['shares'] * exit_price
                        trade['entry_value'] = trade['shares'] * entry_price
                        trade['pnl_dollars'] = trade['exit_value'] - trade['entry_value']
                        trade['pnl_pct'] = ((exit_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                        pnl_dollars = trade['pnl_dollars']
                elif exit_price < entry_price and pnl_dollars > 0:
                    # Exit price is lower than entry, should be LOSS - recalculate
                    if 'shares' in trade and trade['shares'] > 0:
                        trade['exit_value'] = trade['shares'] * exit_price
                        trade['entry_value'] = trade['shares'] * entry_price
                        trade['pnl_dollars'] = trade['exit_value'] - trade['entry_value']
                        trade['pnl_pct'] = ((exit_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                        pnl_dollars = trade['pnl_dollars']
            
            trade['status'] = 'WIN' if pnl_dollars > 0 else 'LOSS'
        
        # Final validation: ensure no null tickers slipped through
        final_trades = [t for t in trades if t.get('ticker') and str(t.get('ticker')).strip()]
        
        # Sort by exit time (most recent first)
        final_trades.sort(key=lambda x: x.get('exit_time', ''), reverse=True)
        
        return jsonify(final_trades)
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/start', methods=['POST'])
def start_trading():
    """Start the trading bot"""
    global trading_bot
    if not trading_bot:
        logger.error("Trading bot not initialized - global trading_bot is None")
        return jsonify({'error': 'Trading bot not initialized. Please restart the application.'}), 503
    
    try:
        # Check if bot is already running
        if hasattr(trading_bot, 'running') and trading_bot.running:
            logger.info("Bot is already running")
            return jsonify({'message': 'Trading bot is already running', 'running': True}), 200
        
        # Start bot in a separate thread
        import threading
        
        # Check if bot thread already exists and is alive
        if hasattr(trading_bot, '_bot_thread') and trading_bot._bot_thread and trading_bot._bot_thread.is_alive():
            logger.info("Bot thread already exists and is alive")
            return jsonify({'message': 'Trading bot thread already exists', 'running': True}), 200
        
        def run_bot():
            try:
                # Get interval from bot or default to 15
                interval = getattr(trading_bot, '_check_interval', 15)
                logger.info(f"Starting trading bot in background thread (interval: {interval} seconds)")
                trading_bot.run_continuous(interval_seconds=interval)
            except Exception as e:
                logger.error(f"Error in trading bot thread: {e}")
                import traceback
                logger.error(traceback.format_exc())
                if hasattr(trading_bot, 'running'):
                    trading_bot.running = False
        
        # Create and start bot thread
        bot_thread = threading.Thread(target=run_bot, daemon=True, name="TradingBotThread")
        bot_thread.start()
        trading_bot._bot_thread = bot_thread
        
        # Give it a moment to start
        import time
        time.sleep(0.3)
        
        # Verify it started
        if hasattr(trading_bot, 'running') and trading_bot.running:
            logger.info("Trading bot started successfully")
            return jsonify({'message': 'Trading bot started successfully', 'running': True}), 200
        else:
            logger.warning("Trading bot thread started but running flag not set")
            return jsonify({'message': 'Trading bot thread started', 'running': True}), 200
            
    except AttributeError as e:
        logger.error(f"Attribute error starting bot: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Bot attribute error: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/stop', methods=['POST'])
def stop_trading():
    """Stop the trading bot gracefully"""
    global trading_bot
    if not trading_bot:
        logger.error("Trading bot not initialized - global trading_bot is None")
        return jsonify({'error': 'Trading bot not initialized. Please restart the application.'}), 503
    
    try:
        # Check if bot is running
        is_running = hasattr(trading_bot, 'running') and trading_bot.running
        
        if not is_running:
            logger.info("Bot is already stopped")
            return jsonify({'message': 'Trading bot is already stopped', 'running': False}), 200
        
        # Gracefully stop the bot
        logger.info("Stopping trading bot gracefully...")
        trading_bot.stop()
        
        # Wait a moment for the thread to stop
        import time
        time.sleep(0.5)
        
        # Verify it stopped
        if hasattr(trading_bot, 'running'):
            is_stopped = not trading_bot.running
            if is_stopped:
                logger.info("Trading bot stopped successfully")
                return jsonify({'message': 'Trading bot stopped gracefully', 'running': False}), 200
            else:
                logger.warning("Stop command sent but bot still appears to be running")
                return jsonify({'message': 'Stop command sent', 'running': False}), 200
        else:
            return jsonify({'message': 'Trading bot stopped', 'running': False}), 200
            
    except Exception as e:
        logger.error(f"Error stopping bot: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/statistics')
def get_statistics():
    """Get trading statistics"""
    if not trading_bot:
        return jsonify({'error': 'Trading bot not initialized'}), 503
    
    try:
        # Get statistics from database
        if hasattr(trading_bot, 'db'):
            stats = trading_bot.db.get_statistics()
            return jsonify(stats)
        
        # Fallback to in-memory calculation
        trades = trading_bot.trade_history
        
        if not trades:
            return jsonify({
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'best_trade': None,
                'worst_trade': None
            })
        
        winning_trades = [t for t in trades if t.pnl_dollars > 0]
        losing_trades = [t for t in trades if t.pnl_dollars <= 0]
        
        win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0
        total_pnl = sum(t.pnl_dollars for t in trades)
        avg_pnl = total_pnl / len(trades) if trades else 0
        
        best_trade = max(trades, key=lambda x: x.pnl_dollars) if trades else None
        worst_trade = min(trades, key=lambda x: x.pnl_dollars) if trades else None
        
        stats = {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'best_trade': {
                'ticker': best_trade.ticker,
                'pnl_pct': best_trade.pnl_pct,
                'pnl_dollars': best_trade.pnl_dollars
            } if best_trade else None,
            'worst_trade': {
                'ticker': worst_trade.ticker,
                'pnl_pct': worst_trade.pnl_pct,
                'pnl_dollars': worst_trade.pnl_dollars
            } if worst_trade else None
        }
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return jsonify({'error': str(e)}), 500


def run_web_server(host='127.0.0.1', port=5000, debug=False):
    """Run the Flask web server"""
    logger.info(f"Starting web interface on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)


if __name__ == '__main__':
    # For testing without bot
    run_web_server(debug=True)

