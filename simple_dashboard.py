"""
Simple Dashboard for Trading Bot
Basic web dashboard without flask_socketio dependency
"""
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import threading
import time
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to store bot instance
bot_instance = None

@app.route('/')
def dashboard():
    """Main dashboard page"""
    # Get real bot status data
    bot_status = get_bot_status()
    positions_data = get_positions_data()
    monitored_tickers = get_monitored_tickers()
    
    return render_template('enhanced_dashboard.html', 
                         bot_status=bot_status, 
                         positions=positions_data,
                         monitored_tickers=monitored_tickers)

def get_bot_status():
    """Get bot status from real bot instance"""
    if bot_instance:
        try:
            # Try to get status from bot
            if hasattr(bot_instance, 'get_bot_status'):
                status = bot_instance.get_bot_status()
                # Remove paused status since pause/resume is removed
                if 'paused' in status:
                    del status['paused']
                return status
            else:
                # Fallback to basic attributes
                return {
                    'current_capital': getattr(bot_instance, 'current_capital', 10000),
                    'daily_profit': getattr(bot_instance, 'daily_profit', 0),
                    'running': getattr(bot_instance, 'running', False),
                    'active_positions': len(getattr(bot_instance, 'active_positions', {}))
                }
        except Exception as e:
            print(f"Error getting bot status: {e}")
            return {
                'current_capital': 10000,
                'daily_profit': 0,
                'running': False,
                'active_positions': 0
            }
    else:
        return {
            'current_capital': 10000,
            'daily_profit': 0,
            'running': False,
            'active_positions': 0
        }

def get_monitored_tickers():
    """Get monitored tickers from bot instance"""
    if not bot_instance:
        return []
    
    try:
        # Try different methods to get monitored tickers
        monitored_tickers = []
        
        # Method 1: Check for top_gainers_data
        if hasattr(bot_instance, 'top_gainers_data'):
            for i, ticker_data in enumerate(bot_instance.top_gainers_data[:10], 1):
                monitored_tickers.append({
                    'symbol': ticker_data.get('symbol', ticker_data.get('ticker', 'N/A')),
                    'price': ticker_data.get('price', 0),
                    'change_pct': ticker_data.get('change_pct', ticker_data.get('change_percent', 0)),
                    'volume': ticker_data.get('volume', 0),
                    'rank': i
                })
        
        # Method 2: Check for watchlist
        elif hasattr(bot_instance, 'watchlist'):
            for i, ticker in enumerate(bot_instance.watchlist[:10], 1):
                # Try to get current price data
                price_data = {}
                if hasattr(bot_instance, 'get_current_price'):
                    try:
                        price_data = bot_instance.get_current_price(ticker)
                    except:
                        pass
                
                monitored_tickers.append({
                    'symbol': ticker,
                    'price': price_data.get('price', 0),
                    'change_pct': price_data.get('change_pct', 0),
                    'volume': price_data.get('volume', 0),
                    'rank': i
                })
        
        # Method 3: Check for scanned symbols
        elif hasattr(bot_instance, 'scanned_symbols'):
            for i, symbol in enumerate(bot_instance.scanned_symbols[:10], 1):
                monitored_tickers.append({
                    'symbol': symbol,
                    'price': 0,
                    'change_pct': 0,
                    'volume': 0,
                    'rank': i
                })
        
        # Method 4: Use active positions as monitored tickers
        elif hasattr(bot_instance, 'active_positions') and bot_instance.active_positions:
            for i, (ticker, position) in enumerate(bot_instance.active_positions.items(), 1):
                monitored_tickers.append({
                    'symbol': ticker,
                    'price': getattr(position, 'current_price', 0),
                    'change_pct': 0,  # Could calculate from entry price
                    'volume': 0,
                    'rank': i
                })
        
        return monitored_tickers
        
    except Exception as e:
        print(f"Error getting monitored tickers: {e}")
        return []

def get_positions_data():
    """Get positions data from bot instance"""
    if not bot_instance or not hasattr(bot_instance, 'active_positions'):
        return {'positions': [], 'total_value': 0, 'total_unrealized': 0, 'count': 0}
    
    positions_data = []
    total_value = 0
    total_unrealized = 0
    
    for ticker, position in bot_instance.active_positions.items():
        pos_data = {
            'ticker': ticker,
            'shares': round(getattr(position, 'shares', 0)),  # Round to whole number
            'entry_price': getattr(position, 'entry_price', 0),
            'current_price': getattr(position, 'current_price', 0),
            'current_value': getattr(position, 'current_value', 0),
            'unrealized_pnl': getattr(position, 'unrealized_pnl_dollars', 0),
            'unrealized_pnl_pct': getattr(position, 'unrealized_pnl_pct', 0),
            'position_type': getattr(position, 'position_type', 'unknown'),
            'entry_pattern': getattr(position, 'entry_pattern', ''),
            'entry_time': getattr(position, 'entry_time', datetime.now().isoformat())
        }
        positions_data.append(pos_data)
        total_value += pos_data['current_value']
        total_unrealized += pos_data['unrealized_pnl']
    
    return {
        'positions': positions_data,
        'total_value': total_value,
        'total_unrealized': total_unrealized,
        'count': len(positions_data)
    }

@app.route('/api/status')
def api_status():
    """API endpoint for bot status"""
    return jsonify(get_bot_status())

@app.route('/api/positions')
def api_positions():
    """API endpoint for positions data"""
    return jsonify(get_positions_data())

@app.route('/api/tickers')
def api_tickers():
    """API endpoint for monitored tickers"""
    return jsonify(get_monitored_tickers())

@app.route('/api/start', methods=['POST'])
def api_start_bot():
    """Start trading bot"""
    try:
        if bot_instance:
            if hasattr(bot_instance, 'start'):
                bot_instance.start()
            else:
                setattr(bot_instance, 'running', True)
            return jsonify({'success': True, 'message': 'Bot started'})
        else:
            return jsonify({'success': False, 'message': 'Bot not available'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/stop', methods=['POST'])
def api_stop_bot():
    """Stop trading bot"""
    try:
        if bot_instance:
            if hasattr(bot_instance, 'stop'):
                bot_instance.stop()
            else:
                setattr(bot_instance, 'running', False)
            return jsonify({'success': True, 'message': 'Bot stopped'})
        else:
            return jsonify({'success': False, 'message': 'Bot not available'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/position/close', methods=['POST'])
def api_close_position():
    """Close a specific position"""
    try:
        ticker = request.json.get('ticker')
        if bot_instance and ticker:
            if hasattr(bot_instance, 'close_position'):
                bot_instance.close_position(ticker)
            elif hasattr(bot_instance, 'active_positions') and ticker in bot_instance.active_positions:
                del bot_instance.active_positions[ticker]
            return jsonify({'success': True, 'message': f'Position {ticker} closed'})
        else:
            return jsonify({'success': False, 'message': 'Invalid request'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/trades')
def trades_page():
    """Completed trades page"""
    trades_data = get_trades_data()
    
    # Calculate summary stats
    total_trades = len(trades_data)
    winning_trades = len([t for t in trades_data if t.get('pnl', 0) > 0])
    losing_trades = len([t for t in trades_data if t.get('pnl', 0) < 0])
    total_pnl = sum(t.get('pnl', 0) for t in trades_data)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    return render_template('trades.html', 
                         trades=trades_data,
                         total_trades=total_trades,
                         winning_trades=winning_trades,
                         losing_trades=losing_trades,
                         total_pnl=total_pnl,
                         win_rate=win_rate)

def get_trades_data():
    """Get completed trades data from bot instance"""
    if not bot_instance:
        return []
    
    try:
        trade_history = getattr(bot_instance, 'trade_history', [])
        trades_data = []
        
        for trade in trade_history[-100:]:  # Get last 100 trades
            trade_data = {
                'ticker': getattr(trade, 'ticker', ''),
                'entry_time': getattr(trade, 'entry_time', datetime.now()).isoformat(),
                'exit_time': getattr(trade, 'exit_time', datetime.now()).isoformat(),
                'entry_price': getattr(trade, 'entry_price', 0),
                'exit_price': getattr(trade, 'exit_price', 0),
                'shares': round(getattr(trade, 'shares', 0)),  # Round to whole number
                'pnl': getattr(trade, 'pnl_dollars', 0),
                'pnl_pct': getattr(trade, 'pnl_pct', 0),
                'entry_pattern': getattr(trade, 'entry_pattern', ''),
                'exit_reason': getattr(trade, 'exit_reason', ''),
                'confidence': getattr(trade, 'confidence', 0)
            }
            
            # Calculate duration
            try:
                entry_dt = datetime.fromisoformat(trade_data['entry_time'].replace('Z', '+00:00'))
                exit_dt = datetime.fromisoformat(trade_data['exit_time'].replace('Z', '+00:00'))
                duration = exit_dt - entry_dt
                hours = duration.total_seconds() / 3600
                if hours < 1:
                    trade_data['duration'] = f"{int(duration.total_seconds() / 60)}m"
                else:
                    trade_data['duration'] = f"{int(hours)}h {int((hours % 1) * 60)}m"
            except:
                trade_data['duration'] = 'N/A'
            
            trades_data.append(trade_data)
        
        # Sort by exit time (newest first)
        trades_data.sort(key=lambda x: x.get('exit_time', ''), reverse=True)
        return trades_data
        
    except Exception as e:
        print(f"Error getting trades data: {e}")
        return []

def set_bot_instance(bot):
    """Set the bot instance for the dashboard"""
    global bot_instance
    bot_instance = bot

def run_dashboard(port=5000):
    """Run the dashboard server"""
    print(f"🌐 Dashboard starting on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    run_dashboard()
