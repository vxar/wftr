"""
Trading Web Interface
Flask web application for monitoring and controlling the trading bot
"""
from flask import Flask, render_template, jsonify, request
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Global variables
app = Flask(__name__)
trading_bot = None

def set_trading_bot(bot):
    """Set the trading bot instance"""
    global trading_bot
    trading_bot = bot

def run_web_server(host='127.0.0.1', port=5000, debug=False):
    """Run the web server"""
    app.run(host=host, port=port, debug=debug)

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get bot status"""
    if trading_bot:
        return jsonify(trading_bot.get_bot_status())
    return jsonify({'error': 'Bot not initialized'})

@app.route('/api/start', methods=['POST'])
def start_bot():
    """Start the bot"""
    if trading_bot:
        try:
            trading_bot.start()
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'error': str(e)})
    return jsonify({'error': 'Bot not initialized'})

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """Stop the bot"""
    if trading_bot:
        try:
            trading_bot.stop()
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'error': str(e)})
    return jsonify({'error': 'Bot not initialized'})
