"""
Pre-Market Analyzer
Analyzes pre-market data (7:00-9:30) to predict stock movement at 9:30 market open
"""
import pandas as pd
from typing import List, Dict, Optional, Tuple
from analysis.pattern_detector import PatternDetector, PatternSignal
from core.realtime_trader import RealtimeTrader, TradeSignal


class PreMarketAnalyzer:
    """Analyzes pre-market data to predict 9:30 market open movement"""
    
    def __init__(self, 
                 min_confidence: float = 0.75,
                 min_entry_price_increase: float = 10.0):
        """
        Args:
            min_confidence: Minimum pattern confidence
            min_entry_price_increase: Minimum expected gain (%)
        """
        self.pattern_detector = PatternDetector()
        self.min_confidence = min_confidence
        self.min_entry_price_increase = min_entry_price_increase
    
    def analyze_premarket(self, df: pd.DataFrame, ticker: str) -> List[Dict]:
        """
        Analyze pre-market data (7:00-9:30) to identify potential trades at 9:30
        
        Args:
            df: DataFrame with all data including pre-market
            ticker: Stock ticker
            
        Returns:
            List of potential trade signals ready for 9:30 execution
        """
        if len(df) < 100:
            return []
        
        # Filter to pre-market hours only (7:00-9:30)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.copy()
        
        # Convert to timezone-aware if needed
        if df['timestamp'].iloc[0].tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')
        
        # Filter pre-market hours (7:00-9:30)
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['time_value'] = df['hour'] * 60 + df['minute']
        
        premarket_mask = (df['time_value'] >= 420) & (df['time_value'] < 570)  # 7:00-9:30
        df_premarket = df[premarket_mask].copy()
        
        if len(df_premarket) < 30:
            return []
        
        # Calculate indicators for pre-market data
        df_premarket = self.pattern_detector.calculate_indicators(df_premarket)
        
        # Get the last pre-market data point (closest to 9:30)
        last_premarket_idx = len(df_premarket) - 1
        if last_premarket_idx < 30:
            return []
        
        # Detect patterns in pre-market
        # Get current date for pattern detection
        from datetime import datetime
        current_date = datetime.now().strftime('%Y-%m-%d')
        signals = self.pattern_detector.detect_patterns(df_premarket, ticker, current_date)
        
        potential_trades = []
        
        for signal in signals:
            if signal.pattern_type != 'bullish':
                continue
            
            if signal.confidence < self.min_confidence:
                continue
            
            # Validate the signal is strong enough
            if not self._validate_premarket_signal(df_premarket, last_premarket_idx, signal):
                continue
            
            # Get the last pre-market price (will be entry price at 9:30)
            last_price = df_premarket.iloc[last_premarket_idx]['close']
            
            # Calculate expected movement
            expected_gain = ((signal.target_price - last_price) / last_price) * 100 if signal.target_price else 0
            
            if expected_gain < self.min_entry_price_increase:
                continue
            
            # Create trade signal for 9:30 execution
            trade_signal = {
                'ticker': ticker,
                'entry_price': last_price,
                'target_price': signal.target_price or last_price * 1.18,  # 18% target
                'stop_loss': signal.stop_loss or last_price * 0.97,  # 3% stop loss
                'pattern': signal.pattern_name,
                'confidence': signal.confidence,
                'expected_gain_pct': expected_gain,
                'premarket_rsi': df_premarket.iloc[last_premarket_idx].get('rsi', None),
                'premarket_macd': df_premarket.iloc[last_premarket_idx].get('macd', None),
                'premarket_volume_ratio': df_premarket.iloc[last_premarket_idx].get('volume_ratio', None),
                'premarket_sma5': df_premarket.iloc[last_premarket_idx].get('sma_5', None),
                'premarket_sma10': df_premarket.iloc[last_premarket_idx].get('sma_10', None),
                'premarket_sma20': df_premarket.iloc[last_premarket_idx].get('sma_20', None),
                'signal_time': df_premarket.iloc[last_premarket_idx]['timestamp'],
                'reason': f"Pre-market signal: {signal.pattern_name}"
            }
            
            potential_trades.append(trade_signal)
        
        return potential_trades
    
    def _validate_premarket_signal(self, df: pd.DataFrame, idx: int, signal: PatternSignal) -> bool:
        """
        Validate that pre-market signal is strong enough for 9:30 entry
        """
        if idx < 20:
            return False
        
        current = df.iloc[idx]
        lookback_10 = df.iloc[max(0, idx-10):idx]
        
        # Must have strong volume in pre-market
        volume_ratio = current.get('volume_ratio', 0)
        if volume_ratio < 1.5:
            return False
        
        # Price should be trending up in pre-market
        if len(lookback_10) >= 5:
            recent_closes = lookback_10['close'].tail(5).values
            if len(recent_closes) >= 3:
                # Check if price is increasing
                price_increase = ((current.get('close', 0) - recent_closes[0]) / recent_closes[0]) * 100
                if price_increase < 1.0:  # At least 1% increase in pre-market
                    return False
        
        # MACD should be bullish
        macd = current.get('macd', 0)
        macd_signal = current.get('macd_signal', 0)
        if macd <= macd_signal:
            return False
        
        # Price should be above key moving averages
        if not (current.get('close', 0) > current.get('sma_5', 0) and
                current.get('close', 0) > current.get('sma_10', 0)):
            return False
        
        return True

