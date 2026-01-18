"""
PreMarket Analyzer
Analyzes premarket data to identify potential trading opportunities
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class PreMarketAnalyzer:
    """
    Analyzes premarket data to identify entry signals
    """
    
    def __init__(self, min_confidence: float = 0.7, min_entry_price_increase: float = 0.02):
        """
        Initialize the premarket analyzer
        
        Args:
            min_confidence: Minimum confidence level for signals
            min_entry_price_increase: Minimum price increase percentage
        """
        self.min_confidence = min_confidence
        self.min_entry_price_increase = min_entry_price_increase
        
    def analyze_premarket(self, df: pd.DataFrame, ticker: str) -> List[Dict]:
        """
        Analyze premarket data for entry signals
        
        Args:
            df: DataFrame with premarket data
            ticker: Stock ticker symbol
            
        Returns:
            List of potential entry signals
        """
        try:
            signals = []
            
            if df.empty or len(df) < 10:
                return signals
                
            # Calculate basic indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['volume_avg'] = df['volume'].rolling(window=20).mean()
            df['price_change'] = df['close'].pct_change()
            
            # Look for gap up patterns
            if len(df) >= 2:
                prev_close = df['close'].iloc[-2]
                current_price = df['close'].iloc[-1]
                gap_pct = (current_price - prev_close) / prev_close
                
                if gap_pct > self.min_entry_price_increase:
                    # Check volume confirmation
                    current_volume = df['volume'].iloc[-1]
                    avg_volume = df['volume_avg'].iloc[-1]
                    
                    if current_volume > avg_volume * 1.5:  # Volume spike
                        confidence = min(0.9, gap_pct * 10)  # Scale confidence with gap size
                        
                        if confidence >= self.min_confidence:
                            signal = {
                                'ticker': ticker,
                                'entry_price': current_price,
                                'confidence': confidence,
                                'gap_percentage': gap_pct,
                                'volume_ratio': current_volume / avg_volume,
                                'timestamp': df.index[-1] if hasattr(df.index[-1], 'isoformat') else datetime.now(),
                                'signal_type': 'gap_up'
                            }
                            signals.append(signal)
                            
            return signals
            
        except Exception as e:
            logger.error(f"Error analyzing premarket data for {ticker}: {e}")
            return []
