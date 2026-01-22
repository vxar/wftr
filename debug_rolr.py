#!/usr/bin/env python3
"""
Debug ROLR entry issues
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.simulation.pure_trade_simulator import PureTradeSimulator, SimulationConfig

def debug_rolr():
    """Debug ROLR entry issues"""
    print("=== Debugging ROLR Entry Issues ===")
    
    # Create simulator
    config = SimulationConfig(
        ticker='ROLR',
        detection_time='2026-01-21 16:00:00',
        initial_capital=2500.0,
        max_positions=1,
        commission_per_trade=0.005,
        data_folder='simulation_data',
        stop_loss_pct=0.12,
        take_profit_pct=0.25,
        min_hold_minutes=5
    )
    
    simulator = PureTradeSimulator(config)
    
    # Load data and find first entry signal
    full_data = simulator.load_data()
    detection_dt = simulator._parse_detection_time()
    detection_idx = full_data.index.get_loc(detection_dt)
    
    print(f"Detection time: {detection_dt}")
    print(f"Detection index: {detection_idx}")
    
    # Find first entry signal
    entry_found = False
    for i, (timestamp, row) in enumerate(full_data.iloc[detection_idx:].iterrows()):
        try:
            # Set current data index
            simulator.current_data_index = detection_idx + i
            
            # Create market data dict
            market_data = {
                'symbol': simulator.config.ticker,
                'price': row['close'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'volume': row['volume'],
                'timestamp': timestamp,
                'rsi': row.get('rsi', 50),
                'macd': row.get('macd'),
                'macd_signal': row.get('macd_signal'),
                'macd_hist': row.get('macd_hist', 0),
                'volume_ratio': simulator._calculate_volume_ratio(row, full_data, i),
                'volatility_score': 0.5,
                'momentum_score': 0.6,
                'trend_alignment': 0.7
            }
            
            # Use RealtimeTrader to analyze
            entry_signal, exit_signals = simulator.realtime_trader.analyze_data(
                full_data.iloc[detection_idx:detection_idx+i+1], 
                simulator.config.ticker, 
                current_price=market_data['price']
            )
            
            if entry_signal:
                print(f"\n--- Entry Signal Found at {timestamp} ---")
                print(f"Pattern: {entry_signal.pattern_name}")
                print(f"Confidence: {entry_signal.confidence}")
                print(f"Volume Ratio: {market_data.get('volume_ratio', 1.0)}")
                print(f"Current Price: {market_data['price']}")
                
                # Check position manager evaluation
                should_enter, position_type, reason = simulator.position_manager.evaluate_entry_signal(
                    ticker=simulator.config.ticker,
                    current_price=entry_signal.price,
                    signal_strength=entry_signal.confidence,
                    multi_timeframe_analysis={
                        'momentum_score': 0.6,
                        'trend_alignment': 0.7,
                        'sma_5': row.get('sma_5', row['close']),
                        'sma_15': row.get('sma_15', row['close']),
                        'sma_50': row.get('sma_50', row['close']),
                        'rsi': row.get('rsi', 50),
                        'volatility_score': 0.3
                    },
                    volume_data={'volume_ratio': market_data.get('volume_ratio', 1.0)},
                    pattern_info={'pattern_name': entry_signal.pattern_name}
                )
                
                print(f"Should Enter: {should_enter}")
                print(f"Position Type: {position_type}")
                print(f"Reason: {reason}")
                
                if should_enter:
                    print("✅ ENTRY SHOULD BE ACCEPTED")
                    entry_found = True
                else:
                    print("❌ ENTRY REJECTED")
                
                if entry_found:
                    break  # Stop after first entry signal
            else:
                continue
        
        print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    debug_rolr()
