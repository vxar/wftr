#!/usr/bin/env python3
"""
Test entry signal evaluation
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.simulation.pure_trade_simulator import PureTradeSimulator, SimulationConfig

def test_entry_evaluation():
    """Test entry signal evaluation"""
    print("=== Testing Entry Signal Evaluation ===")
    
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
    
    # Test evaluate_entry_signal directly
    test_result = simulator.position_manager.evaluate_entry_signal(
        ticker='ROLR',
        current_price=10.95,
        signal_strength=0.82,
        multi_timeframe_analysis={
            'momentum_score': 0.6,
            'trend_alignment': 0.7,
            'sma_5': 10.95,
            'sma_15': 10.95,
            'sma_50': 10.95,
            'rsi': 50,
            'volatility_score': 0.3
        },
        volume_data={'volume_ratio': 6.17},
        pattern_info={'pattern_name': 'MACD_Acceleration_Breakout'}
    )
    
    print(f"Entry result type: {type(test_result)}")
    print(f"Entry result value: {test_result}")
    print(f"Entry result length: {len(test_result)}")
    
    if len(test_result) == 3:
        should_enter = test_result[0]
        position_type = test_result[1] 
        reason = test_result[2]
        print(f"Should enter: {should_enter}")
        print(f"Position type: {position_type}")
        print(f"Reason: {reason}")
    else:
        print(f"ERROR: Expected tuple of length 3, got {len(test_result)}")

if __name__ == "__main__":
    test_entry_evaluation()
