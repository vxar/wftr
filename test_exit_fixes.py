#!/usr/bin/env python3
"""
Test script to verify the exit logic fixes
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.intelligent_position_manager import IntelligentPositionManager, PositionType, ExitPlan
from datetime import datetime, timedelta
import pytz

def test_enhanced_surge_detection():
    """Test FIX 1: Enhanced surge detection"""
    print("=== Testing Enhanced Surge Detection ===")
    
    manager = IntelligentPositionManager()
    
    # Test cases that should now be classified as SURGE
    test_cases = [
        {
            "name": "PRICE_VOLUME_SURGE pattern",
            "signal_strength": 0.8,
            "multi_timeframe_analysis": {"momentum_score": 0.6},
            "volume_data": {"volume_ratio": 4.0},
            "pattern_info": {"pattern_name": "PRICE_VOLUME_SURGE"}
        },
        {
            "name": "CONTINUATION_SURGE pattern",
            "signal_strength": 0.8,
            "multi_timeframe_analysis": {"momentum_score": 0.6},
            "volume_data": {"volume_ratio": 3.5},
            "pattern_info": {"pattern_name": "CONTINUATION_SURGE"}
        },
        {
            "name": "High volume + moderate momentum",
            "signal_strength": 0.8,
            "multi_timeframe_analysis": {"momentum_score": 0.6},
            "volume_data": {"volume_ratio": 6.0},
            "pattern_info": {"pattern_name": "Volume_Breakout_Momentum"}
        },
        {
            "name": "Moderate volume + high signal strength",
            "signal_strength": 0.8,
            "multi_timeframe_analysis": {"momentum_score": 0.6},
            "volume_data": {"volume_ratio": 4.0},
            "pattern_info": {"pattern_name": "Volume_Breakout_Momentum"}
        }
    ]
    
    for test_case in test_cases:
        position_type = manager._determine_position_type(
            test_case["signal_strength"],
            test_case["multi_timeframe_analysis"],
            test_case["volume_data"],
            test_case["pattern_info"]
        )
        
        expected = PositionType.SURGE
        status = "✓ PASS" if position_type == expected else "✗ FAIL"
        print(f"{status}: {test_case['name']} -> {position_type.value} (expected: {expected.value})")

def test_surge_exit_conditions():
    """Test FIX 2: Enhanced surge exit conditions"""
    print("\n=== Testing Surge Exit Conditions ===")
    
    manager = IntelligentPositionManager()
    et_timezone = pytz.timezone('America/New_York')
    
    # Create a test SURGE position
    from src.core.intelligent_position_manager import ActivePosition
    position = ActivePosition(
        ticker="TEST",
        entry_time=datetime.now(et_timezone) - timedelta(minutes=10),
        entry_price=10.0,
        shares=1000,
        original_shares=1000,
        entry_value=10000,
        position_type=PositionType.SURGE,
        exit_plan=manager.position_configs[PositionType.SURGE],
        current_price=8.5,  # 15% loss
        unrealized_pnl_pct=-15.0,
        unrealized_pnl_dollars=-1500.0
    )
    position.current_stop_loss = 8.8  # 12% stop loss hit
    
    # Test market data scenarios
    test_scenarios = [
        {
            "name": "Recovering with strong volume",
            "market_data": {
                "volume_ratio": 3.0,
                "price_change_pct": 2.0
            },
            "should_exit": False
        },
        {
            "name": "Not recovering, weak volume",
            "market_data": {
                "volume_ratio": 1.0,
                "price_change_pct": -5.0
            },
            "should_exit": True
        },
        {
            "name": "Above entry with moderate volume",
            "market_data": {
                "volume_ratio": 2.5,
                "price_change_pct": 0.0
            },
            "should_exit": False
        }
    ]
    
    for scenario in test_scenarios:
        result = manager._is_recovering(position, scenario["market_data"])
        should_exit = not result  # If recovering, don't exit
        
        status = "✓ PASS" if should_exit == scenario["should_exit"] else "✗ FAIL"
        print(f"{status}: {scenario['name']} -> should_exit={should_exit} (expected: {scenario['should_exit']})")

def test_dynamic_adjustments():
    """Test FIX 4: Dynamic adjustments"""
    print("\n=== Testing Dynamic Adjustments ===")
    
    manager = IntelligentPositionManager()
    
    # Create a test position
    from src.core.intelligent_position_manager import ActivePosition
    position = ActivePosition(
        ticker="TEST",
        entry_time=datetime.now(pytz.timezone('America/New_York')),
        entry_price=10.0,
        shares=1000,
        original_shares=1000,
        entry_value=10000,
        position_type=PositionType.SURGE,
        exit_plan=manager.position_configs[PositionType.SURGE],
        current_price=12.0,
        unrealized_pnl_pct=20.0,
        unrealized_pnl_dollars=2000.0
    )
    
    # Store original values
    original_partial_levels = list(position.exit_plan.partial_profit_levels)
    original_final_target = position.exit_plan.final_target
    
    # Test extreme momentum
    market_data = {"momentum_5min": 30.0, "volume_ratio": 6.0}
    manager._adjust_exit_thresholds_by_momentum(position, market_data)
    
    # Check if adjustments were applied
    adjusted = (
        position.exit_plan.partial_profit_levels != original_partial_levels or
        position.exit_plan.final_target != original_final_target
    )
    
    status = "✓ PASS" if adjusted else "✗ FAIL"
    print(f"{status}: Dynamic adjustment applied -> {adjusted}")
    
    if adjusted:
        print(f"  Original partial levels: {original_partial_levels}")
        print(f"  Adjusted partial levels: {position.exit_plan.partial_profit_levels}")
        print(f"  Original final target: {original_final_target}%")
        print(f"  Adjusted final target: {position.exit_plan.final_target}%")

def test_exit_delays():
    """Test FIX 4: Exit delays"""
    print("\n=== Testing Exit Delays ===")
    
    manager = IntelligentPositionManager()
    et_timezone = pytz.timezone('America/New_York')
    
    # Create a test position
    from src.core.intelligent_position_manager import ActivePosition
    position = ActivePosition(
        ticker="TEST",
        entry_time=datetime.now(et_timezone) - timedelta(minutes=2),
        entry_price=10.0,
        shares=1000,
        original_shares=1000,
        entry_value=10000,
        position_type=PositionType.SURGE,
        exit_plan=manager.position_configs[PositionType.SURGE],
        current_price=11.5,  # 15% profit
        unrealized_pnl_pct=15.0,
        unrealized_pnl_dollars=1500.0
    )
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Strong momentum, early exit",
            "market_data": {"momentum_5min": 25.0, "volume_ratio": 4.0},
            "should_delay": True
        },
        {
            "name": "Weak momentum, normal exit",
            "market_data": {"momentum_5min": 5.0, "volume_ratio": 2.0},
            "should_delay": False
        },
        {
            "name": "High volume, moderate momentum",
            "market_data": {"momentum_5min": 18.0, "volume_ratio": 6.0},
            "should_delay": True
        }
    ]
    
    for scenario in test_scenarios:
        should_delay = manager._should_delay_exit(position, scenario["market_data"], datetime.now(et_timezone))
        
        status = "✓ PASS" if should_delay == scenario["should_delay"] else "✗ FAIL"
        print(f"{status}: {scenario['name']} -> delay={should_delay} (expected: {scenario['should_delay']})")

if __name__ == "__main__":
    print("Testing Exit Logic Fixes")
    print("=" * 50)
    
    test_enhanced_surge_detection()
    test_surge_exit_conditions()
    test_dynamic_adjustments()
    test_exit_delays()
    
    print("\n" + "=" * 50)
    print("Test completed!")
