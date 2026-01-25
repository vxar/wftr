"""
Improved Entry Timing Configuration for DRCT-style surges
This file contains the optimized parameters to fix late entry issues
"""

# Current vs Improved Configuration
SURGE_CONFIG_IMPROVEMENTS = {
    "current": {
        "surge_min_volume_ratio": 100.0,  # Too conservative
        "surge_min_volume": 50000,
        "surge_min_price_increase": 30.0,
        "surge_continuation_min_volume": 500000,
        "confirmation_bars": 3,  # Wait for 3 bars confirmation
        "baseline_lookback": 15  # 15-minute baseline
    },
    "improved": {
        "surge_min_volume_ratio": 30.0,  # Much more responsive
        "surge_min_volume": 30000,  # Lower absolute threshold
        "surge_min_price_increase": 15.0,  # Lower price threshold
        "surge_continuation_min_volume": 200000,  # Lower continuation threshold
        "confirmation_bars": 1,  # Enter on 1-bar confirmation
        "baseline_lookback": 5  # 5-minute baseline for early detection
    }
}

def get_improved_surge_config():
    """Return the improved surge configuration"""
    return SURGE_CONFIG_IMPROVEMENTS["improved"]

def calculate_entry_improvements():
    """Calculate potential entry price improvements"""
    improvements = {
        "first_entry": {
            "actual_entry": "04:20:00",
            "actual_price": 3.38,
            "optimal_entry": "04:19:00", 
            "optimal_price": 3.18,
            "improvement_pct": ((3.38 - 3.18) / 3.38) * 100,
            "minutes_earlier": 1
        },
        "second_entry": {
            "actual_entry": "07:09:00",
            "actual_price": 5.13,
            "optimal_entry": "07:07:00",
            "optimal_price": 4.47,
            "improvement_pct": ((5.13 - 4.47) / 5.13) * 100,
            "minutes_earlier": 2
        },
        "third_entry": {
            "actual_entry": "14:58:00", 
            "actual_price": 4.46,
            "optimal_entry": "14:51:00",
            "optimal_price": 4.20,
            "improvement_pct": ((4.46 - 4.20) / 4.46) * 100,
            "minutes_earlier": 7
        }
    }
    
    total_improvement = sum([imp["improvement_pct"] for imp in improvements.values()])
    avg_improvement = total_improvement / len(improvements)
    
    print("Entry Timing Improvements Analysis:")
    print("=" * 50)
    for name, data in improvements.items():
        print(f"{name.replace('_', ' ').title()}:")
        print(f"  Actual: {data['actual_entry']} @ ${data['actual_price']:.4f}")
        print(f"  Optimal: {data['optimal_entry']} @ ${data['optimal_price']:.4f}")
        print(f"  Improvement: {data['improvement_pct']:.1f}% ({data['minutes_earlier']} min earlier)")
        print()
    
    print(f"Average improvement: {avg_improvement:.1f}%")
    return improvements

if __name__ == "__main__":
    calculate_entry_improvements()
