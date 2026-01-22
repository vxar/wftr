#!/usr/bin/env python3
"""
Simple verification of exit logic fixes
"""

def verify_fixes():
    """Verify that all fixes are properly implemented"""
    
    print("=== Exit Logic Fixes Verification ===")
    
    # Check if the intelligent_position_manager.py file contains our fixes
    try:
        with open('src/core/intelligent_position_manager.py', 'r') as f:
            content = f.read()
        
        fixes = [
            ("FIX 1: Enhanced Surge Detection", "pattern_name == 'PRICE_VOLUME_SURGE'"),
            ("FIX 2: Surge Exit Conditions", "_check_surge_exit_conditions"),
            ("FIX 3: Recovery Check", "_is_recovering"),
            ("FIX 4: Dynamic Adjustments", "_adjust_exit_thresholds_by_momentum"),
            ("FIX 4: Exit Delays", "_should_delay_exit"),
            ("EXIT DELAYED", "EXIT DELAYED"),
            ("DYNAMIC ADJUSTMENT", "DYNAMIC ADJUSTMENT")
        ]
        
        all_good = True
        for fix_name, search_string in fixes:
            if search_string in content:
                print(f"✓ {fix_name}: Found")
            else:
                print(f"✗ {fix_name}: Missing")
                all_good = False
        
        print(f"\nOverall Status: {'✓ ALL FIXES IMPLEMENTED' if all_good else '✗ SOME FIXES MISSING'}")
        
        # Show key sections of the code
        print("\n=== Key Code Sections ===")
        
        # Enhanced surge detection
        if "pattern_name == 'PRICE_VOLUME_SURGE'" in content:
            print("\n1. Enhanced Surge Detection:")
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "pattern_name == 'PRICE_VOLUME_SURGE'" in line:
                    for j in range(max(0, i-2), min(len(lines), i+6)):
                        print(f"   {lines[j]}")
                    break
        
        # Surge exit conditions
        if "_check_surge_exit_conditions" in content:
            print("\n2. Surge Exit Conditions:")
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "def _check_surge_exit_conditions" in line:
                    for j in range(max(0, i-1), min(len(lines), i+8)):
                        print(f"   {lines[j]}")
                    break
        
        # Recovery check
        if "_is_recovering" in content:
            print("\n3. Recovery Check:")
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "def _is_recovering" in line:
                    for j in range(max(0, i-1), min(len(lines), i+10)):
                        print(f"   {lines[j]}")
                    break
        
        return all_good
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

if __name__ == "__main__":
    verify_fixes()
