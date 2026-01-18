#!/usr/bin/env python3
"""Validate simulator setup without running full simulation"""

import ast
import os
from pathlib import Path

def check_syntax(file_path):
    """Check if Python file has valid syntax"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def main():
    print("🔍 Validating simulator setup...")
    
    # Check key files exist and have valid syntax
    files_to_check = [
        "src/simulation/pure_trade_simulator.py",
        "src/core/intelligent_position_manager.py", 
        "src/data/webull_data_api.py",
        "run_simulator.py",
        "run_simulator_full.py"
    ]
    
    all_good = True
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            is_valid, error = check_syntax(file_path)
            if is_valid:
                print(f"✅ {file_path} - Valid syntax")
            else:
                print(f"❌ {file_path} - Syntax error: {error}")
                all_good = False
        else:
            print(f"❌ {file_path} - File not found")
            all_good = False
    
    # Check imports in main simulator
    try:
        with open("src/simulation/pure_trade_simulator.py", 'r') as f:
            content = f.read()
        
        required_imports = [
            "from src.core.intelligent_position_manager import",
            "from src.data.webull_data_api import"
        ]
        
        for imp in required_imports:
            if imp in content:
                print(f"✅ Found required import: {imp}")
            else:
                print(f"❌ Missing import: {imp}")
                all_good = False
                
    except Exception as e:
        print(f"❌ Error checking imports: {e}")
        all_good = False
    
    if all_good:
        print("\n🎉 Simulator setup validation PASSED!")
        print("The simulator should work correctly.")
    else:
        print("\n❌ Simulator setup validation FAILED!")
        print("Please fix the issues above.")

if __name__ == "__main__":
    main()
