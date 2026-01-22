#!/usr/bin/env python3
"""
Simple test to check Webull API response
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.WebullUtil import fetch_top_gainers

def test_webull_api():
    """Test Webull API directly"""
    print("Testing Webull API...")
    
    try:
        # Test different rank types
        for rank_type in ['preMarket', '1d', 'afterMarket']:
            print(f"\n=== Testing {rank_type} ===")
            gainers = fetch_top_gainers(rankType=rank_type, pageSize=5)
            
            if gainers:
                print(f"Type: {type(gainers)}")
                print(f"Length: {len(gainers)}")
                
                if isinstance(gainers, list):
                    for i, gainer in enumerate(gainers[:3]):
                        print(f"  {i+1}. {type(gainer)}: {gainer}")
                elif isinstance(gainers, dict):
                    if 'data' in gainers:
                        data_list = gainers['data']
                        print(f"  Data list length: {len(data_list)}")
                        for i, gainer in enumerate(data_list[:3]):
                            print(f"  {i+1}. {type(gainer)}: {gainer}")
                    else:
                        print(f"  Keys: {list(gainers.keys())}")
                else:
                    print(f"  Raw response: {gainers}")
            else:
                print("  No gainers returned")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_webull_api()
