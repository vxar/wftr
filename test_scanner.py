#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.WebullUtil import fetch_top_gainers, get_rank_type

def test_scanner():
    print("Testing Webull API response...")
    
    rank_type = get_rank_type()
    print(f"Current rank type: {rank_type}")
    
    raw_gainers = fetch_top_gainers(rankType=rank_type, pageSize=3)
    
    if not raw_gainers:
        print("No gainers returned")
        return
    
    # Extract the actual data from the response
    gainers_list = raw_gainers
    if isinstance(raw_gainers, dict) and 'data' in raw_gainers:
        gainers_list = raw_gainers['data']
    
    print(f"Response type: {type(raw_gainers)}")
    print(f"Gainers list type: {type(gainers_list)}")
    print(f"Number of gainers: {len(gainers_list)}")
    
    for i, gainer in enumerate(gainers_list[:2]):
        print(f"\nGainer {i+1}:")
        print(f"  Type: {type(gainer)}")
        if isinstance(gainer, dict):
            print(f"  Keys: {list(gainer.keys())}")
            # Try to find symbol
            for key in ['symbol', 'ticker', 'code', 'disSymbol']:
                if key in gainer:
                    print(f"  {key}: {gainer[key]}")
        else:
            print(f"  Value: {gainer}")

if __name__ == "__main__":
    test_scanner()
