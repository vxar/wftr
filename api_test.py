import requests
import json

def test_api():
    headers = {
        "device-type": "Web",
        "did": "xtw0doz2stnl2xghaa0hnba6h7kkslni",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 Edg/128.0.0.0",
        "ver": "4.9.5"
    }

    url = "https://quotes-gw.webullfintech.com/api/wlas/ranking/topGainers?regionId=6&rankType=1d&pageIndex=1&pageSize=2"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        print(f"Response type: {type(data)}")
        print(f"Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        if isinstance(data, dict) and 'data' in data:
            gainers = data['data']
            print(f"Number of gainers: {len(gainers)}")
            if gainers:
                first_gainer = gainers[0]
                print(f"First gainer type: {type(first_gainer)}")
                if isinstance(first_gainer, dict):
                    print(f"First gainer keys: {list(first_gainer.keys())}")
                    # Check for symbol fields
                    for key in ['symbol', 'ticker', 'code', 'disSymbol', 'name']:
                        if key in first_gainer:
                            print(f"  {key}: {first_gainer[key]}")
        elif isinstance(data, list):
            print(f"Direct list with {len(data)} items")
            if data:
                first = data[0]
                print(f"First item keys: {list(first.keys()) if isinstance(first, dict) else 'Not a dict'}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
