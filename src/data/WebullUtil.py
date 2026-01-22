import requests
import pandas as pd
from datetime import datetime, time as dt_time
import traceback
from ..utils.utils import logger
import json

log = logger.get_logger()
PM = 'preMarket'
AM = 'afterMarket'
REG = '1d'


def find_tickerid_for_symbol(symbol):
    tickerid = None
    try:
        url = f"https://quotes-gw.webullfintech.com/api/search/pc/tickers?keyword={symbol}&pageIndex=1&pageSize=1"
        headers = {
            "device-type": "Web",
            "did": "xtw0doz2stnl2xghaa0hnba6h7kkslni",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 Edg/128.0.0.0",
            "ver": "4.9.5"
        }

        response = requests.get(url, headers=headers)
        data = response.json()
        
        # Check if data exists and has results
        if 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
            tickerid = data['data'][0]['tickerId']
        else:
            log.debug(f"find_tickerid_for_symbol: No ticker found for symbol {symbol}")
    except Exception as e:
        log.error(f"find_tickerid_for_symbol: " + str(e) + "\n" + traceback.format_exc())

    return tickerid


def get_stock_quote(tickerid):
    quote = None
    try:
        url = f"https://quotes-gw.webullfintech.com/api/bgw/quote/realtime?ids={tickerid}&includeSecu=1&delay=0&more=1&includeQuote=1"
        headers = {
            "device-type": "Web",
            "did": "xtw0doz2stnl2xghaa0hnba6h7kkslni",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 Edg/128.0.0.0",
            "ver": "4.9.5"
        }

        response = requests.get(url, headers=headers, timeout=10)
        
        # Check response status
        if response.status_code != 200:
            log.error(f"get_stock_quote: HTTP {response.status_code} for tickerId {tickerid}")
            return None
        
        data = response.json()
        
        # Check if data is a list and has at least one element
        if isinstance(data, list):
            if len(data) > 0:
                quote = data[0]
            else:
                log.error(f"get_stock_quote: Empty list returned for tickerId {tickerid}")
                quote = None
        elif isinstance(data, dict):
            # Sometimes the API returns a dict directly
            # Check if it's an error response
            if 'code' in data and data.get('code') != 0:
                log.error(f"get_stock_quote: API error for tickerId {tickerid}: {data.get('msg', 'Unknown error')}")
                quote = None
            else:
                quote = data
        else:
            log.error(f"get_stock_quote: Unexpected data format for tickerId {tickerid}: {type(data)}, data: {str(data)[:200]}")
            quote = None
    except (KeyError, IndexError, TypeError) as e:
        log.error(f"get_stock_quote: Error accessing quote data for tickerId {tickerid}: {e}")
        quote = None
    except requests.exceptions.RequestException as e:
        log.error(f"get_stock_quote: Network error for tickerId {tickerid}: {e}")
        quote = None
    except Exception as e:
        log.error(f"get_stock_quote: {e}\n" + traceback.format_exc())
        quote = None
    return quote


def fetch_top_gainers(rankType, pageSize):
    json_data = None
    try:
        headers = {
            "device-type": "Web",
            "did": "xtw0doz2stnl2xghaa0hnba6h7kkslni",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 Edg/128.0.0.0",
            "ver": "4.9.5"
        }

        url = f"https://quotes-gw.webullfintech.com/api/wlas/ranking/topGainers?" \
              f"regionId=6&" \
              f"rankType={rankType}&pageIndex=1" \
              f"&pageSize={pageSize}"

        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an error for a failed request
        json_data = response.json()
    except Exception as e:
        log.error(f"fetch_top_gainers: " + str(e) + "\n" + traceback.format_exc())

    return json_data


def fetch_swing_stocks(min_price=10, max_price=250, rsi_min=40, rsi_max=60):
    json_data = None
    try:
        url = f"https://quotes-gw.webullfintech.com/api/wlas/screener/ng/query"

        headers = {"Content-Type": "application/json"}
        data = {
            "fetch": 2000,
            "rules": {
                "wlas.screener.rule.region": "securities.region.name.6",
                "wlas.screener.rule.price": "gte={0}&lte={1}".format(str(min_price), str(max_price)),
                "wlas.screener.rule.rsi": "gte=40&lte=60",
                "wlas.screener.rule.volume": "gte=500000",
                "wlas.screener.group.technical.signals": None,
                "wlas.screener.rule.recommend": None
            },
            "sort": {
                "rule": "wlas.screener.rule.price",
                "desc": False
            },
            "attach": {
                "hkexPrivilege": False
            }
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))

        response.raise_for_status()  # Raises an error for a failed request
        json_data = response.json()
    except Exception as e:
        log.error(f"fetch_top_gainers: " + str(e) + "\n" + traceback.format_exc())

    return json_data


def fetch_data_array(ticker_id, symbol=None, timeframe='m1', count=800):
    """
    Fetch stock data array from Webull API
    
    Args:
        ticker_id: Webull ticker ID
        symbol: Stock symbol (optional, used if ticker_id is None)
        timeframe: Time interval - 'm1' (1-minute) or 'm5' (5-minute)
        count: Number of periods to fetch (max 1200)
    
    Returns:
        DataFrame with timestamp index and OHLCV data
    """
    df = None
    try:
        if ticker_id is None and symbol is not None:
            ticker_id = find_tickerid_for_symbol(symbol)

        url = f"https://quotes-gw.webullfintech.com/api/quote/charts/query-mini?tickerId={ticker_id}" \
              f"&type={timeframe}" \
              f"&count={count}" \
              f"&restorationType=1" \
              f"&loadFactor=1" \
              f"&extendTrading=1"
        headers = {
            "device-type": "Web",
            "did": "xtw0doz2stnl2xghaa0hnba6h7kkslni",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 Edg/128.0.0.0",
            "ver": "4.9.5"
        }
        response = requests.get(url, headers=headers)
        data = response.json()
        data_array = data[0]['data']
        columns = ["timestamp", "open", "close", "high", "low", "preClose", "volume", "vwap"]
        df = pd.DataFrame([x.split(',') for x in data_array], columns=columns)
        df = df.sort_values(by='timestamp', ascending=True)

        df['open'] = pd.to_numeric(df['open'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        df['preClose'] = pd.to_numeric(df['preClose'])
        df['volume'] = pd.to_numeric(df['volume'])
        try:
            df['vwap'] = pd.to_numeric(df['vwap'])
        except Exception as ne:
            # do nothing, ignore null vwap values
            traceback.format_exc()

        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit="s")
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        df['timestamp'] = df['timestamp'].dt.tz_convert('US/Eastern')
        df.set_index('timestamp', inplace=True)
    except Exception as e:
        log.error(f"fetch_data_array: " + str(e) + "\n" + traceback.format_exc())

    return df

def get_rank_type(dt=None):
    if dt is None:
        now = datetime.now().time()
    else:
        now = dt.time()

    pm_start = dt_time(4, 0)  # Correct way to create a time object with alias
    pm_end = dt_time(9, 30)

    reg_start = dt_time(9, 30)
    reg_end = dt_time(16, 0)

    am_start = dt_time(16, 0)
    am_end = dt_time(20, 0)

    if pm_start <= now < pm_end:
        rankType = PM
        # print("Running pre market")
    elif reg_start <= now < reg_end:
        rankType = REG
        # print("Running regular trading hours")
    elif am_start <= now < am_end:
        rankType = AM
        # print("Running after market")
    else:
        rankType = PM
        # print("The current time is outside the specified intervals. set for next day pre-market")

    return rankType

def calculate_relative_volume(ticker_id=None, symbol=None, window=14):
    try:
        if ticker_id is None:
            ticker_id = find_tickerid_for_symbol(symbol)
        day_df = fetch_data_array(ticker_id, timeframe='d1', count=50)

        today = datetime.now().date()
        if day_df.index[-1].date() == today:
            day_df = day_df.iloc[:-1]

        today_quote = get_stock_quote(ticker_id)

        today_df = [pd.to_numeric(today_quote['open']), pd.to_numeric(today_quote['close']), pd.to_numeric(today_quote['high']),
                    pd.to_numeric(today_quote['low']), pd.to_numeric(today_quote['preClose']), pd.to_numeric(today_quote['volume']), 0]
        # print(today_df)
        day_df.loc[today] = today_df

        # t_vol = pd.to_numeric(today_quote['volume'])
        # d10_vol = pd.to_numeric(today_quote['avgVol10D'])
        # print(t_vol / d10_vol)

        day_df['relative_volume'] = day_df['volume'] / day_df['volume'].rolling(window).mean()

        # print(day_df.tail(5).to_string())
        return day_df['relative_volume'].iloc[-1]
    except Exception as e:
        log.error(f"calculate_relative_volume: " + str(e) + "\n" + traceback.format_exc())
    return None



# print(get_stock_quote('950165986'))
#print(calculate_relative_volume(symbol='BBAI'))

