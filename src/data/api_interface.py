"""
API Interface for Real-Time Data
Provides abstract interface for fetching 1-minute stock data
"""
import pandas as pd
from typing import Optional, Dict, List
from abc import ABC, abstractmethod
from datetime import datetime


class DataAPI(ABC):
    """Abstract base class for data API interfaces"""
    
    @abstractmethod
    def get_1min_data(self, ticker: str, minutes: int = 800) -> pd.DataFrame:
        """
        Fetch 1-minute data for a ticker
        
        Args:
            ticker: Stock ticker symbol
            minutes: Number of minutes of historical data to fetch (default: 800)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        pass
    
    @abstractmethod
    def get_current_price(self, ticker: str) -> float:
        """Get current price for a ticker"""
        pass


class CSVDataAPI(DataAPI):
    """Example implementation using CSV files (for testing)"""
    
    def __init__(self, data_dir: str = "test_data"):
        self.data_dir = data_dir
    
    def get_1min_data(self, ticker: str, minutes: int = 800) -> pd.DataFrame:
        """Load data from CSV file(s) (for testing purposes)"""
        from pathlib import Path
        
        data_path = Path(self.data_dir)
        # Find all 1-minute CSV files for ticker
        pattern = f"{ticker}-1m_*.csv"
        files = list(data_path.glob(pattern))
        
        if not files:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Load and combine all files, sorted by timestamp
        all_data = []
        for file in files:
            try:
                df = pd.read_csv(file)
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                all_data.append(df)
            except Exception as e:
                print(f"Warning: Could not load {file.name}: {e}")
                continue
        
        if not all_data:
            raise ValueError(f"Could not load any data for ticker {ticker}")
        
        # Combine all dataframes
        df_combined = pd.concat(all_data, ignore_index=True)
        df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates (in case of overlapping data)
        df_combined = df_combined.drop_duplicates(subset=['timestamp'], keep='last')
        df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
        
        # Return last N minutes
        if len(df_combined) > minutes:
            df_combined = df_combined.iloc[-minutes:].reset_index(drop=True)
        
        return df_combined[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    def get_current_price(self, ticker: str) -> float:
        """Get current price from most recent data"""
        df = self.get_1min_data(ticker, minutes=1)
        return df.iloc[-1]['close']


class CustomDataAPI(DataAPI):
    """
    Template for custom API implementation
    Replace the methods with your actual API calls
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize with your API credentials
        
        Args:
            api_key: Your API key
            base_url: Base URL for API endpoints
        """
        self.api_key = api_key
        self.base_url = base_url
    
    def get_1min_data(self, ticker: str, minutes: int = 800) -> pd.DataFrame:
        """
        Implement your API call here
        
        Example structure:
        - Make API request to get 1-minute candles
        - Parse response into DataFrame
        - Ensure columns: timestamp, open, high, low, close, volume
        - Return DataFrame sorted by timestamp
        """
        # TODO: Replace with your actual API call
        # Example:
        # response = requests.get(f"{self.base_url}/candles/{ticker}", 
        #                        params={"interval": "1m", "limit": minutes},
        #                        headers={"Authorization": f"Bearer {self.api_key}"})
        # data = response.json()
        # df = pd.DataFrame(data)
        # return df
        
        raise NotImplementedError("Implement your API call here")
    
    def get_current_price(self, ticker: str) -> float:
        """
        Implement your API call to get current price
        
        Example:
        - Make API request to get latest quote
        - Return current price
        """
        # TODO: Replace with your actual API call
        # Example:
        # response = requests.get(f"{self.base_url}/quote/{ticker}",
        #                        headers={"Authorization": f"Bearer {self.api_key}"})
        # return response.json()['price']
        
        raise NotImplementedError("Implement your API call here")

