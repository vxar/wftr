"""
Trading Database Module
Handles persistence of trades and positions using SQLite
"""
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Trade record for database"""
    ticker: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    shares: float
    entry_value: float
    exit_value: float
    pnl_pct: float
    pnl_dollars: float
    entry_pattern: str
    exit_reason: str
    confidence: float


@dataclass
class PositionRecord:
    """Position record for database"""
    ticker: str
    entry_time: datetime
    entry_price: float
    shares: float
    entry_value: float
    entry_pattern: str
    confidence: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    is_active: bool = True


class TradingDatabase:
    """Database handler for trading data"""
    
    def __init__(self, db_path: str = "trading_data.db"):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database tables if they don't exist"""
        try:
            # Use WAL mode for better concurrency and add timeout
            self.conn = sqlite3.connect(
                self.db_path, 
                check_same_thread=False,
                timeout=30.0  # 30 second timeout for database operations
            )
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            
            # Enable WAL mode for better concurrent access
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA busy_timeout=30000")  # 30 second busy timeout
            
            cursor = self.conn.cursor()
            
            # Create trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    shares REAL NOT NULL,
                    entry_value REAL NOT NULL,
                    exit_value REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    pnl_dollars REAL NOT NULL,
                    entry_pattern TEXT NOT NULL,
                    exit_reason TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create positions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL UNIQUE,
                    entry_time TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    shares REAL NOT NULL,
                    entry_value REAL NOT NULL,
                    entry_pattern TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    target_price REAL,
                    stop_loss REAL,
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create rejected_entries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rejected_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    price REAL NOT NULL,
                    reason TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    date TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions(ticker)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_active ON positions(is_active)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rejected_entries_date ON rejected_entries(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rejected_entries_ticker ON rejected_entries(ticker)")
            
            self.conn.commit()
            
            # Check and add missing columns (migration)
            self._migrate_database()
            
            logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def _migrate_database(self):
        """Add missing columns to existing tables (migration)"""
        try:
            cursor = self.conn.cursor()
            
            # Check if updated_at column exists in positions table
            cursor.execute("PRAGMA table_info(positions)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'updated_at' not in columns:
                logger.info("Adding 'updated_at' column to positions table")
                cursor.execute("ALTER TABLE positions ADD COLUMN updated_at TEXT DEFAULT CURRENT_TIMESTAMP")
                self.conn.commit()
            
        except Exception as e:
            logger.warning(f"Error during database migration: {e}")
            # Don't raise - migration failures shouldn't break the app
    
    def add_trade(self, trade: TradeRecord) -> int:
        """
        Add a completed trade to the database
        
        Args:
            trade: TradeRecord object
            
        Returns:
            Trade ID
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO trades (
                    ticker, entry_time, exit_time, entry_price, exit_price,
                    shares, entry_value, exit_value, pnl_pct, pnl_dollars,
                    entry_pattern, exit_reason, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.ticker,
                trade.entry_time.isoformat() if isinstance(trade.entry_time, datetime) else trade.entry_time,
                trade.exit_time.isoformat() if isinstance(trade.exit_time, datetime) else trade.exit_time,
                trade.entry_price,
                trade.exit_price,
                trade.shares,
                trade.entry_value,
                trade.exit_value,
                trade.pnl_pct,
                trade.pnl_dollars,
                trade.entry_pattern,
                trade.exit_reason,
                trade.confidence
            ))
            self.conn.commit()
            trade_id = cursor.lastrowid
            logger.debug(f"Trade added to database: {trade.ticker} (ID: {trade_id})")
            return trade_id
        except Exception as e:
            logger.error(f"Error adding trade to database: {e}")
            self.conn.rollback()
            raise
    
    def add_position(self, position: PositionRecord) -> int:
        """
        Add or update a position in the database
        
        Args:
            position: PositionRecord object
            
        Returns:
            Position ID
        """
        try:
            cursor = self.conn.cursor()
            
            # Check if position already exists
            cursor.execute("SELECT id FROM positions WHERE ticker = ? AND is_active = 1", (position.ticker,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing position
                cursor.execute("""
                    UPDATE positions SET
                        entry_time = ?,
                        entry_price = ?,
                        shares = ?,
                        entry_value = ?,
                        entry_pattern = ?,
                        confidence = ?,
                        target_price = ?,
                        stop_loss = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE ticker = ? AND is_active = 1
                """, (
                    position.entry_time.isoformat() if isinstance(position.entry_time, datetime) else position.entry_time,
                    position.entry_price,
                    position.shares,
                    position.entry_value,
                    position.entry_pattern,
                    position.confidence,
                    position.target_price,
                    position.stop_loss,
                    position.ticker
                ))
                position_id = existing['id']
            else:
                # Insert new position
                cursor.execute("""
                    INSERT INTO positions (
                        ticker, entry_time, entry_price, shares, entry_value,
                        entry_pattern, confidence, target_price, stop_loss, is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    position.ticker,
                    position.entry_time.isoformat() if isinstance(position.entry_time, datetime) else position.entry_time,
                    position.entry_price,
                    position.shares,
                    position.entry_value,
                    position.entry_pattern,
                    position.confidence,
                    position.target_price,
                    position.stop_loss,
                    1 if position.is_active else 0
                ))
                position_id = cursor.lastrowid
            
            self.conn.commit()
            logger.debug(f"Position saved to database: {position.ticker} (ID: {position_id})")
            return position_id
        except Exception as e:
            logger.error(f"Error adding position to database: {e}")
            self.conn.rollback()
            raise
    
    def update_position(self, ticker: str, target_price: Optional[float] = None, 
                       stop_loss: Optional[float] = None, shares: Optional[float] = None,
                       entry_value: Optional[float] = None):
        """
        Update target price, stop loss, shares, and/or entry value for a position
        
        Args:
            ticker: Stock ticker symbol
            target_price: New target price (optional)
            stop_loss: New stop loss price (optional)
            shares: New shares count (optional, for partial exits)
            entry_value: New entry value (optional, for partial exits)
        """
        try:
            cursor = self.conn.cursor()
            updates = []
            params = []
            
            if target_price is not None:
                updates.append("target_price = ?")
                params.append(target_price)
            
            if stop_loss is not None:
                updates.append("stop_loss = ?")
                params.append(stop_loss)
            
            if shares is not None:
                updates.append("shares = ?")
                params.append(shares)
            
            if entry_value is not None:
                updates.append("entry_value = ?")
                params.append(entry_value)
            
            if not updates:
                return  # Nothing to update
            
            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(ticker)  # Add ticker for WHERE clause
            
            query = f"""
                UPDATE positions SET
                    {', '.join(updates)}
                WHERE ticker = ? AND is_active = 1
            """
            cursor.execute(query, params)
            self.conn.commit()
            logger.debug(f"Position updated in database: {ticker} (target={target_price}, stop={stop_loss}, shares={shares})")
        except Exception as e:
            logger.error(f"Error updating position in database: {e}")
            self.conn.rollback()
            raise
    
    def close_position(self, ticker: str):
        """
        Delete a position from the database when trade is completed
        Completed trades are stored in the trades table, positions table is only for active positions
        
        Args:
            ticker: Stock ticker symbol
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                DELETE FROM positions
                WHERE ticker = ? AND is_active = 1
            """, (ticker,))
            deleted_count = cursor.rowcount
            self.conn.commit()
            if deleted_count > 0:
                logger.debug(f"Position deleted from database: {ticker}")
            else:
                logger.warning(f"Position not found or already closed: {ticker}")
        except Exception as e:
            logger.error(f"Error closing position in database: {e}")
            self.conn.rollback()
            raise
    
    def cleanup_inactive_positions(self):
        """
        Remove all inactive positions from the database
        This should be called periodically to clean up any stale positions
        """
        max_retries = 3
        retry_delay = 0.1  # Start with 100ms delay
        
        for attempt in range(max_retries):
            try:
                cursor = self.conn.cursor()
                cursor.execute("""
                    DELETE FROM positions
                    WHERE is_active = 0
                """)
                deleted_count = cursor.rowcount
                self.conn.commit()
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} inactive position(s) from database")
                return deleted_count
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    # Retry with exponential backoff
                    import time
                    time.sleep(retry_delay * (2 ** attempt))
                    logger.debug(f"Database locked, retrying cleanup_inactive_positions (attempt {attempt + 1}/{max_retries})")
                    continue
                else:
                    logger.error(f"Error cleaning up inactive positions: {e}")
                    self.conn.rollback()
                    return 0
            except Exception as e:
                logger.error(f"Error cleaning up inactive positions: {e}")
                self.conn.rollback()
                return 0
        return 0
    
    def cleanup_orphaned_positions(self):
        """
        Remove positions that have corresponding completed trades
        This ensures positions table only contains truly active positions
        """
        max_retries = 3
        retry_delay = 0.1  # Start with 100ms delay
        
        for attempt in range(max_retries):
            try:
                cursor = self.conn.cursor()
                # Find positions that have completed trades
                cursor.execute("""
                    DELETE FROM positions
                    WHERE ticker IN (
                        SELECT DISTINCT ticker FROM trades
                    )
                """)
                deleted_count = cursor.rowcount
                self.conn.commit()
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} orphaned position(s) (positions with completed trades)")
                return deleted_count
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    # Retry with exponential backoff
                    import time
                    time.sleep(retry_delay * (2 ** attempt))
                    logger.debug(f"Database locked, retrying cleanup_orphaned_positions (attempt {attempt + 1}/{max_retries})")
                    continue
                else:
                    logger.error(f"Error cleaning up orphaned positions: {e}")
                    self.conn.rollback()
                    return 0
            except Exception as e:
                logger.error(f"Error cleaning up orphaned positions: {e}")
                self.conn.rollback()
                return 0
        return 0
    
    def get_all_trades(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all completed trades
        
        Args:
            limit: Maximum number of trades to return (None for all)
            
        Returns:
            List of trade dictionaries
        """
        try:
            cursor = self.conn.cursor()
            
            # Use SELECT * - we know the schema, handle missing columns gracefully
            # Removed PRAGMA table_info - it was causing column detection failures
            # Filter out null/empty tickers at the SQL level for better performance
            query = "SELECT * FROM trades WHERE ticker IS NOT NULL AND ticker != '' ORDER BY exit_time DESC"
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            trades = []
            for row in rows:
                trade_dict = {}
                
                # Safely extract all columns (row factory is set to sqlite3.Row)
                # Required columns - skip if missing (prevents null ticker trades)
                try:
                    trade_dict['id'] = row['id']
                except (KeyError, IndexError):
                    continue  # Skip if id is missing
                
                try:
                    ticker = row['ticker']
                    # Double-check: skip if ticker is None, empty, or whitespace-only
                    if not ticker or not str(ticker).strip():
                        continue
                    trade_dict['ticker'] = str(ticker).strip()
                except (KeyError, IndexError):
                    continue  # Skip if ticker is missing
                
                # Optional columns with safe defaults
                for col in ['entry_time', 'exit_time', 'entry_price', 'exit_price', 
                           'shares', 'entry_value', 'exit_value', 'pnl_pct', 
                           'pnl_dollars', 'entry_pattern', 'exit_reason', 'confidence', 'created_at']:
                    try:
                        trade_dict[col] = row[col]
                    except (KeyError, IndexError):
                        # Use defaults for missing columns
                        if col in ['entry_price', 'exit_price', 'shares', 'entry_value', 
                                  'exit_value', 'pnl_pct', 'pnl_dollars', 'confidence']:
                            trade_dict[col] = 0.0
                        elif col in ['entry_pattern', 'exit_reason']:
                            trade_dict[col] = 'Unknown'
                        else:
                            trade_dict[col] = None
                
                trades.append(trade_dict)
            
            return trades
        except Exception as e:
            logger.error(f"Error getting trades from database: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_active_positions(self) -> List[Dict]:
        """
        Get all active positions
        
        Returns:
            List of position dictionaries
        """
        try:
            cursor = self.conn.cursor()
            
            # First, check which columns exist in the positions table
            cursor.execute("PRAGMA table_info(positions)")
            column_info = cursor.fetchall()
            
            # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
            # Handle both Row and tuple formats
            available_columns = {}
            for col in column_info:
                try:
                    # Try Row access first (if row_factory is set)
                    if hasattr(col, 'keys'):
                        col_name = col['name']
                    else:
                        # Fallback to tuple access (cid, name, type, ...)
                        col_name = col[1] if len(col) > 1 else None
                    
                    if col_name:
                        available_columns[col_name] = True
                except (KeyError, IndexError, TypeError) as e:
                    logger.debug(f"Error extracting column name: {e}")
                    continue
            
            # Build SELECT query with only existing columns
            base_columns = ['id', 'ticker', 'entry_time', 'entry_price', 'shares', 
                          'entry_value', 'entry_pattern', 'confidence', 'target_price', 
                          'stop_loss', 'is_active']
            optional_columns = ['created_at', 'updated_at']
            
            # Select only columns that exist
            select_columns = [col for col in base_columns if col in available_columns]
            select_columns.extend([col for col in optional_columns if col in available_columns])
            
            if not select_columns:
                # If no columns found, try a simple SELECT * as fallback
                logger.warning("No columns found in positions table via PRAGMA, using SELECT * as fallback")
                query = "SELECT * FROM positions WHERE is_active = 1 AND ticker IS NOT NULL AND ticker != '' ORDER BY entry_time DESC"
                cursor.execute(query)
                rows = cursor.fetchall()
                if not rows:
                    return []
                # Use the first row to determine available columns
                if hasattr(rows[0], 'keys'):
                    select_columns = list(rows[0].keys())
                else:
                    # Fallback: use standard columns
                    select_columns = base_columns
            
            # Clean up any corrupted positions (where id or ticker is NULL)
            # This prevents them from being selected and causing warnings
            max_retries = 3
            retry_delay = 0.1
            
            for attempt in range(max_retries):
                try:
                    cleanup_cursor = self.conn.cursor()
                    deleted = cleanup_cursor.execute(
                        "DELETE FROM positions WHERE (id IS NULL OR ticker IS NULL OR ticker = '') AND is_active = 1"
                    ).rowcount
                    if deleted > 0:
                        self.conn.commit()
                        logger.info(f"Cleaned up {deleted} corrupted position(s) from database")
                    break  # Success, exit retry loop
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                        # Retry with exponential backoff
                        import time
                        time.sleep(retry_delay * (2 ** attempt))
                        logger.debug(f"Database locked, retrying cleanup_corrupted_positions (attempt {attempt + 1}/{max_retries})")
                        continue
                    else:
                        logger.warning(f"Error cleaning up corrupted positions: {e}")
                        break
                except Exception as e:
                    logger.warning(f"Error cleaning up corrupted positions: {e}")
                    break
            
            # Filter out null/empty tickers at SQL level
            # Also filter out rows where id is NULL (corrupted rows)
            query = f"""
                SELECT {', '.join(select_columns)} FROM positions 
                WHERE is_active = 1 
                  AND ticker IS NOT NULL 
                  AND ticker != ''
                  AND id IS NOT NULL
                ORDER BY entry_time DESC
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            
            positions = []
            for row in rows:
                position_dict = {}
                columns_accessed = 0
                
                # Map columns by name (row factory is set to sqlite3.Row)
                for col in select_columns:
                    try:
                        if col == 'is_active':
                            position_dict[col] = bool(row[col])
                        else:
                            position_dict[col] = row[col]
                        columns_accessed += 1
                    except (KeyError, IndexError, TypeError) as e:
                        # If column access fails, use default
                        if col in ['created_at', 'updated_at', 'target_price', 'stop_loss']:
                            position_dict[col] = None
                        elif col == 'is_active':
                            position_dict[col] = True
                        elif col in ['entry_price', 'shares', 'entry_value', 'confidence']:
                            position_dict[col] = 0.0
                        elif col == 'entry_pattern':
                            position_dict[col] = 'Unknown'
                        else:
                            position_dict[col] = None
                
                # If we couldn't access any columns, this row is corrupted - skip it
                if columns_accessed == 0:
                    logger.warning(f"Skipping corrupted position row - no columns could be accessed. Row type: {type(row)}")
                    # Try to get row ID for deletion if possible
                    try:
                        # Try multiple ways to get the ID
                        row_id = None
                        if hasattr(row, '__getitem__'):
                            try:
                                row_id = row[0] if len(row) > 0 else None
                            except (IndexError, TypeError):
                                pass
                        if not row_id and hasattr(row, 'keys'):
                            try:
                                row_id = row.get('id') if 'id' in row.keys() else None
                            except:
                                pass
                        
                        if row_id:
                            logger.info(f"Attempting to delete corrupted position with ID: {row_id}")
                            cursor.execute("DELETE FROM positions WHERE id = ?", (row_id,))
                            self.conn.commit()
                    except Exception as e:
                        logger.debug(f"Could not delete corrupted row: {e}")
                    continue
                
                # Ensure required fields have values and ticker is valid
                if 'id' not in position_dict or 'ticker' not in position_dict:
                    logger.warning(f"Skipping position with missing id or ticker: {position_dict}")
                    # Try to delete this invalid position
                    try:
                        if 'id' in position_dict and position_dict['id']:
                            cursor.execute("DELETE FROM positions WHERE id = ?", (position_dict['id'],))
                            self.conn.commit()
                            logger.info(f"Deleted invalid position with ID: {position_dict['id']}")
                    except:
                        pass
                    continue
                
                # Double-check ticker is not null/empty/whitespace
                ticker = position_dict.get('ticker')
                if not ticker or not str(ticker).strip():
                    logger.warning(f"Skipping position with invalid ticker: {position_dict}")
                    # Try to delete this invalid position
                    try:
                        if position_dict.get('id'):
                            cursor.execute("DELETE FROM positions WHERE id = ?", (position_dict['id'],))
                            self.conn.commit()
                            logger.info(f"Deleted position with invalid ticker, ID: {position_dict['id']}")
                    except:
                        pass
                    continue
                
                # Normalize ticker to string
                position_dict['ticker'] = str(ticker).strip()
                
                positions.append(position_dict)
            
            return positions
        except Exception as e:
            logger.error(f"Error getting active positions from database: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_trades_by_ticker(self, ticker: str) -> List[Dict]:
        """
        Get all trades for a specific ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of trade dictionaries
        """
        try:
            cursor = self.conn.cursor()
            
            # Use SELECT * - we know the schema, handle missing columns gracefully
            # Filter out null/empty tickers at SQL level
            cursor.execute("""
                SELECT * FROM trades 
                WHERE ticker = ? AND ticker IS NOT NULL AND ticker != ''
                ORDER BY exit_time DESC
            """, (ticker,))
            rows = cursor.fetchall()
            
            trades = []
            for row in rows:
                trade_dict = {}
                
                # Safely extract all columns (row factory is set to sqlite3.Row)
                # Required columns - skip if missing
                try:
                    trade_dict['id'] = row['id']
                except (KeyError, IndexError):
                    continue  # Skip if id is missing
                
                try:
                    ticker_val = row['ticker']
                    # Double-check: skip if ticker is None, empty, or whitespace-only
                    if not ticker_val or not str(ticker_val).strip():
                        continue
                    trade_dict['ticker'] = str(ticker_val).strip()
                except (KeyError, IndexError):
                    continue  # Skip if ticker is missing (this prevents null ticker trades)
                
                # Optional columns with safe defaults
                for col in ['entry_time', 'exit_time', 'entry_price', 'exit_price', 
                           'shares', 'entry_value', 'exit_value', 'pnl_pct', 
                           'pnl_dollars', 'entry_pattern', 'exit_reason', 'confidence', 'created_at']:
                    try:
                        trade_dict[col] = row[col]
                    except (KeyError, IndexError):
                        # Use defaults for missing columns
                        if col in ['entry_price', 'exit_price', 'shares', 'entry_value', 
                                  'exit_value', 'pnl_pct', 'pnl_dollars', 'confidence']:
                            trade_dict[col] = 0.0
                        elif col in ['entry_pattern', 'exit_reason']:
                            trade_dict[col] = 'Unknown'
                        else:
                            trade_dict[col] = None
                
                trades.append(trade_dict)
            
            return trades
        except Exception as e:
            logger.error(f"Error getting trades for {ticker}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_statistics(self) -> Dict:
        """
        Get trading statistics from database
        Recalculates all values to ensure accuracy
        
        Returns:
            Dictionary with statistics
        """
        try:
            cursor = self.conn.cursor()
            
            # Use SELECT * - we know the schema, handle missing columns gracefully
            # Removed PRAGMA table_info - it was causing column detection failures
            # Filter out null/empty tickers at SQL level for better performance
            cursor.execute("SELECT * FROM trades WHERE ticker IS NOT NULL AND ticker != '' ORDER BY exit_time DESC")
            all_trades = cursor.fetchall()
            
            # Convert sqlite3.Row objects to dictionaries and filter out null/empty tickers
            valid_trades = []
            for row in all_trades:
                try:
                    # Convert Row to dict (row factory is set to sqlite3.Row)
                    ticker = row['ticker']
                    # Double-check: skip if ticker is None, empty, or whitespace-only
                    if not ticker or not str(ticker).strip():
                        continue
                    ticker = str(ticker).strip()
                    
                    # Convert row to dictionary
                    trade = {
                        'ticker': ticker,
                        'shares': row['shares'] if 'shares' in row.keys() else 0,
                        'entry_price': row['entry_price'] if 'entry_price' in row.keys() else 0,
                        'exit_price': row['exit_price'] if 'exit_price' in row.keys() else 0,
                        'entry_value': row['entry_value'] if 'entry_value' in row.keys() else 0,
                        'pnl_dollars': row['pnl_dollars'] if 'pnl_dollars' in row.keys() else 0,
                        'pnl_pct': row['pnl_pct'] if 'pnl_pct' in row.keys() else 0
                    }
                    valid_trades.append(trade)
                except (KeyError, IndexError, TypeError):
                    continue  # Skip trades with missing ticker
            
            total_trades = len(valid_trades)
            
            if total_trades == 0:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_pnl': 0,
                    'best_trade': None,
                    'worst_trade': None
                }
            
            # Recalculate P&L for each trade to ensure accuracy
            # Use valid_trades which already filters out null tickers
            recalculated_trades = []
            for trade in valid_trades:
                # Safely get column values (ticker already validated)
                try:
                    ticker = trade['ticker']
                except (KeyError, IndexError, TypeError):
                    continue  # Skip if ticker is missing (shouldn't happen after filtering)
                
                try:
                    shares = float(trade['shares']) if trade['shares'] is not None else 0
                except (KeyError, IndexError, TypeError):
                    shares = 0
                
                try:
                    entry_price = float(trade['entry_price']) if trade['entry_price'] is not None else 0
                except (KeyError, IndexError, TypeError):
                    entry_price = 0
                
                try:
                    exit_price = float(trade['exit_price']) if trade['exit_price'] is not None else 0
                except (KeyError, IndexError, TypeError):
                    exit_price = 0
                
                try:
                    entry_value = float(trade['entry_value']) if trade['entry_value'] is not None else 0
                except (KeyError, IndexError, TypeError):
                    entry_value = 0
                
                # Recalculate values
                if shares > 0 and entry_price > 0 and exit_price > 0:
                    # Recalculate entry_value if wrong
                    expected_entry_value = shares * entry_price
                    if abs(entry_value - expected_entry_value) > 0.01:
                        entry_value = expected_entry_value
                    
                    # Recalculate exit_value and P&L
                    exit_value = shares * exit_price
                    pnl_dollars = exit_value - entry_value
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                    
                    recalculated_trades.append({
                        'ticker': ticker,
                        'pnl_dollars': pnl_dollars,
                        'pnl_pct': pnl_pct
                    })
                else:
                    # Use stored values if recalculation not possible
                    try:
                        pnl_dollars = float(trade['pnl_dollars']) if trade['pnl_dollars'] is not None else 0
                    except (KeyError, IndexError, TypeError):
                        pnl_dollars = 0
                    
                    try:
                        pnl_pct = float(trade['pnl_pct']) if trade['pnl_pct'] is not None else 0
                    except (KeyError, IndexError, TypeError):
                        pnl_pct = 0
                    
                    recalculated_trades.append({
                        'ticker': ticker,
                        'pnl_dollars': pnl_dollars,
                        'pnl_pct': pnl_pct
                    })
            
            # Calculate statistics from recalculated trades
            winning_trades = sum(1 for t in recalculated_trades if t['pnl_dollars'] > 0)
            losing_trades = total_trades - winning_trades
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl = sum(t['pnl_dollars'] for t in recalculated_trades)
            avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
            
            # Best trade (highest P&L) from recalculated trades
            best_trade_data = max(recalculated_trades, key=lambda x: x['pnl_dollars']) if recalculated_trades else None
            best_trade = None
            if best_trade_data:
                best_trade = {
                    'ticker': best_trade_data['ticker'],
                    'pnl_pct': best_trade_data['pnl_pct'],
                    'pnl_dollars': best_trade_data['pnl_dollars']
                }
            
            # Worst trade (lowest P&L) from recalculated trades
            worst_trade_data = min(recalculated_trades, key=lambda x: x['pnl_dollars']) if recalculated_trades else None
            worst_trade = None
            if worst_trade_data:
                worst_pnl = worst_trade_data['pnl_dollars']
                # Only show worst trade if it's different from best trade, or if it's a loss
                if best_trade and best_trade['pnl_dollars'] == worst_pnl:
                    # If worst trade is the same as best trade (only one trade), only show if it's a loss
                    if worst_pnl < 0:
                        worst_trade = {
                            'ticker': worst_trade_data['ticker'],
                            'pnl_pct': worst_trade_data['pnl_pct'],
                            'pnl_dollars': worst_trade_data['pnl_dollars']
                        }
                else:
                    # Different trade, show it
                    worst_trade = {
                        'ticker': worst_trade_data['ticker'],
                        'pnl_pct': worst_trade_data['pnl_pct'],
                        'pnl_dollars': worst_trade_data['pnl_dollars']
                    }
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'best_trade': best_trade,
                'worst_trade': worst_trade
            }
        except Exception as e:
            logger.error(f"Error getting statistics from database: {e}")
            return {}
    
    def get_current_capital_from_db(self, initial_capital: float) -> float:
        """
        Calculate current cash capital from database
        Recalculates P&L for accuracy
        
        Args:
            initial_capital: Initial starting capital
            
        Returns:
            Current cash capital (initial + completed trades P&L - active positions entry values)
        """
        try:
            cursor = self.conn.cursor()
            
            # Get all trades and recalculate P&L for accuracy
            cursor.execute("SELECT * FROM trades")
            all_trades = cursor.fetchall()
            
            # Recalculate total P&L from all completed trades
            total_pnl = 0.0
            for trade in all_trades:
                shares = float(trade['shares']) if trade['shares'] is not None else 0
                entry_price = float(trade['entry_price']) if trade['entry_price'] is not None else 0
                exit_price = float(trade['exit_price']) if trade['exit_price'] is not None else 0
                entry_value = float(trade['entry_value']) if trade['entry_value'] is not None else 0
                
                if shares > 0 and entry_price > 0 and exit_price > 0:
                    # Recalculate entry_value if wrong
                    expected_entry_value = shares * entry_price
                    if abs(entry_value - expected_entry_value) > 0.01:
                        entry_value = expected_entry_value
                    
                    # Recalculate exit_value and P&L
                    exit_value = shares * exit_price
                    pnl_dollars = exit_value - entry_value
                    total_pnl += pnl_dollars
                else:
                    # Use stored value if recalculation not possible
                    pnl_dollars = float(trade['pnl_dollars']) if trade['pnl_dollars'] is not None else 0
                    total_pnl += pnl_dollars
            
            # Get sum of entry values for active positions (money currently tied up)
            cursor.execute("SELECT SUM(entry_value) as total_entry_value FROM positions WHERE is_active = 1")
            active_entry_row = cursor.fetchone()
            active_entry_value = (active_entry_row['total_entry_value'] if active_entry_row and active_entry_row['total_entry_value'] is not None else 0)
            
            # Current capital = initial + all completed trades P&L - money tied up in active positions
            current_capital = initial_capital + total_pnl - active_entry_value
            
            return current_capital
        except Exception as e:
            logger.error(f"Error calculating current capital from database: {e}")
            return initial_capital
    
    def get_daily_profit_from_db(self, initial_capital: float, date: Optional[str] = None) -> Dict:
        """
        Calculate daily profit from database for a specific date
        
        Args:
            initial_capital: Initial starting capital
            date: Date in YYYY-MM-DD format (None for today)
            
        Returns:
            Dictionary with daily_profit, daily_start_capital, and portfolio_value
        """
        try:
            from datetime import datetime
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')
            
            cursor = self.conn.cursor()
            
            # Get all trades from this date
            cursor.execute("""
                SELECT * FROM trades 
                WHERE DATE(exit_time) = ?
                ORDER BY exit_time ASC
            """, (date,))
            daily_trades = cursor.fetchall()
            
            # Recalculate daily P&L from trades for accuracy
            daily_pnl = 0.0
            for trade in daily_trades:
                shares = float(trade['shares']) if trade['shares'] is not None else 0
                entry_price = float(trade['entry_price']) if trade['entry_price'] is not None else 0
                exit_price = float(trade['exit_price']) if trade['exit_price'] is not None else 0
                entry_value = float(trade['entry_value']) if trade['entry_value'] is not None else 0
                
                if shares > 0 and entry_price > 0 and exit_price > 0:
                    # Recalculate entry_value if wrong
                    expected_entry_value = shares * entry_price
                    if abs(entry_value - expected_entry_value) > 0.01:
                        entry_value = expected_entry_value
                    
                    # Recalculate exit_value and P&L
                    exit_value = shares * exit_price
                    pnl_dollars = exit_value - entry_value
                    daily_pnl += pnl_dollars
                else:
                    # Use stored value if recalculation not possible
                    pnl_dollars = float(trade['pnl_dollars']) if trade['pnl_dollars'] is not None else 0
                    daily_pnl += pnl_dollars
            
            # Get portfolio value at start of day (before today's trades)
            # Recalculate P&L from all trades before today for accuracy
            cursor.execute("""
                SELECT * FROM trades 
                WHERE DATE(exit_time) < ?
            """, (date,))
            prev_trades = cursor.fetchall()
            
            prev_pnl = 0.0
            for trade in prev_trades:
                shares = float(trade['shares']) if trade['shares'] is not None else 0
                entry_price = float(trade['entry_price']) if trade['entry_price'] is not None else 0
                exit_price = float(trade['exit_price']) if trade['exit_price'] is not None else 0
                entry_value = float(trade['entry_value']) if trade['entry_value'] is not None else 0
                
                if shares > 0 and entry_price > 0 and exit_price > 0:
                    # Recalculate entry_value if wrong
                    expected_entry_value = shares * entry_price
                    if abs(entry_value - expected_entry_value) > 0.01:
                        entry_value = expected_entry_value
                    
                    # Recalculate exit_value and P&L
                    exit_value = shares * exit_price
                    pnl_dollars = exit_value - entry_value
                    prev_pnl += pnl_dollars
                else:
                    # Use stored value if recalculation not possible
                    pnl_dollars = float(trade['pnl_dollars']) if trade['pnl_dollars'] is not None else 0
                    prev_pnl += pnl_dollars
            
            daily_start_capital = initial_capital + prev_pnl
            
            # Current portfolio value = daily_start_capital + daily_pnl
            portfolio_value = daily_start_capital + daily_pnl
            
            return {
                'daily_profit': daily_pnl,
                'daily_start_capital': daily_start_capital,
                'portfolio_value': portfolio_value,
                'daily_trades_count': len(daily_trades)
            }
        except Exception as e:
            logger.error(f"Error calculating daily profit from database: {e}")
            return {
                'daily_profit': 0.0,
                'daily_start_capital': initial_capital,
                'portfolio_value': initial_capital,
                'daily_trades_count': 0
            }
    
    def add_rejected_entry(self, ticker: str, price: float, reason: str, timestamp: datetime):
        """
        Add a rejected entry to the database
        
        Args:
            ticker: Stock ticker symbol
            price: Entry price that was rejected
            reason: Reason for rejection
            timestamp: Timestamp of the rejection
        """
        try:
            cursor = self.conn.cursor()
            date_str = timestamp.strftime('%Y-%m-%d')
            timestamp_str = timestamp.isoformat()
            
            cursor.execute("""
                INSERT INTO rejected_entries (ticker, price, reason, timestamp, date)
                VALUES (?, ?, ?, ?, ?)
            """, (ticker, price, reason, timestamp_str, date_str))
            
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error adding rejected entry to database: {e}")
            raise
    
    def get_rejected_entries(self, date: Optional[str] = None, limit: int = 200) -> List[Dict]:
        """
        Get rejected entries from database
        
        Args:
            date: Date string in YYYY-MM-DD format. If None, returns today's entries
            limit: Maximum number of entries to return
        
        Returns:
            List of rejected entry dictionaries
        """
        try:
            cursor = self.conn.cursor()
            
            if date:
                date_str = date
            else:
                # Get today's date
                from datetime import datetime
                date_str = datetime.now().strftime('%Y-%m-%d')
            
            cursor.execute("""
                SELECT ticker, price, reason, timestamp
                FROM rejected_entries
                WHERE date = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (date_str, limit))
            
            rows = cursor.fetchall()
            entries = []
            for row in rows:
                entries.append({
                    'ticker': row[0],
                    'price': row[1],
                    'reason': row[2],
                    'timestamp': row[3]  # ISO format string
                })
            
            return entries
        except Exception as e:
            logger.error(f"Error getting rejected entries from database: {e}")
            return []
    
    def clear_rejected_entries_for_ticker(self, ticker: str):
        """
        Clear rejected entries for a specific ticker (when position is entered)
        
        Args:
            ticker: Ticker symbol to clear entries for
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                DELETE FROM rejected_entries
                WHERE ticker = ?
            """, (ticker,))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error clearing rejected entries for ticker {ticker}: {e}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

