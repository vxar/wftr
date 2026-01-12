"""
Script to unlock SQLite database
This script will:
1. Check for lock files
2. Close any open connections
3. Clean up WAL files if needed
4. Verify database is accessible
"""
import os
import sqlite3
import sys
from pathlib import Path

def unlock_database(db_path="trading_data.db"):
    """Unlock SQLite database by cleaning up lock files and closing connections"""
    
    print(f"Attempting to unlock database: {db_path}")
    print("=" * 60)
    
    # Check if database file exists
    if not os.path.exists(db_path):
        print(f"ERROR: Database file '{db_path}' not found!")
        return False
    
    # Check for WAL files (Write-Ahead Logging files)
    wal_file = f"{db_path}-wal"
    shm_file = f"{db_path}-shm"
    
    print(f"\n1. Checking for lock files...")
    if os.path.exists(wal_file):
        print(f"   Found WAL file: {wal_file}")
        try:
            # Try to checkpoint WAL file (merge it back into main database)
            conn = sqlite3.connect(db_path, timeout=5.0)
            conn.execute("PRAGMA wal_checkpoint(FULL)")
            conn.close()
            print("   [OK] Checkpointed WAL file successfully")
        except Exception as e:
            print(f"   [WARNING] Could not checkpoint WAL: {e}")
    
    if os.path.exists(shm_file):
        print(f"   Found SHM file: {shm_file}")
    
    # Try to connect and verify database is accessible
    print(f"\n2. Testing database connection...")
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        cursor = conn.cursor()
        
        # Test query
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        table_count = cursor.fetchone()[0]
        print(f"   [OK] Database is accessible")
        print(f"   [OK] Found {table_count} table(s)")
        
        # Check for any locks
        cursor.execute("PRAGMA journal_mode")
        journal_mode = cursor.fetchone()[0]
        print(f"   [OK] Journal mode: {journal_mode}")
        
        # If in WAL mode, try to checkpoint
        if journal_mode.upper() == "WAL":
            print(f"\n3. Checkpointing WAL file...")
            try:
                conn.execute("PRAGMA wal_checkpoint(FULL)")
                print("   [OK] WAL checkpoint completed")
            except Exception as e:
                print(f"   [WARNING] WAL checkpoint warning: {e}")
        
        conn.close()
        print(f"\n[SUCCESS] Database unlocked and accessible!")
        return True
        
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e).lower():
            print(f"   [ERROR] Database is still locked: {e}")
            print(f"\n   Possible solutions:")
            print(f"   1. Stop any running Python processes that might be using the database")
            print(f"   2. Close any database viewers (DB Browser, etc.)")
            print(f"   3. Restart your computer if processes are stuck")
            print(f"   4. Manually delete lock files (risky - may cause data loss):")
            print(f"      - {wal_file}")
            print(f"      - {shm_file}")
            return False
        else:
            print(f"   [ERROR] Database error: {e}")
            return False
    except Exception as e:
        print(f"   [ERROR] Unexpected error: {e}")
        return False

def force_unlock_database(db_path="trading_data.db"):
    """Force unlock by deleting WAL and SHM files (USE WITH CAUTION)"""
    
    print(f"\n[WARNING] FORCE UNLOCK - This may cause data loss if database is in use!")
    response = input("Are you sure you want to force unlock? (yes/no): ")
    
    if response.lower() != "yes":
        print("Cancelled.")
        return False
    
    wal_file = f"{db_path}-wal"
    shm_file = f"{db_path}-shm"
    
    deleted = []
    try:
        if os.path.exists(wal_file):
            os.remove(wal_file)
            deleted.append(wal_file)
            print(f"   [OK] Deleted {wal_file}")
        
        if os.path.exists(shm_file):
            os.remove(shm_file)
            deleted.append(shm_file)
            print(f"   [OK] Deleted {shm_file}")
        
        if deleted:
            print(f"\n[SUCCESS] Force unlock completed. Deleted {len(deleted)} lock file(s)")
            print("  [WARNING] If database was in use, you may have lost recent changes!")
            return True
        else:
            print("\n   No lock files found to delete")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Error during force unlock: {e}")
        return False

if __name__ == "__main__":
    db_path = "trading_data.db"
    
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    print("=" * 60)
    print("SQLite Database Unlock Utility")
    print("=" * 60)
    
    # Try normal unlock first
    success = unlock_database(db_path)
    
    if not success:
        print("\n" + "=" * 60)
        print("Normal unlock failed. Would you like to try force unlock?")
        print("=" * 60)
        force_unlock_database(db_path)
    else:
        print("\n" + "=" * 60)
        print("Database is now unlocked and ready to use!")
        print("=" * 60)
