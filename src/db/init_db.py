import sys
import os

# Add project root to path so imports work correctly
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Import the new consolidated function
from src.db.db_utils import get_db_connection, create_tables

def main():
    print("Initializing database schema...")
    
    # 1. Connect
    conn = get_db_connection()
    if conn is None:
        print("Failed to get DB connection. Exiting.")
        sys.exit(1)

    # 2. Create Tables
    # This creates 'cities', 'mapillary_images', and 'mapillary_detections'
    # using the new Option B schema (timestamps and separate statuses).
    create_tables(conn)

    # 3. Close
    conn.close()
    print("Database schema is fully initialized and ready.")

if __name__ == "__main__":
    main()