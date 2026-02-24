import sqlite3
import os
import csv
import sys
from pathlib import Path

# --- CONFIGURATION ---
# The database file path provided
DB_PATH = "data/pipeline.db" 

# The folder where all cropped images are joined
# Based on your working dir: /media/arabanus/500Gb/dataViz
IMAGES_DIR = "cropped_people"

# Output report file
OUTPUT_FILE = "reconciliation_report.csv"

def get_db_filenames(db_path):
    """
    Connects to DB and returns a set of filenames from the 'crop_path' column.
    Handles extraction of filename from full windows/linux paths.
    """
    if not os.path.exists(db_path):
        print(f"❌ Error: Database not found at {db_path}")
        sys.exit(1)

    print(f"--- Loading filenames from Database ({db_path}) ---")
    filenames = set()
    
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # Try the schema-provided name first, fallback to old name if needed
        table_name = "mapillary_detections"
        try:
            cur.execute(f"SELECT crop_path FROM {table_name}")
        except sqlite3.OperationalError:
            print(f"⚠️  Table '{table_name}' not found. Trying 'detections'...")
            table_name = "detections"
            cur.execute(f"SELECT crop_path FROM {table_name}")

        rows = cur.fetchall()
        
        for row in rows:
            path_str = row[0]
            if path_str:
                # Robust filename extraction for both Windows ('\') and Linux ('/') paths
                # os.path.basename depends on the OS running the script.
                # To be safe for mixed data, we split by both separators.
                name = path_str.replace('\\', '/').split('/')[-1]
                filenames.add(name)
        
        conn.close()
        print(f"✅ Loaded {len(filenames)} records from DB.")
        return filenames

    except Exception as e:
        print(f"❌ Database Error: {e}")
        sys.exit(1)

def get_disk_filenames(folder_path):
    """
    Scans the directory and returns a set of actual filenames on disk.
    """
    if not os.path.exists(folder_path):
        print(f"❌ Error: Image directory not found at {folder_path}")
        sys.exit(1)

    print(f"--- Scanning Disk ({folder_path}) ---")
    filenames = set()
    
    try:
        # os.scandir is much faster than os.listdir for large directories
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if entry.is_file():
                    filenames.add(entry.name)
    except Exception as e:
        print(f"❌ Disk Scan Error: {e}")
        sys.exit(1)
        
    print(f"✅ Found {len(filenames)} files on disk.")
    return filenames

def generate_report(missing, orphans):
    """
    Writes the results to a CSV file.
    """
    print(f"--- Generating Report ({OUTPUT_FILE}) ---")
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'Status', 'Action_Required'])
        
        # 1. Write Missing (Critical)
        for name in missing:
            writer.writerow([name, 'MISSING_FROM_DISK', 'Re-download or Remove from DB'])
            
        # 2. Write Orphans (Cleanup)
        for name in orphans:
            writer.writerow([name, 'ORPHAN_ON_DISK', 'Safe to Delete'])

    print(f"📝 Report saved.")

def main():
    # 1. Load Data (In-Memory Sets)
    db_set = get_db_filenames(DB_PATH)
    disk_set = get_disk_filenames(IMAGES_DIR)
    
    # 2. Perform Set Math (The Magic)
    # Files in DB but NOT on Disk
    missing_images = db_set - disk_set
    
    # Files on Disk but NOT in DB
    orphaned_images = disk_set - db_set
    
    # 3. Output Results
    print("\n" + "="*40)
    print(f"SUMMARY REPORT")
    print("="*40)
    print(f"Total DB Records:   {len(db_set):,}")
    print(f"Total Disk Files:   {len(disk_set):,}")
    print("-" * 40)
    print(f"❌ MISSING IMAGES:  {len(missing_images):,} (In DB, not on Disk)")
    print(f"⚠️  ORPHAN IMAGES:   {len(orphaned_images):,} (On Disk, not in DB)")
    print("="*40 + "\n")
    
    if len(missing_images) > 0 or len(orphaned_images) > 0:
        generate_report(missing_images, orphaned_images)
    else:
        print("✅ Perfect Sync! No issues found.")

if __name__ == "__main__":
    main()