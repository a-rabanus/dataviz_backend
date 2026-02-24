import os
import sys
import argparse
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- CONFIGURATION ---
# Reduced workers to prevent Antivirus/Filesystem deadlock
MAX_WORKERS = 8  
CHUNK_SIZE = 10000 

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from db.db_utils_old import get_db_connection
except ImportError:
    print("Error: Could not import 'src'.")
    sys.exit(1)

def check_files_batch(batch):
    """
    Worker: Checks a batch of files.
    """
    results = []
    total_size = 0
    
    for row_id, file_path in batch:
        # Sanity check for valid strings
        if not file_path:
            continue
            
        if os.path.exists(file_path):
            try:
                size = os.path.getsize(file_path)
                results.append((row_id, file_path))
                total_size += size
            except OSError:
                continue
    return results, total_size

def delete_files_batch(batch):
    deleted_ids = []
    for row_id, file_path in batch:
        try:
            os.remove(file_path)
            deleted_ids.append(row_id)
        except OSError:
            continue
    return deleted_ids

def chunk_data(data, size):
    for i in range(0, len(data), size):
        yield data[i:i + size]

def run_cleanup(live_mode=False):
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database.")
        return

    cur = conn.cursor()
    print("--- 1. Database Analysis ---")
    
    query = """
        SELECT m.id, m.file_path 
        FROM mapillary_images m
        LEFT JOIN mapillary_detections d ON m.id = d.original_image_id
        WHERE d.original_image_id IS NULL
    """
    
    try:
        print("Querying DB (This may take a moment)...")
        cur.execute(query)
        rows = cur.fetchall()
    except sqlite3.OperationalError as e:
        print(f"Database Error: {e}")
        return

    if not rows:
        print("Clean. No actions needed.")
        return

    print(f"Found {len(rows)} potential garbage entries.")

    # --- SANITY CHECK ---
    print("\n--- 2. Disk Sanity Check ---")
    test_file = rows[0][1]
    print(f"Attempting to read 1st file: {test_file}")
    if os.path.exists(test_file):
        print("✅ Success: File system is accessible.")
    else:
        print(f"⚠️  Note: Test file not found (This might be normal if it was already deleted).")
    
    # Create chunks
    batches = list(chunk_data(rows, CHUNK_SIZE))
    print(f"\n--- 3. Verifying Disk Existence (Batched: {len(batches)} chunks | {MAX_WORKERS} threads) ---")

    garbage_items = []
    total_garbage_size = 0

    # Using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(check_files_batch, batch) for batch in batches]
        
        # We use a loop with timeout to detect hangs
        pbar = tqdm(total=len(futures), desc="Verifying Batches", unit="chunk")
        for future in as_completed(futures):
            try:
                batch_results, batch_size = future.result(timeout=300) # 5 min timeout per batch
                if batch_results:
                    garbage_items.extend(batch_results)
                    total_garbage_size += batch_size
                pbar.update(1)
            except Exception as e:
                print(f"\n❌ Worker Thread Error: {e}")
        pbar.close()

    size_mb = total_garbage_size / (1024 * 1024)
    size_gb = size_mb / 1024

    print(f"\n--- 4. Summary ---")
    print(f"Garbage Found:      {len(garbage_items)} items")
    print(f"Space to Reclaim:   {size_mb:.2f} MB ({size_gb:.2f} GB)")

    if len(garbage_items) == 0:
        return

    # Execution
    if live_mode:
        print("\nWARNING: LIVE MODE ACTIVE.")
        confirm = input(f"Permanently DELETE {len(garbage_items)} files AND database rows? (yes/no): ")
        
        if confirm.lower() == "yes":
            print(f"\n--- 5. Deleting Files ---")
            ids_to_drop = []
            
            del_batches = list(chunk_data(garbage_items, CHUNK_SIZE))
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(delete_files_batch, batch) for batch in del_batches]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Deleting Batches", unit="chunk"):
                    ids_to_drop.extend(future.result())
            
            if ids_to_drop:
                print(f"\n--- 6. Cleaning Database ({len(ids_to_drop)} rows) ---")
                try:
                    cur = conn.cursor()
                    db_batch_size = 900
                    total_db_chunks = (len(ids_to_drop) // db_batch_size) + 1
                    
                    for i in tqdm(range(0, len(ids_to_drop), db_batch_size), desc="Updating DB", total=total_db_chunks):
                        batch = ids_to_drop[i:i + db_batch_size]
                        cur.executemany("DELETE FROM mapillary_images WHERE id = ?", [(x,) for x in batch])
                    conn.commit()
                except sqlite3.Error as e:
                    print(f"Database deletion failed: {e}")
                    conn.rollback()

            print(f"\nOperation complete. Removed {len(ids_to_drop)} files/rows.")

            # VACUUM
            print("\n--- 7. Database Optimization ---")
            vac_confirm = input("Run VACUUM? (yes/no): ")
            if vac_confirm.lower() == "yes":
                print("Running VACUUM...")
                conn.isolation_level = None 
                conn.execute("VACUUM")
                print("VACUUM complete.")
        else:
            print("\nAborted.")
    else:
        print("\nDRY RUN. Use '--live' to execute.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true', help='Execute deletion')
    args = parser.parse_args()
    
    run_cleanup(live_mode=args.live)