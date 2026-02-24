"""
Pipeline Stage 1: Mass Image Downloader (Double-Parallel)
Arch: [Tile Queue] -> 6 Scanners -> [Image Queue] -> 20 Downloaders -> [DB]
"""

import os
import sys
import asyncio
import aiohttp
import json
import time
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from datetime import datetime, timezone

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.db.db_utils import (
    get_db_connection, create_tables,
    get_and_lock_city_for_download, mark_city_download_complete,
    get_existing_image_ids, insert_image_records_batch
)
from pipelines.image_tools import tile_bbox

load_dotenv()
MAPILLARY_TOKEN = os.getenv("MAPILLARY_ACCESS_TOKEN")
IMAGE_ROOT_DIR = "mapillary_images"

# --- CONFIGURATION ---
TILE_SIZE = 0.02       # ~5.5km tiles
API_LIMIT = 2000        # Max images per API call

# PARALLELISM SETTINGS
SCANNER_WORKERS = 30   # Number of concurrent API fetchers
DOWNLOAD_WORKERS = 50   # Number of concurrent Image downloaders
DB_BATCH_SIZE = 50      # Flush to DB every 50 images
# ---------------------

async def fetch_images_in_tile(session, tile_bbox):
    """Fetches image IDs and metadata for a geo-tile."""
    bbox_str = f"{tile_bbox[0]:.6f},{tile_bbox[1]:.6f},{tile_bbox[2]:.6f},{tile_bbox[3]:.6f}"
    url = "https://graph.mapillary.com/images"
    params = {
        'access_token': MAPILLARY_TOKEN,
        'fields': 'id,captured_at,geometry,thumb_2048_url',
        'bbox': bbox_str,
        'limit': API_LIMIT 
    }
    
    try:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                results = data.get('data', [])
                if len(results) >= API_LIMIT:
                    # Just a warning, we process what we got
                    pass
                return results
            else:
                return []
    except Exception as e:
        print(f"API Error: {e}")
        return []

async def download_image_bytes(session, url):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.read()
    except Exception:
        pass
    return None

def write_file(path, content):
    with open(path, 'wb') as f:
        f.write(content)

# --- WORKER: SCANNER ---
async def worker_scanner(tile_queue, image_queue, session, seen_ids, pbar):
    """Consumes Tiles -> Produces Image Metadata"""
    while True:
        try:
            tile = tile_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        images = await fetch_images_in_tile(session, tile)
        
        count_new = 0
        for img in images:
            img_id = int(img['id'])
            # Atomic check for duplicates (seen_ids is not thread-safe but IS async-safe)
            if img_id not in seen_ids:
                seen_ids.add(img_id)
                await image_queue.put(img)
                count_new += 1
        
        pbar.update(1)
        pbar.set_description(f"Scanning (Queued {count_new} new)")
        tile_queue.task_done()

# --- WORKER: DOWNLOADER ---
async def worker_downloader(image_queue, city_dir, city_id, db_conn, pbar):
    """Consumes Image Metadata -> Downloads -> Inserts to DB"""
    db_buffer = []
    
    async with aiohttp.ClientSession() as session:
        while True:
            item = await image_queue.get()
            
            # Sentinel for Shutdown
            if item is None:
                if db_buffer:
                    insert_image_records_batch(db_conn, db_buffer)
                image_queue.task_done()
                break
            
            img_data = item
            img_id = img_data['id']
            url = img_data.get('thumb_2048_url')
            
            if url:
                content = await download_image_bytes(session, url)
                if content:
                    filename = f"{img_id}.jpg"
                    file_path = os.path.join(city_dir, filename)
                    
                    # Async file write (runs in thread pool)
                    await asyncio.to_thread(write_file, file_path, content)
                    
                    # Parse Time
                    ts_str = img_data.get('captured_at')
                    captured_at = None
                    if ts_str:
                        try:
                            if isinstance(ts_str, (int, float)):
                                captured_at = datetime.fromtimestamp(ts_str/1000, timezone.utc)
                            else:
                                captured_at = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                        except: pass

                    # UPDATED: Keys must match named placeholders in db_utils (:id_image, :id_city, :file_path_image)
                    record = {
                        'id_image': int(img_id),
                        'id_city': city_id,
                        'captured_at': captured_at,
                        'location': json.dumps(img_data.get('geometry')),
                        'file_path_image': file_path
                    }
                    db_buffer.append(record)
                    pbar.update(1)

            # Flush Buffer
            if len(db_buffer) >= DB_BATCH_SIZE:
                insert_image_records_batch(db_conn, db_buffer)
                db_buffer = []

            image_queue.task_done()

async def process_city(city, db_conn):
    # UPDATED: Use new column keys
    city_id = city['id_city']
    city_name = city['name']
    print(f"\n>>> Starting PARALLEL DOWNLOAD for: {city_name}")

    city_dir = os.path.join(IMAGE_ROOT_DIR, city_name.replace(" ", "_").replace(",", ""))
    os.makedirs(city_dir, exist_ok=True)

    # 1. Prepare State
    existing_ids = get_existing_image_ids(db_conn, city_id)
    seen_ids = set(existing_ids)
    print(f"Known images: {len(existing_ids)}")

    # 2. Fill Tile Queue
    # UPDATED: Use new bbox column
    tiles = list(tile_bbox(city['bbox_cities'], tile_size_deg=TILE_SIZE))
    tile_queue = asyncio.Queue()
    for t in tiles:
        tile_queue.put_nowait(t)
    
    print(f"Tiles to scan: {len(tiles)}")

    # 3. Image Queue (Maxsize prevents RAM explosion if scanning is too fast)
    image_queue = asyncio.Queue(maxsize=10000)

    # 4. Start Workers
    scan_pbar = tqdm(total=len(tiles), desc="Scanning Tiles", unit="tile", position=0)
    dl_pbar = tqdm(desc="Downloading", unit="img", position=1)
    
    # Start Scanners
    scanner_tasks = []
    async with aiohttp.ClientSession() as scan_session:
        for _ in range(SCANNER_WORKERS):
            t = asyncio.create_task(worker_scanner(tile_queue, image_queue, scan_session, seen_ids, scan_pbar))
            scanner_tasks.append(t)

        # Start Downloaders
        downloader_tasks = []
        for _ in range(DOWNLOAD_WORKERS):
            t = asyncio.create_task(worker_downloader(image_queue, city_dir, city_id, db_conn, dl_pbar))
            downloader_tasks.append(t)

        # 5. Wait for Scanning to Finish
        await asyncio.gather(*scanner_tasks)
        scan_pbar.close()
        print("\nScanning complete. Waiting for downloads...")

        # 6. Stop Downloaders
        for _ in range(DOWNLOAD_WORKERS):
            await image_queue.put(None)
        
        await asyncio.gather(*downloader_tasks)
        dl_pbar.close()

    mark_city_download_complete(db_conn, city_id)
    print(f"Finished city: {city_name}")

async def main():
    if not MAPILLARY_TOKEN:
        print("ERROR: MAPILLARY_ACCESS_TOKEN not set.")
        return

    conn = get_db_connection()
    create_tables(conn)

    print(f"Parallel Pipeline Started. Scanners: {SCANNER_WORKERS}, Downloaders: {DOWNLOAD_WORKERS}")
    
    while True:
        city = get_and_lock_city_for_download(conn)
        if not city:
            print("No 'pending' cities. Sleeping 10s...")
            time.sleep(10)
            continue
            
        try:
            await process_city(city, conn)
        except Exception as e:
            print(f"CRITICAL FAILURE on city {city['name']}: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main())