import os
import sys
import math
import json
import time
import shutil
import asyncio
import aiohttp
from datetime import datetime, timezone
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.db.db_utils import (
    get_db_connection, 
    get_and_lock_city_for_download, 
    mark_city_download_complete,
    insert_image_records_batch
)

load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
MAPILLARY_TOKEN = os.getenv("MAPILLARY_TOKEN")
IMAGE_ROOT_DIR = os.path.join(PROJECT_ROOT, "data", "raw_images")

TILE_SIZE = 0.02
API_LIMIT = 2000
SCANNER_WORKERS = 50
DOWNLOAD_WORKERS = 100
DB_BATCH_SIZE = 50
MIN_TILE_SIZE = 0.0005

def tile_bbox(main_bbox, tile_size_deg=0.02):
    if isinstance(main_bbox, dict):
        min_lon, min_lat = main_bbox['west'], main_bbox['south']
        max_lon, max_lat = main_bbox['east'], main_bbox['north']
    else:
        min_lon, min_lat, max_lon, max_lat = main_bbox
    
    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat
    
    n_lon = math.ceil(lon_span / tile_size_deg)
    n_lat = math.ceil(lat_span / tile_size_deg)
    
    for i in range(n_lon):
        for j in range(n_lat):
            x0 = min_lon + (i * tile_size_deg)
            y0 = min_lat + (j * tile_size_deg)
            x1 = min(x0 + tile_size_deg, max_lon)
            y1 = min(y0 + tile_size_deg, max_lat)
            yield [x0, y0, x1, y1]

def split_tile(tile_bbox):
    min_lon, min_lat, max_lon, max_lat = tile_bbox
    mid_lon = (min_lon + max_lon) / 2.0
    mid_lat = (min_lat + max_lat) / 2.0
    return [
        [min_lon, min_lat, mid_lon, mid_lat],
        [mid_lon, min_lat, max_lon, mid_lat],
        [min_lon, mid_lat, mid_lon, max_lat],
        [mid_lon, mid_lat, max_lon, max_lat]
    ]

async def fetch_images_in_tile(session, tile_bbox):
    bbox_str = f"{tile_bbox[0]:.6f},{tile_bbox[1]:.6f},{tile_bbox[2]:.6f},{tile_bbox[3]:.6f}"
    url = "https://graph.mapillary.com/images"
    headers = {'Authorization': f'OAuth {MAPILLARY_TOKEN}'}
    params = {
        'fields': 'id,captured_at,geometry,thumb_2048_url',
        'bbox': bbox_str,
        'limit': API_LIMIT
    }
    try:
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                results = data.get('data', [])
                if results:
                    tqdm.write(f"SCANNER: Found {len(results)} images in tile.")
                return True, results
            elif response.status in (500, 502, 503, 504):
                err_text = await response.text()
                tqdm.write(f"SCANNER API HTTP {response.status} (Density/Timeout): {err_text}")
                return False, []
            else:
                err_text = await response.text()
                tqdm.write(f"SCANNER API HTTP {response.status}: {err_text}")
                return True, []
    except asyncio.TimeoutError:
        tqdm.write("SCANNER API ERROR: Timeout")
        return False, []
    except Exception as e:
        tqdm.write(f"SCANNER API ERROR: {type(e).__name__}: {str(e)}")
        return False, []

async def download_image_bytes(session, url, img_id):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.read()
            else:
                tqdm.write(f"DOWNLOADER HTTP {response.status} for image {img_id}")
    except Exception as e:
        tqdm.write(f"DOWNLOADER NET ERROR for image {img_id}: {type(e).__name__}")
    return None

def write_file(path, content):
    try:
        with open(path, 'wb') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"IO ERROR writing {path}: {e}")
        return False

async def worker_scanner(tile_queue, image_queue, session, seen_ids, pbar):
    while True:
        try:
            tile = tile_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        
        success, images = await fetch_images_in_tile(session, tile)
        
        if not success:
            lon_span = tile[2] - tile[0]
            lat_span = tile[3] - tile[1]
            if lon_span > MIN_TILE_SIZE and lat_span > MIN_TILE_SIZE:
                sub_tiles = split_tile(tile)
                for st in sub_tiles:
                    tile_queue.put_nowait(st)
                pbar.total += 3
                pbar.refresh()
                tqdm.write(f"SCANNER: Subdivided dense tile into 4 sub-tiles (new span: {lon_span/2:.5f}).")
            else:
                tqdm.write("SCANNER: Tile resolution limit reached. Dropping tile.")
            
            pbar.update(1)
            tile_queue.task_done()
            continue

        queued_count = 0
        for img in images:
            img_id = int(img['id'])
            if img_id not in seen_ids:
                seen_ids.add(img_id)
                await image_queue.put(img)
                queued_count += 1
        
        if queued_count > 0:
            tqdm.write(f"SCANNER: Queued {queued_count} new images. Queue size approx: {image_queue.qsize()}")
            
        pbar.update(1)
        tile_queue.task_done()

async def worker_downloader(image_queue, city_dir, city_id, db_conn, pbar):
    db_buffer = []
    async with aiohttp.ClientSession() as session:
        while True:
            item = await image_queue.get()
            if item is None:
                if db_buffer:
                    insert_image_records_batch(db_conn, db_buffer)
                image_queue.task_done()
                break
            
            img_id = item['id']
            url = item.get('thumb_2048_url')
            
            if not url:
                tqdm.write(f"DOWNLOADER: Missing URL for image {img_id}")
                image_queue.task_done()
                continue
                
            content = await download_image_bytes(session, url, img_id)
            if content:
                file_path = os.path.join(city_dir, f"{img_id}.jpg")
                success = await asyncio.to_thread(write_file, file_path, content)
                
                if success:
                    captured_at = None
                    ts_str = item.get('captured_at')
                    if ts_str:
                        try:
                            if isinstance(ts_str, (int, float)):
                                dt = datetime.fromtimestamp(ts_str/1000, timezone.utc)
                            else:
                                dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                            captured_at = dt.strftime('%Y-%m-%d-%H')
                        except Exception:
                            pass

                    record = {
                        'id_image': int(img_id),
                        'id_city': city_id,
                        'captured_at': captured_at,
                        'location': json.dumps(item.get('geometry')),
                        'file_path_image': file_path
                    }
                    db_buffer.append(record)
                    pbar.update(1)
            
            if len(db_buffer) >= DB_BATCH_SIZE:
                insert_image_records_batch(db_conn, db_buffer)
                tqdm.write(f"DOWNLOADER: Flushed {DB_BATCH_SIZE} records to DB.")
                db_buffer = []
            
            image_queue.task_done()

async def process_city(city, db_conn):
    city_id = city['id_city']
    city_name = city['name']
    print(f"\nProcessing: {city_name}")

    city_dir = os.path.join(IMAGE_ROOT_DIR, city_name.replace(" ", "_").replace(",", ""))
    
    if os.path.exists(city_dir):
        print(f"Purging existing directory: {city_dir}")
        shutil.rmtree(city_dir)
    os.makedirs(city_dir)

    print(f"Purging existing database records for city_id: {city_id}")
    with db_conn:
        db_conn.execute("DELETE FROM images_detected WHERE id_city = ?", (city_id,))

    seen_ids = set()
    
    tiles = list(tile_bbox(city['bbox_cities'], tile_size_deg=TILE_SIZE))
    tile_queue = asyncio.Queue()
    for t in tiles:
        tile_queue.put_nowait(t)

    image_queue = asyncio.Queue(maxsize=10000)

    scan_pbar = tqdm(total=len(tiles), desc="Scanning Tiles", unit="tile", position=0)
    dl_pbar = tqdm(desc="Downloading", unit="img", position=1)

    scanner_tasks = []
    async with aiohttp.ClientSession() as scan_session:
        for _ in range(SCANNER_WORKERS):
            t = asyncio.create_task(worker_scanner(tile_queue, image_queue, scan_session, seen_ids, scan_pbar))
            scanner_tasks.append(t)

        downloader_tasks = []
        for _ in range(DOWNLOAD_WORKERS):
            t = asyncio.create_task(worker_downloader(image_queue, city_dir, city_id, db_conn, dl_pbar))
            downloader_tasks.append(t)

        await asyncio.gather(*scanner_tasks)
        scan_pbar.close()
        tqdm.write("SCANNER PHASE COMPLETE. Waiting for downloaders to clear queue.")

        for _ in range(DOWNLOAD_WORKERS):
            await image_queue.put(None)
        await asyncio.gather(*downloader_tasks)
        dl_pbar.close()

    mark_city_download_complete(db_conn, city_id)

async def main():
    if not MAPILLARY_TOKEN:
        sys.exit(1)

    conn = get_db_connection()
    if not conn:
        sys.exit(1)

    while True:
        city = get_and_lock_city_for_download(conn)
        if not city:
            break
        try:
            await process_city(city, conn)
        except Exception as e:
            print(f"Error on {city['name']}: {e}")
            time.sleep(5)

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())