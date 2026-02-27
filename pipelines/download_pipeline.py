"""Asynchronously fetches and downloads Mapillary images using bounding box tiling and Nominatim POIs."""

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
    insert_image_records_batch,
    get_existing_image_ids,
    get_all_existing_image_ids
)

load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
MAPILLARY_TOKEN = os.getenv("MAPILLARY_TOKEN")
IMAGE_ROOT_DIR = os.path.join(PROJECT_ROOT, "data", "raw_images")
USER_AGENT = os.getenv("USER_AGENT")

TILE_SIZE = 0.02
API_LIMIT = 2000
SCANNER_WORKERS = 70
DOWNLOAD_WORKERS = 140
DB_BATCH_SIZE = 50
MIN_TILE_SIZE = 0.0005
POI_MARGIN = 0.005

POIS = [
    "Altstadt", "Hauptbahnhof", "Marktplatz", "Fußgängerzone", 
    "Universität", "Schule", "Einkaufszentrum", "Park", 
    "Rathaus", "Zentraler Omnibusbahnhof", "Theater", 
    "Stadion", "Promenade", "Museum", "Klinikum"
]

def tile_bbox(main_bbox, tile_size_deg=0.02):
    """Divides a master bounding box into smaller coordinate grids."""
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
    """Subdivides a single bounding box into four quadrants."""
    min_lon, min_lat, max_lon, max_lat = tile_bbox
    mid_lon = (min_lon + max_lon) / 2.0
    mid_lat = (min_lat + max_lat) / 2.0
    return [
        [min_lon, min_lat, mid_lon, mid_lat],
        [mid_lon, min_lat, max_lon, mid_lat],
        [min_lon, mid_lat, mid_lon, max_lat],
        [mid_lon, mid_lat, max_lon, max_lat]
    ]

async def fetch_poi_bboxes(session, city_name):
    """Queries Nominatim API for bounding boxes of specified Points of Interest within a city."""
    bboxes = []
    headers = {'User-Agent': USER_AGENT}
    for poi in POIS:
        query = f"{poi}, {city_name}"
        url = f"https://nominatim.openstreetmap.org/search?q={query}&format=json&limit=1"
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and data[0].get('boundingbox'):
                        b = data[0]['boundingbox']
                        expanded_bbox = [
                            float(b[2]) - POI_MARGIN,
                            float(b[0]) - POI_MARGIN,
                            float(b[3]) + POI_MARGIN,
                            float(b[1]) + POI_MARGIN 
                        ]
                        bboxes.append(expanded_bbox)
        except Exception as e:
            tqdm.write(f"NOMINATIM ERROR für '{query}': {e}")
        await asyncio.sleep(1.1)
    return bboxes

async def fetch_images_in_tile(session, tile_bbox):
    """Requests Mapillary image metadata within a specific coordinate grid."""
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
                    tqdm.write(f"SCANNER: {len(results)} Bilder in Kachel gefunden.")
                return True, results
            elif response.status in (500, 502, 503, 504):
                return False, []
            else:
                return True, []
    except asyncio.TimeoutError:
        return False, []
    except Exception:
        return False, []

async def download_image_bytes(session, url, img_id):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.read()
    except Exception:
        pass
    return None

def write_file(path, content):
    try:
        with open(path, 'wb') as f:
            f.write(content)
        return True
    except Exception:
        return False

async def worker_scanner(tile_queue, image_queue, session, seen_ids, pbar):
    """Consumes coordinate tiles, queries the Mapillary API, and queues distinct image metadata for download."""
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
            pbar.update(1)
            tile_queue.task_done()
            continue

        for img in images:
            img_id = int(img['id'])
            if img_id not in seen_ids:
                seen_ids.add(img_id)
                await image_queue.put(img)
            
        pbar.update(1)
        tile_queue.task_done()

async def worker_downloader(image_queue, city_dir, city_id, db_conn, pbar):
    """Consumes image metadata, downloads thumbnail binaries, and writes records to the database."""
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
                db_buffer = []
            
            image_queue.task_done()

async def process_city(city, db_conn):
    city_id = city['id_city']
    city_name = city['name']
    print(f"\nVerarbeitung: {city_name}")

    city_dir = os.path.join(IMAGE_ROOT_DIR, city_name.replace(" ", "_").replace(",", ""))
    os.makedirs(city_dir, exist_ok=True)

    db_ids = get_existing_image_ids(db_conn, city_id)
    disk_ids = {int(f.split('.')[0]) for f in os.listdir(city_dir) if f.endswith('.jpg')}

    valid_local_ids = db_ids.intersection(disk_ids)

    orphaned_db = db_ids - disk_ids
    if orphaned_db:
        print(f"Lösche {len(orphaned_db)} verwaiste Datenbankeinträge.")
        with db_conn:
            db_conn.executemany(
                "DELETE FROM images_detected WHERE id_city = ? AND id_image = ?", 
                [(city_id, oid) for oid in orphaned_db]
            )

    orphaned_disk = disk_ids - db_ids
    if orphaned_disk:
        print(f"Lösche {len(orphaned_disk)} verwaiste Dateien.")
        for oid in orphaned_disk:
            try:
                os.remove(os.path.join(city_dir, f"{oid}.jpg"))
            except OSError:
                pass

    global_ids = get_all_existing_image_ids(db_conn)
    seen_ids = valid_local_ids.union(global_ids)

    print("Frage Nominatim POIs ab...")
    async with aiohttp.ClientSession() as session:
        poi_bboxes = await fetch_poi_bboxes(session, city_name)

    if not poi_bboxes:
        print("Keine POIs gefunden. Nutze Zentrums-Fallback.")
        main_bbox = city['bbox_cities']
        center_lon = (main_bbox['west'] + main_bbox['east']) / 2
        center_lat = (main_bbox['south'] + main_bbox['north']) / 2
        poi_bboxes = [[center_lon - POI_MARGIN, center_lat - POI_MARGIN, center_lon + POI_MARGIN, center_lat + POI_MARGIN]]

    unique_tiles = set()
    for bbox in poi_bboxes:
        for t in tile_bbox(bbox, tile_size_deg=TILE_SIZE):
            unique_tiles.add(tuple(t))
    
    tiles = [list(t) for t in unique_tiles]
    print(f"{len(tiles)} einzigartige Kacheln über {len(poi_bboxes)} POI-Zonen generiert.")

    tile_queue = asyncio.Queue()
    for t in tiles:
        tile_queue.put_nowait(t)

    image_queue = asyncio.Queue(maxsize=10000)

    scan_pbar = tqdm(total=len(tiles), desc="Kacheln scannen", unit="tile", position=0)
    dl_pbar = tqdm(desc="Download", unit="img", position=1)

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
            print(f"Kritischer Fehler bei {city['name']}: {e}")
            time.sleep(5)

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())