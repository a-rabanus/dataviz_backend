import os
import sqlite3
import json
import time
from sqlite3 import Error

# Define the path for the SQLite database file
DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "pipeline.db")

def get_db_connection(timeout=30.0):
    """Establishes and returns a sqlite3 connection object."""
    os.makedirs(DB_DIR, exist_ok=True)
    try:
        # Increase timeout to handle locking during heavy concurrency
        conn = sqlite3.connect(DB_PATH, timeout=timeout)
        conn.row_factory = sqlite3.Row
        
        # Enable Write-Ahead Logging for concurrency (Readers don't block Writers)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA synchronous = NORMAL;") 
        
        return conn
    except Error as e:
        print(f"ERROR: Could not connect to the SQLite database: {e}")
        return None

def create_tables(conn):
    """Creates tables based on the new schema."""
    
    # 1. Cities Table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS cities (
        id_city INTEGER PRIMARY KEY AUTOINCREMENT,
        name VARCHAR(255) NOT NULL UNIQUE,
        search_term VARCHAR(255),
        bbox_cities TEXT,
        population INTEGER,
        download_status TEXT DEFAULT 'pending',
        downloaded_at DATETIME,
        analysis_status TEXT DEFAULT 'pending',
        analyzed_at DATETIME,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 2. Images Detected Table (Renamed from mapillary_images)
    # id_image is the PK (BIGINT).
    conn.execute("""
    CREATE TABLE IF NOT EXISTS images_detected (
        id_image BIGINT PRIMARY KEY,
        id_city INTEGER,
        captured_at DATETIME,
        location TEXT,
        file_path_image TEXT,
        processing_status TEXT DEFAULT 'pending',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(id_city) REFERENCES cities(id_city) ON DELETE CASCADE
    );
    """)

    # 3. Person Detected Table (Renamed from mapillary_detections)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS person_detected (
        id_person INTEGER PRIMARY KEY AUTOINCREMENT,
        id_city INTEGER,
        image_id BIGINT,
        captured_at DATETIME,
        location TEXT,
        confidence INTEGER, 
        bbox_person TEXT,
        crop_path TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        clothing_status TEXT DEFAULT 'pending',
        FOREIGN KEY(id_city) REFERENCES cities(id_city) ON DELETE CASCADE
    );
    """)
    
    # Index for person processing status
    conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_clothing_status ON person_detected(clothing_status);
    """)
    
    # 4. Clothing Items Table (Renamed from clothing_measurements)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS clothing_item_detected (
        id_item INTEGER PRIMARY KEY AUTOINCREMENT,
        id_detection INTEGER,
        category VARCHAR(50),
        confidence REAL,
        color_h REAL,
        color_s REAL,
        color_v REAL,
        texture_score REAL,
        area_ratio REAL,
        bbox_item TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(id_detection) REFERENCES person_detected(id_person) ON DELETE CASCADE
    );
    """)

    print("Database tables verified (New Schema).")

# --- PRODUCER (DOWNLOADER) FUNCTIONS ---

def get_and_lock_city_for_download(conn):
    """Finds a city pending download and locks it."""
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            
            # Select using new column names
            query = "SELECT id_city, name, bbox_cities, population FROM cities WHERE download_status = 'pending' ORDER BY population DESC LIMIT 1"
            cursor.execute(query)
            row = cursor.fetchone()
            
            if row:
                city = dict(row)
                if city['bbox_cities']:
                    city['bbox_cities'] = json.loads(city['bbox_cities'])
                
                cursor.execute("UPDATE cities SET download_status = 'processing' WHERE id_city = ?", (city['id_city'],))
                return city
            return None
    except Exception as e:
        print(f"Error locking city: {e}")
        return None

def mark_city_download_complete(conn, city_id):
    try:
        with conn:
            conn.execute(
                "UPDATE cities SET download_status = 'done', downloaded_at = CURRENT_TIMESTAMP WHERE id_city = ?", 
                (city_id,)
            )
    except Error as e:
        print(f"Error marking city complete: {e}")

def get_existing_image_ids(conn, city_id):
    cursor = conn.cursor()
    # Updated table: images_detected, column: id_image, id_city
    cursor.execute("SELECT id_image FROM images_detected WHERE id_city = ?", (city_id,))
    rows = cursor.fetchall()
    return {row['id_image'] for row in rows}

def insert_image_records_batch(conn, records):
    """
    Bulk insert images.
    records: list of dicts {'id_image', 'id_city', 'captured_at', 'location', 'file_path_image'}
    """
    if not records:
        return
    try:
        with conn:
            conn.executemany("""
            INSERT INTO images_detected (id_image, id_city, captured_at, location, file_path_image, processing_status)
            VALUES (:id_image, :id_city, :captured_at, :location, :file_path_image, 'pending')
            ON CONFLICT(id_image) DO NOTHING
            """, records)
    except Error as e:
        print(f"Error bulk inserting images: {e}")

# --- CONSUMER (DETECTOR) FUNCTIONS ---

def claim_batch_for_analysis(conn, batch_size=32, worker_id="gpu_worker"):
    """
    Atomically claims a batch of 'pending' images for processing.
    """
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            
            # 1. Select IDs to lock (Using id_image)
            query = f"""
                SELECT id_image FROM images_detected 
                WHERE processing_status = 'pending' 
                LIMIT {batch_size}
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            
            if not rows:
                return []
            
            ids_to_lock = [row['id_image'] for row in rows]
            
            # 2. Mark them as processing
            placeholders = ','.join(['?'] * len(ids_to_lock))
            update_query = f"""
                UPDATE images_detected 
                SET processing_status = 'processing' 
                WHERE id_image IN ({placeholders})
            """
            cursor.execute(update_query, ids_to_lock)
            
            # 3. Retrieve the full data
            select_query = f"""
                SELECT * FROM images_detected 
                WHERE id_image IN ({placeholders})
            """
            cursor.execute(select_query, ids_to_lock)
            locked_images = cursor.fetchall()
            
            return [dict(row) for row in locked_images]
            
    except Error as e:
        print(f"Error claiming batch: {e}")
        return []

def mark_batch_analysis_complete(conn, image_ids):
    """Marks a list of image database IDs as completed."""
    if not image_ids:
        return
    try:
        with conn:
            placeholders = ','.join(['?'] * len(image_ids))
            sql = f"UPDATE images_detected SET processing_status = 'completed' WHERE id_image IN ({placeholders})"
            conn.execute(sql, tuple(image_ids))
    except Error as e:
        print(f"Error marking batch complete: {e}")

def insert_detections_batch(conn, detections):
    """
    Bulk insert detections.
    detections: list of dicts with new keys
    """
    if not detections:
        return
    try:
        with conn:
            # Table: person_detected
            conn.executemany("""
            INSERT INTO person_detected (
                id_city, image_id, captured_at, location, confidence, bbox_person, crop_path
            )
            VALUES (
                :id_city, :image_id, :captured_at, :location, :confidence, :bbox_person, :crop_path
            )
            """, detections)
    except Error as e:
        print(f"Error inserting detections: {e}")

def get_pending_crops(conn, batch_size=1000):
    """Fetches crops that haven't been analyzed for clothing yet."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id_person, crop_path FROM person_detected WHERE crop_path IS NOT NULL LIMIT ?", (batch_size,))
        return [dict(row) for row in cursor.fetchall()]
    except Error as e:
        print(f"Error fetching crops: {e}")
        return []

def insert_clothing_batch(conn, items):
    """Bulk insert clothing vectors."""
    insert_clothing_measurements(conn, items)

def claim_detections_for_analysis(conn, batch_size=32):
    """
    Atomically claims a batch of 'pending' people detections.
    """
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            
            query = f"""
                SELECT id_person FROM person_detected 
                WHERE clothing_status = 'pending' 
                AND crop_path IS NOT NULL
                LIMIT {batch_size}
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            
            if not rows:
                return []
            
            ids_to_lock = [row['id_person'] for row in rows]
            
            # 2. Mark as processing
            placeholders = ','.join(['?'] * len(ids_to_lock))
            update_query = f"""
                UPDATE person_detected 
                SET clothing_status = 'processing' 
                WHERE id_person IN ({placeholders})
            """
            cursor.execute(update_query, ids_to_lock)
            
            # 3. Return data
            select_query = f"""
                SELECT id_person, crop_path FROM person_detected 
                WHERE id_person IN ({placeholders})
            """
            cursor.execute(select_query, ids_to_lock)
            return [dict(row) for row in cursor.fetchall()]
            
    except Exception as e:
        print(f"Error claiming detections: {e}")
        return []

def mark_clothing_analysis_complete(conn, detection_ids):
    """Marks detection rows as 'completed'."""
    if not detection_ids: return
    try:
        with conn:
            placeholders = ','.join(['?'] * len(detection_ids))
            sql = f"UPDATE person_detected SET clothing_status = 'completed' WHERE id_person IN ({placeholders})"
            conn.execute(sql, tuple(detection_ids))
    except Exception as e:
        print(f"Error marking clothing analysis complete: {e}")

def insert_clothing_measurements(conn, measurements):
    """Bulk insert the extracted vectors."""
    if not measurements: return
    try:
        with conn:
            # Table: clothing_item_detected
            conn.executemany("""
            INSERT INTO clothing_item_detected (
                id_detection, category, confidence, 
                color_h, color_s, color_v, texture_score, 
                area_ratio, bbox_item
            ) VALUES (
                :id_detection, :category, :confidence, 
                :color_h, :color_s, :color_v, :texture_score, 
                :area_ratio, :bbox_item
            )
            """, measurements)
    except Exception as e:
        print(f"Error inserting measurements: {e}")