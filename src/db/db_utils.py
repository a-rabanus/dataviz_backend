import sqlite3
import os
from sqlite3 import Error
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DB_DIR = os.path.join(PROJECT_ROOT, 'data')
DB_PATH = os.path.join(DB_DIR, 'pipeline.db')

def get_db_connection(timeout=30.0):
    os.makedirs(DB_DIR, exist_ok=True)
    try:
        conn = sqlite3.connect(DB_PATH, timeout=timeout)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA synchronous = NORMAL;") 
        return conn
    except Error as e:
        print(f"ERROR: Could not connect to the SQLite database: {e}")
        return None

def create_tables(conn):
    schema = """
    BEGIN TRANSACTION;
    CREATE TABLE IF NOT EXISTS "cities" (
        "id_city"	INTEGER,
        "name"	VARCHAR(255) NOT NULL UNIQUE,
        "search_term"	VARCHAR(255),
        "bbox_cities"	TEXT,
        "population"	INTEGER,
        "download_status"	TEXT DEFAULT 'pending',
        "downloaded_at"	DATETIME,
        "analysis_status"	TEXT DEFAULT 'pending',
        "analyzed_at"	DATETIME,
        "created_at"	DATETIME DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY("id_city" AUTOINCREMENT)
    );
    CREATE TABLE IF NOT EXISTS "clothing_item_detected" (
        "id_item"	INTEGER,
        "id_detection"	INTEGER,
        "category"	VARCHAR(50),
        "confidence"	REAL,
        "color_h"	REAL,
        "color_s"	REAL,
        "color_v"	REAL,
        "texture_score"	REAL,
        "area_ratio"	REAL,
        "bbox_item"	TEXT,
        "created_at"	DATETIME DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY("id_item" AUTOINCREMENT),
        FOREIGN KEY("id_detection") REFERENCES "person_detected"("id_person") ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS "images_detected" (
        "id_image"	BIGING,
        "id_city"	INTEGER,
        "captured_at"	DATETIME,
        "location"	TEXT,
        "file_path_image"	TEXT,
        "processing_status"	TEXT,
        "created_at"	DATETIME,
        PRIMARY KEY("id_image"),
        FOREIGN KEY("id_city") REFERENCES "cities"("id_city")
    );
    CREATE TABLE IF NOT EXISTS "person_detected" (
        "id_person"	INTEGER,
        "id_city"	INTEGER,
        "id_image"	BIGINT,
        "captured_at"	DATETIME,
        "location"	TEXT,
        "confidence"	INTEGER,
        "bbox_person"	TEXT,
        "crop_path"	TEXT,
        "created_at"	DATETIME DEFAULT CURRENT_TIMESTAMP,
        "clothing_status"	TEXT DEFAULT 'pending',
        PRIMARY KEY("id_person" AUTOINCREMENT),
        FOREIGN KEY("id_city") REFERENCES "cities"("id_city") ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS "idx_clothing_status" ON "person_detected" (
        "clothing_status"
    );
    COMMIT;
    """
    try:
        conn.executescript(schema)
    except Error as e:
        print(f"ERROR: Failed to create tables: {e}")

def get_and_lock_city_for_download(conn):
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            query = """
                SELECT id_city, name, bbox_cities, population 
                FROM cities 
                WHERE download_status IN ('pending', 'processing') 
                ORDER BY population DESC 
                LIMIT 1
            """
            cursor.execute(query)
            row = cursor.fetchone()
            if row:
                city = dict(row)
                if city['bbox_cities']:
                    city['bbox_cities'] = json.loads(city['bbox_cities'])
                cursor.execute(
                    "UPDATE cities SET download_status = 'processing' WHERE id_city = ?", 
                    (city['id_city'],)
                )
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
    except Exception as e:
        print(f"Error marking city complete: {e}")

def get_existing_image_ids(conn, city_id):
    cursor = conn.cursor()
    cursor.execute("SELECT id_image FROM images_detected WHERE id_city = ?", (city_id,))
    rows = cursor.fetchall()
    return {row['id_image'] for row in rows}

def insert_image_records_batch(conn, records):
    if not records:
        return
    try:
        with conn:
            conn.executemany("""
                INSERT OR REPLACE INTO images_detected 
                (id_image, id_city, captured_at, location, file_path_image, processing_status, created_at)
                VALUES (:id_image, :id_city, :captured_at, :location, :file_path_image, 'pending', CURRENT_TIMESTAMP)
            """, records)
    except Exception as e:
        print(f"Error bulk inserting images: {e}")

def claim_batch_for_analysis(conn, batch_size=32):
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            query = f"""
                SELECT id_image, id_city, captured_at, location, file_path_image 
                FROM images_detected 
                WHERE processing_status = 'pending' 
                LIMIT {batch_size}
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            if not rows:
                return []
            
            images = [dict(row) for row in rows]
            ids_to_lock = [img['id_image'] for img in images]
            placeholders = ','.join(['?'] * len(ids_to_lock))
            
            update_query = f"""
                UPDATE images_detected 
                SET processing_status = 'processing' 
                WHERE id_image IN ({placeholders})
            """
            cursor.execute(update_query, ids_to_lock)
            return images
    except Exception as e:
        print(f"Error claiming batch: {e}")
        return []

def mark_batch_analysis_complete(conn, image_ids):
    if not image_ids:
        return
    try:
        with conn:
            placeholders = ','.join(['?'] * len(image_ids))
            sql = f"UPDATE images_detected SET processing_status = 'completed' WHERE id_image IN ({placeholders})"
            conn.execute(sql, tuple(image_ids))
    except Exception as e:
        print(f"Error marking batch complete: {e}")

def insert_detections_batch(conn, detections):
    if not detections:
        return
    try:
        with conn:
            conn.executemany("""
                INSERT INTO person_detected (
                    id_city, id_image, captured_at, location, confidence, bbox_person, crop_path
                )
                VALUES (
                    :id_city, :id_image, :captured_at, :location, :confidence, :bbox_person, :crop_path
                )
            """, detections)
    except Exception as e:
        print(f"Error inserting detections: {e}")