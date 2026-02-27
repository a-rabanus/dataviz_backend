"""Queries Nominatim API to acquire bounding boxes for target cities and initializes database records."""

import os
import sys
import time
import json
import requests
import sqlite3
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.db.db_utils import get_db_connection, create_tables

USER_AGENT = os.getenv("USER_AGENT")
CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'worldcities.csv')

def get_city_bbox(city_name, headers):
    """Requests and extracts geographic bounding box coordinates for a specified city name."""
    url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(city_name)}&format=json&limit=5"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data:
            return None

        best_match = next((res for res in data if res.get('class') == 'place' and res.get('type') == 'city' and res.get('boundingbox')), None)
        if not best_match:
            best_match = data[0] if data and data[0].get('boundingbox') else None

        if not best_match:
            return None

        bbox_osm = best_match['boundingbox']
        formatted_bbox = {
            "west": float(bbox_osm[2]),
            "south": float(bbox_osm[0]),
            "east": float(bbox_osm[3]),
            "north": float(bbox_osm[1])
        }
        return {
            "name": best_match['display_name'],
            "search_term": city_name,
            "bbox_cities": formatted_bbox
        }
    except requests.RequestException:
        return None

def extract_cities():
    """Filters a static CSV to identify German cities within the target population range."""
    df = pd.read_csv(CSV_PATH)
    filtered = df[
        (df['country'].str.contains('germany', case=False, na=False)) &
        (df['population'] >= 150000) &
        (df['population'] <= 5000000)
    ]
    city_list = []
    for _, row in filtered.iterrows():
        term = f"{row['city_ascii']}, {row['country']}"
        city_list.append((term, int(row['population'])))
    return sorted(list(set(city_list)), key=lambda x: x[0])

def main():
    db_conn = get_db_connection()
    if not db_conn:
        sys.exit(1)
    
    create_tables(db_conn)
    cities = extract_cities()
    headers = {'User-Agent': USER_AGENT}

    cursor = db_conn.cursor()
    
    tqdm.write(f"Targeting {len(cities)} cities.")
    
    for term, pop in tqdm(cities, desc="Processing Nominatim API", unit="city"):
        cursor.execute("SELECT id_city FROM cities WHERE search_term = ?", (term,))
        if cursor.fetchone():
            tqdm.write(f"SKIP: {term} (exists in DB)")
            continue

        tqdm.write(f"FETCH: {term} (API request)")
        city_data = get_city_bbox(term, headers)
        
        if city_data:
            try:
                sql_insert = """
                INSERT INTO cities (name, search_term, bbox_cities, population)
                VALUES (?, ?, ?, ?)
                """
                cursor.execute(sql_insert, (
                    city_data['name'], 
                    term, 
                    json.dumps(city_data['bbox_cities']), 
                    pop
                ))
                db_conn.commit()
                tqdm.write(f"INSERT: {term} -> {city_data['name']}")
            except sqlite3.IntegrityError:
                tqdm.write(f"COLLISION IGNORED: {city_data['name']}")
        else:
            tqdm.write(f"FAIL: No bounding box for {term}")
            
        time.sleep(1)

    db_conn.close()

if __name__ == "__main__":
    main()