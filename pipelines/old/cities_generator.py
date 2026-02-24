"""
This script populates the 'cities' table in the database.
It dynamically generates the city list by filtering a CSV
of world cities by population, then uses the OpenStreetMap
Nominatim API to find the bounding box for each.
"""

import requests
import json
import time
import csv
import os
import sys

from dotenv import load_dotenv

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# ----------------

from db.db_utils_old import get_db_connection, create_tables

load_dotenv()

USER_AGENT = 'CityBoxFinder/1.0 (anton.rabanus@study.hs-duesseldorf.de)'

def get_city_bbox(city_name, headers):
    """Fetches the bounding box for a single city from Nominatim."""
    url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(city_name)}&format=json&limit=5"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            print(f"WARNING: No results found for '{city_name}'")
            return None

        best_match = None
        for result in data:
            if (result.get('class') == 'place' and 
                result.get('type') == 'city' and 
                result.get('boundingbox')):
                best_match = result
                break 

        if not best_match:
            print(f"  -> WARNING: No 'city' type match for '{city_name}'. Falling back to first result.")
            if data and data[0].get('boundingbox'):
                best_match = data[0]
            else:
                print(f"WARNING: No usable results with a bounding box for '{city_name}'. Skipping.")
                return None
            
        bbox_osm = best_match['boundingbox'] # [south, north, west, east]
        
        formatted_bbox = {
            "west": float(bbox_osm[2]),
            "south": float(bbox_osm[0]),
            "east": float(bbox_osm[3]),
            "north": float(bbox_osm[1])
        }
        
        city_data = {
            "name": best_match['display_name'],
            "search_term": city_name,
            "bbox": formatted_bbox
        }
        
        print(f"SUCCESS: Found '{best_match['display_name']}'")
        return city_data
            
    except requests.RequestException as e:
        print(f"EXCEPTION: {e} for '{city_name}'")
        return None

def check_db_for_search_term(conn, search_term):
    """Fast check: Returns True if we have already processed this exact search term."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM cities WHERE search_term = ?", (search_term,))
        return cursor.fetchone() is not None
    except Exception:
        return False

def check_or_create_city(conn, city_name, city_bbox, search_term, population):
    """
    Checks if a city exists. Updates missing data (search_term, population).
    Creates new entry if not exists.
    """
    try:
        with conn:
            cursor = conn.cursor()
            
            # 1. Check if city exists by UNIQUE NAME (Nominatim name)
            cursor.execute("SELECT id, download_status, search_term, population FROM cities WHERE name = ?", (city_name,))
            result = cursor.fetchone()
            
            if result:
                # City exists
                city_id = result['id']
                status = result['download_status']
                
                # SELF-HEALING: Update missing columns
                updates = []
                params = []
                
                if not result['search_term']:
                    updates.append("search_term = ?")
                    params.append(search_term)
                
                if not result['population']:
                    updates.append("population = ?")
                    params.append(population)
                    
                if updates:
                    sql_update = f"UPDATE cities SET {', '.join(updates)} WHERE id = ?"
                    params.append(city_id)
                    cursor.execute(sql_update, tuple(params))
                    print(f"  -> Updated existing city data (population/search_term).")
                
                return city_id, status, False
            else:
                # 2. City doesn't exist, insert it
                print(f"  -> Adding new city '{city_name}' (Pop: {population}) to database.")
                sql_insert = """
                INSERT INTO cities (name, bbox, search_term, population, download_status, analysis_status) 
                VALUES (?, ?, ?, ?, 'pending', 'pending')
                """
                cursor.execute(sql_insert, (city_name, json.dumps(city_bbox), search_term, population))
                
                city_id = cursor.lastrowid
                return city_id, 'pending', True
                
    except Exception as e:
        print(f"Error checking/creating city '{city_name}': {e}")
        return None, None, False

def load_cities_from_csv(csv_path, min_population):
    """Loads cities from CSV. Returns list of tuples (search_term, population)."""
    city_list = []
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(csv_path):
        alt_path = os.path.join(script_dir, os.path.basename(csv_path))
        if os.path.exists(alt_path):
            csv_path = alt_path
    
    print(f"Reading from {csv_path}...")
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    pop_str = row.get('population', '0')
                    if not pop_str: continue 
                    
                    population = float(pop_str)
                    
                    if population >= min_population:
                        city = row['city_ascii'] 
                        country = row['country']
                        term = f"{city}, {country}"
                        # Append tuple (term, int_population)
                        city_list.append((term, int(population)))
                except (ValueError, TypeError):
                    continue 
    except FileNotFoundError:
        print(f"ERROR: Could not find '{csv_path}'.")
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
        
    return city_list

def main():
    if USER_AGENT == 'MyCityBoxFinder/1.0 (your-email@example.com)':
        print("ERROR: Please edit the 'USER_AGENT' variable in this script")
        return

    # --- Connect to Database ---
    print("Initializing database connection...")
    db_conn = get_db_connection()
    if db_conn is None:
        return
        
    create_tables(db_conn)

    # --- Load Cities ---
    MIN_POPULATION = 500000 
    INPUT_CSV = "config/worldcities.csv"
    city_data_list = load_cities_from_csv(INPUT_CSV, MIN_POPULATION)
    
    if city_data_list is None:
        db_conn.close()
        return
        
    # Sort and Deduplicate
    city_data_list = sorted(list(set(city_data_list)), key=lambda x: x[0])
    print(f"Found {len(city_data_list)} unique cities matching criteria.")

    # --- Process ---
    headers = {'User-Agent': USER_AGENT}
    
    print(f"Starting geocoding...")
    
    for i, (city_search_term, population) in enumerate(city_data_list):
        progress = f"({i + 1}/{len(city_data_list)})"
        
        # --- Check DB first (Fast Skip) ---
        if check_db_for_search_term(db_conn, city_search_term):
            # We assume if it's in DB, the self-healing logic in previous runs fixed the population
            # If you want to force check population update even for skipped ones,
            # you'd need to remove this `continue` and rely on check_or_create_city logic.
            # But for speed, we skip if search_term is known.
            print(f"{progress} Skipping '{city_search_term}' (Already in DB)")
            continue

        print(f"\n{progress} Processing: '{city_search_term}' (Pop: {population})")
        
        city_data = get_city_bbox(city_search_term, headers)
        
        if city_data:
            city_id, status, was_created = check_or_create_city(
                db_conn, 
                city_data['name'], 
                city_data['bbox'],
                city_search_term,
                population
            )
            
            if city_id and status == 'done':
                print("  -> City exists and is DONE.")
            elif city_id and was_created:
                print("  -> New city added (Status: pending).")
            elif city_id:
                print(f"  -> City exists (Status: {status}).")
        
        time.sleep(1)

    db_conn.close()
    print("Process complete.")

if __name__ == "__main__":
    main()