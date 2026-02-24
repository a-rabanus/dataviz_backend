import numpy as np
import math

def tile_bbox(main_bbox, tile_size_deg=0.02): # <--- CHANGED DEFAULT TO 0.02
    """
    Splits a large bounding box into smaller square tiles.
    
    Args:
        main_bbox: Dictionary with keys 'west', 'south', 'east', 'north'
        tile_size_deg: The side length of each square tile in degrees.
                       0.0025 ~ 250m
                       0.01   ~ 1km
                       0.02   ~ 2km (Recommended for download efficiency)

    Yields:
        A list [min_lon, min_lat, max_lon, max_lat] for each tile.
    """
    
    # Handle Dict input (west, south, etc.)
    if isinstance(main_bbox, dict):
        min_lon = main_bbox['west']
        min_lat = main_bbox['south']
        max_lon = main_bbox['east']
        max_lat = main_bbox['north']
    else:
        # Fallback if list is passed
        min_lon, min_lat, max_lon, max_lat = main_bbox
    
    # Calculate number of steps
    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat
    
    n_lon = math.ceil(lon_span / tile_size_deg)
    n_lat = math.ceil(lat_span / tile_size_deg)
    
    # Generate tiles
    for i in range(n_lon):
        for j in range(n_lat):
            x0 = min_lon + (i * tile_size_deg)
            y0 = min_lat + (j * tile_size_deg)
            x1 = min(x0 + tile_size_deg, max_lon)
            y1 = min(y0 + tile_size_deg, max_lat)
            
            yield [x0, y0, x1, y1]