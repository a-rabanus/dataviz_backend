"""
Pipeline Stage 2: People Detection (Dynamic Batch Size & Full Logging)
Usage: 
    python pipelines/run_detection_pipeline.py --gpu 0 --batch_size 16
    python pipelines/run_detection_pipeline.py --gpu 1 --batch_size 32
"""

import os
import sys
import cv2
import json
import time
import csv
import argparse
import torch
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.db.db_utils import (
    get_db_connection,
    create_tables,
    claim_batch_for_analysis,
    mark_batch_analysis_complete,
    insert_detections_batch
)

# --- GLOBAL CONFIG ---
CONF_THRESHOLD = 0.75   
IMG_SIZE = 1024         
CROP_DIR = "cropped_people"
LOG_FILE = "performance_log.csv"

# Background thread pool for file deletions
executor = ThreadPoolExecutor(max_workers=4)

def ensure_dirs():
    os.makedirs(CROP_DIR, exist_ok=True)

def delete_file_async(path):
    try:
        os.remove(path)
    except OSError:
        pass

def log_performance(gpu_id, batch_size, images_count, duration, total_imgs, total_time, people_found, files_deleted):
    """Writes stats to CSV and prints detailed terminal summary."""
    file_exists = os.path.isfile(LOG_FILE)
    
    current_fps = images_count / duration if duration > 0 else 0
    avg_fps = total_imgs / total_time if total_time > 0 else 0
    
    # 1. Write to CSV
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Add Header if new file
        if not file_exists:
            writer.writerow(['Timestamp', 'GPU_ID', 'Batch_Size', 'Images', 'Batch_Time_Sec', 'Current_FPS', 'Average_FPS', 'People_Found'])
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            gpu_id,
            batch_size,
            images_count,
            f"{duration:.4f}",
            f"{current_fps:.2f}",
            f"{avg_fps:.2f}",
            people_found  # <--- Added to CSV
        ])

    # 2. Print Detailed Terminal Output
    print(f"[{gpu_id}] Batch ({images_count} img) finished in {duration:.2f}s ({current_fps:.1f} FPS | Avg: {avg_fps:.1f})")
    print(f"    - Found {people_found} people. Deleting {files_deleted} empty files.")
    print("-" * 50)

def run_consumer(gpu_id, batch_size):
    device = f'cuda:{gpu_id}'
    
    # Dynamic Model Filename
    model_file = f"models/yolov8n_batch{batch_size}.engine"

    print(f"[{gpu_id}] Initializing Worker on {device}...")
    print(f"[{gpu_id}] Config: Batch={batch_size}, Engine={model_file}")

    conn = get_db_connection()
    if not conn:
        print("Failed to connect to DB.")
        return

    ensure_dirs()
    create_tables(conn) 

    if not os.path.exists(model_file):
        print(f"❌ Error: Engine file '{model_file}' not found!")
        print(f"   Please run: yolo export ... batch={batch_size} ...")
        return

    print(f"Loading Model: {model_file}...")
    try:
        model = YOLO(model_file, task='detect')
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print(f"Worker {gpu_id} ready. Polling for batches...")

    # --- BENCHMARKING VARS ---
    session_start = time.time()
    total_images_processed = 0
    
    while True:
        t0 = time.time() 

        # 1. Claim Batch
        images = claim_batch_for_analysis(conn, batch_size=batch_size)
        
        if not images:
            time.sleep(2)
            continue
            
        # 2. Filter Valid Files
        valid_images = []
        file_paths = []
        for img in images:
            if os.path.exists(img['file_path']):
                valid_images.append(img)
                file_paths.append(img['file_path'])
        
        if not valid_images:
            mark_batch_analysis_complete(conn, [img['id'] for img in images])
            continue

        try:
            # 3. Batch Inference
            results = model(
                file_paths, 
                device=device,
                imgsz=IMG_SIZE, 
                rect=False, 
                verbose=False, 
                stream=False
            )
            
            detections_to_insert = []
            files_queued_for_deletion = 0
            
            # 4. Process Results
            for i, result in enumerate(results):
                db_img = valid_images[i]
                orig_img_path = file_paths[i]
                found_person = False
                img_array = result.orig_img 

                for box in result.boxes:
                    if int(box.cls) == 0 and float(box.conf) >= CONF_THRESHOLD:
                        found_person = True
                        conf = float(box.conf)
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        h, w, _ = img_array.shape
                        bbox_norm = [x1/w, y1/h, x2/w, y2/h]
                        
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        crop = img_array[y1:y2, x1:x2]
                        
                        if crop.size > 0:
                            crop_filename = f"{db_img['image_id']}_p{len(detections_to_insert)}_{int(conf*100)}.jpg"
                            crop_full_path = os.path.join(CROP_DIR, crop_filename)
                            cv2.imwrite(crop_full_path, crop)
                            
                            detections_to_insert.append({
                                'city_id': db_img['city_id'],
                                'original_image_id': db_img['image_id'],
                                'captured_at': db_img['captured_at'],
                                'location': db_img['location'],
                                'confidence': int(conf * 100),
                                'bounding_box_detection': json.dumps(bbox_norm),
                                'crop_path': crop_full_path
                            })
                
                if not found_person:
                    executor.submit(delete_file_async, orig_img_path)
                    files_queued_for_deletion += 1
            
            # 5. Insert & Finalize
            if detections_to_insert:
                insert_detections_batch(conn, detections_to_insert)
            
            mark_batch_analysis_complete(conn, [img['id'] for img in images])
            
            # --- LOGGING ---
            t_end = time.time()
            batch_duration = t_end - t0
            
            total_images_processed += len(valid_images)
            total_session_time = t_end - session_start
            
            log_performance(
                gpu_id, 
                batch_size, 
                len(valid_images), 
                batch_duration, 
                total_images_processed, 
                total_session_time,
                len(detections_to_insert), # <--- Passed People Found
                files_queued_for_deletion  # <--- Passed Files Deleted
            )

        except Exception as e:
            print(f"[{gpu_id}] Error: {e}")
            mark_batch_analysis_complete(conn, [img['id'] for img in images])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (0 or 1)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (must match your engine file)')
    args = parser.parse_args()
    
    run_consumer(args.gpu, args.batch_size)