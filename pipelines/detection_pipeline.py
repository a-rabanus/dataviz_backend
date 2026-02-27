"""Executes human detection on image batches using YOLOv8, extracts cropped bounding boxes, and records metadata."""

import os
import sys
import cv2
import json
import time
import csv
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.db.db_utils import (
    get_db_connection,
    create_tables,
    claim_batch_for_analysis,
    mark_batch_analysis_complete,
    insert_detections_batch
)

CONF_THRESHOLD = 0.75
IMG_SIZE = 1024
CROP_DIR = os.path.join(PROJECT_ROOT, "data", "cropped_people")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "performance_log.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Increased workers for parallel I/O (reads, writes, deletes)
executor = ThreadPoolExecutor(max_workers=8)

def ensure_dirs():
    os.makedirs(CROP_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

def delete_file_async(path):
    try:
        os.remove(path)
    except OSError:
        pass

def write_crop_async(path, crop_array):
    try:
        cv2.imwrite(path, crop_array)
    except Exception as e:
        print(f"Error saving crop {path}: {e}")

def log_performance(gpu_id, batch_size, images_count, duration, total_imgs, total_time, people_found, files_deleted):
    """Appends batch processing metrics to the performance CSV log."""
    file_exists = os.path.isfile(LOG_FILE)
    current_fps = images_count / duration if duration > 0 else 0
    avg_fps = total_imgs / total_time if total_time > 0 else 0

    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'GPU_ID', 'Batch_Size', 'Images', 'Batch_Time_Sec', 'Current_FPS', 'Average_FPS', 'People_Found'])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            gpu_id, batch_size, images_count, f"{duration:.4f}",
            f"{current_fps:.2f}", f"{avg_fps:.2f}", people_found
        ])

    print(f"[{gpu_id}] Batch ({images_count} img): {duration:.2f}s | {current_fps:.1f} FPS | Avg: {avg_fps:.1f} FPS")
    print(f"[{gpu_id}] Found {people_found} people. Deleted {files_deleted} empty files.\n" + "-" * 50)

def resolve_model(batch_size):
    """Locates and returns the path to the optimal YOLO model (TensorRT engine or PyTorch fallback)."""
    engine_path = os.path.join(MODELS_DIR, f"yolov8n_batch{batch_size}.engine")
    if os.path.exists(engine_path):
        return engine_path
    
    pt_path = os.path.join(MODELS_DIR, "yolov8n.pt")
    if os.path.exists(pt_path):
        print(f"WARN: Engine file missing. Falling back to {pt_path}")
        return pt_path
        
    sys.exit(f"ERROR: No model found in {MODELS_DIR}. Ensure yolov8n.pt or .engine exists.")

def run_consumer(gpu_id, batch_size):
    """Pulls unanalyzed images from the database, runs YOLO detection, crops instances, and updates database state."""
    device = f'cuda:{gpu_id}'
    model_file = resolve_model(batch_size)

    conn = get_db_connection()
    if not conn:
        sys.exit("ERROR: Database connection failed.")

    ensure_dirs()
    create_tables(conn)

    print(f"[{gpu_id}] Loading Model: {model_file} on {device}")
    try:
        model = YOLO(model_file, task='detect')
    except Exception as e:
        sys.exit(f"ERROR: Model load failed: {e}")

    session_start = time.time()
    total_images_processed = 0

    while True:
        t0 = time.time()
        images = claim_batch_for_analysis(conn, batch_size=batch_size)
        
        if not images:
            time.sleep(2)
            continue

        valid_images = []
        file_paths = []
        invalid_ids = []

        for img in images:
            if os.path.exists(img['file_path_image']):
                valid_images.append(img)
                file_paths.append(img['file_path_image'])
            else:
                invalid_ids.append(img['id_image'])

        if invalid_ids:
            mark_batch_analysis_complete(conn, invalid_ids)

        if not valid_images:
            continue

        try:
            # Inference: ultralytics handles batching internally if a list of paths is provided.
            # For optimal speed with TensorRT, ensure the model was exported with dynamic=False 
            # and the batch parameter matches your execution batch_size.
            results = model(file_paths, device=device, imgsz=IMG_SIZE, verbose=False)
            
            detections_to_insert = []
            files_queued_for_deletion = 0
            processed_ids = []

            for i, result in enumerate(results):
                db_img = valid_images[i]
                orig_img_path = file_paths[i]
                found_person = False
                
                # Fetch original image array returned by the predictor
                img_array = result.orig_img
                
                processed_ids.append(db_img['id_image'])

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
                            crop_filename = f"{db_img['id_image']}_p{len(detections_to_insert)}_{int(conf*100)}.jpg"
                            crop_full_path = os.path.join(CROP_DIR, crop_filename)
                            
                            # Offload I/O write to thread pool
                            executor.submit(write_crop_async, crop_full_path, crop)
                            
                            detections_to_insert.append({
                                'id_city': db_img['id_city'],
                                'id_image': db_img['id_image'],
                                'captured_at': db_img['captured_at'],
                                'location': db_img['location'],
                                'confidence': int(conf * 100),
                                'bbox_person': json.dumps(bbox_norm),
                                'crop_path': crop_full_path
                            })

                if not found_person:
                    executor.submit(delete_file_async, orig_img_path)
                    files_queued_for_deletion += 1

            if detections_to_insert:
                insert_detections_batch(conn, detections_to_insert)
            mark_batch_analysis_complete(conn, processed_ids)

            t_end = time.time()
            batch_duration = t_end - t0
            total_images_processed += len(valid_images)
            total_session_time = t_end - session_start

            log_performance(
                gpu_id, batch_size, len(valid_images), batch_duration,
                total_images_processed, total_session_time,
                len(detections_to_insert), files_queued_for_deletion
            )

        except Exception as e:
            print(f"[{gpu_id}] ERROR during inference: {e}")
            mark_batch_analysis_complete(conn, [img['id_image'] for img in valid_images])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    run_consumer(args.gpu, args.batch_size)