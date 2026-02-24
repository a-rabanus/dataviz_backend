"""
Pipeline Stage 3: Clothing Analysis & Vector Extraction
Usage: python pipelines/run_clothing_analysis.py --batch_size 64
"""

import os
import sys
import cv2
import json
import time
import argparse
import torch
import numpy as np
from tqdm import tqdm

# Detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.db.db_utils import (
    get_db_connection,
    create_tables,
    claim_detections_for_analysis,
    mark_clothing_analysis_complete,
    insert_clothing_measurements
)

# --- CONFIG ---
FASHION_MODEL = "models/model_final.pth"
CONFIDENCE_THRESH = 0.5

# ADD THIS: Point to where the folder is strictly on this Linux machine
# Since you are running from /dataViz, and the folder is likely /dataViz/cropped_people:
LOCAL_CROP_DIR = "cropped_people"

# Anatomy logic from your live_fusion_viewer.py
# (Truncated for brevity, but using the same structure)
CLOTHING_ANATOMY = {
    1: {"name": "Short Sleeve Top", "req": ["SHOULDER"], "forbid": ["WRIST"], "group": "TOP"},
    2: {"name": "Long Sleeve Top", "req": ["SHOULDER", "WRIST"], "forbid": [], "group": "TOP"},
    3: {"name": "Short Outwear", "req": ["SHOULDER"], "forbid": ["WRIST"], "group": "OUTWEAR"},
    4: {"name": "Long Outwear", "req": ["SHOULDER", "WRIST"], "forbid": [], "group": "OUTWEAR"},
    5: {"name": "Vest", "req": ["SHOULDER"], "forbid": ["ELBOW", "WRIST"], "group": "TOP"},
    6: {"name": "Sling", "req": [], "forbid": ["SHOULDER", "ELBOW"], "group": "TOP"},
    7: {"name": "Shorts", "req": ["HIP"], "forbid": ["ANKLE"], "group": "BOTTOM"},
    8: {"name": "Trousers", "req": ["ANKLE"], "forbid": [], "group": "BOTTOM"},
    9: {"name": "Skirt", "req": ["HIP"], "forbid": [], "group": "BOTTOM"},
    10: {"name": "Short Dress", "req": ["SHOULDER"], "forbid": ["ANKLE"], "group": "DRESS"},
    11: {"name": "Long Dress", "req": ["SHOULDER", "ANKLE"], "forbid": [], "group": "DRESS"},
    12: {"name": "Vest Dress", "req": ["SHOULDER"], "forbid": ["ELBOW"], "group": "DRESS"},
    13: {"name": "Sling Dress", "req": [], "forbid": ["SHOULDER", "ELBOW"], "group": "DRESS"}
}

class BatchProcessor:
    def __init__(self):
        print("Loading Models...")
        # 1. Fashion Model
        self.cfg_f = get_cfg()
        self.cfg_f.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg_f.MODEL.ROI_HEADS.NUM_CLASSES = 14
        self.cfg_f.MODEL.WEIGHTS = FASHION_MODEL
        self.cfg_f.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESH
        self.pred_fashion = DefaultPredictor(self.cfg_f)

        # 2. Pose Model
        self.cfg_p = get_cfg()
        self.cfg_p.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        self.cfg_p.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        self.pred_pose = DefaultPredictor(self.cfg_p)

    def extract_features(self, image_bgr, mask):
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        pixels = hsv[mask > 0]
        
        if len(pixels) == 0: return 0, 0, 0, 0

        # Hue Circular Mean
        h_rad = pixels[:, 0] * (2 * np.pi / 180.0)
        avg_h = np.arctan2(np.sum(np.sin(h_rad)), np.sum(np.cos(h_rad))) / (2 * np.pi) % 1.0
        
        avg_s = np.mean(pixels[:, 1]) / 255.0
        avg_v = np.mean(pixels[:, 2]) / 255.0
        
        # Texture
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        texture = cv2.Laplacian(gray, cv2.CV_64F).var()
        texture = min(texture / 1000.0, 1.0) 

        return avg_h, avg_s, avg_v, texture

    def process_batch(self, images, ids):
        results_to_save = []
        
        for i, img in enumerate(images):
            db_id = ids[i]
            
            # Inference
            out_f = self.pred_fashion(img)["instances"].to("cpu")
            out_p = self.pred_pose(img)["instances"].to("cpu")
            
            # Relaxed Fusion: Just get keypoints if available
            pose_kps = None
            if len(out_p) > 0:
                pose_kps = out_p.pred_keypoints[np.argmax(out_p.pred_boxes.area().numpy())].numpy()
            
            if len(out_f) > 0:
                boxes = out_f.pred_boxes.tensor.numpy()
                classes = out_f.pred_classes.numpy()
                scores = out_f.scores.numpy()
                masks = out_f.pred_masks.numpy()
                
                for k in range(len(boxes)):
                    cls_id = classes[k] + 1
                    mask = masks[k]
                    
                    # Logic: If skeleton exists, validate. If not, accept (Relaxed).
                    # (You can copy exact check_joints logic from live_fusion_viewer here if strictness needed)
                    
                    h, s, v, tex = self.extract_features(img, mask)
                    
                    results_to_save.append({
                        "detection_id": db_id,
                        "category": str(cls_id), 
                        "confidence": float(scores[k]),
                        "color_h": float(h),
                        "color_s": float(s),
                        "color_v": float(v),
                        "texture_score": float(tex),
                        "area_ratio": float(np.sum(mask) / (img.shape[0]*img.shape[1])),
                        "bbox_json": json.dumps(boxes[k].tolist())
                    })
                    
        return results_to_save

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    conn = get_db_connection()
    if not conn: return

    # Ensure table/columns exist
    create_tables(conn)

    processor = BatchProcessor()
    
    print(f"Starting Clothing Analysis (Batch Size: {args.batch_size})...")

    while True:
        # 1. Claim Batch
        rows = claim_detections_for_analysis(conn, batch_size=args.batch_size)
        
        if not rows:
            print("No pending detections found. Sleeping 5s...")
            time.sleep(5)
            continue
            
        # 2. Load Images
        batch_imgs = []
        batch_ids = []

        for row in rows:
            # --- PATH FIX START ---
            # 1. Grab the raw path from DB (e.g., "D:\cropped_people\123_p0.jpg")
            raw_path = row['crop_path']
            
            # 2. Force convert backslashes to forward slashes just in case
            clean_path = raw_path.replace('\\', '/')
            
            # 3. Extract just the filename (e.g., "123_p0.jpg")
            filename = os.path.basename(clean_path)
            
            # 4. Join with your actual Linux directory
            real_path = os.path.join(LOCAL_CROP_DIR, filename)
            # --- PATH FIX END ---

            if not os.path.exists(real_path):
                # Print the 'real_path' so you see exactly where it's looking now
                print(f"Warning: Crop missing at {real_path} (DB originally said: {raw_path})")
                continue
                
            img = cv2.imread(real_path)
            if img is not None:
                batch_imgs.append(img)
                batch_ids.append(row['id'])
        
        if not batch_imgs:
            # Mark as done anyway so we don't get stuck on missing files
            mark_clothing_analysis_complete(conn, [r['id'] for r in rows])
            continue

        # 3. Process
        try:
            vectors = processor.process_batch(batch_imgs, batch_ids)
            
            # 4. Insert & Mark Complete
            if vectors:
                insert_clothing_measurements(conn, vectors)
            
            mark_clothing_analysis_complete(conn, [r['id'] for r in rows])
            print(f"Processed {len(batch_imgs)} people -> {len(vectors)} clothing items.")

        except Exception as e:
            print(f"Batch Error: {e}")
            # Release or mark failed? For now mark complete to avoid infinite loop
            mark_clothing_analysis_complete(conn, [r['id'] for r in rows])

if __name__ == "__main__":
    main()