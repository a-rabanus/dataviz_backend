"""
Pipeline Stage 3: Optimized Clothing Analysis
Architecture:
  [DB Claim] -> [Dataset (4 Workers Read/Prep)] -> [Queue] -> [GPU Batch Inference] -> [GPU Feature Extractor] -> [DB Write]
"""

import os
import sys
import cv2
import json
import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Detectron2
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
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
LOCAL_CROP_DIR = "cropped_people"  # Point to your local Linux folder
CONFIDENCE_THRESH = 0.5
DB_CHUNK_SIZE = 2000 # How many rows to claim from DB at once (Logic layer)

# --- GPU HELPER FUNCTIONS ---

def torch_rgb_to_hsv(image_tensor):
    """
    Convert (C, H, W) RGB tensor to HSV (C, H, W) on GPU.
    Image assumed to be 0-255 float.
    """
    # Normalize to 0-1
    img = image_tensor / 255.0
    r, g, b = img[0], img[1], img[2]

    max_val, _ = img.max(dim=0)
    min_val, _ = img.min(dim=0)
    diff = max_val - min_val

    # Hue Calculation
    h = torch.zeros_like(max_val)
    mask = diff > 0
    
    # Standard HSV logic
    # If R is max
    mask_r = (max_val == r) & mask
    h[mask_r] = (g[mask_r] - b[mask_r]) / diff[mask_r] % 6
    
    # If G is max
    mask_g = (max_val == g) & mask
    h[mask_g] = (b[mask_g] - r[mask_g]) / diff[mask_g] + 2
    
    # If B is max
    mask_b = (max_val == b) & mask
    h[mask_b] = (r[mask_b] - g[mask_b]) / diff[mask_b] + 4
    
    h = h / 6.0 # Normalize 0-1

    # Saturation
    s = torch.zeros_like(max_val)
    s[mask] = diff[mask] / max_val[mask]

    # Value
    v = max_val

    return torch.stack([h, s, v], dim=0)

def torch_texture_score(image_tensor, mask_tensor):
    """
    Calculate Laplacian variance on GPU as texture proxy.
    image_tensor: (3, H, W)
    mask_tensor: (H, W) boolean
    """
    # Convert to grayscale (approx)
    gray = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
    gray = gray.unsqueeze(0).unsqueeze(0) # (1, 1, H, W) needed for conv2d

    # Laplacian Kernel
    kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], 
                          device=image_tensor.device, dtype=torch.float32)
    
    # Apply Convolution
    laplacian = F.conv2d(gray, kernel, padding=1)
    laplacian = laplacian.squeeze() # (H, W)

    # Mask and calculate variance
    masked_lap = laplacian[mask_tensor]
    
    if masked_lap.numel() == 0:
        return 0.0
    
    return torch.var(masked_lap).item() / 1000.0  # Normalize roughly

# --- DATA LOADING ---

class ClothingDataset(Dataset):
    def __init__(self, db_rows):
        self.rows = db_rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        # 1. FORCE SINGLE THREADING FOR WORKER SAFETY
        cv2.setNumThreads(0) 

        row = self.rows[idx]
        raw_path = row['crop_path']
        
        # Path Correction
        clean_path = raw_path.replace('\\', '/')
        filename = os.path.basename(clean_path)
        real_path = os.path.join(LOCAL_CROP_DIR, filename)

        if not os.path.exists(real_path):
            return None 

        # Read Image
        img = cv2.imread(real_path)
        if img is None:
            return None

        # --- NEW: RESIZE TO PREVENT VRAM EXPLOSION ---
        # Resize so longest edge is max 512px
        h, w = img.shape[:2]
        scale = 512.0 / max(h, w)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
        # ---------------------------------------------

        # Convert to Tensor (C, H, W)
        img_tensor = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        
        return {
            "image": img_tensor,
            "height": img.shape[0],
            "width": img.shape[1],
            "detection_id": row['id']
        }

def collate_fn(batch):
    # Filter out Nones (missing files)
    return [x for x in batch if x is not None]

# --- MAIN PROCESSOR ---

class OptimizedProcessor:
    def __init__(self):
        print("Initializing Optimized Models...")
        
        # 1. Fashion Config
        cfg_f = get_cfg()
        cfg_f.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg_f.MODEL.ROI_HEADS.NUM_CLASSES = 14
        cfg_f.MODEL.WEIGHTS = FASHION_MODEL
        cfg_f.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESH
        
        # Build Model Directly (No DefaultPredictor)
        self.model_f = build_model(cfg_f)
        self.model_f.eval()
        checkpointer = DetectionCheckpointer(self.model_f)
        checkpointer.load(cfg_f.MODEL.WEIGHTS)
        
        # 2. Pose Config
        cfg_p = get_cfg()
        cfg_p.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        cfg_p.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        
        self.model_p = build_model(cfg_p)
        self.model_p.eval()
        checkpointer_p = DetectionCheckpointer(self.model_p)
        checkpointer_p.load(model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))

    def process_batch(self, batched_inputs):
        """
        batched_inputs: List[Dict] from DataLoader
        """
        results = []
        
        with torch.no_grad():
            # Run Inference (Automatic batching handled by model.inference internal loop, 
            # but pre-fetching is handled by DataLoader)
            preds_f = self.model_f(batched_inputs)
            # Optional: preds_p = self.model_p(batched_inputs) # If you need skeletons
            
            # Loop through batch results
            for i, output in enumerate(preds_f):
                instances = output["instances"]
                input_data = batched_inputs[i]
                img_tensor = input_data["image"] # On CPU or GPU?
                db_id = input_data["detection_id"]
                
                if len(instances) == 0:
                    continue
                
                # Move image to GPU for feature extraction if it isn't already
                if img_tensor.device != instances.pred_masks.device:
                    img_tensor = img_tensor.to(instances.pred_masks.device)

                # Vectorized Feature Extraction
                # Pre-calculate HSV for whole image once
                hsv_image = torch_rgb_to_hsv(img_tensor) # (3, H, W)
                
                # Iterate detected clothes
                for k in range(len(instances)):
                    mask = instances.pred_masks[k] # (H, W) boolean
                    cls_id = instances.pred_classes[k].item() + 1
                    score = instances.scores[k].item()
                    box = instances.pred_boxes[k].tensor.cpu().numpy()[0]
                    
                    # 1. Mask the HSV image
                    # We flatten to simplify mean calc
                    # (3, H, W) -> (3, N_pixels)
                    hsv_pixels = hsv_image[:, mask] 
                    
                    if hsv_pixels.shape[1] == 0:
                        continue
                        
                    # 2. Calculate Stats (GPU)
                    # Hue Circular Mean
                    h_rad = hsv_pixels[0] * (2 * np.pi)
                    h_sin = torch.sum(torch.sin(h_rad))
                    h_cos = torch.sum(torch.cos(h_rad))
                    avg_h = torch.atan2(h_sin, h_cos) / (2 * np.pi) % 1.0
                    
                    avg_s = torch.mean(hsv_pixels[1])
                    avg_v = torch.mean(hsv_pixels[2])
                    
                    # Texture
                    tex_score = torch_texture_score(img_tensor, mask)
                    
                    # Append result (move scalars to CPU)
                    results.append({
                        "detection_id": db_id,
                        "category": str(cls_id),
                        "confidence": score,
                        "color_h": avg_h.item(),
                        "color_s": avg_s.item(),
                        "color_v": avg_v.item(),
                        "texture_score": min(tex_score, 1.0),
                        "area_ratio": (mask.sum() / (mask.shape[0] * mask.shape[1])).item(),
                        "bbox_json": json.dumps(box.tolist())
                    })

        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help="Inference batch size")
    parser.add_argument('--workers', type=int, default=4, help="Data loader workers")
    args = parser.parse_args()

    conn = get_db_connection()
    create_tables(conn)
    
    processor = OptimizedProcessor()
    
    print(f"🚀 Optimized Pipeline Started. Workers: {args.workers}, Batch: {args.batch_size}")

    while True:
        # 1. Claim a LARGE chunk from DB (e.g. 2000 images)
        # We claim more than the batch size to keep the pipeline fed
        rows = claim_detections_for_analysis(conn, batch_size=DB_CHUNK_SIZE)
        
        if not rows:
            print("No pending detections. Sleeping 5s...")
            time.sleep(5)
            continue
            
        print(f"Loaded {len(rows)} tasks from DB. Starting Batch Processing...")
        
        # 2. Setup DataLoader
        dataset = ClothingDataset(rows)
        loader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.workers,
            collate_fn=collate_fn,
            pin_memory=True # Faster transfer to GPU
        )
        
        # 3. Process
        total_items = 0
        pbar = tqdm(total=len(rows), unit="img")
        
        try:
            for batch in loader:
                pbar.update(args.batch_size)
                if not batch: continue
                
                # Inference & Feature Extract
                results = processor.process_batch(batch)
                
                # Write to DB
                insert_clothing_measurements(conn, results)
                total_items += len(results)
                pbar.update(len(batch))
                
            pbar.close()
            
            # 4. Mark all rows in this chunk as complete
            ids = [r['id'] for r in rows]
            mark_clothing_analysis_complete(conn, ids)
            print(f"Chunk Complete. Extracted {total_items} clothing items.")
            
        except Exception as e:
            print(f"CRITICAL PIPELINE ERROR: {e}")
            import traceback
            traceback.print_exc()
            # Safety: Mark as done to prevent infinite loops on corrupt data
            ids = [r['id'] for r in rows]
            mark_clothing_analysis_complete(conn, ids)

if __name__ == "__main__":
    # Fix for multiprocessing on some Linux distros
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()