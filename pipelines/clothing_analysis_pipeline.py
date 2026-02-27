"""Executes clothing instance segmentation, extracts HSV/texture features, and refines classification using pose keypoints via Detectron2."""

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

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2 import model_zoo
from detectron2.layers import batched_nms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.db.db_utils import (
    get_db_connection,
    create_tables,
    claim_detections_for_analysis,
    mark_clothing_analysis_complete,
    insert_clothing_measurements
)

FASHION_MODEL = os.path.join(PROJECT_ROOT, "models", "model_final.pth")
LOCAL_CROP_DIR = os.path.join(PROJECT_ROOT, "data", "cropped_people")
CONFIDENCE_THRESH = 0.5
NO_POSE_CONFIDENCE_THRESH = 0.75
IOU_THRESHOLD = 0.5
DB_CHUNK_SIZE = 2000

KP = {
    "NOSE": 0, "L_EYE": 1, "R_EYE": 2, "L_EAR": 3, "R_EAR": 4,
    "L_SHOULDER": 5, "R_SHOULDER": 6, "L_ELBOW": 7, "R_ELBOW": 8,
    "L_WRIST": 9, "R_WRIST": 10, "L_HIP": 11, "R_HIP": 12,
    "L_KNEE": 13, "R_KNEE": 14, "L_ANKLE": 15, "R_ANKLE": 16
}

CLOTHING_ANATOMY = {
    1:  {"name": "Short Sleeve Top",    "req": ["SHOULDER"], "forbid": ["WRIST"], "group": "TOP"},
    2:  {"name": "Long Sleeve Top",     "req": ["SHOULDER", "WRIST"], "forbid": [], "group": "TOP"},
    3:  {"name": "Short Outwear",       "req": ["SHOULDER"], "forbid": ["WRIST"], "group": "OUTWEAR"},
    4:  {"name": "Long Outwear",        "req": ["SHOULDER", "WRIST"], "forbid": [], "group": "OUTWEAR"},
    5:  {"name": "Vest",                "req": ["SHOULDER"], "forbid": ["ELBOW", "WRIST"], "group": "TOP"},
    6:  {"name": "Sling",               "req": [], "forbid": ["SHOULDER", "ELBOW"], "group": "TOP"},
    7:  {"name": "Shorts",              "req": ["HIP"], "forbid": ["ANKLE"], "group": "BOTTOM"},
    8:  {"name": "Trousers",            "req": ["ANKLE"], "forbid": [], "group": "BOTTOM"},
    9:  {"name": "Skirt",               "req": ["HIP"], "forbid": [], "group": "BOTTOM"},
    10: {"name": "Short Dress",         "req": ["SHOULDER"], "forbid": ["ANKLE"], "group": "DRESS"},
    11: {"name": "Long Dress",          "req": ["SHOULDER", "ANKLE"], "forbid": [], "group": "DRESS"},
    12: {"name": "Vest Dress",          "req": ["SHOULDER"], "forbid": ["ELBOW"], "group": "DRESS"},
    13: {"name": "Sling Dress",         "req": [], "forbid": ["SHOULDER", "ELBOW"], "group": "DRESS"}
}

def torch_rgb_to_hsv(image_tensor):
    """Converts a normalized RGB image tensor to an HSV tensor."""
    img = image_tensor / 255.0
    r, g, b = img[0], img[1], img[2]
    max_val, _ = img.max(dim=0)
    min_val, _ = img.min(dim=0)
    diff = max_val - min_val

    h = torch.zeros_like(max_val)
    mask = diff > 0
    
    mask_r = (max_val == r) & mask
    h[mask_r] = (g[mask_r] - b[mask_r]) / diff[mask_r] % 6
    
    mask_g = (max_val == g) & mask
    h[mask_g] = (b[mask_g] - r[mask_g]) / diff[mask_g] + 2
    
    mask_b = (max_val == b) & mask
    h[mask_b] = (r[mask_b] - g[mask_b]) / diff[mask_b] + 4
    h = h / 6.0

    s = torch.zeros_like(max_val)
    s[mask] = diff[mask] / max_val[mask]
    v = max_val
    return torch.stack([h, s, v], dim=0)

def torch_texture_score(image_tensor, mask_tensor):
    """Calculates texture score via Laplacian variance on masked image regions."""
    gray = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
    gray = gray.unsqueeze(0).unsqueeze(0)
    kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], device=image_tensor.device, dtype=torch.float32)
    laplacian = F.conv2d(gray, kernel, padding=1).squeeze()
    masked_lap = laplacian[mask_tensor]
    if masked_lap.numel() == 0:
        return 0.0
    return torch.var(masked_lap).item() / 1000.0

class ClothingDataset(Dataset):
    """PyTorch Dataset for loading and resizing cropped human images."""
    def __init__(self, db_rows):
        self.rows = db_rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        row = self.rows[idx]
        raw_path = row['crop_path']
        clean_path = raw_path.replace('\\', '/')
        filename = os.path.basename(clean_path)
        real_path = os.path.join(LOCAL_CROP_DIR, filename)

        if not os.path.exists(real_path):
            return None

        img = cv2.imread(real_path)
        if img is None:
            return None

        h, w = img.shape[:2]
        scale = 512.0 / max(h, w)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))

        img_tensor = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        return {
            "image": img_tensor,
            "height": img.shape[0],
            "width": img.shape[1],
            "detection_id": row['id_person']
        }

def collate_fn(batch):
    return [x for x in batch if x is not None]

class OptimizedProcessor:
    """Manages Detectron2 mask and keypoint model inference."""
    def __init__(self):
        cfg_f = get_cfg()
        cfg_f.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg_f.MODEL.ROI_HEADS.NUM_CLASSES = 14
        cfg_f.MODEL.WEIGHTS = FASHION_MODEL
        cfg_f.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESH
        self.model_f = build_model(cfg_f)
        self.model_f.eval()
        DetectionCheckpointer(self.model_f).load(cfg_f.MODEL.WEIGHTS)

        cfg_p = get_cfg()
        cfg_p.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        cfg_p.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.model_p = build_model(cfg_p)
        self.model_p.eval()
        DetectionCheckpointer(self.model_p).load(model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))

    def check_joints(self, mask, keypoints, joint_names):
        """Verifies if specified anatomical joints overlap with the instance mask."""
        for name in joint_names:
            for side in ["L_", "R_"]:
                full_name = side + name
                if full_name not in KP: continue 
                idx = KP[full_name]
                x, y, vis = keypoints[idx]
                if vis > 0.05:
                    ix, iy = int(x), int(y)
                    if 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1]:
                        if mask[iy, ix]: return True
        return False

    def refine_class(self, original_class, mask, keypoints):
        """Adjusts clothing classification based on anatomical keypoint constraints."""
        if original_class not in CLOTHING_ANATOMY:
            return original_class, "No Rules"

        rules = CLOTHING_ANATOMY[original_class]
        current_group = rules["group"]
        
        orig_req = self.check_joints(mask, keypoints, rules["req"]) if rules["req"] else True
        orig_forbid = self.check_joints(mask, keypoints, rules["forbid"]) if rules["forbid"] else False
        
        if orig_req and not orig_forbid:
            return original_class, "Confirmed"

        for class_id, r in CLOTHING_ANATOMY.items():
            if r["group"] != current_group: continue
            if class_id == original_class: continue
            
            cand_req = self.check_joints(mask, keypoints, r["req"]) if r["req"] else True
            cand_forbid = self.check_joints(mask, keypoints, r["forbid"]) if r["forbid"] else False
            
            if cand_req and not cand_forbid:
                return class_id, "Switched"

        return original_class, "Geometry Mismatch"

    def process_batch(self, batched_inputs):
        """Executes inference, applies NMS, refines classes, and calculates color/texture metrics."""
        results = []
        with torch.no_grad():
            preds_f = self.model_f(batched_inputs)
            preds_p = self.model_p(batched_inputs)

            for i in range(len(preds_f)):
                inst_f = preds_f[i]["instances"]
                inst_p = preds_p[i]["instances"]
                input_data = batched_inputs[i]
                img_tensor = input_data["image"]
                db_id = input_data["detection_id"]

                if len(inst_f) == 0:
                    continue

                keep = batched_nms(inst_f.pred_boxes.tensor, inst_f.scores, inst_f.pred_classes, IOU_THRESHOLD)
                inst_f = inst_f[keep]

                pose_kps = None
                if len(inst_p) > 0:
                    idx = torch.argmax(inst_p.pred_boxes.area())
                    pose_kps = inst_p.pred_keypoints[idx].cpu().numpy()

                valid_detections = []
                for k in range(len(inst_f)):
                    cls_id = inst_f.pred_classes[k].item() + 1
                    score = inst_f.scores[k].item()
                    mask_cpu = inst_f.pred_masks[k].cpu().numpy()

                    if pose_kps is None:
                        if score < NO_POSE_CONFIDENCE_THRESH:
                            continue
                    else:
                        cls_id, outcome = self.refine_class(cls_id, mask_cpu, pose_kps)
                        if outcome == "Geometry Mismatch":
                            continue

                    group = CLOTHING_ANATOMY.get(cls_id, {}).get("group", "OTHER")
                    valid_detections.append({
                        "idx": k,
                        "cls_id": cls_id,
                        "score": score,
                        "group": group
                    })

                best_per_group = {}
                for det in valid_detections:
                    grp = det["group"]
                    if grp not in best_per_group or det["score"] > best_per_group[grp]["score"]:
                        best_per_group[grp] = det

                if img_tensor.device != inst_f.pred_masks.device:
                    img_tensor = img_tensor.to(inst_f.pred_masks.device)
                hsv_image = torch_rgb_to_hsv(img_tensor)

                for grp, det in best_per_group.items():
                    k = det["idx"]
                    cls_id = det["cls_id"]
                    score = det["score"]
                    mask = inst_f.pred_masks[k]
                    box = inst_f.pred_boxes[k].tensor.cpu().numpy()[0]

                    hsv_pixels = hsv_image[:, mask]
                    if hsv_pixels.shape[1] == 0:
                        continue

                    h_rad = hsv_pixels[0] * (2 * np.pi)
                    h_sin = torch.sum(torch.sin(h_rad))
                    h_cos = torch.sum(torch.cos(h_rad))
                    avg_h = torch.atan2(h_sin, h_cos) / (2 * np.pi) % 1.0
                    avg_s = torch.mean(hsv_pixels[1])
                    avg_v = torch.mean(hsv_pixels[2])

                    tex_score = torch_texture_score(img_tensor, mask)

                    results.append({
                        "id_person": db_id,
                        "category": str(cls_id),
                        "confidence": score,
                        "color_h": avg_h.item(),
                        "color_s": avg_s.item(),
                        "color_v": avg_v.item(),
                        "texture_score": min(tex_score, 1.0),
                        "area_ratio": (mask.sum() / (mask.shape[0] * mask.shape[1])).item(),
                        "bbox_item": json.dumps(box.tolist())
                    })

        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    conn = get_db_connection()
    create_tables(conn)
    processor = OptimizedProcessor()

    while True:
        rows = claim_detections_for_analysis(conn, batch_size=DB_CHUNK_SIZE)
        if not rows:
            time.sleep(5)
            continue

        dataset = ClothingDataset(rows)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

        total_items = 0
        pbar = tqdm(total=len(rows), unit="img")
        try:
            for batch in loader:
                pbar.update(args.batch_size)
                if not batch: continue
                
                results = processor.process_batch(batch)
                insert_clothing_measurements(conn, results)
                total_items += len(results)
                pbar.update(len(batch))
            
            pbar.close()
            ids = [r['id_person'] for r in rows]
            mark_clothing_analysis_complete(conn, ids)

        except Exception as e:
            ids = [r['id_person'] for r in rows]
            mark_clothing_analysis_complete(conn, ids)

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()