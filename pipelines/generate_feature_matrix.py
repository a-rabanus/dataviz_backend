import os
import sqlite3
import numpy as np
import polars as pl
import cv2
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from tqdm import tqdm

DB_PATH = "data/pipeline.db"
OUTPUT_DIR = "data/feature_matrices"
SEED = 42
BATCH_SIZE = 4096
EPOCHS = 15
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CATEGORY_MAP = {
    1: 'short_sleeve_top', 2: 'long_sleeve_top',
    3: 'short_sleeve_outwear', 4: 'long_sleeve_outwear', 5: 'vest',
    6: 'sling', 7: 'shorts', 8: 'trousers', 9: 'skirt',
    10: 'short_sleeve_dress', 11: 'long_sleeve_dress', 12: 'vest_dress', 13: 'sling_dress'
}

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        l1 = 256 if input_dim > 10 else 64
        l2 = 128 if input_dim > 10 else 32
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, l1),
            nn.BatchNorm1d(l1),
            nn.LeakyReLU(0.2),
            nn.Linear(l1, l2),
            nn.BatchNorm1d(l2),
            nn.LeakyReLU(0.2),
            nn.Linear(l2, 2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(2, l2),
            nn.LeakyReLU(0.2),
            nn.Linear(l2, l1),
            nn.LeakyReLU(0.2),
            nn.Linear(l1, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def train_autoencoder(features_np):
    input_dim = features_np.shape[1]
    mean = np.mean(features_np, axis=0)
    std = np.std(features_np, axis=0) + 1e-8
    features_norm = (features_np - mean) / std
    
    tensor_x = torch.tensor(features_norm, dtype=torch.float32).to(DEVICE)
    dataset = TensorDataset(tensor_x)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = Autoencoder(input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    print(f"[{datetime.now()}] [GPU] Training Autoencoder (In: {input_dim} -> Latent: 2)...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in loader:
            x = batch[0]
            optimizer.zero_grad()
            latent, reconstructed = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    model.eval()
    coordinates = []
    inference_loader = DataLoader(TensorDataset(tensor_x), batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for batch in tqdm(inference_loader, desc="Generating Embeddings"):
            x = batch[0]
            latent, _ = model(x)
            coordinates.append(latent.cpu().numpy())
    return np.vstack(coordinates)

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def hsv_to_lab_opencv(hsv_array):
    h_scaled = hsv_array[:, 0] * 179
    s_scaled = hsv_array[:, 1] * 255
    v_scaled = hsv_array[:, 2] * 255
    img_hsv = np.dstack((h_scaled, s_scaled, v_scaled)).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    return img_lab.reshape(-1, 3).astype(np.float32)

def hsv_to_hex_array(hsv_array):
    h_scaled = hsv_array[:, 0] * 179
    s_scaled = hsv_array[:, 1] * 255
    v_scaled = hsv_array[:, 2] * 255
    img_hsv = np.dstack((h_scaled, s_scaled, v_scaled)).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    hex_list = ["#{:02x}{:02x}{:02x}".format(r, g, b) for r, g, b in img_rgb.reshape(-1, 3)]
    return np.array(hex_list)

def load_data():
    print(f"[{datetime.now()}] Loading data from {DB_PATH}...")
    conn = get_db_connection()
    query = """
        SELECT 
            c.id_item, 
            c.id_person AS id_detection, 
            CAST(c.category AS INTEGER) AS category, 
            c.confidence, 
            c.color_h, 
            c.color_s, 
            c.color_v, 
            c.texture_score, 
            c.area_ratio, 
            p.id_image,
            p.crop_path,
            i.captured_at,
            city.name AS full_city_name
        FROM clothing_item_detected c
        JOIN person_detected p ON c.id_person = p.id_person
        JOIN images_detected i ON p.id_image = i.id_image
        JOIN cities city ON i.id_city = city.id_city
        WHERE c.category IS NOT NULL
    """
    df = pl.read_database(query, conn)
    conn.close()
    
    df = df.with_columns([
        (pl.lit("data/cropped_people/") + pl.col("crop_path").str.replace_all(r"\\", "/").str.split("/").list.last()).alias("crop_path"),
        pl.col("captured_at").str.slice(0, 10).alias("date"),
        pl.col("captured_at").str.slice(11, 2).alias("time"),
        pl.col("full_city_name").str.split(",").list.get(0).str.strip_chars().alias("city"),
        pl.col("full_city_name").str.split(",").list.get(1).str.strip_chars().alias("state")
    ])

    print(f"[{datetime.now()}] Loaded {len(df)} rows.")
    return df

def process_item_matrix(df):
    print(f"[{datetime.now()}] Processing Item Matrix...")
    hsv = df.select(['color_h', 'color_s', 'color_v']).to_numpy()
    lab = hsv_to_lab_opencv(hsv)
    hex_colors = hsv_to_hex_array(hsv)
    
    features = np.hstack([
        lab,
        df['texture_score'].to_numpy().reshape(-1, 1),
        df['area_ratio'].to_numpy().reshape(-1, 1)
    ])
    
    embedding = train_autoencoder(features)
    
    output_df = df.select([
        pl.col('id_item').alias('id'),
        pl.col('category'),
        pl.col('id_image'),
        pl.col('crop_path'),
        pl.col('date'),
        pl.col('time'),
        pl.col('city'),
        pl.col('state')
    ]).with_columns([
        pl.Series(embedding[:, 0]).alias('x'),
        pl.Series(embedding[:, 1]).alias('y'),
        pl.Series(hex_colors).alias('color')
    ])
    
    out_path = os.path.join(OUTPUT_DIR, "item_feature_matrix.csv")
    output_df.write_csv(out_path)
    print(f"[{datetime.now()}] Saved {out_path}")

def generate_pairs():
    pairs = []
    cats = sorted(CATEGORY_MAP.keys())
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            pairs.append((cats[i], cats[j]))
    return sorted(list(set(pairs)))

def process_outfit_matrix(df):
    print(f"[{datetime.now()}] Processing Outfit Matrix...")
    df_dedup = df.sort("confidence", descending=True).unique(subset=["id_detection", "category"])
    
    hsv = df_dedup.select(['color_h', 'color_s', 'color_v']).to_numpy()
    lab = hsv_to_lab_opencv(hsv)
    
    df_dedup = df_dedup.with_columns([
        pl.Series(lab[:, 0]).alias('L'),
        pl.Series(lab[:, 1]).alias('A'),
        pl.Series(lab[:, 2]).alias('B')
    ])
    
    print(f"[{datetime.now()}] Pivoting data...")
    df_dedup = df_dedup.with_columns(pl.col("category").cast(pl.String))
    
    pivot_df = df_dedup.pivot(
        values=["L", "A", "B", "texture_score", "area_ratio"],
        index="id_detection",
        columns="category",
        aggregate_function="first"
    )
    
    meta_df = df_dedup.group_by("id_detection").agg([
        pl.col("id_image").first(),
        pl.col("crop_path").first(),
        pl.col("date").first(),
        pl.col("time").first(),
        pl.col("city").first(),
        pl.col("state").first(),
        pl.col("category").alias("category_list")
    ])
    
    pairs = generate_pairs()
    print(f"[{datetime.now()}] Calculating features for {len(pairs)} pairs...")
    
    exprs = []
    for c in CATEGORY_MAP.keys():
        col_name = f"L_{c}"
        if col_name in pivot_df.columns:
            exprs.append(pl.col(col_name).is_not_null().cast(pl.Int8).alias(f"has_{c}"))
        else:
            exprs.append(pl.lit(0).alias(f"has_{c}"))
            
    for c1, c2 in tqdm(pairs, desc="Building Pair Expressions"):
        c1, c2 = str(c1), str(c2)
        if f"L_{c1}" not in pivot_df.columns or f"L_{c2}" not in pivot_df.columns:
            exprs.append(pl.lit(None).cast(pl.Float32).alias(f"dist_color_{c1}_{c2}"))
            exprs.append(pl.lit(None).cast(pl.Float32).alias(f"dist_tex_{c1}_{c2}"))
            exprs.append(pl.lit(None).cast(pl.Float32).alias(f"dist_area_{c1}_{c2}"))
            exprs.append(pl.lit(0).alias(f"has_pair_{c1}_{c2}"))
            continue

        color_dist = ((pl.col(f"L_{c1}") - pl.col(f"L_{c2}")).pow(2) + 
                      (pl.col(f"A_{c1}") - pl.col(f"A_{c2}")).pow(2) + 
                      (pl.col(f"B_{c1}") - pl.col(f"B_{c2}")).pow(2)).sqrt().alias(f"dist_color_{c1}_{c2}")
        tex_dist = (pl.col(f"texture_score_{c1}") - pl.col(f"texture_score_{c2}")).abs().alias(f"dist_tex_{c1}_{c2}")
        area_dist = (pl.col(f"area_ratio_{c1}") - pl.col(f"area_ratio_{c2}")).abs().alias(f"dist_area_{c1}_{c2}")
        pair_flag = (pl.col(f"L_{c1}").is_not_null() & pl.col(f"L_{c2}").is_not_null()).cast(pl.Int8).alias(f"has_pair_{c1}_{c2}")
        
        exprs.extend([color_dist, tex_dist, area_dist, pair_flag])

    features_df = pivot_df.with_columns(exprs)
    
    target_cols = [c for c in features_df.columns if c.startswith('dist_') or c.startswith('has_')]
    dist_cols = [c for c in target_cols if c.startswith('dist_')]
    flag_cols = [c for c in target_cols if c.startswith('has_')]
    
    print(f"[{datetime.now()}] Performing Mean Imputation...")
    fill_exprs = []
    for col in dist_cols:
        fill_exprs.append(pl.col(col).fill_null(pl.col(col).mean()))
    for col in flag_cols:
        fill_exprs.append(pl.col(col).fill_null(0))
        
    final_features_df = features_df.with_columns(fill_exprs)
    final_features_df = final_features_df.with_columns(pl.all().fill_null(0))
    
    feature_matrix = final_features_df.select(target_cols).to_numpy()
    print(f"[{datetime.now()}] Feature Matrix Shape: {feature_matrix.shape}")
    
    embedding = train_autoencoder(feature_matrix)
    
    result_df = final_features_df.select("id_detection").with_columns([
        pl.Series(embedding[:, 0]).alias('x'),
        pl.Series(embedding[:, 1]).alias('y')
    ])
    
    final_output = result_df.join(meta_df, on="id_detection", how="left")
    final_output = final_output.rename({"id_detection": "id"})
    final_output = final_output.with_columns(pl.col("category_list").list.join("|"))
    
    out_path = os.path.join(OUTPUT_DIR, "outfit_feature_matrix.csv")
    final_output.write_csv(out_path)
    print(f"[{datetime.now()}] Saved {out_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if torch.cuda.is_available():
        print(f"[{datetime.now()}] GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        print(f"[{datetime.now()}] WARNING: GPU not detected. Training will be slow.")
        
    df = load_data()
    if df.is_empty():
        print("Database is empty or no valid detections found.")
        return
        
    process_item_matrix(df)
    process_outfit_matrix(df)

if __name__ == "__main__":
    main()