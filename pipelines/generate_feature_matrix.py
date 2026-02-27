"""Trains PyTorch autoencoders to reduce clothing and outfit features into 2D latent spaces."""

import os
import sqlite3
import numpy as np
import polars as pl
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DB_PATH = "data/pipeline.db"
OUTPUT_DIR = "data/feature_matrices"
BATCH_SIZE = 16384
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
    """Defines a fully connected encoder-decoder architecture for dimensionality reduction."""
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
    """Normalizes input features and trains the autoencoder to produce a 2D latent representation."""
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
    scaler = torch.amp.GradScaler('cuda')
    
    model.train()
    for _ in range(EPOCHS):
        for batch in loader:
            x = batch[0]
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                latent, reconstructed = model(x)
                loss = criterion(reconstructed, x)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.float16):
            latent, _ = model(tensor_x)
    return latent.cpu().numpy().astype(np.float32)

def hsv_to_lab_opencv(hsv_array):
    h_scaled = hsv_array[:, 0] * 179
    s_scaled = hsv_array[:, 1] * 255
    v_scaled = hsv_array[:, 2] * 255
    img_hsv = np.dstack((h_scaled, s_scaled, v_scaled)).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    return img_lab.reshape(-1, 3).astype(np.float32)

def hsv_to_hex_vectorized(hsv_array):
    h_scaled = (hsv_array[:, 0] * 179).astype(np.uint8)
    s_scaled = (hsv_array[:, 1] * 255).astype(np.uint8)
    v_scaled = (hsv_array[:, 2] * 255).astype(np.uint8)
    
    hsv_uint8 = np.stack([h_scaled, s_scaled, v_scaled], axis=1).reshape(-1, 1, 3)
    rgb_uint8 = cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2RGB).reshape(-1, 3)
    
    hex_chars = np.array([f"#{r:02x}{g:02x}{b:02x}" for r, g, b in rgb_uint8])
    return hex_chars

def load_data():
    """Extracts, parses, and formats clothing measurements and spatio-temporal data from the database."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT 
            c.id_item, 
            c.id_person, 
            CAST(c.category AS INTEGER) AS category, 
            c.color_h, c.color_s, c.color_v, 
            c.texture_score, c.area_ratio, 
            p.id_image, p.crop_path, p.captured_at,
            CAST(json_extract(p.location, '$.coordinates[0]') AS REAL) AS lon,
            CAST(json_extract(p.location, '$.coordinates[1]') AS REAL) AS lat,
            city.name AS full_city_name
        FROM clothing_item_detected c
        JOIN person_detected p ON c.id_person = p.id_person
        JOIN cities city ON p.id_city = city.id_city
        WHERE c.category IS NOT NULL
    """
    df = pl.read_database(query, conn)
    conn.close()

    df = df.with_columns([
        (pl.lit("data/cropped/people/") + pl.col("crop_path").str.replace_all(r"\\", "/").str.split("/").list.last()).alias("crop_path"),
        pl.col("captured_at").str.slice(0, 10).alias("date"),
        pl.col("captured_at").str.slice(11, 8).alias("time"),
        pl.col("full_city_name").str.split(",").list.get(0).str.strip_chars().alias("city"),
        pl.col("full_city_name").str.split(",").list.get(1).str.strip_chars().alias("state")
    ])

    df = df.with_columns(pl.col("captured_at").str.slice(0, 19).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False).alias("dt_parsed"))

    df = df.with_columns([
        (np.cos(np.radians(pl.col("lat"))) * np.cos(np.radians(pl.col("lon")))).fill_null(0).alias("loc_x"),
        (np.cos(np.radians(pl.col("lat"))) * np.sin(np.radians(pl.col("lon")))).fill_null(0).alias("loc_y"),
        (np.sin(np.radians(pl.col("lat")))).fill_null(0).alias("loc_z"),
        (np.sin(2 * np.pi * pl.col("dt_parsed").dt.month() / 12)).fill_null(0).alias("time_month_sin"),
        (np.cos(2 * np.pi * pl.col("dt_parsed").dt.month() / 12)).fill_null(0).alias("time_month_cos"),
        (np.sin(2 * np.pi * pl.col("dt_parsed").dt.weekday() / 7)).fill_null(0).alias("time_day_sin"),
        (np.cos(2 * np.pi * pl.col("dt_parsed").dt.weekday() / 7)).fill_null(0).alias("time_day_cos"),
        (np.sin(2 * np.pi * pl.col("dt_parsed").dt.hour() / 24)).fill_null(0).alias("time_hour_sin"),
        (np.cos(2 * np.pi * pl.col("dt_parsed").dt.hour() / 24)).fill_null(0).alias("time_hour_cos"),
        pl.col("category").cast(pl.String).alias("category_list") 
    ])
    return df

def generate_matrices(df, mode="item"):
    """Compiles visual, spatial, and temporal features into configured matrices and executes autoencoder training."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if mode == "item":
        hsv_raw = df.select(['color_h', 'color_s', 'color_v']).to_numpy()
        lab = hsv_to_lab_opencv(hsv_raw)
        hex_colors = hsv_to_hex_vectorized(hsv_raw)
        
        base_features = np.hstack([
            lab,
            df['texture_score'].to_numpy().reshape(-1, 1),
            df['area_ratio'].to_numpy().reshape(-1, 1)
        ])
        loc_features = df.select(['loc_x', 'loc_y', 'loc_z']).to_numpy()
        time_features = df.select(['time_month_sin', 'time_month_cos', 'time_day_sin', 'time_day_cos', 'time_hour_sin', 'time_hour_cos']).to_numpy()
        
        meta = df.select(['id_item', 'id_image', 'crop_path', 'date', 'time', 'city', 'state', 'category_list']).rename({"id_item": "id"})
        meta = meta.with_columns(pl.Series("color", hex_colors))
        prefix = "item"

    elif mode == "outfit":
        df_dedup = df.sort("texture_score", descending=True).unique(subset=["id_person", "category"])
        hsv_raw = df_dedup.select(['color_h', 'color_s', 'color_v']).to_numpy()
        lab = hsv_to_lab_opencv(hsv_raw)
        
        # We define outfit color as the color of the most prominent item in the outfit (by area)
        df_dominant = df_dedup.sort("area_ratio", descending=True).unique(subset=["id_person"])
        hsv_dom = df_dominant.select(['color_h', 'color_s', 'color_v']).to_numpy()
        dom_hex = hsv_to_hex_vectorized(hsv_dom)
        dom_map = pl.DataFrame({"id": df_dominant["id_person"], "color": dom_hex})

        df_dedup = df_dedup.with_columns([
            pl.Series(lab[:, 0]).alias('L'), pl.Series(lab[:, 1]).alias('A'), pl.Series(lab[:, 2]).alias('B')
        ])
        
        pivot_df = df_dedup.pivot(
            values=["L", "A", "B", "texture_score", "area_ratio"],
            index="id_person", columns="category", aggregate_function="first"
        )
        
        exprs = []
        for c1 in CATEGORY_MAP.keys():
            for c2 in range(c1 + 1, max(CATEGORY_MAP.keys()) + 1):
                c1_str, c2_str = str(c1), str(c2)
                if f"L_{c1_str}" not in pivot_df.columns or f"L_{c2_str}" not in pivot_df.columns:
                    continue
                exprs.extend([
                    ((pl.col(f"L_{c1_str}") - pl.col(f"L_{c2_str}")).pow(2) + 
                     (pl.col(f"A_{c1_str}") - pl.col(f"A_{c2_str}")).pow(2) + 
                     (pl.col(f"B_{c1_str}") - pl.col(f"B_{c2_str}")).pow(2)).sqrt().fill_null(0).alias(f"dist_color_{c1}_{c2}"),
                    (pl.col(f"texture_score_{c1_str}") - pl.col(f"texture_score_{c2_str}")).abs().fill_null(0).alias(f"dist_tex_{c1}_{c2}"),
                    (pl.col(f"area_ratio_{c1_str}") - pl.col(f"area_ratio_{c2_str}")).abs().fill_null(0).alias(f"dist_area_{c1}_{c2}"),
                    (pl.col(f"L_{c1_str}").is_not_null() & pl.col(f"L_{c2_str}").is_not_null()).cast(pl.Int8).fill_null(0).alias(f"has_pair_{c1}_{c2}")
                ])
        
        features_df = pivot_df.with_columns(exprs).fill_null(0).sort("id_person")
        target_cols = [c for c in features_df.columns if c.startswith('dist_') or c.startswith('has_')]
        
        meta = df_dedup.group_by("id_person").agg([
            pl.col("id_image").first(), pl.col("crop_path").first(), pl.col("date").first(), pl.col("time").first(),
            pl.col("city").first(), pl.col("state").first(),
            pl.col("category").cast(pl.String).alias("cat_list"),
            pl.col("loc_x").first(), pl.col("loc_y").first(), pl.col("loc_z").first(),
            pl.col("time_month_sin").first(), pl.col("time_month_cos").first(), pl.col("time_day_sin").first(),
            pl.col("time_day_cos").first(), pl.col("time_hour_sin").first(), pl.col("time_hour_cos").first()
        ]).with_columns([
            pl.col("cat_list").list.join("|").alias("category_list")
        ]).rename({"id_person": "id"}).sort("id")
        
        meta = meta.join(dom_map, on="id", how="left")
        
        base_features = features_df.select(target_cols).to_numpy()
        loc_features = meta.select(['loc_x', 'loc_y', 'loc_z']).to_numpy()
        time_features = meta.select(['time_month_sin', 'time_month_cos', 'time_day_sin', 'time_day_cos', 'time_hour_sin', 'time_hour_cos']).to_numpy()
        meta = meta.drop(['cat_list', 'loc_x', 'loc_y', 'loc_z', 'time_month_sin', 'time_month_cos', 'time_day_sin', 'time_day_cos', 'time_hour_sin', 'time_hour_cos'])
        prefix = "outfit"

    configs = [
        ("base", base_features),
        ("time", np.hstack([base_features, time_features])),
        ("loc", np.hstack([base_features, loc_features])),
        ("time_loc", np.hstack([base_features, time_features, loc_features]))
    ]

    for suffix, matrix in configs:
        latent = train_autoencoder(matrix)
        out = meta.with_columns([
            pl.Series("x", latent[:, 0]),
            pl.Series("y", latent[:, 1])
        ])
        
        out = out.select(["id", "x", "y", "id_image", "crop_path", "date", "time", "city", "state", "category_list", "color"])
        out.write_csv(os.path.join(OUTPUT_DIR, f"{prefix}_{suffix}.csv"))

def main():
    torch.backends.cudnn.benchmark = True
    df = load_data()
    if df.is_empty():
        return
    generate_matrices(df, mode="item")
    generate_matrices(df, mode="outfit")

if __name__ == "__main__":
    main()