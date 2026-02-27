"""Generates high-resolution scatter plots from 2D latent feature matrices."""

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import os

DATA_DIR = "data/feature_matrices"
FIG_SIZE = (32, 24)
DPI = 240
MARKER_SIZE = 0.05
ALPHA = 0.3

def visualize_file(filepath, filename):
    """Reads latent coordinates and renders an 8K scatter plot colored by category or outfit item count."""
    print(f"[LOAD] Reading {filename}...")
    df = pl.read_csv(filepath)
    
    if df.is_empty():
        print(f"[SKIP] {filename} is empty.")
        return

    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    
    print(f"[PLOT] Rendering 8K image (7680x5760)...")
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    
    if "item" in filename.lower():
        print(" -> Mode: Item (Coloring by Category)")
        categories = df['category_list'].fill_null("0").cast(pl.Int32).to_numpy()
        scatter = ax.scatter(x, y, c=categories, cmap='tab20', s=MARKER_SIZE, alpha=ALPHA)
        
        unique_cats = np.unique(categories)
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cat {c}', 
                   markerfacecolor=scatter.cmap(scatter.norm(c)), markersize=10) for c in unique_cats]
        ax.legend(handles=handles, title="Category", loc='upper right', fontsize='medium')

    else:
        print(" -> Mode: Outfit (Coloring by Item Count)")
        counts = df['category_list'].fill_null("").str.split("|").list.len().to_numpy()
        scatter = ax.scatter(x, y, c=counts, cmap='magma', s=MARKER_SIZE, alpha=ALPHA, vmin=1, vmax=5)
        
        cbar = plt.colorbar(scatter, ax=ax, label="Items in Outfit", fraction=0.02, pad=0.01)
        cbar.ax.tick_params(labelsize='medium')

    ax.set_title(f"{filename}\n(N={len(df):,})", color='white', fontsize=24)
    ax.set_axis_off()
    
    output_path = filepath.replace('.csv', '_8k.png')
    plt.tight_layout()
    plt.savefig(output_path, facecolor='black')
    plt.close()
    print(f"[DONE] Saved {output_path}")

def main():
    plt.style.use('dark_background')
    
    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] Directory {DATA_DIR} not found.")
        return
        
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    
    if not files:
        print(f"[ERROR] No CSV files found in {DATA_DIR}.")
        return
        
    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)
        visualize_file(filepath, filename)

if __name__ == "__main__":
    main()