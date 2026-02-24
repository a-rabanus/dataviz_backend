'''
Generate "Functional" Mode:
```Bash
python generate_feature_matrices_pytorch.py --mode functional
mv outfit_feature_matrix.csv data/outfit_feature_matrix_mode-functional.csv
mv item_feature_matrix.csv data/item_feature_matrix_mode-functional.csv
```

Generate "Full" Mode:
```Bash
python generate_feature_matrices_pytorch.py --mode full
mv outfit_feature_matrix.csv data/outfit_feature_matrix_mode-full.csv
mv item_feature_matrix.csv data/item_feature_matrix_mode-full.csv
```
Run Visualizer:
```Bash
python visualize_matrices.py
```
'''
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import Normalize

# --- Configuration ---
DATA_DIR = "data/images/"
FILES = [
    "outfit_feature_matrix_mode-full.csv",
    "item_feature_matrix_mode-full.csv",
    "outfit_feature_matrix_mode-functional.csv",
    "item_feature_matrix_mode-functional.csv"
]

# Style for better point cloud visibility
plt.style.use('dark_background')

# --- 8K Resolution Settings ---
# figsize=(32, 24) * dpi=240  =>  7680 x 5760 pixels
FIG_SIZE = (32, 24)
DPI = 240
# Marker size needs to be tiny at this resolution to see density detail
MARKER_SIZE = 0.05  
ALPHA = 0.3

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct RGB color'''
    return plt.cm.get_cmap(name, n)

def visualize_file(filename):
    file_path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(file_path):
        print(f"[SKIP] File not found: {file_path}")
        return

    print(f"[LOAD] Reading {filename}...")
    df = pl.read_csv(file_path)
    
    # Extract coordinates
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    
    # --- Plot Setup (Updated for 8K) ---
    print(f"[PLOT] Rendering 8K image (7680x5760)... hold tight.")
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    
    # --- Coloring Logic ---
    if "item_" in filename:
        # ITEM MATRIX: Color by Category (1-13)
        print(" -> Mode: Item (Coloring by Category)")
        categories = df['category'].to_numpy()
        scatter = ax.scatter(x, y, c=categories, cmap='tab20', s=MARKER_SIZE, alpha=ALPHA)
        
        # Legend for categories
        unique_cats = np.unique(categories)
        # Make legend markers bigger than plot markers so they are visible
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cat {c}', 
                   markerfacecolor=scatter.cmap(scatter.norm(c)), markersize=10) for c in unique_cats]
        # Move legend inside plot boundary for cleaner export
        ax.legend(handles=handles, title="Category", loc='upper right', fontsize='medium')

    else:
        # OUTFIT MATRIX: Color by Complexity (Item Count)
        print(" -> Mode: Outfit (Coloring by Item Count)")
        counts = df['category_list'].fill_null("").str.split("|").list.len().to_numpy()
        
        scatter = ax.scatter(x, y, c=counts, cmap='magma', s=MARKER_SIZE, alpha=ALPHA, vmin=1, vmax=5)
        # Adjust colorbar size relative to the huge figure
        cbar = plt.colorbar(scatter, ax=ax, label="Items in Outfit", fraction=0.02, pad=0.01)
        cbar.ax.tick_params(labelsize='medium')

    # --- Decoration ---
    ax.set_title(f"{filename}\n(N={len(df):,})", color='white', fontsize=24)
    ax.set_axis_off()
    
    # --- Save ---
    # Add _8k suffix to differentiate
    output_path = file_path.replace('.csv', '_8k.png')
    # tight_layout can be slow/buggy at huge resolutions, specify rect if needed
    plt.tight_layout()
    plt.savefig(output_path, facecolor='black')
    plt.close()
    print(f"[DONE] Saved {output_path}")

def main():
    print(f"Checking {DATA_DIR} for matrices...")
    for f in FILES:
        visualize_file(f)

if __name__ == "__main__":
    main()