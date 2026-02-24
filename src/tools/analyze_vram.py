import os
import random
import sys
import time
from PIL import Image

def analyze_vram_usage(folder_path, sample_size=5000):
    print(f"--- Scanning directory: {folder_path} ---")
    
    all_files = []
    
    # 1. Quick Scan to get all filenames
    try:
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if entry.is_file():
                    all_files.append(entry.path)
    except FileNotFoundError:
        print(f"Error: Directory '{folder_path}' not found.")
        sys.exit(1)

    total_files = len(all_files)
    print(f"Total files found: {total_files:,}")

    # 2. Sample random images
    if total_files > sample_size:
        print(f"Randomly sampling {sample_size:,} images for VRAM estimation...")
        sampled_files = random.sample(all_files, sample_size)
    else:
        sampled_files = all_files

    print("Analyzing image dimensions (reading headers)...")
    
    tensor_sizes_mb = []
    resolutions = [] # Store (width, height)
    
    # 3. Analyze Dimensions
    start_time = time.time()
    
    for file_path in sampled_files:
        try:
            with Image.open(file_path) as img:
                w, h = img.size
                resolutions.append((w, h))
                
                # Calculate Raw Tensor Size in VRAM (Float32)
                # Formula: Height * Width * 3 channels * 4 bytes
                raw_bytes = w * h * 3 * 4
                tensor_sizes_mb.append(raw_bytes / (1024 * 1024))
        except Exception:
            # Skip non-image files or errors
            continue

    elapsed = time.time() - start_time
    count = len(tensor_sizes_mb)
    
    if count == 0:
        print("No valid images found in sample.")
        return

    # 4. Sort and Calculate Statistics
    tensor_sizes_mb.sort()
    resolutions.sort(key=lambda x: x[0]*x[1]) # Sort by total pixels
    
    avg_vram = sum(tensor_sizes_mb) / count
    max_vram = tensor_sizes_mb[-1]
    p95_vram = tensor_sizes_mb[int(count * 0.95)]
    p99_vram = tensor_sizes_mb[int(count * 0.99)]
    
    min_res = resolutions[0]
    max_res = resolutions[-1]
    avg_pixels = sum(w*h for w,h in resolutions) / count
    avg_res_side = int(avg_pixels**0.5) # Approximate average side length

    print(f"\nAnalysis complete in {elapsed:.2f} seconds based on {count} samples.")
    
    print("\n--- Image Resolution Stats ---")
    print(f"Min Resolution: {min_res[0]}x{min_res[1]}")
    print(f"Max Resolution: {max_res[0]}x{max_res[1]}")
    print(f"Avg Resolution: ~{avg_res_side}x{avg_res_side} (approx)")

    print("\n--- VRAM Usage Per Image (Float32 / Batch Size 1) ---")
    print(f"{'Metric':<20} | {'VRAM Required':<15}")
    print("-" * 40)
    print(f"{'Average Image':<20} | {avg_vram:.2f} MB")
    print(f"{'95th Percentile':<20} | {p95_vram:.2f} MB")
    print(f"{'Max (Worst Case)':<20} | {max_vram:.2f} MB")

    print("\n--- Batch Size Estimator (Float32) ---")
    print("Estimated max batch size for varying VRAM capacities:")
    print("Note: This assumes 20% VRAM overhead for model weights/gradients.")
    print("-" * 65)
    
    vram_sizes = [8, 12, 16, 24, 40, 80] # Common GPU VRAMs in GB
    
    # Using 99th percentile for safety
    safety_size_mb = p99_vram 
    
    print(f"{'GPU VRAM':<10} | {'Safe Batch Size (approx)':<25}")
    for vram in vram_sizes:
        # Convert GB to MB, reserve 20% overhead
        available_mem = (vram * 1024) * 0.8
        batch_size = int(available_mem / safety_size_mb)
        print(f"{vram:<2} GB     | {batch_size:,} images")

if __name__ == "__main__":
    # Ensure this points to your directory
    target_dir = "cropped_people"
    analyze_vram_usage(target_dir)