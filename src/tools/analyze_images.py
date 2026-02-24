import os
import statistics
import sys
import time

def analyze_directory(folder_path):
    print(f"--- Scanning directory: {folder_path} ---")
    print("This may take a moment for 1 million files...")
    
    file_sizes = []
    start_time = time.time()
    
    try:
        # os.scandir is an iterator and is much faster than os.listdir
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if entry.is_file():
                    # st_size is in bytes
                    file_sizes.append(entry.stat().st_size)
                    
    except FileNotFoundError:
        print(f"Error: The directory '{folder_path}' was not found.")
        sys.exit(1)

    elapsed = time.time() - start_time
    count = len(file_sizes)
    
    if count == 0:
        print("No files found in the directory.")
        return

    print(f"\nScan complete in {elapsed:.2f} seconds.")
    print(f"Total images found: {count:,}")
    
    # Sort for percentiles
    file_sizes.sort()
    
    # Conversions
    def to_mb(bytes_val):
        return bytes_val / (1024 * 1024)

    min_size = file_sizes[0]
    max_size = file_sizes[-1]
    total_size = sum(file_sizes)
    avg_size = total_size / count
    median_size = statistics.median(file_sizes)
    stdev = statistics.stdev(file_sizes) if count > 1 else 0
    
    # Percentiles for outlier detection
    p95 = file_sizes[int(count * 0.95)]
    p99 = file_sizes[int(count * 0.99)]

    print("\n--- File Size Statistics ---")
    print(f"{'Metric':<20} | {'Bytes':<15} | {'Megabytes (MB)':<15}")
    print("-" * 55)
    print(f"{'Total Volume':<20} | {total_size:<15,.0f} | {to_mb(total_size):<15.2f}")
    print(f"{'Average Size':<20} | {avg_size:<15,.0f} | {to_mb(avg_size):<15.4f}")
    print(f"{'Median Size':<20} | {median_size:<15,.0f} | {to_mb(median_size):<15.4f}")
    print(f"{'Min Size':<20} | {min_size:<15,.0f} | {to_mb(min_size):<15.6f}")
    print(f"{'Max Size':<20} | {max_size:<15,.0f} | {to_mb(max_size):<15.2f}")
    print(f"{'Std Deviation':<20} | {stdev:<15,.0f} | {to_mb(stdev):<15.4f}")
    
    print("\n--- Variance & Outliers ---")
    print(f"95% of images are smaller than: {to_mb(p95):.2f} MB")
    print(f"99% of images are smaller than: {to_mb(p99):.2f} MB")
    
    # Suggestion based on batch processing needs
    print("\n--- Batch Processing Estimate ---")
    print(f"If you use the 99th percentile ({to_mb(p99):.2f} MB) as a safety buffer:")
    print(f"1 GB of disk data ~= {1024 / to_mb(p99):.0f} images (Safe Estimate)")
    print(f"1 GB of disk data ~= {1024 / to_mb(avg_size):.0f} images (Aggressive Estimate)")

if __name__ == "__main__":
    target_dir = "cropped_people"
    analyze_directory(target_dir)