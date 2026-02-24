import subprocess
import time
import signal
import sys

# --- CONFIG ---
WORKERS_PER_GPU = 2      # Run 3 scripts on EACH GPU
GPUS = [0, 1]            # List of GPU IDs to use
BATCH_SIZE = 16         # Your sweet spot
# ----------------

processes = []

def signal_handler(sig, frame):
    print("\n🛑 Shutting down swarm...")
    for p in processes:
        p.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

print(f"🚀 Launching Swarm: {len(GPUS)} GPUs x {WORKERS_PER_GPU} Workers = {len(GPUS)*WORKERS_PER_GPU} Total Processes")

for gpu_id in GPUS:
    for i in range(WORKERS_PER_GPU):
        print(f"   [+] Launching Worker {i+1} on GPU {gpu_id}...")
        cmd = [
            sys.executable, 
            "pipelines/run_detection_pipeline.py", 
            "--gpu", str(gpu_id), 
            "--batch_size", str(BATCH_SIZE)
        ]
        # Start process without blocking
        p = subprocess.Popen(cmd)
        processes.append(p)
        # Small stagger to prevent database locking issues at startup
        time.sleep(2) 

print("✅ All workers running. Press Ctrl+C to stop.")

# Keep main script alive
while True:
    time.sleep(1)