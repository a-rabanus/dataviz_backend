import tensorrt as trt
import os
import sys

# ================= CONFIGURATION SECTION =================
# SELECT YOUR TARGET BATCH SIZE HERE

# --- OPTION A: Batch 64 (Aggressive but possible on A4000) ---
TARGET_BATCH = 128
ENGINE_FILE = f"models/yolov8n_batch{TARGET_BATCH}.engine"
WORKSPACE_SIZE = 6 * (1 << 30)  # 6 GB Scratchpad

# --- OPTION B: Batch 128 (Very Risky on 16GB VRAM) ---
# TARGET_BATCH = 128
# ENGINE_FILE = "yolov8n_b128.engine"
# WORKSPACE_SIZE = 4 * (1 << 30) # Reduced to 4GB to save VRAM for the huge batch

# Common Settings
ONNX_FILE = "models/yolov8n.onnx"
IMG_SIZE = 1024
# =========================================================

def build_engine():
    print(f"--- Starting Engine Builder for Batch {TARGET_BATCH} ---")
    
    # Init TensorRT
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    # 1. Create Network (Explicit Batch is required for dynamic shapes)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, logger)
    
    # 2. Set Memory Pool
    # This dictates how much RAM TRT can use to test algorithms.
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_SIZE)
    print(f"Workspace limit set to: {WORKSPACE_SIZE / (1 << 30):.2f} GB")

    # 3. Parse ONNX Model
    if not os.path.exists(ONNX_FILE):
        print(f"❌ Error: {ONNX_FILE} not found. Did you run 'yolo export ... format=onnx'?")
        return

    print(f"Parsing {ONNX_FILE}...")
    with open(ONNX_FILE, 'rb') as model:
        if not parser.parse(model.read()):
            print("❌ ERROR: Failed to parse ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    # 4. Define Optimization Profile (CRITICAL STEP)
    # This tells TensorRT what input shapes to expect.
    print(f"Configuring optimization profile for Max Batch={TARGET_BATCH}...")
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    
    # (Min, Opt, Max)
    # Min: Smallest batch you'll ever send (1)
    # Opt: The most common batch size (Target / 2 is usually a sweet spot for performance)
    # Max: The absolute limit (Target)
    profile.set_shape(
        input_name, 
        (1, 3, IMG_SIZE, IMG_SIZE),              # Min
        (TARGET_BATCH // 2, 3, IMG_SIZE, IMG_SIZE), # Opt (Optimized for half load)
        (TARGET_BATCH, 3, IMG_SIZE, IMG_SIZE)    # Max
    )
    config.add_optimization_profile(profile)
    
    # 5. Enable FP16 (Half Precision)
    if builder.platform_has_fast_fp16:
        print("Enabling FP16 acceleration.")
        config.set_flag(trt.BuilderFlag.FP16)

    # 6. Build Engine
    print("Building Engine... (This will take a while, monitor VRAM!)")
    try:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine:
            with open(ENGINE_FILE, 'wb') as f:
                f.write(serialized_engine)
            print(f"✅ Success! Engine saved to {ENGINE_FILE}")
        else:
            print("❌ Build failed (returned None). Likely Out Of Memory.")
    except Exception as e:
        print(f"❌ Exception during build: {e}")

if __name__ == "__main__":
    build_engine()