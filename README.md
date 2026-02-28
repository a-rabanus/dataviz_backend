# Mapillary Urban Clothing and Person Detection Pipeline

## Core Architecture

Automated data pipeline for downloading street-level imagery, detecting pedestrians, extracting clothing segmentation features, and reducing dimensionality for latent space visualization.

## Module Definitions

## System Pipeline Architecture

The system executes sequentially across independent modules synchronized via `data/pipeline.db` utilizing SQLite WAL mode.

1. Geographic Scoping
Script: `populate_cities_table.py`
Logic: Queries Nominatim API to extract spatial bounding boxes for target cities. Implements spherical distortion correction mapping.
Storage: Target coordinates and status flags stored in the `cities` table.
2. Image Acquisition
Script: `download_pipeline.py`
API: Mapillary API
Logic: Executes asynchronous `aiohttp` requests with `unique_tiles` deduplication. Resolves bounding boxes into sub-grids for API compliance.
Storage: Source files saved locally. Metadata written to `images_detected` keyed by globally unique Mapillary `image_id`.
3. Pedestrian Detection & Cropping
Script: `detection_pipeline.py`
Model: YOLOv8 (TensorRT engine `yolov8n.engine`)
Logic: Isolates human targets using a 75% confidence threshold. Crops bounding box coordinates and discards source imagery lacking target detections to minimize storage overhead.
Storage: Crops written to `data/cropped_people/`. Spatial relationships stored in `person_detected` with natural key deduplication `(image_id, bbox_person)`.
4. Clothing Analysis
Script: `clothing_analysis_pipeline.py`
Model: Detectron2 (Dual network: Mask R-CNN + Keypoint R-CNN)

Logic: Extracts initial segmentation masks. Applies anatomical pose filtering via joint visibility to refine or discard clothing classes. Enforces strict categorical limits per person. Calculates vectorized HSV color values and Laplacian texture variance.
Storage: Attributes written to `clothing_item_detected`.

5. Dimensionality Reduction & Feature Matrix
Script: `generate_feature_matrix.py`
Method: PyTorch Autoencoder (32D Bottleneck) + UMAP Projection

Logic: Applies asymmetrical feature scalar weighting (Color/Texture: 2.0x, Spatiotemporal metadata: 0.1x) to prevent variance domination. Compresses high-dimensional vectors (LAB color, texture, area ratio, cyclic time, location) into 32 dimensions, then projects to 2D topological coordinates via UMAP.
Storage: CSV files (`item_*.csv`, `outfit_*.csv`) exported to `data/feature_matrices/` containing spatial coordinates `x, y` and literal hex `color` strings.
## Execution Sequence

1. Initialize database: `python init_db.py`
2. Seed geographic targets: `python populate_cities_table.py`
3. Acquire raw images: `python download_pipeline.py` (Requires `MAPILLARY_TOKEN` in `.env`)
4. Generate person crops: `python detection_pipeline.py --gpu 0 --batch_size 16`
5. Extract clothing features: `python clothing_analysis_pipeline.py`
6. Train autoencoders: `python generate_feature_matrix.py`
7. Render plots: `python visualize_matrices.py`
k
## System Prerequisitesk
* **Operating System**: Linux
* **Hardware**: CUDA-capable GPU required for YOLOv8 and Detectron2 inference.

## Environment Setup
1.  Initialize a Python virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
2.  Install dependencies from `requirements.txt`. Specify the PyTorch CUDA 12.1 index for hardware acceleration:
    ```bash
    pip install -r requirements.txt --extra-index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```
3.  Install Detectron2 from the repository source if it fails to resolve via PyPI:
    ```bash
    python -m pip install 'git+[https://github.com/facebookresearch/detectron2.git](https://github.com/facebookresearch/detectron2.git)'
    ```

## Configuration
Define API and client environment variables. Create a `.env` file in the project root containing:
```env
MAPILLARY_TOKEN=<your_mapillary_api_token>
USER_AGENT=<your_custom_user_agent_string>
```

_Note: The Mapillary Token is mandatory for download_pipeline.py execution. The User Agent is mandatory for Nominatim API queries in populate_cities_table.py._