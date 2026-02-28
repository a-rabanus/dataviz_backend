# Mapillary Urban Clothing and Person Detection Pipeline

## Core Architecture

Automated data pipeline for downloading street-level imagery, detecting pedestrians, extracting clothing segmentation features, and reducing dimensionality for latent space visualization.

## Module Definitions

* **Database Management (`init_db.py`, `db_utils.py`)**:
Constructs SQLite schema and manages asynchronous state transitions. Enforces WAL mode and concurrency locks for parallel workers.
* **Geographic Seeding (`populate_cities_table.py`)**:
Filters target cities by population parameters and queries the Nominatim API to extract and store geographic bounding boxes.
* **Image Acquisition (`download_pipeline.py`)**:
Executes concurrent asynchronous requests to the Mapillary API. Implements coordinate grid tiling and Nominatim POI targeting to optimize image retrieval density.
* **Pedestrian Detection (`detection_pipeline.py`)**:
Processes image batches through YOLOv8. Extracts bounding box crops of detected persons and logs performance metrics. Deletes source images lacking target classes.
* **Clothing Segmentation (`clothing_analysis_pipeline.py`)**:
Ingests person crops into Detectron2. Evaluates Mask R-CNN segmentation against Keypoint R-CNN anatomical constraints to eliminate invalid garments. Calculates HSV color distribution, spatial area ratios, and Laplacian texture variance.
* **Dimensionality Reduction (`generate_feature_matrix.py`)**:
Aggregates spatiotemporal and visual features. Trains PyTorch autoencoders to compress high-dimensional item and outfit representations into 2D latent space coordinates.
* **Data Visualization (`visualize_matrices.py`)**:
Consumes output matrices to render 8K resolution scatter plots mapped by clothing category and outfit composition.

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