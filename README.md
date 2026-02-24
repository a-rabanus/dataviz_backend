# Urban Fashion Mapping Pipeline

Project inspired by meme starterkits, applying data science to urban fashion trends by analyzing street-level imagery and generating 2D semantic embedding spaces.

## Pipeline Implementation

The system consists of independent modules synchronized via a SQLite database `pipeline.db`.

### 1. Geographic Scoping

* **Scripts:** `pipelines/cities_generator.py`, `image_tools.py`
* **Data Source:** `assets/worldcities.csv` (SimpleMaps) and Nominatim API.
* **Logic:** Filters for cities with population > 500k. Generates JSON bounding boxes for target areas.
* **Output:** Bounding boxes stored in the `cities` table.

### 2. Image Acquisition

* **Scripts:** `run_download_pipeline.py`
* **API:** Mapillary API.
* **Logic:** Breaks city bounding boxes into subtiles (max 2sqkm) for API compliance.
* **Storage:** Images saved to `mapillary_images/`. Metadata recorded in `images_detected`.

### 3. Pedestrian Detection & Cropping

* **Scripts:** `run_detection_pipeline.py`
* **Model:** YOLOv8.
* **Logic:** Detects pedestrians in downloaded imagery. Uses detection bounding boxes to crop individuals.
* **Storage:** Cropped images saved to `cropped_people/`. Entries added to `person_detected`.

### 4. Clothing Analysis

* **Scripts:** `run_clothing_analysis.py`
* **Model:** Detectron2 (trained on DeepFashion2).
* **Logic:** Extracts clothing type, color, and texture from cropped images.
* **Output:** Attributes stored in `clothing_item_detected`.

### 5. Feature Matrix Generation

* **Scripts:** `generate_feature_matrices.py`
* **Method:** Deep Autoencoder.
* **Logic:** Compresses high-dimensional clothing vectors into a 2D latent space.
* **Output:** `.csv` file formatted for frontend visualization.

---

## Frontend & UI

A Svelte 5 application utilizing Cosmos.gl for high-performance WebGL point cloud rendering.

### Architecture

* **Physics-Driven Graph:** Transitioned from static geometric layouts to a force-directed system. Points represent clothing items; layout is determined by simulated springs and repulsion.
* **Lifecycle Management:** Uses Svelte `onMount` and `bind:this` to manage the imperative Cosmos.gl engine within a declarative framework.
* **Memory Management:** Implements `graph.destroy()` on component unmount to clear GPU resources.

### Tech Stack

| Library | Version | Detail |
| --- | --- | --- |
| svelte | 5.46.0 | Snippet/Children patterns for layout management |
| @cosmos.gl/graph | 2.6.2 | GPU-accelerated graph simulation |
| vite | 6.4.1 | Build and import analysis |

---

## Performance and Monitoring

The following tools are used to identify resource-heavy elements and optimize the visualization:

* [rStats](https://spite.github.io/rstats/)
* [stats.js](https://github.com/mrdoob/stats.js)

---

## Environment Configuration

Create a `.env` file in the root directory:

```ini
MAPILLARY_ACCESS_TOKEN="YOUR_TOKEN_HERE"

```

## Database Schema

The system uses `src/db/db_utils.py` for all database interactions. The schema includes:

* `cities`: Geographic boundaries and scan status.
* `images_detected`: Raw Mapillary image metadata.
* `person_detected`: Links cropped images to original sources.
* `clothing_item_detected`: Extracted features and attributes.

---

## References

* [DeepFashion2 Dataset](https://github.com/switchablenorms/DeepFashion2)
* [Figma Design Prototype](https://www.figma.com/design/xWFdUtSlsngfeJKi33LZEx/Portfolio?node-id=0-1&t=JhhHZ7DehM6H0LSy-1)