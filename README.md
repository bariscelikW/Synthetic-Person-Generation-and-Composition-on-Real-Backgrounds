# Video Dataset Generation Pipeline

This repository contains a complete pipeline for generating synthetic datasets from video footage using diffusion models and automated frame extraction.

## 📁 Project Structure

The project consists of three main components:

1. **`extract_videos_frames.py`** - Video frame extraction utility
2. **`diffusion_part.ipynb`** - Diffusion model training for synthetic person generation
3. **`generate_dataset_scratch.ipynb`** - Dataset composition pipeline

---

## 🚀 Quick Start

### Prerequisites

Make sure you have Python 3.8+ installed. Then install the required dependencies:

```bash
# for all notebooks and py file
pip install -r requirements.txt
```

---

## 📹 1. Frame Extraction (`extract_video_frames.py`)

Extract frames from CCTV or security camera footage at a specified frame rate.

### Features
- Extract frames at custom FPS (frames per second)
- Automatic quality optimization (JPEG quality: 88)
- Progress tracking
- Optional frame limit

### Usage

```bash
# Extract 1 frame per second
python extract_video_frames.py input_video.mp4 -o output_frames -f 1.0

# Extract 1 frame every 5 seconds (0.2 fps)
python extract_video_frames.py input_video.mp4 -o output_frames -f 0.2

# Extract maximum 1000 frames
python extract_video_frames.py input_video.mp4 -o output_frames -f 1.0 --max 1000
```

### Arguments
- `video`: Path to input video file (mp4, avi, mov, etc.)
- `-o, --output`: Output folder for frames (default: ./frames)
- `-f, --fps`: Target FPS to extract (default: 1.0)
- `--max`: Maximum number of frames to extract (optional)

### Requirements
```
opencv-python
```

---

## 🎨 2. Diffusion Model Training (`diffusion_part.ipynb`)

Train a diffusion model to generate synthetic person images. This is Part 1 of the dataset generation pipeline.

### Overview
This notebook implements a diffusion-based generative model to create synthetic person images. The generated synthetic persons are later used in the dataset composition process.

### Key Features
- Custom diffusion model architecture
- PyTorch-based training pipeline
- Progress tracking with tqdm
- Image preprocessing and augmentation
- Model checkpointing

### Workflow
1. **Data Preparation**: Load and preprocess person images
2. **Model Training**: Train the diffusion model
3. **Generation**: Generate 500 synthetic person images
4. **Export**: Save generated images for use in dataset composition

### Requirements
```
torch
torchvision
Pillow
tqdm
google-colab (optional, only for Google Colab)
pyyaml==6.0.2
```

### Output
The notebook generates **500 synthetic person images** that will be used in the next step.

> **Note:** `diffusion_part.ipynb` was developed and trained on **Google Colab**. File paths in the notebook (e.g., `/content/drive/MyDrive/...`) are Colab-specific. If you run it locally, update these paths to match your local directory structure.

---

## 🏗️ 3. Dataset Generation (`generate_dataset_scratch.ipynb`)

Compose synthetic persons with real backgrounds to create a complete dataset. This is Part 2 of the pipeline.

### Overview
After training the diffusion model and generating synthetic persons, this notebook combines them with annotated background images to create a realistic dataset.

### Key Features
- **CVAT Annotation Support**: Converts CVAT RLE masks to NumPy masks
- **Automated Composition**: Places synthetic persons on walkable areas
- **Mask Processing**: Handles segmentation masks for accurate placement
- **Quality Control**: Validates walkable area percentages

### Workflow

#### Step 1: Prepare Annotation Files
The notebook expects CVAT annotations in COCO format:
- Location: `background_with_masks/annotations/instances_default.json`
- Alternative: `cvat_export/*.json`

#### Step 2: Convert CVAT Masks
Automatically converts CVAT RLE format to NumPy masks:
- Processes 100+ background images
- Extracts walkable areas from segmentation masks
- Saves to `segmentation_cache_cvat/` directory
- Generates metadata with statistics

#### Step 3: Compose Dataset
Places synthetic persons on backgrounds:
- Uses walkable area masks for accurate placement
- Ensures realistic positioning
- Maintains proper scaling and proportions

### Requirements
```
numpy
opencv-python
matplotlib
tqdm
ultralytics
```

### File Structure
```
project/
├── background_with_masks/
│   ├── annotations/
│   │   └── instances_default.json
│   └── images/
├── synthetic_persons/         # Output from diffusion_part.ipynb
├── segmentation_cache_cvat/   # Generated masks
│   ├── frame_*.npy
│   └── metadata.json
└── final_dataset/             # Final composed images
```


---

## 🔄 Complete Pipeline Workflow

### Step-by-Step Process

1. **Extract Frames from Video**
   ```bash
   python extract_video_frames.py cctv_footage.mp4 -o backgrounds -f 0.2
   ```

2. **Annotate Backgrounds** (External)
   - Use CVAT to annotate walkable areas
   - Export annotations in COCO format
   - Place in `background_with_masks/annotations/`

3. **Train Diffusion Model**
   - Open `diffusion_part.ipynb`
   - Run all cells to train the model
   - Generate 500 synthetic persons

4. **Generate Dataset**
   - Open `generate_dataset_scratch.ipynb`
   - Run all cells to compose final dataset
   - Output: Synthetic persons placed on real backgrounds

---

## 📊 Output Format

The final dataset contains:
- **Background images**: Real frames from video footage
- **Synthetic persons**: AI-generated humans
- **Segmentation masks**: Walkable area annotations
- **Metadata**: Statistics and placement information

---

## 📊 Presentation and Results

Detailed information about our methodology, experiments, and final results can be found in our project presentation:
[View Project Presentation (Google Drive)](https://drive.google.com/file/d/1Sd2_wMA-6koPYTUuXsrlSDvrcTzxo-De/view?usp=sharing)

---
This project is a Deep Learning Class Project developed by Barış Çelik and Sude Naz Öztürk. All ideas and contributions are shared equally.
