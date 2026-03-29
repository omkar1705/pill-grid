# Pill Grid Detection Project

This project uses a YOLOv8 model to detect pills in a grid from images.

## Project Structure

```
.
├── add_data.py
├── make_synthetic_dataset_augmented.py
├── predict.py
├── train.py
├── yolo11n.pt
├── requirements.txt
├── datasets/
│   └── medical-pills/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       ├── labels/
│       │   ├── train/
│       │   └── val/
│       └── medical-pills.yaml
├── evaluation/
├── input_images/
├── output_dataset/
└── runs/
    └── detect/
        ├── predict/
        └── train/
```

## Setup

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd pill-grid
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Preparation

The project expects the data to be in YOLO format. The `datasets/medical-pills` directory shows an example of the expected structure.

- `images/train` and `images/val`: Training and validation images.
- `labels/train` and `labels/val`: Training and validation labels in `.txt` format.
- `medical-pills.yaml`: The dataset configuration file.

You can use the provided scripts to help with data preparation:

- `add_data.py`: Moves images and labels from a source directory to the training and validation directories with a specified split.
- `make_synthetic_dataset_augmented.py`: Creates an augmented dataset from a directory of input images.

### 2. Training

To train the model, run the `train.py` script:

```bash
python train.py
```

This will use the configuration from `datasets/medical-pills/medical-pills.yaml` to train the model. The training results, including the trained model weights, will be saved in the `runs/detect/train` directory.

### 3. Prediction

To make predictions on new images, you can use the `predict.py` script. You will need to modify the script to load your trained model and the images you want to predict on.

The `predict.py` script contains functions to load the model and perform inference. The `_get_yolo` function loads the model weights. By default, it loads from `runs/detect/train/weights/best.pt`. You can change this path to your trained model.

```python
# In predict.py
def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        import ultralytics

        ultralytics.checks()
        from ultralytics import YOLO

        weights = os.environ.get(
            "PILL_GRID_WEIGHTS",
            "runs/detect/train/weights/best.pt", # Make sure this points to your trained model
        )
        _yolo_model = YOLO(weights)
    return _yolo_model
```

You can then call the functions in `predict.py` with your own images to get detections.
