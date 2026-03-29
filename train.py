import ultralytics
ultralytics.checks()
from ultralytics import YOLO
# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="datasets/medical-pills/medical-pills.yaml", epochs=20, imgsz=640)
