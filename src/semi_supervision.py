"""
Semi-supervised training of YOLOv8 model on staff and non-staff images.
"""
from ultralytics import YOLO , settings
import mlflow
from dotenv import load_dotenv

load_dotenv()

# Update dataaset directory
settings.update({
    "datasets_dir": "./data",
    "mlflow": True,
})
mlflow.set_tracking_uri("https://dagshub.com/tchiayan/footfallcamp.mlflow")

# Load pretrained YOLOv8 model
model = YOLO("YOLOv8s.pt")

# Train the model on a custom dataset
model.train(
    data="./data/data.yaml", 
    epochs=10,
    imgsz=640
)