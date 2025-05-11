"""
Semi-supervised training of YOLOv8 model on staff and non-staff images.
"""
import yaml
import subprocess
import os
import shutil

from dotenv import load_dotenv
from ultralytics import YOLO , settings
from roboflow import Roboflow

load_dotenv()

# Update dataaset directory
settings.update({
    "datasets_dir": "./data",
    "mlflow": True,
})

def model_training()->None:
    """
    Train the semi-supervised YOLOv8 model on staff and non-staff images.
    
    Returns:
        None
    """
    # Clear the previous logging
    shutil.rmtree("runs", ignore_errors=True)
    
    # Load pretrained YOLOv8 model
    model = YOLO("YOLOv8s.pt")

    # Train the model on a custom dataset
    model.train(
        data="./data/data.yaml", 
        epochs=20,
        imgsz=640
    )


def upload_model()->None:
    """
    Upload the trained model to Roboflow.
    
    Returns:
        None
    """
    # load configuration file 
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    workspace = config['workspace']
    project = config['project']
    version = config['version']
    
    # Initialize Roboflow
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace(workspace).project(project)
    version = project.version(version)
    
    # Upload the model to Roboflow
    version.deploy(model_type="yolov8",
                   model_path="runs/detect/train/weights",
                   filename="best.pt")
        
if __name__ == "__main__": 
    # Train the model
    model_training()
    
    # Upload the model to Roboflow
    upload_model()