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

def model_training(epochs:int=10)->None:
    """
    Train the semi-supervised YOLOv8 model on staff and non-staff images.
    
    Args:
        epochs (int): The number of epochs to train the model. Default is 10.
        
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
        epochs=epochs,
        imgsz=640
    )


def upload_model(workspace:str, project:str, version:int)->None:
    """
    Upload the trained model to Roboflow.
    
    Args:
        workspace (str): The name of the Roboflow workspace.
        project (str): The name of the Roboflow project.
        version (int): The version of the dataset to upload the model to.

    Returns:
        None
    """
    # Initialize Roboflow
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace(workspace).project(project)
    version = project.version(version)
    
    # Upload the model to Roboflow
    version.deploy(model_type="yolov8",
                   model_path="runs/detect/train/weights",
                   filename="best.pt")
        
if __name__ == "__main__": 
    # load configuration file 
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    workspace = config['workspace']
    project = config['project']
    version = config['version']
    epochs = config['epochs']
    
    # Train the model
    model_training(epochs)
    
    # Upload the model to Roboflow
    upload_model(workspace, project, version)
