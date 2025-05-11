"""
Download and prepare datasets for training and evaluation.
"""
import os 
import subprocess
import yaml

def download_dataset(workspace: str, project: str, version: int)-> None:
    """
    Download the dataset from Roboflow. 
    
    Args:
        workspace (str): The name of the Roboflow workspace. 
        project (str): The name of the Roboflow project. 
        version (int): The version of the dataset to download.
    
    Returns:
        None
    """
    # Create the dataset directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data", exist_ok=True)
        
    # Empty the dataset directory if it exists
    os.system("rm -rf data/*")
    
    # Download the dataset from Roboflow
    subprocess.run([
        "roboflow", "download", "-f", "yolov8", "-l", "data", 
        f"{workspace}/{project}/{version}"    
    ], check=True)

if __name__ == "__main__":
    # Load yaml file 
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    download_dataset(config['workspace'], config['project'], config['version'])