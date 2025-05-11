"""
Download and prepare datasets for training and evaluation.
"""
import os
import shutil
import subprocess
import yaml

from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()


def download_dataset(workspace: str, project: str, version: int) -> None:
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
    shutil.rmtree("data", ignore_errors=True)

    # Download the dataset from Roboflow
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace(workspace).project(project)
    version = project.version(version)
    version.download(model_format="yolov8", location="./data")


if __name__ == "__main__":
    # Load yaml file
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    download_dataset(config["workspace"], config["project"], config["version"])
