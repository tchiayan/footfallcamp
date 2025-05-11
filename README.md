# Vision Staff Identification
The main objectives of this project as follows:
1. Identifying frame in the clip have the stuff present 
2. Locate the staff xy coordinates when present in the clip

## Initial planning / ideas
Base on the requirements, the following decision was made
1. Use of `Roboflow` for easy data annotation. This can speed up the process of data 
labelling as it provide the GUI interface for bounding box creation. 
2. Use of `Ultralytics` tool to train `Yolov8` model. Assuming that this project is used
for internal showcase for non-profit, this will not incurs any charges or fee based on 
the license. Futhermore, the `yolov8` model is also compatible with roboflow for 
assisted labelling (Semi-supervison)
3. In this example, I'm using small model for `yolov8` as weak semi-supervise model. 
However, in real scenario, it should be larger model or more capability model for 
semi-supervise as data annotation quality can be improve. Data quality decides the
quality of the trained model. For the real deployment of the inference model, it depend 
on the situation, where the hardware capability is the contraint; there is a trade of 
between model accuracy performance and realtime inference latency. For this demo 
purpose, I just stick to same `yolov8s` for semi-supervision model for labelling and 
inference model. 
4. Use of `dvc` for data versioning. We can use the `roboflow` for versioning as well. 
Benefit of using `dvc` for versioning: (1) easily to make change to another annotation
provider (2) Easy for collabration and data sharing. (3) Can determine choice of the 
storage provide (has the ability to help to optimize the storage cost). (4) Can setup
the training pipeline and reproduce back the result. 
5. In this example, I'm also using `Dagshub`, just for this demo purpose to use 
`MLFlow` as experiment tracking purpose.

## Project Structure
The following is the main project structure.
```
├── data/               # Dataset files managed by DVC
├── runs/               # Training outputs and model checkpoints (local only)
├── src/                # Source code
│   └── semi_supervision.py  # Semi-supervised learning for data labeling
├── yolov8s.pt          # YOLOv8 small pre-trained model
├── dvc.yaml            # DVC pipeline
├── params.yaml         # Parameters / Hyperparameters Configuration
└── requirements.txt    # Project dependencies
```

## Getting Started

### Prerequisites
- Python 3.11 or higher
- Required packages: Install with `pip install -r requirements.txt`
- Add required environment variables as follow: 
```shell
MLFLOW_TRACKING_URI=<URL>
MLFLOW_EXPERIMENT_NAME=<STRING>
MLFLOW_TRACKING_USERNAME=<STRING>
MLFLOW_TRACKING_PASSWORD=<CREDENTIAL>
ROBOFLOW_API_KEY=<CREDENTIAL>
```

## Steps
1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Update DVC credential using following command: 
   ```
   dvc remote modify origin --local access_key_id <ACCESS_KEY_ID>
   dvc remote modify origin --local secret_access_key <SECRET_ACCESS_KEY>
   ```
4. Pull data using DVC: (optional)
   ```
   dvc pull
   ```

## Workflow
1. Step annotion: Use roboflow to annotate the image. In this demo, I annotated 
non-staff as well. `Roboflow` can easily filter out the unwanted class or create null
object frame (to improve the false positive rate). 
2. Once annotated, create the dataset with augmentation steps (Blurness, contrast, 
rotation, skewness, etc), which can also be tuned for better model performance. 
3. Update the `version` in `params.yaml` according to which dataset version that you 
want to train. 
4. Run the DVC pipeline using this command: `dvc repro`. In this demo, I'm only 
including `epochs` hyperparameters for tuning. For real project, it can be advance 
tuning strategy such as bayesian optimization, successive halving, hyperband etc. The
model experiments will be tracked in `Dagshub MLFLOW`
5. Repeat the step 1 until step 5 to improve the model, data quality. 

## Drawback
In this demo, it can't really represent the real case as there are limited sample 
for staff  (black t-shirt with staff logo, white t-shirt with staff logo, walking
back and forth in repeated direction, while other non-staff mostly static at a 
location). The model might be overfitted to this scenario only. 

## License
This project is licensed under the GNU General Public License v3.0 - see the 
[LICENSE](LICENSE) file for details.
