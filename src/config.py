import os 
from typing import Tuple
from dataclasses import dataclass


@dataclass(kw_only=True)
class Config:
    # Model specific configuration
    MODEL_NUM_CLASSES: int = 39
    LOCAL_DOWNLOADED_MODEL_PATH: str = "models/model.pt"
    
    # Image specific configuration
    IMAGE_UPLOAD_PATH : str = "static/uploads/"
    IMAGE_RESIZE_SHAPE: Tuple = (224, 224)
    TENSOR_INPUT_DATA_SHAPE: Tuple = (-1, 3, 224, 224)

    # default metadata configuration
    DISEASE_INFO_CSV_PATH: str = "data/disease_info.csv"
    SUPPLEMENT_INFO_CSV_PATH: str = "data/supplement_info.csv"

    # AWS specific configuration
    AWS_ACCESS_KEY_ID : str = os.environ['AWS_ACCESS_KEY_ID']
    AWS_SECRET_ACCESS_KEY : str = os.environ['AWS_SECRET_ACCESS_KEY']
    AWS_ACCESS_REGION : str = "ap-south-1"
    AWS_BUCKET_NAME: str = "rootskart-users"
    AWS_BUCKET_KEY: str = "plant-disease-model/plant_disease_model_1.pt"


default_config = Config()
