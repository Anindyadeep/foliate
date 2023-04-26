import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import List
from flask import jsonify
from src.config import default_config
from torchvision.transforms.functional import to_tensor


def get_df_infos(disease_info_path: str, supplement_info_path: str) -> List[pd.DataFrame]:
    disease_info = pd.read_csv(disease_info_path, encoding="cp1252")
    supplement_info = pd.read_csv(supplement_info_path, encoding="cp1252")
    return disease_info, supplement_info


def predict(
    model: torch.nn.Module,
    image_path: str,
    disease_info: pd.DataFrame,
    supplement_info: pd.DataFrame,
) -> int:
    image = Image.open(image_path)
    image = image.resize(default_config.IMAGE_RESIZE_SHAPE)
    input_data = to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)

    title = disease_info["disease_name"][index]
    description = disease_info["description"][index]
    prevent = disease_info["Possible Steps"][index]
    image_url = disease_info['image_url'][index]
    
    supplement_buy_link = supplement_info["buy link"][index]
    supplement_name = supplement_info["supplement name"][index]
    supplement_image_url = supplement_info["supplement image"][index]

    return {
        "title": title,
        "description": description,
        "prevent": prevent,
        "image_url" : image_url, 
        "pred": int(index),
        "supplement_name": supplement_name,
        "supplement_image_url": supplement_image_url,
        "supplement_buy_link": supplement_buy_link,
    }

