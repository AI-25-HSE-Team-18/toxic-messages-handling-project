import os 
import json
import pickle 
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

def load_config(config_path: str = "../config.json"):
    
    # path = BASE_DIR / Path(config_path)
    path = config_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    for key in ["available_predictors", "available_encoders", "default_predictor", "default_encoder"]:
        if key not in config:
            raise ValueError(f"Missing key '{key}' in config file")
    
    return config

def load_local_model(model_path: str): 
    
    # TODO: fix strange paths 
    model_path = BASE_DIR.parent.parent / Path(model_path)

    # model_path = Path(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model

def load_local_encoder(encoder_path: str): 

    # TODO: fix strange paths 
    encoder_path = BASE_DIR.parent.parent /Path(encoder_path)
    if not encoder_path.exists():
        raise FileNotFoundError(f"Model file not found at: {encoder_path}")
    
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    return encoder

def load_loc_enc_json(path: str) -> dict: 
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config