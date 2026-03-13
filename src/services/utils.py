import os 
import json
import pickle 
import boto3
import io
import pandas as pd 

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


class Boto3Base: 
    def __init__(self, config_path='src/config_s3.json'):
        self.config: dict = load_config(config_path)
    
    def get_client(self):
        
        endpoint_url = self.config.get('aws_endpoint_url')
        access_key, secret_key = self.config.get('aws_access_key_id'), \
                                self.config.get('aws_secret_access_key')
        
        session = boto3.session.Session()
        s3_client = session.client(
            service_name='s3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        self.s3_client = s3_client

        return s3_client
    
class Boto3Reader(Boto3Base): 
    def __init__(self, config_path='src/config_s3.json', bucket_name='toxic-messages-bucket-1'):
        super().__init__(config_path)

        self.get_client()
        self.bucket_name = bucket_name


    def get_boto3_obj(self, object_path: str) -> bytes: 
        """:param str object_path: full object name without bucket"""
        
        return self.s3_client.get_object(Bucket=self.bucket_name, Key=object_path)
    
    def get_boto3_csv(self, object_path: str) -> pd.DataFrame: 
        obj = self.get_boto3_obj(object_path)['Body'].read()
        df = pd.read_csv(io.BytesIO(obj), encoding='utf8', index_col=0)

        return df
    
    def get_boto3_json(self, object_path: str): 
        obj = self.get_boto3_obj(object_path)['Body'].read()
        return json.loads(obj)