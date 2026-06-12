"""
Model class to predict toxicity type of the messages. 
"""

import ast
import time

import torch
import logging

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from scipy.sparse import hstack

from services.text_preprocessor import (LinearSVMPreprocessor,
                                   LinearSVMPreprocessorSI,
                                   LinearSVMPreprocessorRaw,
                                   BertPreprocessor)

from services.text_preprocessor import TextPreprocessor
from core.config import MODEL_CONFIG
from services.utils import load_config, load_pickle, load_local_encoder, load_encoder
import tempfile
from services.utils import Boto3Reader

from prometheus_client import Counter, Histogram
from transformers import BertForSequenceClassification, BertTokenizer


# prometheus info: 
MODEL_INFERENCE_DURATION = Histogram(
    "model_inference_duration_seconds",
    "Time spent on model preprocessing + inference (per model)",
    ["model_id"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

MODEL_INFERENCE_TOTAL = Counter(
    "model_inference_total",
    "Total number of inference calls, labelled by model and predicted class",
    ["model_id", "prediction_label"],
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# logger: 
logger = logging.getLogger(__name__)

class Model: 
    def __init__(self, config_path=MODEL_CONFIG, worker_id=0):
        self.worker_id = worker_id
        self.config_path = config_path
        self.config_model = None
        self._label_decode: Dict[int, str] = {}
        
        # load passed config: 
        self._set_config()
        self._load_weights()
        self._load_optional_encoder()
        self._load_label_decode()

    @property
    def model_id(self) -> str:
        return self.config_model.get("description", f"model_{self.worker_id}")

    @property
    def is_multilabel(self) -> bool:
        return bool(self.config_model.get("is_multilabel", False))

    def _set_config(self): 
        """Select current model and encoders from the config"""

        base_config = load_config(self.config_path)

        # Find the predictor config in available_predictors
        # by using current worker id 
        # (worker id is for multiprocessing, for the future):
        config_model: dict = [conf for conf in base_config.get("predictors")][self.worker_id]

        self.config_model = config_model
        self.storage_type = config_model.get("storage_type")

    def _load_weights(self):
        """Loading weights and encoders if passed"""

        self._model_path = self.config_model.get("model_path")

        if self._model_path.endswith(".pkl"):
            self._model_weights = load_pickle(self.config_model)
        else: 
            self.load_custom_weights()
    
    def _load_optional_encoder(self): 
        """Method to load encoder if passed in config, otherwise set to None"""

        if self.config_model.get("encoder_path") is not None:
            self.encoder = load_encoder(config=self.config_model)
        else: 
            self.encoder = None

    def _load_label_decode(self):
        """Load integer→label name mapping when label_map_path is configured."""
        label_map_path = self.config_model.get("label_map_path")
        if not label_map_path:
            return
        full_path = _PROJECT_ROOT / Path(label_map_path)
        try:
            with open(full_path, "r", encoding="utf-8") as fh:
                mapping = ast.literal_eval(fh.read())
            self._label_decode = {int(k): v for k, v in mapping.get("decode", {}).items()}
        except Exception as exc:
            logger.warning("Could not load label map from %s: %s", full_path, exc)

    @abstractmethod 
    def load_custom_weights(self):
        """Method to load any of non-pickle models, e.g. from torch, tensorflow, etc."""
        
        pass

    @abstractmethod
    def preprocess(self, text: str) -> Any:
        """
        Must return model input array, e.g. sparse matrix for classic ml models,
        tensor for deep learning models, etc."""
        pass

    @abstractmethod
    def predict(self, inputs):
        """Do prediction based on input and return pred"""
        pass

    # def predict_full(self, text: str) -> Tuple[str, int, str, float]:
    #     """
    #     Preprocess + predict with timing.
    #     Returns (model_id, prediction_int, prediction_label, processing_time_ms).
    #     """

    #     start = time.perf_counter()
    #     inputs = self.preprocess(text)
    #     pred_int = int(self.predict(inputs))
    #     elapsed_ms = (time.perf_counter() - start) * 1000

    #     if self.is_multilabel:
    #         pred_label = self._label_decode.get(pred_int, str(pred_int))
    #     else:
    #         pred_label = "toxic" if pred_int == 1 else "non_toxic"

    #     MODEL_INFERENCE_DURATION.labels(model_id=self.model_id).observe(elapsed_ms / 1000.0)
    #     MODEL_INFERENCE_TOTAL.labels(model_id=self.model_id, prediction_label=pred_label).inc()

    #     return (self.model_id, pred_int, pred_label, elapsed_ms)
    
    def predict_log_prometheus(self, text: str) -> Tuple[str, int, str, float]:
        """
        Preprocess + predict with timing + Prometheus 
        Returns (model_id, prediction_int, prediction_label, processing_time_ms).
        """

        start = time.perf_counter()
        inputs = self.preprocess(text)
        pred_int = int(self.predict(inputs))
        elapsed_ms = (time.perf_counter() - start) * 1000

        if self.is_multilabel:
            pred_label = self._label_decode.get(pred_int, str(pred_int))
        else:
            pred_label = "toxic" if pred_int == 1 else "non_toxic"

        MODEL_INFERENCE_DURATION.labels(model_id=self.model_id).observe(elapsed_ms / 1000.0)
        MODEL_INFERENCE_TOTAL.labels(model_id=self.model_id, prediction_label=pred_label).inc()

        return (self.model_id, pred_int, pred_label, elapsed_ms)


class LinearSVMModel(Model):
    """Pickle model assuming use of encoder and different types of preprocessors"""
    
    def __init__(self, 
                 config_path=MODEL_CONFIG, 
                 worker_id=0,
                 preprocessor_type="LinearSVMPreprocessorSI"
                 ):
        super().__init__(config_path, worker_id)

        # text preprocessor is additional for LinearSVM 
        # it composes lexical features: 
        self.text_preprocessor = None

        # dict for classic ml models only,
        # for different combinations switching in preprocess() method:
        self.PREPROCESSOR_REGISTRY = {
            "LinearSVMPreprocessor": LinearSVMPreprocessor,
            "LinearSVMPreprocessorSI": LinearSVMPreprocessorSI,
            "LinearSVMPreprocessorRaw": LinearSVMPreprocessorRaw
        }
        
        # instance preprocessor class and load its data from config: 
        preprocessor_class: TextPreprocessor = self.PREPROCESSOR_REGISTRY.get(
                preprocessor_type
            )
        self.text_preprocessor = preprocessor_class(config=self.config_model)

    def preprocess(self, text: str):
        """The main preprocessor logic to compose input modelarrays"""

        # preprocess in text domain:  
        text_preprocessed, numc_features = self.text_preprocessor.preprocess(text)
        
        # encode text: 
        encoded_text = self.encoder.transform([text_preprocessed])

        # hstack if has num_features: 
        if numc_features is not None: 
            inputs = hstack([encoded_text, numc_features])
        else: 
            inputs = encoded_text

        return inputs

    def predict(self, inputs) -> int: 
        """Do prediction based on input sparce matrix"""
        
        pred = self._model_weights.predict(inputs)

        # return pred
        return float(pred[0])


class BertClassifierModel(Model):
    """
    Wraps a fine-tuned BertForSequenceClassification model.
    """

    def __init__(self, config_path=MODEL_CONFIG, worker_id=0):
        # torch / transformers are loaded lazily inside load_custom_weights: 
        self._bert_model = None
        self._tokenizer = None
        self._device = None
        self._max_len: int = 128
        super().__init__(config_path, worker_id)
        self.text_preprocessor = BertPreprocessor(config=self.config_model)

    def load_custom_weights(self):
        """Load BertForSequenceClassification and its tokenizer."""

        self._max_len = int(self.config_model.get("max_len", 128))
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.storage_type == "local":
            model_dir = _PROJECT_ROOT / Path(self._model_path)
            if not model_dir.exists():
                raise FileNotFoundError(f"BERT model directory not found: {model_dir}")

        elif self.storage_type == "s3":

            bucket = self.config_model.get("bucket_name", "toxic-messages-bucket-1")
            model_prefix = self._model_path.rstrip("/")  # e.g. "models/v1"

            tmp = Path(tempfile.mkdtemp(prefix="bert_s3_"))
            reader = Boto3Reader(config_path=self.config_path, bucket_name=bucket)

            bert_files = self.config_model.get("bert_files")

            for fname in bert_files:
                s3_key = f"{model_prefix}/{fname}"
                logger.info("Downloading s3://%s/%s …", bucket, s3_key)
                obj = reader.get_boto3_obj(s3_key)
                local_path = tmp / fname
                with open(local_path, "wb") as fh:
                    fh.write(obj["Body"].read())
                logger.info("Saved %s (%d bytes)", local_path, local_path.stat().st_size)

            model_dir = tmp

        else:
            raise ValueError(f"Unsupported storage_type for BertClassifierModel: {self.storage_type}")

        self._bert_model = BertForSequenceClassification.from_pretrained(str(model_dir))
        self._bert_model.to(self._device)
        self._bert_model.eval()

        self._tokenizer = BertTokenizer.from_pretrained(str(model_dir))
        logger.info("BertClassifierModel loaded from %s on %s", model_dir, self._device)

    def preprocess(self, text: str):
        """Tokenize text and return (input_ids, attention_mask) tensors on the model device."""
        
        # simple preprocess: 
        text_preprocessed = self.text_preprocessor.preprocess(text)
        
        encoding = self._tokenizer(
            text_preprocessed,
            max_length=self._max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            encoding["input_ids"].to(self._device),
            encoding["attention_mask"].to(self._device),
        )

    def predict(self, inputs) -> int:
        """Run forward pass and return integer class index."""

        input_ids, attention_mask = inputs
        with torch.no_grad():
            outputs = self._bert_model(input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=1).item()
        return int(pred)
