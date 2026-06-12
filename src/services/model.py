"""
Model class to predict toxicity type of the messages. 
"""

from abc import ABC, abstractmethod
from typing import Any
from scipy.sparse import hstack
import logging

from services.text_preprocessor import (LinearSVMPreprocessor, 
                                   LinearSVMPreprocessorSI, 
                                   LinearSVMPreprocessorRaw)

# from services.preprocessor import Preprocessor, PREPROCESSOR_REGISTRY
from services.text_preprocessor import TextPreprocessor
from core.config import MODEL_CONFIG
from services.utils import load_config, load_pickle, load_local_encoder, load_encoder

logger = logging.getLogger(__name__)


class Model: 
    def __init__(self, config_path=MODEL_CONFIG, worker_id=0):
        self.worker_id = worker_id
        self.config_path = config_path
        self.config_model = None
        
        # load passed config: 
        self._set_config()
        self._load_weights()
        self._load_optional_encoder()

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