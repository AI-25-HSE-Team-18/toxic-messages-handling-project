"""
Model class to predict toxicity type of the messages. 
"""

from scipy.sparse import hstack
from services.preprocessor import Preprocessor, PREPROCESSOR_REGISTRY
from services.utils import load_config, load_local_model, load_local_encoder


class Model: 
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.config_model = None
        self.config_encoder = None

        # usable objects (model weights, encoder object): 
        self.model_weights = None # model weights is using predict() method 
        self.encoder = None

        # current model, encoder info: 
        self.predictor_id = None
        self.encoder_id = None

        # pass preprocessor: 
        self.text_preprocessor = None

class PickleModel(Model):
    def __init__(self, config_path='config.json'):
        super().__init__(config_path)
        
        # load passed config: 
        self.init_config()
        # load model, encoder objects: 
        self.load_models()

    def init_config(self): 
        """Select current model and encoders from the config"""

        base_config = load_config(self.config_path)
        
        # get defaults and set up the current values: 
        default_predictor = base_config.get("default_predictor")
        default_encoder = base_config.get("default_encoder")
        
        config_model: dict = [conf for conf in base_config.get("available_predictors") 
                        if conf.get("predictor_id")==default_predictor][0]
        config_encoder: dict = [conf for conf in base_config.get("available_encoders") 
                        if conf.get("encoder_id")==default_encoder][0]
        
        self.config_model = config_model
        self.config_encoder = config_encoder
        self.predictor_id = default_predictor
        self.encoder_id = base_config.get("default_encoder")
        self.storage_type = config_model.get("storage_type")

    def load_models(self):
        if self.storage_type == "local":
            # load model: 
            self.model_weights = load_local_model(self.config_model.get("model_path"))
            # load encoder: 
            self.encoder = load_local_encoder(self.config_encoder.get("encoder_path"))
        else:
            raise NotImplementedError(
                f"Storage type '{self.storage_type}' is not implemented yet"
                )
        
        # instance preprocessor class and load its data: 
        # strange but.. it works: 
        preprocessor_class: Preprocessor = PREPROCESSOR_REGISTRY.get(
                self.config_model.get("preprocessor_type")
            )
        self.text_preprocessor = preprocessor_class()
        
    def preprocess(self, text: str) -> str:

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
        
        pred = self.model_weights.predict(inputs)

        # return pred
        return float(pred[0]) 