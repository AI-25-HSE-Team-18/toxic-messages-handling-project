"""Implement different versions to predict values. 
Preprocessors can be slightly flexible a
"""

from abc import ABC, abstractmethod
import pymorphy3 as pymorphy2

from core.config import MODEL_CONFIG
from stop_words import get_stop_words
from pathlib import Path

from services.text_utils import map_noninformatives, map_punctuation, \
                                map_profanity, map_emoji_emoticons, del_punct_tokens, get_num_features
from services.utils import Boto3Reader, load_loc_enc_json, load_s3_enc_json, load_s3_txt


BASE_DIR = Path(__file__).resolve().parent


class TextPreprocessor: 
    """Text domain preprocessor: cleaning, mapping, lemmatization, etc."""

    def __init__(self): 
        # self.use_num_features = use_num_features
        pass 
    
    @abstractmethod
    def preprocess(self, text: str) -> str:

        return text  

class LinearSVMPreprocessor(TextPreprocessor):
    """Preprocessor especially for LinearSVM model
    Using different text utils from text_utils"""
    
    def __init__(self, config: dict):
        """Load usable objects"""
        # super().__init__(use_num_features)
        super().__init__()

        storage_type = config.get("storage_type")
        data_dir = Path(config.get("additional_data_path")) if \
            config.get("additional_data_path") is not None else None

        # specific config files: 
        profanity_file = data_dir / 'bad_words_lemmas.txt'
        enc_emoj_file = data_dir / "encoding_emoji.json"
        enc_emot_file = data_dir / "encoding_emoticon.json"
        enc_prof_file = data_dir / "encoding_profanities.json"
        enc_rep_file = data_dir / "encoding_rep_punct.json"
        enc_sep_file = data_dir / "encoding_sep_punct.json"

        if storage_type == "local":

            # profanities text file: 
            with open(profanity_file, 'r', encoding="utf-8") as f:
                profanities = [l.strip() for l in f.readlines()]
            self.profanities: list = profanities
            
            # encoding dicts: 
            self.enc_emoj: dict = load_loc_enc_json(enc_emoj_file)
            self.enc_emot: dict = load_loc_enc_json(enc_emot_file)
            self.enc_prof: dict = load_loc_enc_json(enc_prof_file)
            self.enc_rep: dict = load_loc_enc_json(enc_rep_file)
            self.enc_sep: dict = load_loc_enc_json(enc_sep_file)

        elif storage_type == "s3":
            
            print('enc_emoj_file', str(enc_emoj_file))
            self.enc_emoj: dict = load_s3_enc_json(config, str(enc_emoj_file).replace('\\','/'))
            self.enc_emot: dict = load_s3_enc_json(config, str(enc_emot_file).replace('\\','/'))
            self.enc_prof: dict = load_s3_enc_json(config, str(enc_prof_file).replace('\\','/'))
            self.enc_rep: dict = load_s3_enc_json(config, str(enc_rep_file).replace('\\','/'))
            self.enc_sep: dict = load_s3_enc_json(config, str(enc_sep_file).replace('\\','/'))

            self.profanities: list = load_s3_txt(config, str(profanity_file).replace('\\','/'))

        self.morph = pymorphy2.MorphAnalyzer()
        self.stop_words = set(get_stop_words('russian'))

    def preprocess(self, 
                    text: str,
                    mapping=True,
                    del_stop_words=False,
                    del_punct=True,
                    use_num_features=True
                    ) -> str: 
        
        """Returns preprocessed text
        This method is for flexible changing of preprocessing steps.
        Turn flags on your custom steps in subclasses for faster switches 
        between the models and their methods.
        """
        
        mapping_dict = {
            "url": "[URL]",
            "num": "[NUM]", # 
            "mention": "[MNT]",
            "hashtag": "[HSG]",
            "email": "[EML]",
            "repeat_punct": "[RPP]",
        }

        # if not(mapping)

        if mapping:
            # mapping steps from text part of text_domain_features_0.ipynb: 
            text = map_noninformatives(text, mapping_dict)
            text = map_emoji_emoticons(text, (self.enc_emoj, self.enc_emot))
            text = map_punctuation(text, self.enc_rep, self.enc_sep)
            text = map_profanity(self.morph, text, self.profanities, self.enc_prof)

            # delete stop_words: 
            if del_stop_words:
                text = ' '.join(word for word in text.split() if word.lower() not in self.stop_words)
            
            if del_punct: 
                text = del_punct_tokens(text)

        # add numeric features as sparse matrix: 
        if use_num_features: 
            num_features = get_num_features(text)
        else: 
            num_features = None

        return (text, num_features)

class LinearSVMPreprocessorSI(LinearSVMPreprocessor): 
    def __init__(self, config: dict):
        super().__init__(config)

    def preprocess(self, text, mapping=True, del_stop_words=False, del_punct=True, use_num_features=True):
        return super().preprocess(text, mapping, del_stop_words, del_punct, use_num_features)
    
class LinearSVMPreprocessorRaw(LinearSVMPreprocessor): 
    def __init__(self, config: dict):
        super().__init__(config)

    def preprocess(self, text, mapping=False, del_stop_words=False, del_punct=False, use_num_features=True):
        return super().preprocess(text, mapping, del_stop_words, del_punct, use_num_features)

class BertPreprocessor(TextPreprocessor):
    """Basic preprocessor for BERT.
    Does not include any specific steps because 
    BERT handles the raw data at the train step. 
    """
    def __init__(self, config: dict):
        super().__init__()

    def preprocess(self, text: str) -> str:
        # Return cleaned text as-is; tokenization logic lives in the tokenizer
        return text.strip()
    