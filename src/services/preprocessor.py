"""Implement different versions to predict values. 
Preprocessors can be slightly flexible a
"""


import pymorphy3 as pymorphy2
from stop_words import get_stop_words
import numpy as np
from pathlib import Path
from services.text_utils import map_noninformatives, map_punctuation, \
                                map_profanity, map_emoji_emoticons, get_num_features
from services.utils import load_loc_enc_json


BASE_DIR = Path(__file__).resolve().parent

class Preprocessor: 
    def __init__(self): 
        # self.use_num_features = use_num_features
        pass 

    def preprocess(self, text: str):
        return text  

class LinearSVMTextPreprocessor(Preprocessor):
    """Preprocessor especially for LinearSVM model
    Using different text utils from text_utils"""
    
    def __init__(self):
        """Load usable objects"""
        # super().__init__(use_num_features)
        super().__init__()

        # load configs and meta, morphanalyzer: 
    
        data_dir = BASE_DIR / 'data/LinearSVMBinaryV0'

        # specific config files: 
        profanity_file = data_dir / 'bad_words_lemmas.txt'
        enc_emoj_file = data_dir / "encoding_emoji.json"
        enc_emot_file = data_dir / "encoding_emoticon.json"
        enc_prof_file = data_dir / "encoding_profanities.json"
        enc_rep_file = data_dir / "encoding_rep_punct.json"
        enc_sep_file = data_dir / "encoding_sep_punct.json"

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

        self.morph = pymorphy2.MorphAnalyzer()
        self.stop_words = set(get_stop_words('russian'))

    def preprocess(self, 
                   text: str,
                   del_stop_words=False) -> str: 
        """Returns preprocessed text"""
        
        mapping_dict = {
            "url": "[URL]",
            "num": "[NUM]", # 
            "mention": "[MNT]",
            "hashtag": "[HSG]",
            "email": "[EML]",
            "repeat_punct": "[RPP]",
        }

        # mapping steps from text part of text_domain_features_0.ipynb: 
        text = map_noninformatives(text, mapping_dict)
        text = map_emoji_emoticons(text, (self.enc_emoj, self.enc_emot))
        text = map_punctuation(text, self.enc_rep, self.enc_sep)
        text = map_profanity(self.morph, text, self.profanities, self.enc_prof)

        # delete stop_words: 
        if del_stop_words:
            text = ' '.join(word for word in text.split() if word.lower() not in self.stop_words)

        return text


# class NumericFeaturesPreprocessor(Preprocessor):
#     def __init__(self):
#         super().__init__() 

#         self.num_sparse = None

#     def preprocess_numeric(self): 
#         pass 