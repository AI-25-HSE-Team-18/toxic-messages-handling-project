import html
import re
import emoji
# import pymorphy2

# import kagglehub
import numpy as np
import pandas as pd 
import seaborn as sns 
from scipy.sparse import csr_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt 


def get_num_features(text: str) -> csr_matrix: 
    """Returns numpy sparse matrix from features"""

    def is_all_lower(text):
        text_no_tokens = re.sub(r'\[.*?\]', '', text)
        return int(bool(text_no_tokens) and text_no_tokens.islower())

    def starts_with_cap(text):
        """Returns 1 if more than half of sentences start with a capital letter"""
        text_no_tokens = re.sub(r'\[.*?\]', '', text).strip()
        sentences = re.split(r'[.!?]', text_no_tokens)
        starts_with_cap_each = [s.strip()[0].isupper() for s in sentences if s.strip()]

        if not starts_with_cap_each:
            return 0

        return int(sum(starts_with_cap_each) > (len(starts_with_cap_each) / 2))

    def has_caps(word_list):
        for w in word_list:
            if w.isupper() and not re.match(r'\[.*?\]', w):
                return True
        return False

    def has_repeating_letters(text):
        text_clean = re.sub(r'\[.*?\]', '', text)
        return int(bool(re.search(r'([A-Za-zА-Яа-яЁё])\1{2,}', text_clean, flags=re.IGNORECASE))) 
    
    features = {}

    # Separate / repeating punctuations
    features['count_spp'] = len(re.findall(r'\[SPP_\d+\]', text))
    features['count_rpp'] = len(re.findall(r'\[RPP_\d+\]', text))
    features['punct_after_space'] = int(bool(re.search(r' \[SPP_\d+\]', text)))

    # Emoji / emoticons
    features['has_emoji'] = int(bool(re.search(r'\[EMJ_\d+\]', text)))
    features['has_emoticon'] = int(bool(re.search(r'\[EMT_\d+\]', text)))

    # Tonality / structure
    word_list = re.findall(r'\b\w+\b', text)
    features['has_capslock'] = int(has_caps(word_list))
    features['is_all_lower'] = is_all_lower(text)

    features['has_punctuation_spp'] = int(bool(re.search(r'\[SPP_\d+\]', text)))
    features['has_punctuation_rpp'] = int(bool(re.search(r'\[RPP_\d+\]', text)))

    features['has_fence_ironic_style'] = int(bool(re.search(r'\*.*?\*', text)))

    # Lexical content
    features['count_profanity'] = len(re.findall(r'\[PRF_\d+\]', text))
    features['is_bad_word_incl'] = int(features['count_profanity'] >= 1)

    features['has_pronouns'] = int(bool(re.search(r'\[PRON_\d+\]', text)))

    # Sentence-level features
    features['starts_with_cap'] = starts_with_cap(text)
    features['ends_with_dot'] = int(text.endswith('[SPP_3]'))

    features['has_emotional_sym'] = int(bool(re.search(r'\[RPP_2\]|\[SPP_4\]', text)))
    features['has_repeating_letters_3plus'] = has_repeating_letters(text)

    # Neutral tokens
    features['has_url'] = int(bool(re.search(r'\[URL\]', text)))
    features['has_number'] = int(bool(re.search(r'\[NUM\]', text)))
    features['has_mention'] = int(bool(re.search(r'\[MNT\]', text)))
    features['has_hashtag'] = int(bool(re.search(r'\[HSG\]', text)))

    # features = pd.DataFrame(features)
    features_list = list(features.values())
    features = np.array([features_list], dtype=np.float32)
    num_sparse = csr_matrix(features)

    return num_sparse

def map_noninformatives(text, mapping_dict: dict):
    """Mapping extra str that does not contain useful or informative substrings: 
    Cleaning of URLs, mentiones, hashtags, numbers, emails, HTML symbols, do strip
    Uses the dict of mapping
    
    Regexps are written by GPT"""

    def clean_html(text):
        """Cleaning of html entities. Regexps are written by GPT"""
        # Replace <br> and its variations with a space
        text = re.sub(r'<br\s*/?>', ' ', text, flags=re.IGNORECASE)
        
        # Remove all other HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Decode HTML entities (e.g. &quot;, &amp;, &#39;)
        text = html.unescape(text)
        
        # Remove extra spaces and trim
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    # Replace urls: 
    text = re.sub(r"http\S+|www\S+", mapping_dict.get('url'), text)
    
    # Replace mentions: 
    text = re.sub(r"[@]\w+", mapping_dict.get('mention'), text)
    text = re.sub(r'id\d+\|[^\s]+', mapping_dict.get('mention'), text) # based on info above

    # replace hashtags: 
    text = re.sub(r"[#]\w+", mapping_dict.get('hashtag'), text) 

    # Replace numbers (integer or decimal):
    text = re.sub(r'\b\d+(\.\d+)?\b|NUMBER|number', mapping_dict.get('num'), text, flags=re.IGNORECASE)

    # Replace emails: 
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', mapping_dict.get('email'), text)

    # Delete br and quot, other HTML tags: 
    # text = re.sub(r'<br\s*/?>', ' ', text, flags=re.IGNORECASE)
    # text = re.sub(r'&quot;', '', text, flags=re.IGNORECASE)
    text = clean_html(text)
    
    # collapse multiple spaces and strip: 
    # text = re.sub(r"\s+", " ", text).strip()
    
    return text


emoji_reg_row = r"""(?x)  # verbose mode
    (?:                             # group of emoticons
    (?::|;|=|8)                 # leading eyes for Western style
    (?:-)?                      # optional nose
    (?:\)|\(|D|P|p|O|o|3|/|\\|\||\*|\$|@)   # mouth / expression
        |
    (?:\^\^|_\^|^_\^|^‿^|˘‿˘)   # simple happy eyes/mouth combos such as ^_^ _^ ^_^ ˘‿˘
        |
    (?:T_T|;_;|;\-\;|>_<|>\.<|>_>|<_<)  # sad/embarrassed/frustrated
        |
    (?:<3|♥|♡)                # heart symbols
        |
    (?:¯\\_\(ツ\)_/¯|¯\\_\(ಠ_ಠ\)_/¯)  # shrug / disapproval special ones
        |
    (?:uwu|OwO|UwU|owo)        # internet‑emoticon style
)
"""

def map_emoji_emoticons(text: str, 
                        mapping_dicts: tuple[dict, dict], 
                        emoji_regrow=emoji_reg_row,
                        token_emoji='EMJ',
                        token_emoticon='EMT'):
    
    encoding_emoji, encoding_emoticon = mapping_dicts

    def encode_emoji(encoding_text: str):

        emoji_list = emoji.emoji_list(encoding_text)
        
        for e in sorted(set([item['emoji'] for item in emoji_list])):
            if e not in encoding_emoji: 
                token = f"[{token_emoji}_{len(encoding_emoji)}]" 
                encoding_emoji[e] = token
            encoding_text = encoding_text.replace(e, encoding_emoji[e])

        return encoding_text

    def encode_emoticon(encoding_text: str, regrow=emoji_regrow):

        emoticon_pattern = re.compile(regrow)
        emoticons_found = set(re.findall(emoticon_pattern, encoding_text))
        for emo in emoticons_found:
            if emo not in encoding_emoticon:
                token = f"[{token_emoticon}_{len(encoding_emoticon)}]"
                encoding_emoticon[emo] = token
            encoding_text = encoding_text.replace(emo, encoding_emoticon[emo])
            
        return encoding_text

    text = encode_emoji(text)
    text = encode_emoticon(text)
    
    return text


def map_punctuation(
        text: str, 
        mapping_rep_dict: dict, 
        mapping_sep_dict: dict, 
        rep_regexp = r"([!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~]{2,})",
        # nonrep_regexp=r"[!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~]",
        nonrep_regexp=r"[!\"#$%&'()*+,\-./:;<=>?@^`{|}~]",
        token_sep_punct='SPP', # separate punctuatuin 
        token_seq_punct='RPP' # repeating punctuation
    ) -> str:
    """
    Encode both repeating and single punctuation marks with mapping dictionaries.
    Sequences inside any [ ... ] (even unclosed) are ignored.
    Already existing [TOKEN]-like patterns are ignored too.
    """


    def is_inside_token(pos: int) -> bool:
        """Check if position is inside [TOKEN]-like region."""
        return any(start <= pos < end for start, end in token_spans)
    
    def is_protected(pos: int) -> bool:
        """Return True if position is inside any protected region (token or [ ... ])."""
        
        return is_inside_token(pos) or any(start <= pos < end for start, end in bracket_spans)

    def repl_repeat(match):
        """Handle repeating punctuation"""

        seq = match.group(1)
        start = match.start()

        if is_protected(start):
            return seq

        # Unique punctuation marks, preserving order
        unique_chars = ''.join(sorted(set(seq), key=seq.index))
        normalized = unique_chars[0] if len(unique_chars) == 1 else unique_chars

        if normalized not in mapping_rep_dict:
            token = f"[{token_seq_punct}_{len(mapping_rep_dict)}]"
            mapping_rep_dict[normalized] = token
        else:
            token = mapping_rep_dict[normalized]

        return token

    def repl_single(match):
        """Handle separate punctuation"""

        ch = match.group(0)
        start = match.start()

        if is_protected(start):
            return ch

        if ch not in mapping_sep_dict:
            token = f"[{token_sep_punct}_{len(mapping_sep_dict)}]"
            mapping_sep_dict[ch] = token
        else:
            token = mapping_sep_dict[ch]

        return token
    
    # Compile patterns: 
    rep_pattern = re.compile(rep_regexp)
    sep_pattern = re.compile(nonrep_regexp)

    # Find existing [TOKEN]-like regions: 
    token_spans = []
    for match in re.finditer(r"\[[A-Za-z0-9_]+\]", text):
        token_spans.append((match.start(), match.end()))

    # Detect generic bracket regions (for [ ... ] ), excluding [TOKEN] ones: 
    bracket_spans = []
    open_pos = None
    for i, ch in enumerate(text):
        if is_inside_token(i):
            continue  # Skip positions inside known tokens entirely
        if ch == '[':
            if open_pos is None:
                open_pos = i
        elif ch == ']' and open_pos is not None:
            bracket_spans.append((open_pos, i + 1))
            open_pos = None
    if open_pos is not None:  # text ends with unclosed '['
        bracket_spans.append((open_pos, len(text)))

    # Apply replacements: 
    new_text = rep_pattern.sub(repl_repeat, text)
    new_text = sep_pattern.sub(repl_single, new_text)

    return new_text

def map_profanity(
        morph_analyzer,
        text: str, 
        profanities: list,
        mapping_dict=None, 
        token_profanity='PRF', # profanity 
        
        # replace_char='[PROFANITY]'
        ) -> str:
    """
    Map Russian profanity using [what] in text to tokens,
    """
    
    morph = morph_analyzer

    def repl(match) -> str:

        word = match.group(0)
        lower = word.lower()
        lemma = morph.parse(lower)[0].normal_form
        
        if lemma in profanities: 
            # print('word', word, 'lemma', lemma) # uncomment if you don't need you eyes
            if lemma not in mapping_dict:
                res = f"[{token_profanity}_{len(mapping_dict)}]"
                mapping_dict[lemma] = res
            else:
                res = mapping_dict[lemma]

            # print(res)
        else: 
            res = word
        
        return res

    pattern = re.compile(r"\b[А-Яа-яЁё']+\b")
    
    # print(text)
    return pattern.sub(repl, text)

def del_punct_tokens(text: str):
    return re.sub(r'\[(?!(EMJ|EMT|PRF)_\d+)[^\]]*\]', '', text)