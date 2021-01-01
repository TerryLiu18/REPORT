import os
import sys
import time
import json
import argparse
import random
from tools import save_dict_to_json, txt2iterable, read_dict_from_json
import torch
import torch.nn.functional as F
import numpy as np
import os.path as pth
import pandas as pd

import copy
import util
from util import str2bool
from GloveEmbed import _get_embedding
from tools import txt2iterable, iterable2txt

def _load_word2index(word_file):
    with open(word_file) as jsonfile:
        word_map = json.load(jsonfile)

    vocab_size = len(word_map)
    return word_map, vocab_size
TWEETS_WORD_FILE = pth.join('../load_data15_1473/tweets_words_mapping.json')
tweets_word_map1, _ = _load_word2index(TWEETS_WORD_FILE)
TWEETS_WORD_FILE = pth.join('../load_data16/tweets_words_mapping.json')
tweets_word_map2, _ = _load_word2index(TWEETS_WORD_FILE)
key1 = set(list(tweets_word_map1.keys()))
key2 = set(list(tweets_word_map2.keys()))
print(len(key1))
print(len(key2))
print(len(key2.intersection(key1)))