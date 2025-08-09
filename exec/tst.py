import sys
import os
import timeit
import pandas as pd

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer, AutoModel, BertConfig
from dzdt.model._char_tokenizer_model import CharDzDtTokenizer

from dzdt.model.char_tokenizer import DzDtCharTokenizer

from transformers import tokenization_utils_base, configuration_utils, modeling_utils
# from transformers.models.auto import auto_factory, configuration_auto

# PRJPATH    = '/home/karim/Data/DZDT/'
# CHRTOKURL  = 'models/char_tokenizer.pickle'

# tokenizer = CharDzDtTokenizer.from_pretrained(PRJPATH+ 'models/')

# tokenizer_config = BertConfig.from_json_file("kariminf/tst/char_tokenizer_config.json")

# tokenization_utils_base.TOKENIZER_CONFIG_FILE = "char_tokenizer_config.json"
# auto_factory.WEIGHTS_NAME = "char_pytorch_model.bin"
# auto_factory.CONFIG_NAME = "char_config.json"
# configuration_auto.CONFIG_NAME = "char_config.json"

configuration_utils.CONFIG_NAME = "char_config.json"
modeling_utils.CONFIG_NAME = "char_config.json"
modeling_utils.WEIGHTS_NAME = "char_pytorch_model.bin"

TOKEN_PATH = "/home/karim/Data/DZDT/hug_token.txt"
with open(TOKEN_PATH, "r") as f:
    for l in f:
        hugingface_token = l.rstrip("\n")
        break

tokenizer: CharDzDtTokenizer = CharDzDtTokenizer.from_pretrained("kariminf/tst", token=hugingface_token)

print(tokenizer.special_tokens_map)

# print(tokenization_utils_base.TOKENIZER_CONFIG_FILE)

model = AutoModel.from_pretrained("kariminf/tst", token=hugingface_token)

text = ['Replace', 'Go']
print(tokenizer._tokenize(text))
st = timeit.default_timer()
encoded_input = tokenizer(text, return_tensors='pt',
                          truncation=True, 
                          padding=True,
                          add_special_tokens=True,
                          max_length=20)
print('time encoder=', timeit.default_timer() - st)
print(encoded_input)
st = timeit.default_timer()
encoded_input2 = tokenizer.chartok.encode_words(text, return_tensors='pt')
# for n in encoded_input2:
#     encoded_input2[n] = torch.tensor(encoded_input2[n])
print('time encoder=', timeit.default_timer() - st)
print(encoded_input2)

output = model(**encoded_input, return_dict=True)
output2 = model(**encoded_input2, return_dict=True)

print(output['last_hidden_state'].shape, output2['last_hidden_state'].shape)