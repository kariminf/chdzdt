#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2024 Abdelkrime Aries <kariminfo0@gmail.com>
#
#  ---- AUTHORS ----
# 2024	Abdelkrime Aries <kariminfo0@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


CHAR_TOKENIZER_NAME = "char_tokenizer.pickle"

MAX_CHAR_POSITION    = 20
CHAR_CODE_SIZE = 128
CHAR_NUM_HIDDEN_LAYERS = 4
CHAR_NUM_ATTENTION_HEADS = 4
EPOCHS = 10
BATCHSIZE  = 1000
MAX_WORD_SIZE = 30

MODEL_MAX_LENGTH = 512

URL_TAG = "<url>"
MAIL_TAG = "<mail>"
HASH_TAG = "<hash>"
NBR_TAG = "<nbr>"
REF_TAG = "<tag>"
BREAK_TAG = "<break>"

TAG_LIST = [URL_TAG, MAIL_TAG, HASH_TAG, NBR_TAG, REF_TAG, BREAK_TAG]

CHAR_TOKENIZER_CONFIG_FILE = "char_tokenizer_config.json"
CHAR_CONFIG_NAME = "char_config.json"
CHAR_WEIGHTS_NAME = "char_pytorch_model.bin"
WORD_TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
WORD_CONFIG_NAME = "config.json"
WORD_WEIGHTS_NAME = "pytorch_model.bin"


def char_tokenizer_config():
    from transformers import tokenization_utils_base, configuration_utils, modeling_utils
    tokenization_utils_base.TOKENIZER_CONFIG_FILE = CHAR_TOKENIZER_CONFIG_FILE
    configuration_utils.CONFIG_NAME = CHAR_CONFIG_NAME
    modeling_utils.CONFIG_NAME = CHAR_CONFIG_NAME
    modeling_utils.WEIGHTS_NAME = CHAR_WEIGHTS_NAME
        

def word_tokenizer_config():
    from transformers import tokenization_utils_base, configuration_utils, modeling_utils
    tokenization_utils_base.TOKENIZER_CONFIG_FILE = WORD_TOKENIZER_CONFIG_FILE
    configuration_utils.CONFIG_NAME = WORD_CONFIG_NAME
    modeling_utils.CONFIG_NAME = WORD_CONFIG_NAME
    modeling_utils.WEIGHTS_NAME = WORD_WEIGHTS_NAME