#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2023 Abdelkrime Aries <kariminfo0@gmail.com>
#
#  ---- AUTHORS ----
# 2023-2025	Abdelkrime Aries <kariminfo0@gmail.com>
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

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import BertConfig
from transformers import Trainer, TrainingArguments


from dzdt.model.chdzdt_tok import DzDtCharTokenizer
from dzdt.model.chdzdt_mdl      import MLMLMBertModel
from dzdt.pipeline.hfdatasets       import CSVWordMultiLabel
from dzdt.pipeline.hfcollectors import DataCollatorForMLMandMLC
import pandas as pd
import numpy as np


PRJPATH   = '/home/kariminf/Data/DZDT/'
WORDSURL  = 'data/mutlicls_words.csv'
# WORDSURL  = 'data/test_words.csv'
CHRTOKURL = 'models/chartok.pickle'
CHRBERTURL = 'models/'
MAX_POS   = 20
MDLNAME = 'dzdt-char-bert'
BATCHSIZE = 32


# =============================================
# ======= CREATE TOKENIZER ====================
# =============================================

# Creating a char tokenizer with alphabets: latin, arabic, tifinagh
# It is like training, but we know the vocabulary already
char_tokenizer = DzDtCharTokenizer(max_position=MAX_POS)
char_tokenizer.add_charset(33, 126) # basic-latin
char_tokenizer.add_charset(161, 255) # latin1-suppliment
char_tokenizer.add_charset(1536, 1791) # arabic
char_tokenizer.add_charset(11568, 11647) # tifinagh
char_tokenizer.add_charset(592, 687) # IPA extensions (kabyle use this)

# =============================================
# ========= SAVE TOKENIZER ====================
# =============================================

print('saving the chartokenizer')
char_tokenizer.save(PRJPATH + CHRTOKURL)

print('vocabulary size', char_tokenizer.size)



# =============================================
# ========= GET TRAIN DATA ====================
# =============================================

print('reading dataset')

data = pd.read_csv(PRJPATH + WORDSURL, sep='\t')
data = data.dropna()
words = data.iloc[:, 1].to_numpy().tolist()
classes = data.iloc[:, 2:].to_numpy()
del data
print('encoding words')
tokens = char_tokenizer.encode_words(words)
del words
ndata = np.concatenate((tokens['input_ids'], tokens['attention_mask'], classes), axis=1)
del tokens
del classes

print(ndata.shape)

print('saving encoded data')

with open(PRJPATH + WORDSURL + '2.csv', 'w') as f:#
    for i in range(ndata.shape[0]):
        f.write(','.join(ndata[i, :].astype(str)) + '\n')

# df=pd.DataFrame.from_records(ndata)

# df.to_csv(PRJPATH + WORDSURL + '2.csv', index=False)