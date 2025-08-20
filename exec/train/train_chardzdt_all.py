#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2023 Abdelkrime Aries <kariminfo0@gmail.com>
#
#  ---- AUTHORS ----
# 2023	Abdelkrime Aries <kariminfo0@gmail.com>
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

import numpy as np


PRJPATH   = '/home/karim/Data/DZDT/'
WORDSURL  = 'data/mutlicls_words.csv'
WORDSURL  = 'data/test_words.csv'
CHRTOKURL = 'models/chartok.pickle'
CHRBERTURL = 'models/'
MAX_POS   = 20
MDLNAME = 'dzdt-char-bert'
BATCHSIZE = 32
NBLABELS = 5
HSIZE = 128
EPOCHS = 50

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
# ======== CREATE CHARBERT ====================
# =============================================

# Creating a BERT-based character model with the same vocabulary size as the charTokenizer
config     = BertConfig(
    vocab_size=char_tokenizer.size, 
    max_position_embeddings=MAX_POS,
    hidden_size=HSIZE,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_labels=NBLABELS
    )
char_model = MLMLMBertModel(config=config)

# =============================================
# ========= GET TRAIN DATA ====================
# =============================================

print('preparing dataset')

train_dataset = CSVWordMultiLabel(char_tokenizer, PRJPATH + WORDSURL)

# print(train_dataset.__getitem__(0))

# =============================================
# ========= DATA COLLECTOR ====================
# =============================================

print('creating the dayacollector')

data_collator = DataCollatorForMLMandMLC(
    tokenizer=char_tokenizer, mlm_probability=0.20
)

# =============================================
# ========= TRAIN ARGUMENTS ===================
# =============================================

# Training Arguments
training_args = TrainingArguments(
    output_dir= PRJPATH + CHRBERTURL,
    evaluation_strategy = 'steps',
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCHSIZE,
    logging_steps=5000,
    save_steps=5000,
    save_total_limit=2,
)

# Build a trainer
trainer = Trainer(
    model=char_model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=None
)

print('training charBERT')

trainer.train()

print('saving charBERT model')

trainer.save_model(PRJPATH + CHRBERTURL + MDLNAME)