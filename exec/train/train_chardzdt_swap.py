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

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from functools import reduce
from random import shuffle

from transformers import BertConfig
from transformers import Trainer, TrainingArguments

from dzdt.model.chdzdt_tok import DzDtCharTokenizer
from dzdt.model.modeling_chardzdt      import MLMLMBertModel
from dzdt.train.datasets       import CSVWordMultiLabel, CSVWordMultiLabelSwap, DatasetSwapTrainerCallback
from dzdt.train.collectors import DataCollatorForMLMandMLC
from dzdt.train.trainers import CharBertTrainer

import numpy as np


PRJPATH    = '/home/karim/Data/DZDT/'
WORDSURL   = 'data/mutlicls_words/'
WORDSURL   = 'data/test/'
CHRTOKURL  = 'models/char_tokenizer.pickle'
CHRBERTURL = 'models/'
MDLNAME    = 'dzdt-char-bert'
MAX_POS    = 20
BATCHSIZE  = 1000
NBLABELS   = 5
HSIZE      = 128
EPOCHS     = 10
ROUNDS     = 5

def get_labels(url: str):
    data = pd.read_csv(url, sep='\t')
    return list(data.columns)

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
# ========= GET TRAIN DATA LAB ====================
# =============================================

print('preparing dataset ...')

datasts_urls = []
for csv in os.listdir(PRJPATH + WORDSURL):
    if csv.endswith(".csv"):
        datasts_urls.append(PRJPATH + WORDSURL + csv)

# print(datasts_urls)
dataset = CSVWordMultiLabelSwap(char_tokenizer, datasts_urls)

labels = get_labels(datasts_urls[0])[1:]

EPOCHS *= len(datasts_urls)

# print(labels)

# =============================================
# ======== CREATE CHARBERT ====================
# =============================================

print('creating charBERT ...')

id2label = reduce(lambda x, y: {str(len(x)):y}|x, labels, {})
label2id = reduce(lambda x, y: {y: len(x)}|x, labels, {})

# Creating a BERT-based character model with the same vocabulary size as the charTokenizer
config     = BertConfig(
    vocab_size=char_tokenizer.size, 
    max_position_embeddings=MAX_POS,
    hidden_size=HSIZE,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_labels=len(labels),
    id2label = id2label,
    label2id = label2id,
    return_dict=None
    )

# print(config.id2label)

char_model = MLMLMBertModel(config=config)


# =============================================
# ========= DATA COLLECTOR ====================
# =============================================

print('creating a dataCollector')

data_collator = DataCollatorForMLMandMLC(
    tokenizer=char_tokenizer, mlm_probability=0.20
)


# =============================================
# ========= TRAIN ARGUMENTS ===================
# =============================================

print('training charBERT')

# Training Arguments
training_args = TrainingArguments(
    output_dir= PRJPATH + CHRBERTURL,
    # eval_strategy = 'steps',
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCHSIZE,
    logging_steps=20,
    save_steps=1000,
    save_total_limit=2,
    save_safetensors=False,
    label_names = labels,
    # logging_dir=PRJPATH + 'logs',
)

# =============================================
# =============== TRAINING ===================
# =============================================


# Build a trainer
trainer = CharBertTrainer(
    model=char_model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    eval_dataset=None,
    callbacks=[DatasetSwapTrainerCallback]
)

trainer.train()

print('saving charBERT model')

trainer.save_model(PRJPATH + CHRBERTURL + MDLNAME)