#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2025 Abdelkrime Aries <kariminfo0@gmail.com>
#
#  ---- AUTHORS ----
# 2025	Abdelkrime Aries <kariminfo0@gmail.com>
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

import argparse
import json
import sys
import os
import logging, os
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from typing import Any, Dict, List
import pandas as pd
from functools import reduce
from transformers import BertConfig, BertPreTrainedModel
from transformers import TrainingArguments
from transformers import configuration_utils, modeling_utils

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dzdt.train.datasets import (
    CSVLoader, 
    DataTransformer, 
    GeneralDataset, 
    GeneralSwapDataset, 
    DatasetSwapTrainerCallback
    )
from dzdt.model.chdzdt_tok import CharTokenizer
from dzdt.model.chdzdt_mdl import MLMLMBertModel
from dzdt.train.collectors import DataCollatorForMLMandMLC
from dzdt.train.trainers import CharBertTrainer
from dzdt.tools.const import (
    BATCHSIZE,
    CHAR_NUM_ATTENTION_HEADS, 
    CHAR_NUM_HIDDEN_LAYERS, 
    CHAR_TOKENIZER_NAME,
    EPOCHS, 
    MAX_CHAR_POSITION, 
    CHAR_CODE_SIZE,
    char_tokenizer_config,
    word_tokenizer_config
    )


def get_labels(url: str):
    data = pd.read_csv(url, sep='\t')
    return list(data.columns)

def list_files(url: str, ext: str = ".csv") -> List[str]:
    files_urls = []
    for file in os.listdir(url):
        if file.endswith(ext):
            files_urls.append(os.path.join(url, file)) 
    return files_urls

def load_config(url: str) -> Dict[str, Any]:
    with open(url, "r") as f:
        return json.load(f)
    
def prepare_words_dataset(tokenizer: CharTokenizer, **kwargs) -> GeneralDataset:
    print("Preparing dataset")

    project_path = kwargs.get("project_path", "./")
    project_path = os.path.expanduser(project_path)
    words_url   = kwargs.get("words_url", None)
    file_paths = os.path.join(project_path, words_url)
    if not os.path.exists(file_paths):
        raise FileNotFoundError(f"File or folder '{file_paths}' not found")
    swapping    = os.path.isdir(file_paths)
    words_idx   = kwargs.get("words_idx", 0)
    class_idx   = kwargs.get("class_idx", False)
    text_label  = kwargs.get("text_label", "word")
    
    data_loader = CSVLoader(names=["word", "multi_labels"], idx=[(words_idx, words_idx+1), class_idx])
    data_transformer = DataTransformer()

    newDataset = GeneralDataset
    if swapping:
        file_paths = list_files(file_paths)
        newDataset = GeneralSwapDataset
        
    dataset = newDataset(
        file_paths, 
        data_loader=data_loader, 
        data_transformer=data_transformer,
        tokenizer=tokenizer,
        text_label=text_label
        )
        
    return dataset

def create_char_encoder(tokenizer: CharTokenizer, **kwargs) -> BertPreTrainedModel:
    print("Creating character encoder")

    labels = kwargs.get("labels", [])
    max_char_position = kwargs.get("max_char_position", MAX_CHAR_POSITION)
    hidden_size = kwargs.get("hidden_size", CHAR_CODE_SIZE)
    num_hidden_layers = kwargs.get("num_hidden_layers", CHAR_NUM_HIDDEN_LAYERS)
    num_attention_heads = kwargs.get("num_attention_heads", CHAR_NUM_ATTENTION_HEADS)
    
    id2label = reduce(lambda x, y: {str(len(x)):y}|x, labels, {})
    label2id = reduce(lambda x, y: {y: len(x)}|x, labels, {})

    # Creating a BERT-based character model with the same vocabulary size as the charTokenizer
    config     = BertConfig(
        vocab_size=tokenizer.size,   
        max_position_embeddings=max_char_position,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_labels=len(labels),
        id2label = id2label,
        label2id = label2id
        )

    return MLMLMBertModel(config=config)

def train_model(encoder: BertPreTrainedModel, 
                tokenizer: CharTokenizer,
                train_dataset: GeneralDataset, 
                eval_dataset: GeneralDataset, 
                **kwargs) -> CharBertTrainer:
    
    print("Training character encoder")
    project_path = kwargs.get("project_path", "./")
    project_path = os.path.expanduser(project_path)
    words_url   = kwargs.get("words_url", None)
    file_paths = os.path.join(project_path, words_url)
    swapping    = os.path.isdir(file_paths)

    chardzdt_url = kwargs.get("chardzdt_url", None)
    out_path = os.path.join(project_path, chardzdt_url)
    epochs = kwargs.get("epochs", EPOCHS)
    epochs *= train_dataset.nbrFiles
    batch_size = kwargs.get("batch_size", BATCHSIZE)
    labels = labels = kwargs.get("labels", None)

    data_collator = DataCollatorForMLMandMLC(tokenizer=tokenizer, mlm_probability=0.20)
    training_args = TrainingArguments(
        output_dir= out_path,
        # eval_strategy = 'steps',
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        # learning_rate= 0.00000005144,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        save_safetensors=False,
        label_names = labels,
        # logging_dir=PRJPATH + 'logs',
    )

    callbacks = [DatasetSwapTrainerCallback] if swapping else []


    # Build a trainer
    trainer = CharBertTrainer(
        model=encoder,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks
    )

    trainer.train()
    # trainer.save_model(out_path)
    return trainer

def save_model(trainer: CharBertTrainer,
               tokenizer: CharTokenizer, 
               **kwargs) -> None:
    print("Saving character tokenizer")
    project_path = kwargs.get("project_path", "./")
    project_path = os.path.expanduser(project_path)
    chardzdt_url = kwargs.get("chardzdt_url", "chardzdt")
    path = os.path.join(project_path, chardzdt_url)
    # encoder.save_pretrained(path, filename_prefix="char", safe_serialization=False)
    configuration_utils.CONFIG_NAME = "char_config.json"
    modeling_utils.CONFIG_NAME = "char_config.json"
    modeling_utils.WEIGHTS_NAME = "char_pytorch_model.bin"
    
    trainer.save_model(path)
    path = os.path.join(path, CHAR_TOKENIZER_NAME)

    tokenizer.save(path)


# =============================================
#          Command line functions
# =============================================

def main(args):
    config = load_config(os.path.expanduser(args.config))
    if "input" in args:
        in_url = os.path.expanduser(args.input)
        print("loading model ", in_url)
        tokenizer = CharTokenizer.load(os.path.join(in_url, "char_tokenizer.pickle"))
        char_tokenizer_config()
        encoder = MLMLMBertModel.from_pretrained(in_url)
        word_tokenizer_config()
    else:
        tokenizer = create_char_tokenizer(**config)
        encoder = create_char_encoder(tokenizer, **config)
    train_dataset = prepare_words_dataset(tokenizer, **config)
    trainer = train_model(encoder, tokenizer, train_dataset, None, **config)
    save_model(trainer, tokenizer, **config)


parser = argparse.ArgumentParser(description="train a char-based DzDt")
subparsers = parser.add_subparsers(help="choose training type", required=True)

parser_new = subparsers.add_parser("new", help="train a new model")
parser_new.add_argument("config", help="config file")
parser_new.set_defaults(func=main)

parser_resume = subparsers.add_parser("resume", help="resume training a model")
parser_resume.add_argument("config", help="config file")
parser_resume.add_argument("input", help="input model's location")
parser_resume.set_defaults(func=main)

if __name__ == "__main__":
    argv = sys.argv[1:]

    argv = [
        "new",
        # "resume",
        "~/Data/DZDT/models/chdzdt_ft.json",
        # "~/Data/DZDT/models/chdzdt_4x4x32_20it"
    ]

    args = parser.parse_args(argv)
    args.func(args)
