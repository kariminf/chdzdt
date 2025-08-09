#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2023 Abdelkrime Aries <kariminfo0@gmail.com>
#
#  ---- AUTHORS ----
# 2023-2024	Abdelkrime Aries <kariminfo0@gmail.com>
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

import os
from random import shuffle
from typing import Any, Dict, List, Tuple, Union
import torch
import pandas as pd
from torch.utils.data import Dataset
import logging
from transformers import TrainerCallback, PreTrainedTokenizer
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from dzdt.model.chdzdt_tok import CharTokenizer

class DataLoader:
    """
    Base class for all file loaders.
    It simply takes a file, it loads it and then returns 
    a dictionary of features (feature name, list of values).

    This is an abstract class, the mthod `load` must be overloaded
    """
    def __call__(self, file_path: str) -> Dict[str, List[Any]]: 
        if not os.path.isfile(file_path):
            raise ValueError(f"Input file path {file_path} not found")
        return self.load(file_path)
    
    def load(self, file_path: str) -> Dict[str, List[Any]]: 
        raise NotImplementedError()
    

class CSVLoader(DataLoader):
    def __init__(self, sep="\t", 
                 names: List[str] = None, 
                 idx: List[Tuple[int, int]] = None
                 ):
        super().__init__()
        self.sep = sep
        self.names = names
        self.idx = idx

    def load(self, file_path: str) -> Dict[str, List[Any]]: 
        data = pd.read_csv(file_path, sep=self.sep)
        result = {}
        if self.names is None:
            self.names = list(data.columns)
        if self.idx is None:
            self.idx = [(i, i+1) for i in range(len(self.names))]
        
        for name, idx in zip(self.names, self.idx):
            result[name] = data.iloc[:, idx[0]:idx[1]].to_numpy().tolist()

        return result

class DataTransformer:
    def __call__(self, data: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        return self.transform(data)
    def transform(self, data: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        keys = list(data.keys())
        result = []
        for i in range(len(data[keys[0]])):
            result.append({k: data[k][i] for k in keys})
        return result


class GeneralDataset(Dataset):
    def __init__(self, 
                 file_path: str, 
                 data_loader: DataLoader, 
                 data_transformer: DataTransformer,
                 tokenizer: Union[CharTokenizer, PreTrainedTokenizer] = None,
                 text_label: str = None):
        data = data_loader(file_path)
        if (text_label is not None) and (tokenizer is not None):
            tokenized = tokenizer(data[text_label], return_tensors="pt")
            del data[text_label]
            for feature in data:
                data[feature] =  torch.tensor(data[feature], dtype=torch.float32)
            data = {**data, **tokenized}
        self.nbrFiles = 1
        self.examples = data_transformer(data)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, Any]:
        return self.examples[i]

class GeneralSwapDataset(GeneralDataset):
    def __init__(self, 
                 file_paths: List[str], 
                 data_loader: DataLoader, 
                 data_transformer: DataTransformer, 
                 tokenizer: CharTokenizer | PreTrainedTokenizer = None, 
                 text_label: str = None
                 ):
        self.file_paths = file_paths
        self.nbrFiles = len(self.file_paths)
        self.currentFID = self.nbrFiles
        self.data_loader = data_loader
        self.data_transformer = data_transformer
        self.tokenizer = tokenizer
        self.text_label = text_label
        self.reset_dataset()

    def reset_dataset(self):
        if self.currentFID == self.nbrFiles:
            shuffle(self.file_paths)
            self.currentFID = 0

        super().__init__(
            self.file_paths[self.currentFID], 
            self.data_loader, 
            self.data_transformer, 
            tokenizer=self.tokenizer, 
            text_label=self.text_label
            )
        self.currentFID += 1




def create_example(token_id, attention_mask, token_multi_label):
    return {
        "input_ids": torch.tensor(token_id, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        # "input_type_ids": torch.tensor(tokens["input_type_ids"][i], dtype=torch.long),
        "multi_labels": torch.tensor(token_multi_label, dtype=torch.float32)
    }

class CSVWordMultiLabel(Dataset):

    def __init__(
        self,
        tokenizer: CharTokenizer,
        file_path: str,
        sep="\t"
    ):
        self.tokenizer = tokenizer
        self.sep = sep 
        self.load_file(file_path)

    def load_file(self, file_path: str):
        
        if not os.path.isfile(file_path):
            raise ValueError(f"Input file path {file_path} not found")
        
        # logging.info("loading " + file_path + "...")
        # print("loading " + file_path + "...")

        data = pd.read_csv(file_path, sep=self.sep)
        # data = data.dropna()
        words = data.iloc[:, 0].to_numpy().tolist()
        # print(words[:5])
        classes = data.iloc[:, 1:].to_numpy().tolist()
        # print(classes[:5])
        # return None
        del data
        tokens = self.tokenizer.encode_words(words)
        del words

        

        self.examples = list(map(lambda tok, att, cls: create_example(tok, att, cls), tokens["input_ids"], tokens["attention_mask"], classes))


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return self.examples[i]


class CSVWordMultiLabelSwap(CSVWordMultiLabel):

    def __init__(
        self,
        tokenizer: CharTokenizer,
        file_paths: List[str],
        sep="\t"
    ):
        self.file_paths = file_paths
        shuffle(self.file_paths)
        super().__init__(tokenizer, self.file_paths[0], sep)
        self.currentFID = 1
        self.nbrFiles = len(self.file_paths)

    def reset_dataset(self):
        if self.currentFID == self.nbrFiles:
            shuffle(self.file_paths)
            self.currentFID = 0
        self.load_file(self.file_paths[self.currentFID])
        self.currentFID += 1

class DatasetSwapTrainerCallback(TrainerCallback):
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        train_dataloader = kwargs.pop("train_dataloader")
        train_dataloader.dataset.reset_dataset()

class CSVWordMultiLabelEncoded(Dataset):
    def __init__(self, file_path: str, max_pos: int):
        
        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")

        data = pd.read_csv(file_path, sep=",", header=None).to_numpy()

        self.examples = list(map(lambda line: create_example(line[:max_pos], line[max_pos:2*max_pos], line[2*max_pos:]), data))


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return self.examples[i]