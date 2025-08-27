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


import torch
import numpy as np
from typing import List, Tuple, Dict, Any

from torch.nn import Module
from torch.utils.data import DataLoader
from dataclasses import dataclass

from dzdt.pipeline.recorders import Recorder
from dzdt.pipeline.preprocessor import Embedder, ClsTokEmbedder


@dataclass
class SimpleTesterConfig:
    model: Module
    data_loader: DataLoader
    embedder: Embedder = None
    criteria: List[Any] = None
    recorder: Recorder = Recorder()
    stream: bool = False


class SimpleTester:
    def __init__(self, config: SimpleTesterConfig, device=None):
        self.device = device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config 

        self.config.model.to(self.device)
        self.config.model.eval()

    def forward(self, X):
        X = X.to(self.device)
        with torch.no_grad():
            pred = self.config.model(X)
        return pred

    def forward_stream(self, text):
        return self.forward(self.config.embedder.encode(text))

    def test(self):
        self.config.recorder.start()
        all_preds, all_labels = [], []
        i = 0
        for batch_X, batch_Y in self.config.data_loader:
            i += 1
            self.config.recorder.record(i)
            pred = self.forward_stream(batch_X) if self.config.stream else self.forward(batch_X)

            if pred.shape[-1] == 1:
                # binary: one logit
                pred = torch.sigmoid(pred).squeeze(-1)
                pred = (pred > 0.5).long().cpu().tolist()
            else:
                pred = pred.argmax(dim=-1).cpu().tolist()

            all_preds.extend(pred)
            all_labels.extend(batch_Y)

        results = []
        for criterion in self.config.criteria:
            results.append(criterion(all_labels, all_preds))

        self.config.recorder.finish()
        return results
    
class FTSimpleTester(SimpleTester):

    def forward_stream(self, text):
        with torch.no_grad():
            X = self.config.model.tokenize_encode(text, self.device)
        return self.forward(X)
    
class MaskedSimpleTester(SimpleTester):
    def forward(self, X):
        X = X.to(self.device)
        with torch.no_grad():
            pred = self.config.model(X)
        return pred

    def forward_stream(self, text):
        X, mask = self.config.embedder.encode(text)
        return self.forward(X), mask

    def test(self):
        self.config.recorder.start()
        all_preds, all_labels, all_masks = [], [], []
        i = 0
        for data in self.config.data_loader:
            i += 1
            self.config.recorder.record(i)
            if self.config.stream:
                batch_X, batch_Y = data
                pred, batch_mask = self.forward_stream(batch_X)
            else:
                batch_X, batch_mask, batch_Y = data
                pred = self.forward(batch_X)

            if pred.shape[-1] == 1:
                # binary: one logit
                pred = torch.sigmoid(pred).squeeze(-1)
                pred = (pred > 0.5).long().cpu().view(-1)
            else:
                pred = pred.argmax(dim=-1).cpu().view(-1)

            pred = pred[batch_mask.view(-1)].tolist()

            all_preds.extend(pred)
            all_labels.extend(batch_Y)

        results = []
        for criterion in self.config.criteria:
            results.append(criterion(all_labels, all_preds))

        self.config.recorder.finish()
        return results

@dataclass
class MultiTesterConfig:
    model: Module
    data_loader: DataLoader
    output_features: List[str]
    embedder: Embedder = None
    criteria: List[Any] = None
    recorder: Recorder = Recorder()
    stream: bool = False
    
class MultiTester:
    def __init__(self, config: MultiTesterConfig, device=None):
        self.device = device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config 

        self.config.model.to(self.device)
        self.config.model.eval()


    def forward(self, X):
        X = X.to(self.device)
        with torch.no_grad():
            y = self.config.model(X)
        return y

    def forward_stream(self, text):
        return self.forward(self.config.embedder.encode(text))

    def test(self):
        self.config.recorder.start()
        all_labels = {}
        all_preds = {}

        for feature in self.config.output_features:
            all_preds[feature] = []
            all_labels[feature] = []

        i = 0
        for data in self.config.data_loader:
            i += 1
            self.config.recorder.record(i)
            pred = self.forward_stream(data[0]) if self.config.stream else self.forward(data[0])

            for j, feature in enumerate(self.config.output_features):

                predj = pred[feature].detach().cpu()

                if predj.shape[-1] == 1:
                    # binary: one logit
                    predj = torch.sigmoid(predj).squeeze(-1)
                    predj = (predj > 0.5).long().tolist()
                else:
                    predj = np.argmax(predj, axis=1).tolist()

                all_preds[feature].extend(predj)

                y = data[1:][j]
                if not isinstance(y, list):
                    y = y.tolist()
                all_labels[feature].extend(y)

        results = []
        for criterion in self.config.criteria:
            results.append(criterion(all_labels, all_preds ))

        self.config.recorder.finish()
        return results


@dataclass
class ClsTokMultiTesterConfig:
    cls_model: Module
    tok_model: Module
    data_loader: DataLoader
    output_features: List[str]
    embedder: ClsTokEmbedder = None
    criteria: List[Any] = None
    recorder: Recorder = Recorder()
    stream: bool = False
    
class ClsTokMultiTester:
    def __init__(self, config: ClsTokMultiTesterConfig, device=None):
        self.device = device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cls_model = config.cls_model 
        self.tok_model = config.tok_model 
        self.data_loader = config.data_loader
        self.embedder = config.embedder
        self.recorder = config.recorder
        self.criteria = config.criteria
        self.stream = config.stream
        self.output_features = config.output_features

        self.cls_model.to(self.device)
        self.tok_model.to(self.device)
        
        self.cls_model.eval()
        self.tok_model.eval()


    def forward(self, X_cls, X_tok):
        X_cls, X_tok = X_cls.to(self.device), X_tok.to(self.device)
        with torch.no_grad():
            y_cls = self.cls_model(X_cls)
            y_tok = self.tok_model(X_tok)
        return y_cls, y_tok

    def forward_stream(self, text):
        return self.forward(*self.embedder.encode(text))

    def test(self):
        self.recorder.start()
        all_labels = {}
        all_preds_cls, all_preds_tok = {}, {}

        for feature in self.output_features:
            all_preds_cls[feature] = []
            all_preds_tok[feature] = []
            all_labels[feature] = []

        i = 0
        for data in self.data_loader:
            i += 1
            self.recorder.record(i)
            pred_cls, pred_tok = self.forward_stream(data[0]) if self.stream else self.forward(data[0])

            for j, feature in enumerate(self.output_features):

                pred_clsi = pred_cls[feature].detach().cpu()
                pred_toki = pred_tok[feature].detach().cpu()

                if pred_clsi.shape[-1] == 1:
                    # binary: one logit
                    pred_clsi = torch.sigmoid(pred_clsi).squeeze(-1)
                    pred_clsi = (pred_clsi > 0.5).long().tolist()

                    pred_toki = torch.sigmoid(pred_toki).squeeze(-1)
                    pred_toki = (pred_toki > 0.5).long().tolist()
                else:
                    pred_clsi = np.argmax(pred_clsi, axis=1).tolist()
                    pred_toki = np.argmax(pred_toki, axis=1).tolist()

                all_preds_cls[feature].extend(pred_clsi)
                all_preds_tok[feature].extend(pred_toki)

                y = data[1:][j]
                if not isinstance(y, list):
                    y = y.tolist()
                all_labels[feature].extend(y)

        results = []
        for criterion in self.criteria:
            results.append(criterion(all_labels, (all_preds_cls, all_preds_tok)))

        self.recorder.finish()
        return results
    

@dataclass
class FTMultiTesterConfig:
    model: Module
    data_loader: DataLoader
    output_features: List[str]
    criteria: List[Any] = None
    recorder: Recorder = Recorder()
    stream: bool = False
    
class FTMultiTester:
    def __init__(self, config: FTMultiTesterConfig, device=None):
        self.device = device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.config.model.to(self.device)
        self.config.model.eval()


    def forward(self, X):
        X  = X.to(self.device)
        with torch.no_grad():
            y = self.config.model(X)
        return y

    def forward_stream(self, text):
        return self.forward(self.config.model.tokenize(text))

    def test(self):
        self.config.recorder.start()
        all_labels = {}
        all_preds  = {}

        for feature in self.config.output_features:
            all_preds[feature] = []
            all_labels[feature] = []

        i = 0
        for data in self.config.data_loader:
            i += 1
            self.config.recorder.record(i)
            pred = self.forward_stream(data[0]) if self.config.stream else self.forward(data[0])

            for j, feature in enumerate(self.config.output_features):

                predi = pred[feature].detach().cpu()

                if predi.shape[-1] == 1:
                    # binary: one logit
                    predi = torch.sigmoid(predi).squeeze(-1)
                    predi = (predi > 0.5).long().tolist()
                else:
                    predi = np.argmax(predi, axis=1).tolist()

                all_preds[feature].extend(predi)

                y = data[1:][j]
                if not isinstance(y, list):
                    y = y.tolist()
                all_labels[feature].extend(y)

        results = []
        for criterion in self.config.criteria:
            results.append(criterion(all_labels, all_preds))

        self.config.recorder.finish()
        return results