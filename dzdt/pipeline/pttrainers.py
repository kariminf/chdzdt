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
from torch.nn import Module
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import List, Any

from dzdt.pipeline.recorders import Recorder
from dzdt.pipeline.preprocessor import Embedder, ClsTokEmbedder

@dataclass
class SimpleTrainerConfig:
    model: Module
    data_loader: DataLoader
    optimizer: Optimizer
    criterion: Module
    embedder: Embedder = None
    recorder: Recorder = Recorder()
    stream: bool = False

class SimpleTrainer:
    def __init__(self, config: SimpleTrainerConfig, device=None):
        self.device = device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config 

        self.config.model.to(self.device)
        self.config.model.train()

    def step(self, X, y):
        X, y = X.to(self.device), y.to(self.device)
        self.config.optimizer.zero_grad()
        pred = self.config.model(X)
        loss = self.config.criterion(pred, y)
        loss.backward()
        self.config.optimizer.step()

        return loss.item()

    def epoch(self):
        epoch_loss = 0.0
        for X, y in self.config.data_loader: #batch
            loss = self.step(X, y)
            epoch_loss += loss * X.size(0)

        return epoch_loss / len(self.config.data_loader.dataset)

    def epoch_stream(self):
        epoch_loss = 0.0
        for text, y in self.config.data_loader: #batch
            X = self.config.embedder.encode(text)
            loss = self.step(X, y)
            epoch_loss += loss * len(text)

        return epoch_loss / len(self.config.data_loader.dataset)

    def train(self, epochs=100, gamma = None):
        self.config.recorder.start()
        for epoch in range(epochs):
            epoch_loss = self.epoch_stream() if self.config.stream else self.epoch()
            # print("perfect!!!")
            # exit(0)
            self.config.recorder.record(epoch, epochs, epoch_loss)
            if gamma is not None and epoch_loss <= gamma:
                break
        self.config.recorder.finish()


class FTSimpleTrainer(SimpleTrainer):

    def epoch_stream(self):
        epoch_loss = 0.0
        for text, y in self.config.data_loader: #batch
            X = self.config.model.tokenize_encode(text, self.device)
            loss = self.step(X, y)
            epoch_loss += loss * len(text)

        return epoch_loss / len(self.config.data_loader.dataset)

class MaskedSimpleTrainer(SimpleTrainer):
    def step(self, X_m, y):
        X, mask = X_m
        X, y = X.to(self.device), y.to(self.device)
        self.config.optimizer.zero_grad()
        pred = self.config.model(X)
        loss = self.config.criterion((pred, mask), y)
        loss.backward()
        self.config.optimizer.step()

        return loss.item()


@dataclass
class MultiTrainerConfig:
    model: Module
    optimizer: Optimizer
    data_loader: DataLoader
    output_features: List[str]
    embedder: Embedder = None
    recorder: Recorder = Recorder()
    stream: bool = False

    
class MultiTrainer:
    def __init__(self, config: MultiTrainerConfig, device=None):
        self.device = device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config 

        self.config.model.to(self.device)
        self.config.model.train()

        self.multiclass_criterion = nn.CrossEntropyLoss()
        self.binary_criterion = nn.BCEWithLogitsLoss()


    def step(self, X, y):
        X = X.to(self.device)
        losses = [0.0] * len(self.config.output_features)
        self.config.optimizer.zero_grad()
        pred = self.config.model(X)
        loss = 0.0
        for j, feature in enumerate(self.config.output_features):

            if pred[feature].shape[-1] == 1:
                lossi = self.binary_criterion(pred[feature].squeeze(), y[j])
            else:
                lossi = self.multiclass_criterion(pred[feature], y[j])
            losses[j] += lossi.item()
            loss += lossi
        loss.backward()
        self.config.optimizer.step()
        return losses
        
    def epoch(self):
        epoch_losses = [0.0] * len(self.config.output_features)
        data_size = len(self.config.data_loader.dataset)
        for data in self.config.data_loader: #batch
            X = data[0]
            y = [d.to(self.device) for d in data[1:]]
            losses = self.step(X, y)

            weight = len(X[0])/data_size

            for i in range(len(self.config.output_features)):
                epoch_losses[i] += losses[i] * weight

        return epoch_losses

    def epoch_stream(self):
        epoch_losses = [0.0] * len(self.config.output_features)
        data_size = len(self.config.data_loader.dataset)
        for data in self.config.data_loader: #batch
            X = self.config.embedder.encode(data[0])
            y = [d.to(self.device) for d in data[1:]]
            losses = self.step(X, y)

            weight = len(X[0])/data_size

            for i in range(len(self.config.output_features)):
                epoch_losses[i] += losses[i] * weight

        return epoch_losses


    def train(self, epochs=100, gamma = None):
        self.config.recorder.start()
        for epoch in range(epochs):
            epoch_losses = self.epoch_stream() if self.config.stream else self.epoch()
            self.config.recorder.record(epoch, epochs, (self.config.output_features, epoch_losses))
        self.config.recorder.finish()

@dataclass
class ClsTokMultiTrainerConfig:
    cls_model: Module
    tok_model: Module
    cls_optimizer: Optimizer
    tok_optimizer: Optimizer
    data_loader: DataLoader
    output_features: List[str]
    embedder: ClsTokEmbedder = None
    recorder: Recorder = Recorder()
    stream: bool = False

    
class ClsTokMultiTrainer:
    def __init__(self, config: ClsTokMultiTrainerConfig, device=None):
        self.device = device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config 

        self.config.cls_model.to(self.device)
        self.config.tok_model.to(self.device)
        
        self.config.cls_model.train()
        self.config.tok_model.train()

        self.multiclass_criterion = nn.CrossEntropyLoss()
        self.binary_criterion = nn.BCEWithLogitsLoss()

    def step_one(self, X, y, model, optimizer):
        X = X.to(self.device)
        losses = [0.0] * len(self.config.output_features)
        optimizer.zero_grad()
        pred = model(X)
        loss = 0.0
        for j, feature in enumerate(self.config.output_features):

            if pred[feature].shape[-1] == 1:
                lossi = self.binary_criterion(pred[feature].squeeze(), y[j])
            else:
                lossi = self.multiclass_criterion(pred[feature], y[j])
            losses[j] += lossi.item()
            loss += lossi
        loss.backward()
        optimizer.step()
        return losses

    def step(self, X, y):
        X_cls, X_tok = X
        cls_losses = self.step_one(X_cls, y, self.config.cls_model, self.config.cls_optimizer)
        tok_losses = self.step_one(X_tok, y, self.config.tok_model, self.config.tok_optimizer)

        return cls_losses, tok_losses

    def epoch(self):
        cls_epoch_losses = [0.0] * len(self.config.output_features)
        tok_epoch_losses = [0.0] * len(self.config.output_features)
        data_size = len(self.config.data_loader.dataset)
        for data in self.config.data_loader: #batch
            X = data[0]
            y = [d.to(self.device) for d in data[1:]]
            cls_losses, tok_losses = self.step(X, y)

            weight = len(X[0])/data_size

            for i in range(len(self.config.output_features)):
                cls_epoch_losses[i] += cls_losses[i] * weight
                tok_epoch_losses[i] += tok_losses[i] * weight

        return cls_epoch_losses, tok_epoch_losses

    def epoch_stream(self):
        cls_epoch_losses = [0.0] * len(self.config.output_features)
        tok_epoch_losses = [0.0] * len(self.config.output_features)
        data_size = len(self.config.data_loader.dataset)
        for data in self.config.data_loader: #batch
            X = self.config.embedder.encode(data[0])
            y = [d.to(self.device) for d in data[1:]]
            cls_losses, tok_losses = self.step(X, y)

            weight = len(X[0])/data_size

            for i in range(len(self.config.output_features)):
                cls_epoch_losses[i] += cls_losses[i] * weight
                tok_epoch_losses[i] += tok_losses[i] * weight

        return cls_epoch_losses, tok_epoch_losses


    def train(self, epochs=100, gamma = None):
        self.config.recorder.start()
        for epoch in range(epochs):
            cls_epoch_losses, tok_epoch_losses = self.epoch_stream() if self.config.stream else self.epoch()
            self.config.recorder.record(epoch, epochs, (self.config.output_features, cls_epoch_losses, tok_epoch_losses))
        self.config.recorder.finish()


@dataclass
class FTMultiTrainerConfig:
    model: Module
    optimizer: Optimizer
    data_loader: DataLoader
    output_features: List[str]
    recorder: Recorder = Recorder()
    stream: bool = False

    
class FTMultiTrainer:
    def __init__(self, config: FTMultiTrainerConfig, device=None):
        self.device = device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config 

        self.config.model.to(self.device)
        
        self.config.model.train()

        self.multiclass_criterion = nn.CrossEntropyLoss()
        self.binary_criterion = nn.BCEWithLogitsLoss()

    def step(self, X, y):
        X = X.to(self.device)
        losses = [0.0] * len(self.config.output_features)
        self.config.optimizer.zero_grad()
        pred = self.config.model(X)
        loss = 0.0
        for j, feature in enumerate(self.config.output_features):

            if pred[feature].shape[-1] == 1:
                lossi = self.binary_criterion(pred[feature].squeeze(), y[j])
            else:
                lossi = self.multiclass_criterion(pred[feature], y[j])
            losses[j] += lossi.item()
            loss += lossi
        loss.backward()
        self.config.optimizer.step()
        return losses

    def epoch(self):
        epoch_losses = [0.0] * len(self.config.output_features)
        data_size = len(self.config.data_loader.dataset)
        for data in self.config.data_loader: #batch
            X = data[0]
            y = [d.to(self.device) for d in data[1:]]
            losses = self.step(X, y)

            weight = len(X)/data_size

            for i in range(len(self.config.output_features)):
                epoch_losses[i] += losses[i] * weight

        return epoch_losses

    def epoch_stream(self):
        epoch_losses = [0.0] * len(self.config.output_features)
        data_size = len(self.config.data_loader.dataset)
        for data in self.config.data_loader: #batch
            X = self.config.model.tokenize(data[0])
            y = [d.to(self.device) for d in data[1:]]
            losses = self.step(X, y)

            weight = len(X)/data_size

            for i in range(len(self.config.output_features)):
                epoch_losses[i] += losses[i] * weight

        return epoch_losses


    def train(self, epochs=100, gamma = None):
        self.config.recorder.start()
        for epoch in range(epochs):
            epoch_losses = self.epoch_stream() if self.config.stream else self.epoch()
            self.config.recorder.record(epoch, epochs, (self.config.output_features, epoch_losses))
        self.config.recorder.finish()