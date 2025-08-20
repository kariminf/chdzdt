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
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from dataclasses import dataclass

from dzdt.pipeline.recorders import Recorder
from dzdt.pipeline.preprocessor import Embedder

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
        self.model = config.model 
        self.data_loader = config.data_loader
        self.optimizer = config.optimizer
        self.criterion = config.criterion
        self.embedder = config.embedder
        self.recorder = config.recorder

        self.model.to(self.device)

    def step(self, X, y):
        X, y = X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        pred = self.model(X)
        # print("X.shape", X.shape, "pred.shape", pred.shape, " y.shape", y.shape)
        loss = self.criterion(pred, y)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def epoch(self):
        epoch_loss = 0.0
        for X, y in self.data_loader: #batch
            loss = self.step(X, y)
            epoch_loss += loss * X.size(0)

        return epoch_loss / len(self.data_loader)

    def epoch_stream(self):
        epoch_loss = 0.0
        for text, y in self.data_loader: #batch
            X = self.embedder.encode(text)
            loss = self.step(X, y)
            epoch_loss += loss * X.size(0)

        return epoch_loss

    def train(self, epochs=100, gamma = None, stream=False):
        self.recorder.start()
        for epoch in range(epochs):
            epoch_loss = self.epoch_stream() if stream else self.epoch()
            self.recorder.record(epoch, epochs, epoch_loss)
            if gamma is not None and epoch_loss <= gamma:
                break
        self.recorder.finish()