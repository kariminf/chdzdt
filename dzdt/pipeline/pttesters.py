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
from typing import List, Tuple, Dict, Any

from torch import nn
from torch.utils.data import DataLoader
from dataclasses import dataclass

from dzdt.pipeline.recorders import Recorder
from dzdt.pipeline.preprocessor import Embedder


@dataclass
class SimpleTesterConfig:
    model: nn.Module
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
        self.model = config.model 
        self.data_loader = config.data_loader
        self.embedder = config.embedder
        self.recorder = config.recorder
        self.criteria = config.criteria
        self.stream = config.stream

        self.model.to(self.device)

    def forward(self, X):
        X = X.to(self.device)
        with torch.no_grad():
            pred = self.model(X)
        return pred

    def forward_stream(self, text):
        return self.forward(self.embedder.encode(text))

    def test(self):
        self.recorder.start()
        all_preds, all_labels = [], []
        i = 0
        for batch_X, batch_Y in self.data_loader:
            i += 1
            self.recorder.record(i)
            pred = self.forward_stream(batch_X) if self.stream else self.forward(batch_X)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch_Y)

        results = []
        for criterion in self.criteria:
            results.append(criterion(all_labels, all_preds))

        self.recorder.finish()
