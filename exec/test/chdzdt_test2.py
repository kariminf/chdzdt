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
import timeit
import pandas as pd

import torch
import numpy as np
from transformers import BertConfig

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dzdt.model.chdzdt_mdl import MultiLabelPredictionHead, OneLabelPredictionHead

config     = BertConfig(
        hidden_size=3,
        num_labels=5,
        )

olh = OneLabelPredictionHead(config)

mdl = MultiLabelPredictionHead(config)

t = torch.tensor([
    [-1., 0., 1.],
    [-0.5, 0.25, 1.5]
])

print(olh(t))

print(mdl(t))