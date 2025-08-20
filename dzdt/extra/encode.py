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

import numpy as np
from sklearn.preprocessing import LabelEncoder


def encode_tags(le: LabelEncoder, Y_tags, train=False):
    # flatten all tags
    all_tags = [tag for seq in Y_tags for tag in seq]
    if train:
        le.fit(all_tags)
    # transform each sequence
    Y_encoded = [le.transform(seq) for seq in Y_tags]
    Y_encoded = np.array(Y_encoded, dtype=np.int64)
    return Y_encoded