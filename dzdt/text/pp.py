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

from typing import List, Union
from itertools import repeat

def pad_truncate(sentences: Union[List[str], List[List[str]]], max_words: int) -> List[List[str]]:
    """
    Pads or truncates each sentence in the input to a fixed number of words.

    Parameters:
        sentences (Union[List[str], List[List[str]]]): A list of sentences, either as strings or lists of words.
        max_words (int): The desired number of words per sentence.

    Returns:
        List[List[str]]: A list of sentences, each represented as a list of words with length equal to max_words.
    """
    if not sentences:
        return []
    should_split = isinstance(sentences[0], str)
    sent_words = []
    for sentence in sentences:
        if should_split:
            sentence = sentence.split()
        l = len(sentence)
        if l > max_words:
            sentence = sentence[:max_words]
        else:
            sentence = list(sentence) + list(repeat("", max_words - l))
        sent_words.append(sentence)
    return sent_words