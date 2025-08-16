#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2023 Abdelkrime Aries <kariminfo0@gmail.com>
#
#  ---- AUTHORS ----
# 2023-2025	Abdelkrime Aries <kariminfo0@gmail.com>
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

from typing import Any, Union, List, Optional
from dzdt.tools.chars import CharManager, DOUBLE_SPACE
import numpy as np
# from tokenizers.models import Model
# from tokenizers import Tokenizer

import numpy as np

import pickle
import os

# Latin Extended-D (4 letters will be used as special tokens)

class CharTokenizer(CharManager): #, Model
    # kwargs are added just for compatibility with BERT tokenizers
    def __init__(self, max_position=20, **kwargs) -> None:
        super().__init__()

        self.max_position = max_position

        # [PAD], [UNK], [SEP], [CLS], [MASK]
        self.special = "ꝐꞸꟉꞒꞢ"

        self.pad = 0 
        self.unk = 1 
        self.sep = 2 
        self.cls = 3 
        self.mask = 4 

        self.size = 5
        self.charsets_base = []
        self.chars_base = self.size

        self.add_charset(9984,10175) # Dingbats
        self.add_charset(128512, 128591) # Emoticons
        self.add_charset(9728, 9983) # Miscellaneous Symbols 
        self.add_charset(127744, 128511) # Miscellaneous Symbols and Pictographs 
        self.add_charset(129648, 129782) # Symbols and Pictographs Extended-A
        self.add_charset(129280, 129535) # Supplemental Symbols and Pictographs
        self.add_charset(8192, 8303) # General Punctuation
        self.add_charset(128640, 128764) # Transport and Map Symbols
        self.add_charset(127232, 127487) # Enclosed Alphanumeric Supplement
        self.add_chars(DOUBLE_SPACE + ".")
        self.add_charcode(65039) # emoji variation selector

    def __call__(self, words: Union[str, List[str]], add_cls = True, add_sep=True, return_tensors=None, **kwargs) -> object:
        return self.encode_words(words, add_cls=add_cls, add_sep=add_sep, return_tensors=return_tensors)
    
    def state_dict(self):
        """Return serializable attributes only (not methods)."""
        return {
            "max_position": self.max_position,
            "special": self.special,
            "pad": self.pad,
            "unk": self.unk,
            "sep": self.sep,
            "cls": self.cls,
            "mask": self.mask,
            "size": self.size,
            "charsets_base": self.charsets_base,
            "chars_base": self.chars_base,
            "charsets": self.charsets,
            "chars": self.chars
        }

    def load_state_dict(self, state: dict):
        """Restore attributes from a saved dict."""
        self.max_position = state["max_position"]
        self.special = state["special"]
        self.pad = state["pad"]
        self.unk = state["unk"]
        self.sep = state["sep"]
        self.cls = state["cls"]
        self.mask = state["mask"]
        self.size = state["size"]
        self.charsets_base = state["charsets_base"]
        self.chars_base = state["chars_base"]
        self.charsets = state["charsets"]
        self.chars = state["chars"]

    # Override CharManager 
    # ====================

    # def add_charset(self, begin:int, end: int):
    #     self.charsets_base.append(self.size)
    #     self.size += end - begin + 1
    #     self.chars_base = self.size
    #     super().add_charset(begin, end)

    # def add_charcode(self, charcode: int):
    #     self.size += 1
    #     super().add_charcode(charcode)

    # =============
    # encoding
    # =============

    def encode_char(self, char: Union[str, int]) -> int:

        if isinstance(char, str):
            try:
                id = self.special.index(char)
                return id
            except ValueError:
                char = ord(char[0]) # in case user enters a string, we take the first char
            
        for i, be in enumerate(self.charsets):
            if be[0] <= char <= be[1]:
                return char - be[0] + self.charsets_base[i]
            
        if char in self.chars:
            return self.chars.index(char) + self.chars_base
        
        return self.unk
    
    def encode_word(self, word, add_cls = True, add_sep=True) -> np.ndarray:
        # print("word=", word)
        result = [self.pad] * self.max_position
        i = 0
        if add_cls:
            result[0] = self.cls
            i = 1
        end = self.max_position - i - 1

        for char in word:
            if i > end:
                break
            result[i] = self.encode_char(char)
            i += 1

        if i > end:
            i = end
        if add_sep:
            result[i] = self.sep

        return result
    
    def encode_words(self, words: Union[str, List[str], List[List[str]]], add_cls = True, add_sep=True, return_tensors=None) -> object:
        if isinstance(words, str):
            words = [words]

        # input_ids = list(map(lambda word: self._encode_one_word(word), words))

        # return {"input_ids": input_ids}

        if isinstance(words[0], list):
            input_ids = np.array(list(map(lambda word: self.encode_word(word[0]), words)))
        else:
            input_ids = np.array(list(map(lambda word: self.encode_word(word), words)))
        
        attention_mask = (input_ids != self.pad).astype(int)

        if return_tensors == "pt":
            import torch
            input_ids  = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
    # =============
    # decoding
    # =============
    
    def decode_char(self, id: int) -> str:

        if id < 5:
            return self.special[id]
        
        if id >= self.chars_base:
            return chr(self.chars[id - self.chars_base])
        
        
        low, up = 0, len(self.charsets_base) - 1


        while low < up:
            mid = low + int((up - low)/2)

            if (mid == low ) or (id == self.charsets_base[mid]):
                low, up = mid, mid
            elif id < self.charsets_base[mid]:
                up = mid
            else:
                low = mid
        
        return chr(self.charsets[up][0] + id - self.charsets_base[up])
    
    def decode_word(self, word_code) -> str:
        result = ""
        for char_code in word_code:
            result += self.decode_char(char_code)

        return result
    
    def decode_words(self, words_code: List[List[str]], sizes: List[int] = None) -> List[str]:
        if sizes is None:
            return list(map(lambda word_code: self.decode_word(word_code), words_code))
        # result = 
        return list(map(lambda w: self.decode_word(w[0][1:w[1]+1]), zip(words_code, sizes)))
    
    @staticmethod
    def load_old(url: str) -> "CharTokenizer":
        result = None
        with open(url, "rb") as f:
            result = pickle.load(f)

        return result
    
    def save_old(self, url: str):
        with open(url, "wb") as f:
            pickle.dump(self, f)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "rb") as f:
            state = pickle.load(f)
        tok = cls(max_position=state.get("max_position", 20))
        tok.load_state_dict(state)
        return tok
    
    def tokenize(self, sequence):
        if " " in sequence:
            return sequence.split()
        return list(sequence)


