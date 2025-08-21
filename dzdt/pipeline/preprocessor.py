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


class Embedder:
    def __init__(self, tokenizer, encoder, device=None, pooling: str =None, one_word=False):
        self.device = device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = tokenizer
        self.encoder  = encoder
        self.pooling = pooling
        self.one_word = one_word

        self.encoder.to(self.device)
        self.encoder.eval()

    def encode(self, text):
        if self.one_word: 
            result = []
            for words in text:
                tokens = self.tokenizer(words, 
                                return_tensors="pt", 
                                padding=True, 
                                truncation=True, 
                                add_special_tokens=True).to(self.device)
                with torch.no_grad():
                    outputs = self.encoder(**tokens)
                
                emb = outputs.last_hidden_state.detach()
                if self.pooling == "cls":
                    emb = emb[:, 0, :]
                elif self.pooling == "mean":
                    emb = emb[:, 1:, :].mean(dim=1)
                result.append(emb)
            
            return torch.stack(result)

        tokens = self.tokenizer(text, 
                                return_tensors="pt", 
                                padding=True, 
                                truncation=True, 
                                add_special_tokens=True).to(self.device)
        with torch.no_grad():
            outputs = self.encoder(**tokens)

        emb = outputs.last_hidden_state.detach()

        if self.pooling == "cls":
            return emb[:, 0, :]
        elif self.pooling == "mean":
            return emb[:, 1:, :].mean(dim=1)
        
        return emb
    
class ClsTokEmbedder:
    def __init__(self, tokenizer, encoder, device=None):
        self.device = device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = tokenizer
        self.encoder  = encoder

        self.encoder.to(self.device)
        self.encoder.eval()

    def encode(self, text):
        tokens = self.tokenizer(text, 
                                return_tensors="pt", 
                                padding=True, 
                                truncation=True, 
                                add_special_tokens=True).to(self.device)
        with torch.no_grad():
            outputs = self.encoder(**tokens)

        emb = outputs.last_hidden_state.detach()
        
        return emb[:, 0, :], emb[:, 1:, :].mean(dim=1)
    