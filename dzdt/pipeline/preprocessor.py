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

import unicodedata

from transformers.models.flaubert.tokenization_flaubert import replace_unicode_punct, convert_to_unicode, remove_non_printing_char
try:
    import sacremoses as sm

    moses_normalizer = sm.MosesPunctNormalizer(lang="fr")
except ImportError:
    raise ImportError(
        "You need to install sacremoses to use FlaubertTokenizer. "
        "See https://pypi.org/project/sacremoses/ for installation."
    )
    sm = {}

def preprocess_text(text):
    text = text.replace("``", '"').replace("''", '"')
    text = convert_to_unicode(text)
    text = unicodedata.normalize("NFC", text)

    text = text.lower()

    return text

def moses_pipeline(text):
    text = replace_unicode_punct(text)
    text = moses_normalizer.normalize(text)
    text = remove_non_printing_char(text)
    return text



class Embedder:
    def __init__(self, tokenizer, encoder, device=None, pretokenized=False):
        self.device = device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = tokenizer
        self.encoder  = encoder
        self.pretokenized = pretokenized

        self.encoder.to(self.device)
        self.encoder.eval()

    def _tokenize(self, text):
        return self.tokenizer(
            text,
            return_tensors="pt",
            is_split_into_words=self.pretokenized,
            padding=True,
            truncation=True,
            add_special_tokens=True
        )
    
    def _encode(self, tokens):
        with torch.no_grad():
            outputs = self.encoder(**tokens)

        return outputs.last_hidden_state

    def encode(self, text):
        """
        Abstract method to encode input text.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class BertEmbedder(Embedder):
    def __init__(self, tokenizer, encoder, device=None, pretokenized=False, pooling: str = None,  word_mask=None):
        super().__init__(tokenizer, encoder, device=device, pretokenized=pretokenized)
        self.pooling = pooling
        if (word_mask is not None) and (word_mask not in ["##", "</w>", "fast", "cmp"]):
            raise ValueError(f"word_mask ={word_mask} is not in ['##', '</w>', 'fast', 'cmp'] ")
        self.word_mask = word_mask
        self.special = set()
        for v in self.tokenizer.special_tokens_map.values():
            if isinstance(v, list):
                self.special.update(v)   # add all tokens from the list
            else:
                self.special.add(v)      # add single token

    def _get_mask_fast(self, tokens):
        mask = []
        for i in range(len(tokens["input_ids"])):
            word_ids = tokens.word_ids(batch_index=i)
            labels = []
            past_word = None
            for w in word_ids:
                if w is None:
                    labels.append(False)       # special tokens + padding
                elif w == past_word:
                    labels.append(False)       # continuation subword
                else:
                    labels.append(True)        # first subword
                past_word = w
            mask.append(labels)
        return torch.tensor(mask)
    
    def _get_mask_cmp(self, tokens, text):
        mask = []

        for ids, words in zip(tokens["input_ids"], text):
            seq = self.tokenizer.convert_ids_to_tokens(ids)
            j, l = 0, len(words)
            labels = []
            for i in range(len(ids)): 
                if j == l:
                    labels.append(False)
                    continue
                # current_word = replace_unicode_punct(words[j].lower()) 
                # current_word = current_word.replace("«", '"').replace("»", '"')

                current_word = preprocess_text(words[j])
                current_word = moses_pipeline(current_word)
                current_sub = seq[i]
                if current_sub.endswith("</w>"):
                    current_sub = current_sub[:-4]
                if current_word.startswith(current_sub) or current_sub=="<unk>":
                    labels.append(True)
                    j += 1
                else:
                    labels.append(False)
            mask.append(labels)

            # msk2 = np.array(words) != ""
            # if np.array(labels).sum() != msk2.sum():
            #     print(labels, words, seq, np.array(labels).sum(), msk2.sum())
            #     exit(0)
        
        return torch.tensor(mask)


    def _get_mask_2sharp(self, tokens):
        mask = []
        for ids in tokens["input_ids"]:
            seq_tokens = self.tokenizer.convert_ids_to_tokens(ids)
            labels = []
            for tok in seq_tokens:
                if tok in self.special:
                    labels.append(False)          # ignore special tokens
                elif tok.startswith("##"):       
                    labels.append(False)          # subword continuation
                else:
                    labels.append(True)           # first subword of a word
            mask.append(labels)
        return torch.tensor(mask)
    
    def _get_mask_endw(self, tokens):
        mask = []
        # special_tokens = set(self.tokenizer.special_tokens_map.values())
        for ids in tokens["input_ids"]:
            seq_tokens = self.tokenizer.convert_ids_to_tokens(ids)
            labels = []
            is_new_word = True
            for tok in seq_tokens:
                if tok in self.special:
                    labels.append(False)          # ignore special tokens
                    is_new_word = True                 # initialize the start for the next found one
                elif is_new_word:
                    labels.append(True)
                    is_new_word = tok.endswith("</w>")
                else:
                    labels.append(False)
                    is_new_word = tok.endswith("</w>")

            mask.append(labels)
        return torch.tensor(mask)
        

    def encode(self, text):

        tokens = self._tokenize(text)

        if self.word_mask is not None:
            if self.word_mask == "##":
                mask = self._get_mask_2sharp(tokens)
            elif self.word_mask == "</w>":
                mask = self._get_mask_endw(tokens)
            elif self.word_mask == "fast":
                mask = self._get_mask_fast(tokens)  
            elif self.word_mask == "cmp":
                mask = self._get_mask_cmp(tokens, text) 
            else:
                self.word_mask = None
        
        tokens = tokens.to(self.device)
        emb = self._encode(tokens)

        if self.word_mask:
            return emb, mask

        if self.pooling == "cls":
            emb = emb[:, 0, :]
        elif self.pooling == "mean":
            emb = emb[:, 1:, :].mean(dim=1)
        

        return emb


class DzDTEmbedder(Embedder):
    def __init__(self, tokenizer, encoder, device=None, pooling: str = None, one_word=False):
        super().__init__(tokenizer, encoder, device=device)
        self.pooling = pooling
        self.one_word = one_word

    def encode(self, text):
        if self.one_word: 
            result = []
            for words in text:
                tokens = self._tokenize(words)
                tokens = tokens.to(self.device)
                emb = self._encode(tokens)
                result.append(emb[:, 0, :])
            return torch.stack(result)

        tokens = self._tokenize(text)
        tokens = tokens.to(self.device)
        emb = self._encode(tokens)
        if self.pooling == "cls":
            emb = emb[:, 0, :]
        elif self.pooling == "mean":
            emb = emb[:, 1:, :].mean(dim=1)

        return emb
    

class ClsTokEmbedder(Embedder):
    def __init__(self, tokenizer, encoder, device=None):
        super().__init__(tokenizer, encoder, device=device)

    def encode(self, text):
        tokens = self._tokenize(text)
        tokens = tokens.to(self.device)
        emb = self._encode(tokens)
        return emb[:, 0, :], emb[:, 1:, :].mean(dim=1)
    