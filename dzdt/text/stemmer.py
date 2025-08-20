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

from typing import List, Tuple

# import sys, os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dzdt.tools.struct import Trie, to_trie, trie_get_prefixes


class DzDtStemmer:
    def __init__(self, prefixes: List[str], suffixes: List[str]) -> None:
        self.prefixes = to_trie(prefixes)
        self.suffixes = to_trie([suffix[::-1] for suffix in suffixes]) # reverse

        # print(self.prefixes)

        # for prefix in prefixes:
        #     self.prefixes.add(prefix, True)
        
        # for suffix in suffixes:
        #     self.suffixes.add(suffix[::-1], True) # reverse

    def stem(self, word: str) -> Tuple[str, str, str]:
        prefix, root, suffix = "", word, ""
        # suffixes = self.suffixes.get_prefixes(root[::-1]) # reverse
        suffixes = trie_get_prefixes(self.suffixes, root[::-1]) # reverse
        # print(suffixes, self.suffixes, root[::-1])
        if suffixes:
            suffix = suffixes[-1][::-1] # reverse the last one
            root = root[:-len(suffix)]
        # prefixes = self.prefixes.get_prefixes(root)
        prefixes = trie_get_prefixes(self.prefixes, root)
        if prefixes:
            prefix = prefixes[-1]
            root = root[len(prefix):]
        return prefix, root, suffix
    
    def get_prefixes(self) -> List[str]:
        return list(self.prefixes.keys())
    
    def get_suffixes(self) -> List[str]:
        return [s[::-1] for s in self.suffixes.keys()]

