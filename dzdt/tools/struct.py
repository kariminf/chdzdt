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

from collections import deque
from collections.abc import Iterator
from typing import Any, Dict, List, NewType, Tuple, Union

from dzdt.tools.process import SeqProcessor

# import sys, os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Trie = NewType("Trie", dict)

class Trie(dict): pass

def trie_get_node(node: Trie, key: str) -> Union[Trie, None]:
    for k in key:
        if k not in node:
            return None
        node = node[k] 
    return node

def trie_add_value(node: Trie, key: str, value: Any) -> bool:
    for k in key:
        if k not in node:
            node[k] = {}
        node = node[k] 
    if "" not in node:
        node[""] = value
        return True
    return False

def trie_del_key(node: Trie, key: str) -> None:
    stack_v = deque([])
    stack_k = deque([])
    for k in key:
        if k not in node:
            raise KeyError(key)
        stack_v.appendleft(node)
        stack_k.appendleft(k)
        node = node[k] 
    if "" not in node:
        raise KeyError(key)
    del node[""] 
    for k, n in zip(stack_k, stack_v):
        if not len(n[k]):
            del n[k]

def trie_to_iterator(node: Trie) -> Iterator:
    stack_k = [""]
    stack_v = [node]
    while stack_v: 
        node = stack_v.pop()
        value = stack_k.pop()
        for k, v in node.items():
            if k == "":
                yield value + k, v
            else:
                stack_v.append(v)
                stack_k.append(value + k)

def trie_get_prefixes(node: Trie, key: str, value: Any = None, get_value: bool = False) -> List[str]:
    prefixes = []
    prefix = ""
    for k in key:
        if k not in node:
            break
        node = node[k] 
        prefix = prefix + k
        if ("" in node):
            myvalue = node[""]
            if (value is None) or (value == myvalue):
                prefixes.append((prefix, myvalue) if get_value else prefix)
        
    return prefixes

def to_trie(comp: Union[List[Tuple[str, str]], List[str]]) -> Trie:
    root = Trie()
    if not isinstance(comp[0], tuple):
        comp = zip(comp, [True] * len(comp))
    for key, value in comp:
        trie_add_value(root, key, value)
    return root

class TrieClass:
    def __init__(self) -> None:
        self.__data = {}
        self.__size = 0

    def __sizeof__(self) -> int:
        return self.__data.__sizeof__()

    def __contains__(self, key: str) -> bool:
        return trie_get_node(self.__data, key) is not None

    def __setitem__(self, key: str, value: Any) -> None:
        if trie_add_value(self.__data, key, value):
            self.__size += 1

    def __getitem__(self, key: str) -> Any:
        node = trie_get_node(self.__data, key)
        if node is None:
            return None
        return node.get("")

    def __delitem__(self, key: str) -> None:
        trie_del_key(self.__data, key)

    def __iter__(self) -> Iterator:
        return trie_to_iterator(self.__data)
    
    def __len__(self) -> int:
        return self.__size

    def __repr__(self) -> str:
        return repr(self.__data)
    
    def __str__(self) -> str:
        return str(self.items())
    
    def add(self, key: str, value: Any) -> None:
        if trie_add_value(self.__data, key, value):
            self.__size += 1

    def get(self, key: str) -> Any:
        node = trie_get_node(self.__data, key)
        if node is None:
            return None
        return node.get("")
    
    def keys(self) -> Iterator:
        for k, v in trie_to_iterator(self.__data):
            yield k
    
    def values(self) -> Iterator:
        for k, v in trie_to_iterator(self.__data):
            yield v

    @property
    def data(self):
        return self.__data

    def get_prefixes(self, key: str, value: Any = None, get_value: bool = False) -> Union[List[str], List[Tuple[str, Any]]]:
        return trie_get_prefixes(self.__data, key, value=value, get_value=get_value)
    
    # def init_process(self):
    #     self.buf = []
    #     self.accept = []
    #     self.icon = []
    #     self.node = self.__data

    # def next_process(self, c: str) -> Tuple[str, List[str]]:

    #     if c in self.node:
    #         self.buf.append(c)
    #         self.accept.append("" in self.node[c])
    #         self.icon.append(self.node[c].get("", ""))
    #         self.node = self.node[c]
    #         return "", []

    #     # c not in self.node
    #     icon = ""
    #     rest = self.buf
    #     if self.buf:
    #         try:
    #             i = len(self.buf) - 1 - self.accept[::-1].index(True)
    #             icon = self.icon[i] 
    #             rest = self.buf[i+1:]
    #         except ValueError:
    #             pass

    #     self.init_process()
    #     return icon, rest + [c]

    # def stop_process(self) -> Tuple[str, List[str]]:
    #     result = self.next_process("")
    #     self.init_process()
    #     return result


class TrieValueProcessor(SeqProcessor):
    def __init__(self, trie: Trie) -> None:
        self.trie = trie
        super().__init__()

    def init(self) -> None:
        self.result = ""
        self.remain = []
        self.node = self.trie

    def next(self, element: str) -> List[str]:
        if element in self.node:
            self.node = self.node[element]
            if "" in self.node:
                self.result = self.node[""]
                self.remain = []
            else:
                self.remain.append(element)
            return []
        result = self.stop()
        if element in self.node:
            self.next(element)
        else:
            result = result + [element]
        return result
    
    def stop(self) -> List[str]:
        result = ([self.result] if len(self.result) else []) + self.remain
        self.init()
        return result
    

# class TrieKeyProcessor(SeqProcessor):
#     def __init__(self, trie: Trie) -> None:
#         self.trie = trie
#         super().__init__()

#     def init(self) -> None:
#         self.result = ""
#         self.remain = []
#         self.node = self.trie.data

#     def next(self, element: str) -> List[str]:
#         if element in self.node:
#             self.node = self.node[element]
#             if "" in self.node:
#                 self.result = self.node[""]
#                 self.remain = []
#             else:
#                 self.remain.append(element)
#             return []
#         result = self.stop()
#         if element in self.node:
#             self.next(element)
#         else:
#             result = result + [element]
#         return result
    
#     def stop(self) -> List[str]:
#         result = ([self.result] if len(self.result) else []) + self.remain
#         self.init()
#         return result


class BatchEncoder(dict):
    def to(self, device):
        return BatchEncoder({k: v.to(device) for k, v in self.items()})
    

class ObjectDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ObjectDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value