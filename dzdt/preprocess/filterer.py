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

from html import unescape
import re
from typing import List
from dzdt.tools.chars import NUM, CharManager
from dzdt.tools.process import SeqPipeline, SeqProcessor
from dzdt.tools.struct import Trie, to_trie, trie_get_node

WEB_TAGS = [
    r"^\[\[[^\]]*(\]\])?$",
    r"\[\[",
    r"^=+",
    r"=+$",
    r"^[-\.]+",
    r"\|[^\]]*\]\]",
    r"\]\]",
    r"<\/?\w+[^>]*>",
]



class DuplicateFilterer(SeqProcessor[str]):
    """A processor to delete duplicates
    """

    def __init__(self, max_dup=2) -> None:
        """Create an instance of a processor which deletes duplicates

        Args:
            max_dup (int, optional): maximum number of allowed consicutive duplicates before starting delition. Defaults to 2.
        """
        super().__init__()
        self.__max_dup = max_dup


    def init(self) -> None:
        self.last = ""
        self.dup = 0
    
    @property
    def max_dup(self) -> int:
        return self.__max_dup

    @max_dup.setter
    def max_dup(self, value: int) -> None:
        self.__max_dup = value

    def next(self, e: str) -> List[str]:
        """Check the next char and return a new one.
        If it will be deleted, an empty string will be returned.

        Args:
            e (str): the char to be checked.

        Returns:
            str: the result, either the same char or an empty string.
        """

        if e in NUM: # do not filter numerals
            self.init()
            return [e]

        if e != self.last:
            self.last = e
            self.dup = 0
            return [e]
        
        self.dup += 1
        
        if self.dup < self.__max_dup:
            return [e]

        return []
    
    
class StringFilterer(SeqProcessor[str]):
    def __init__(self, words: List[str]) -> None:
        self.trie = to_trie(words)

    def next(self, e: str) -> List[str]:
        if trie_get_node(self.trie, e) is not None:
            return []
        return e

# This is the same as the prvious one
# class SetFilterer(SeqProcessor[str]):
#     def __init__(self, chars: List[str]) -> None:
#         self.chars = set(chars)

#     def next(self, e: str) -> List[str]:
#         if e in self.chars:
#             return []
#         return e
    
class RegexFilterer(SeqProcessor[str]):
    def __init__(self, regex: List[str]) -> None:
        self.regex = regex

    def next(self, e: str) -> List[str]:
        for regex in self.regex:
            e = re.sub(regex, "", e)

        return [e]
    
class CharsetFilterer(SeqProcessor[str]):
    def __init__(self, char_manager: CharManager) -> None:
        self.char_manager = char_manager

    def next(self, e: str) -> List[str]:
        if self.char_manager.is_valid(e):
            return []
        return [e]


        
DUPLICATE_FILTERER = DuplicateFilterer()
DUPLICATE_FILTERER_PIPE = SeqPipeline([DUPLICATE_FILTERER])

def delete_duplicate_chars(text: str, max_dup=2) -> str:
    """Delete consecutive duplicate chars

    Args:
        text (str): A text.
        max_dup (int, optional): maximum allowed duplication. Defaults to 2.

    Returns:
        str: The text without duplication.
    """
    DUPLICATE_FILTERER_PIPE.init()
    DUPLICATE_FILTERER.max_dup = max_dup
    return "".join(DUPLICATE_FILTERER_PIPE.process([*text]))

def delete_duplicate_words(text: str, max_dup=2) -> str:
    DUPLICATE_FILTERER_PIPE.init()
    DUPLICATE_FILTERER.max_dup = max_dup
    return " ".join(DUPLICATE_FILTERER_PIPE.process(re.split("[ \t]+", text)))


# def minimize(text: str) -> str:
#     text = text.lower()
#     diac = "[" + "".join(TASHKIIL+TANWIIN+[u'ّ', u'ـ']) + "]"
#     return re.sub(diac, "", text) 


WEB_TAGS_FILTERER = RegexFilterer(WEB_TAGS)
WEB_TAGS_FILTERER_PIPE = SeqPipeline([WEB_TAGS_FILTERER])

def filter_web_tags_in_line(line: str) -> str:
    WEB_TAGS_FILTERER_PIPE.init()
    return unescape(WEB_TAGS_FILTERER_PIPE.process([line]))


def filter_characters(text: str) -> str:
    text = re.sub(r"[_]+", " ", text)
    return text