#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2023 Abdelkrime Aries <kariminfo0@gmail.com>
#
#  ---- AUTHORS ----
# 2023	Abdelkrime Aries <kariminfo0@gmail.com>
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
from typing import Any, Callable, List, Tuple, Union

from dzdt.tools.const import BREAK_TAG, HASH_TAG, MAIL_TAG, REF_TAG, URL_TAG, NBR_TAG

import re

from dzdt.tools.process import SeqPipeline, SeqProcessor


# URL_RE = re.compile("(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))")

URL_RE = re.compile("(^|\s)(https?://\w[\w\d.-]*\.\w{2,6})(/[^\s]*)?(\s|$)")
NBR_RE = r"(^|\s\(?)[+-]?\d[\d.,]*(%?\)?\s|%?\)?$)"

SINGLE_QUOTATION = "'‘’‚‛"
DOUBLE_QUOTATION = '"«»“”„‟‹›❛❜❝❞❟❠❮❯⹂〝〞〟＂🙶🙷🙸'
HYPHEN = "-­֊᐀᠆‐‑‧⁃⸗⸚⹀゠﹣－"
DASH = "‒–—┄┅┈┉╌╍⸻⹃〜〰﹘"
BAR = "|┆┇┊┋╎╏︱"
SPACE = "   "

SIMILAR_CHARS = [
    SINGLE_QUOTATION, DOUBLE_QUOTATION, HYPHEN, DASH, BAR, SPACE
]

# "[hkw°٪غمد$]|mah|gb|دج|hz|mp|fps|km|am|pm|ml|مل|"

ENTITIES_SENTENCE = [
    ("https?://\w[\w\d.-]*\.\w{2,6}(/[^\s]*)? ", URL_TAG + " "),
    (" [+-]?\d[\d.,]*((?=%?\))? )", " " + NBR_TAG + r" \2")
]

ENTITIES_WORD = [
    ("^https?://\w[\w\d.-]*\.\w{2,6}(/[^\s]*)?$", URL_TAG),
    ("^\d[\d.,]*$", NBR_TAG),
    ("^\#.+$", HASH_TAG),
    ("^.+@.+\..+$", MAIL_TAG),
    ("^@.+$", REF_TAG),
    ("[\n]", BREAK_TAG)
]

class SeqNormalizer(SeqProcessor[str]):
    def __init__(self, norm_list: Union[List[str], List[List[Any]]]) -> None:
        self.map = {}
        for values in norm_list:
            for value in values[1:]:
                self.map[value] = values[0]

    def next(self, e: str) -> List[str]:
        return [self.map.get(e, e)]
    
class RegexNormalizer(SeqProcessor[str]):
    def __init__(self, regex: List[Tuple[str, str]]) -> None:
        self.map = regex

    def next(self, e: str) -> List[str]:
        for k, v in self.map:
            e = re.sub(k, v, e)
            # m = re.match(k, e)
            # if m: return [v]
        return [e]
    
class FuncNormalizer(SeqProcessor[str]):
    def __init__(self, func: Callable[[str], Union[str, List[str]]]) -> None:
        self.func = func

    def next(self, e: str) -> List[str]:
        res = self.func(e)
        if not isinstance(res, list): res = [res]
        return res

CHAR_NORMALIZER = SeqNormalizer(SIMILAR_CHARS)
CHAR_NORMALIZER_PIPE = SeqPipeline([CHAR_NORMALIZER])

def normalize_chars(text: str) -> str:
    return "".join(CHAR_NORMALIZER_PIPE.process([*text]))


# def normalize_words(text: str, max_dup=2) -> str:

ENTITY_WORD_NORMALIZER = RegexNormalizer(ENTITIES_WORD)

ENTITY_SNT_NORMALIZER = RegexNormalizer(ENTITIES_SENTENCE)
ENTITY_SNT_NORMALIZER_PIPE = SeqPipeline([CHAR_NORMALIZER])

def normalize_entities(text: str) -> str:
    return "".join(ENTITY_SNT_NORMALIZER_PIPE.process(re.split("[ \t]+", text)))

# def normalize_entities(text: str) -> str:
#     text = text.replace(" ", "  ")
#     text = URL_RE.sub(" " + URL_TAG + " ", text)
#     text = re.sub(NBR_RE, r" \1 " + NBR_TAG + r" \2 ", text)
#     return re.sub("\b{2,}", " ", text).strip()

# def filter_tags(text: str) -> str:
#     result = ""
#     for l in text.splitlines(keepends=True):
#         result += filter_tags_in_line(l)
#     return re.sub("\n{2,}", "\n", result)

