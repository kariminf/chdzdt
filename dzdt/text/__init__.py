#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2023 Abdelkrime Aries <kariminfo0@gmail.com>
#
#  ---- AUTHORS ----
# 2023-2024	Abdelkrime Aries <kariminfo0@gmail.com>
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

from typing import List
from dzdt.text.emojicator import EMOJI_PROCESSOR
from dzdt.text.filterer import CharsetFilterer, DuplicateFilterer, RegexFilterer
from dzdt.text.normalizer import CHAR_NORMALIZER, ENTITY_WORD_NORMALIZER, FuncNormalizer, RegexNormalizer
from dzdt.text.tokenizer import CHARS_SPLITTER, SPACING_PROCESSOR, TOKENS_SPLITTER, RegexSplitter, SeqJoiner, SpacingProcessor, TokensSplitter
from dzdt.tools.chars import ARABIC_TASHKIIL_CHARSET, ARABIC_TATWEEL_ORD, CharManager, charset2string
from dzdt.tools.process import SeqPipeline


SPACE_JOINER = SeqJoiner(join_char=" ")
# DOT_SPACER = RegexNormalizer([(r"([^.])(\.+)$", r"\1 \2"), (r"^(\.+)([^.])", r"\1 \2")])
CHAR_FILTERER = CharsetFilterer(
    CharManager().add_charset(*ARABIC_TASHKIIL_CHARSET).add_charcode(ARABIC_TATWEEL_ORD)
    )

CHAR_DPLCT_FILTERER = DuplicateFilterer()
TOKENS_DPLCT_FILTERER = DuplicateFilterer()

LOWER_NORMALIZER = FuncNormalizer(lambda s: s.lower())

RE_CHAR_FILTERER = RegexFilterer([
    r"[" + charset2string(ARABIC_TASHKIIL_CHARSET) + chr(ARABIC_TATWEEL_ORD) + "]"
])

RE_WORDS_SPLITTER = RegexSplitter()


class PreProcessorConfig:
    def __init__(self, 
                 max_char_dup = 0, 
                 max_word_dup = 0,
                 spacing = False,
                 emoticon = False,
                 normalize_chars = False
                 ):
        self.max_char_dup = max_char_dup
        self.max_word_dup = max_word_dup
        self.spacing = spacing
        self.emoticon = emoticon
        self.normalize_chars = normalize_chars

    def get_pipeline(self) -> SeqPipeline:
        processors = []

        if self.emoticon:
            processors.append(EMOJI_PROCESSOR)

        if self.normalize_chars:
            processors.append(CHAR_NORMALIZER)

        if self.max_char_dup > 0:
            processors.append(DuplicateFilterer(max_dup=self.max_char_dup))
        
        if self.spacing:
            processors.append(SpacingProcessor())

        processors.append(TokensSplitter())

        if self.max_word_dup > 0:
            processors.append(DuplicateFilterer(max_dup=self.max_word_dup))

        return SeqPipeline(processors)


class Preprocessor:
    def __init__(self, config: PreProcessorConfig) -> None:
        self.seq_pieline = config.get_pipeline()

    def process(self, text: str) -> List[str]:
        return self.seq_pieline.process([*text])

RN = RegexNormalizer([
    (r"\.\.", " .. "),
    (r",([^\d])", r" , \1"),
])

NOTAG_NORMALIZER = SeqPipeline([
    CHARS_SPLITTER,
    EMOJI_PROCESSOR,
    CHAR_NORMALIZER,
    CHAR_DPLCT_FILTERER,
    SPACING_PROCESSOR,
    TOKENS_SPLITTER,
    TOKENS_DPLCT_FILTERER,
    # RN,
    SPACE_JOINER,
    ])


TAG_NORMALIZER = SeqPipeline([
    CHARS_SPLITTER,
    EMOJI_PROCESSOR,
    CHAR_NORMALIZER,
    CHAR_DPLCT_FILTERER,
    SPACING_PROCESSOR,
    CHAR_FILTERER,
    TOKENS_SPLITTER,
    TOKENS_DPLCT_FILTERER,
    LOWER_NORMALIZER,
    ENTITY_WORD_NORMALIZER,
    SPACE_JOINER,
    ])
