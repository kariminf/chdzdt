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

import re
from typing import List
from dzdt.tools.chars import NUM, VS16, FITZPATRICK_MANAGER, REGIONAL_INDICATOR_MANAGER, SYMBOLS_MANAGER, ZWJ
from dzdt.tools.process import SeqProcessor, SeqPipeline


NUM_DEC = NUM + ",."


class SpacingProcessor(SeqProcessor[str]):
    """A spacing processor.
       Adding spaces before and after some characters like punctuations.
       Adding spaces before and after emojis.
    """

    def init(self):
        """Initializing the processor.
        """
        self.buf = [" "]
        self.http = ""

    def next(self, c: str) -> List[str]:
        """Check the next character and return a list of processed chars.
        The char can be retained in a buffer, and the processor returns an empty list.
        Then, a trigering char can result in emptying the buffer.
        Which means, processing characters can be delayed until checking their following ones.

        Args:
            c (str): A single character.

        Returns:
            List[str]: A list of processed characters.
        """

        # print(c, self.buf)

        # ==============================
        #           SPACE
        # ==============================

        if (not c) or ((self.buf) and (c + self.buf[-1] == "  ")):
            return []
        
        if c == "\n":
            tmp = self.stop() 
            self.buf = [" "]
            return tmp + [" ", "\n"]
        
        if c == " ":
            tmp = self.stop()
            self.buf = [" "]
            return tmp
        
        # ==============================
        #           URLS
        # ==============================

        if self.http == "http":
            return [c]
        
        if len(self.http) or (c == "h"):
            self.http += c
            if self.http not in ["h", "ht", "htt", "http"]:
                self.http = ""
        
        # ==============================
        #           NUMBERS
        # ==============================
        
        if c in NUM:
            tmp = self.buf
            if not tmp or tmp[-1] == " ":
                self.buf = [c]
                return tmp
            
            if tmp[-1] in NUM_DEC:
                self.buf.append(c)
                return []
            self.buf = [c]
            return tmp + [" "]
            
        if c == ".":
            tmp = self.buf
            if tmp and tmp[-1] in " ":
                self.buf += ["."]
                return []
            if tmp and tmp[-1] in NUM:
                self.buf = ["."]
                return tmp
            if tmp and tmp[-1] == ".":
                self.buf = [" "]
                return [" "] + tmp + ["."]
            
            self.buf = ["."]
            return tmp
        
        if c == ",":
            if self.buf and self.buf[-1] in NUM:
                self.buf.append(c)
                return []
            return self.stop() + [" ", ",", " "]
        
        # ==============================
        #           FLAGS
        # ==============================
        # https://en.wikipedia.org/wiki/Regional_indicator_symbol
        # Separate to consicutive ones
        if REGIONAL_INDICATOR_MANAGER.is_valid(c):
            if not self.buf:
                self.buf = [c]
                return [" "]
            tmp = self.stop()
            if tmp[-1] == " ":
                self.buf = [c]
                return tmp
            if REGIONAL_INDICATOR_MANAGER.is_valid(tmp[-1]):
                tmp += [c]
                self.buf = [" "]
                return tmp
            self.buf = [c]
            return tmp + [" "]
        
        # ==============================
        #           COMPOZED EMOJIS
        # ==============================

        # must be here since ZWJ in SYMBOLS_MANAGER
        if c in ZWJ + VS16:
            if self.buf and SYMBOLS_MANAGER.is_valid(self.buf[-1]):
                tmp = self.buf
                self.buf = [c]
                return tmp
            return [] #ignore it if the buffer does not have a symbole
        
        # must be here since FITZPATRICK in SYMBOLS_MANAGER
        if FITZPATRICK_MANAGER.is_valid(c):
            if self.buf and SYMBOLS_MANAGER.is_valid(self.buf[-1]):
                tmp = self.buf
                self.buf = [c]
                return tmp
            return []
        
        if SYMBOLS_MANAGER.is_valid(c):
            tmp = self.stop()
            self.buf = [c]

            if not tmp:
                tmp = [" "]
            elif tmp[-1] not in " " + ZWJ:
                tmp += [" "]
            
            return tmp
        
        # ==============================
        #           OTHERISE
        # ==============================

        if self.buf:
            tmp = self.buf
            self.buf = []
            if ((tmp[-1] in NUM and len(tmp) < 3)
                or (tmp[-1] == "." and tmp[0] != " ")):
                return tmp + [c]
            return tmp + [" ", c]
        
        return [c]
    
    def stop(self) -> List[str]:
        """When reaching the end of the string, the buffer can be full.
        So, we can return the buffer's content and empty it.

        Returns:
            List[str]: The buffer.
        """
        tmp = self.buf
        if tmp and tmp[-1] in ".,":
            tmp = tmp[:-1] + [" ", tmp[-1]]

        self.init()
        return tmp
    

    
class TokensSplitter(SeqProcessor[str]):
    """A processor to tokenize text
    """

    def __init__(self, split_chars:str = " \t") -> None:
        self.split_chars = split_chars
        super().__init__()

    def init(self) -> None:
        self.buf = ""
    
    def next(self, e: str) -> List[str]:
        if e in self.split_chars:
            return self.stop()
        self.buf += e
        return []
    
    def stop(self) -> List[str]:
        tmp = [self.buf] if self.buf else []
        self.init()
        return tmp 

class CharsSplitter(SeqProcessor[str]):
    def next(self, e: str) -> List[str]:
        return [*e]

class SeqJoiner(SeqProcessor[str]):
    def __init__(self, join_char:str = "") -> None:
        self.join_char = join_char
        super().__init__()

    def init(self) -> None:
        self.buf = ""
    
    def next(self, e: str) -> List[str]:
        if not len(self.buf):
            self.buf = e
        else:
            self.buf += self.join_char + e
        return []
    
    def stop(self) -> List[str]:
        tmp = self.buf
        self.init()
        return [tmp] 
    

class RegexSplitter(SeqProcessor[str]):
    def __init__(self, split_regex:str = "[ \t]") -> None:
        self.split_regex = split_regex
        super().__init__()
    
    def next(self, e: str) -> List[str]:
        return re.split(self.split_regex, e)

    

CHARS_SPLITTER = CharsSplitter()

SPACING_PROCESSOR = SpacingProcessor()
SPACING_PROCESSOR_PIPE = SeqPipeline([CHARS_SPLITTER, SPACING_PROCESSOR])

def add_spacing(text: str) -> str:
    SPACING_PROCESSOR_PIPE.init()
    return "".join(SPACING_PROCESSOR_PIPE.process(text))


TOKENS_SPLITTER = TokensSplitter()
TOKENS_SPLITTER_PIPE = SeqPipeline([CHARS_SPLITTER, TOKENS_SPLITTER])

def to_words(text: str) -> List[str]:
    TOKENS_SPLITTER_PIPE.init()
    return "".join(TOKENS_SPLITTER_PIPE.process(text))
