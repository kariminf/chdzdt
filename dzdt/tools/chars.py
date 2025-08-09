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

# Chars which are normalized by a double space (before and after)
from typing import List, Tuple


NUM = "0123456789"
DOUBLE_SPACE = "°\n!\"<>$%=&':()*+-,;[]^`{\\|}~/?؛،؟"
ZWJ = "\u200D" # ZERO WIDTH JOINER
VS16 = "\uFE0F" # VARIATION SELECTOR-16

ARABIC_TASHKIIL_CHARSET = (1611, 1630)
ARABIC_TATWEEL_ORD = 1600

# TASHKIIL	= [u'ِ', u'ُ', u'َ', u'ْ']
# TANWIIN		= [u'ٍ', u'ٌ', u'ً']

DINGBATS_CHARSET = (9984,10175)
EMOTICONS_CHARSET = (128512, 128591)
MISC_SYMB_CHARSET = (9728, 9983)
MISC_SYMB_PICTO_CHARSET = (127744, 128511)
SYMB_PICTO_EXT1_CHARSET = (129648, 129782)
SUPP_SYMB_PICTO_CHARSET = (129280, 129535)
GENERAL_PUNCT_CHARSET = (8192, 8303) # ZWJ is here
TRANSPORT_MAP_SYMB_CHARSET = (128640, 128764)
ENCLOSED_ALPHANUM_SUPP_CHARSET = (127232, 127487)
EMOJI_VAR_SELECTOR_ORD = 65039

MODIF_FITZPATRICK_CHARSET = (127995, 127999) #(skin color)

REGIONAL_INDIC_SYMB_CHARSET = (127462, 127487)


def charset2string(charset: Tuple[int, int]) -> str:
    result = ""
    for i in range(*charset):
        result += chr(i)
    return result


class CharManager:
    """A charset manager. 
    You can add charsets and chars to create a net set.
    Then you can check if a char belongs to this set.
    """
    def __init__(self) -> None:
        self.charsets: List[Tuple[int, int]] = []
        self.chars: List[int] = []

        self.size = 0
        self.charsets_base = []
        self.chars_base = 0

    def add_charset(self, begin:int, end: int) -> "CharManager":
        """Add a charset

        Args:
            begin (int): The first char code in decimal.
            end (int): The last char code in decimal.
        """
        self.charsets_base.append(self.size)
        add = end - begin + 1
        self.size += add
        self.chars_base += add
        self.charsets.append((begin, end))
        return self

    def add_charsets(self, charsets_list: List[Tuple[int, int]]) -> "CharManager":
        for begin, end in charsets_list:
            self.add_charset(begin, end)
        return self

    def add_charcode(self, charcode: int) -> "CharManager":
        """Add a single char.

        Args:
            charcode (int): The char code in decimal.
        """
        self.size += 1
        self.chars.append(charcode)
        return self
    
    def add_chars(self, chars: str) -> "CharManager":
        """Add many chars at once

        Args:
            chars (str): A string containing the chars to be added.
        """
        for char in chars:
            c = ord(char)
            self.add_charcode(c)
        return self

    def is_valid(self, char: str) -> bool:
        """Check if a given char is a valid one; i.e. belonging to this set.

        Args:
            char (str): A single char. If a string is passed, the first one will be considered.

        Returns:
            bool: A boolean expressing if the char belongs to the set or not.
        """
        c = ord(char[0]) #If you pass a string with more one char, the first one will be considered
        if c in self.chars:
            return True
        for begin, end in self.charsets:
            if begin <= c <= end:
                return True
        return False
    

SYMBOLS_MANAGER = CharManager()
SYMBOLS_MANAGER.add_charsets([
    DINGBATS_CHARSET, EMOTICONS_CHARSET, MISC_SYMB_CHARSET,
    MISC_SYMB_PICTO_CHARSET, SYMB_PICTO_EXT1_CHARSET, SUPP_SYMB_PICTO_CHARSET, 
    GENERAL_PUNCT_CHARSET, TRANSPORT_MAP_SYMB_CHARSET, ENCLOSED_ALPHANUM_SUPP_CHARSET,
])
SYMBOLS_MANAGER.add_chars(DOUBLE_SPACE)
SYMBOLS_MANAGER.add_charcode(EMOJI_VAR_SELECTOR_ORD)


REGIONAL_INDICATOR_MANAGER = CharManager()
REGIONAL_INDICATOR_MANAGER.add_charset(*REGIONAL_INDIC_SYMB_CHARSET)



# _COMPONENT_EMOJI_ = CharManager()
# _COMPONENT_EMOJI_.add_charset(127995, 127999) # MODIFIER FITZPATRICK (skin color)
# _COMPONENT_EMOJI_.add_charset(129456, 129459) # Component (hair style)

FITZPATRICK_MANAGER = CharManager()
FITZPATRICK_MANAGER.add_charset(*MODIF_FITZPATRICK_CHARSET)


