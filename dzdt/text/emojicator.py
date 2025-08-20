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

from dzdt.tools.process import SeqPipeline
from dzdt.tools.struct import to_trie, TrieValueProcessor


# https://en.wikipedia.org/wiki/List_of_emoticons
# https://web.archive.org/web/20090411052027/http://messenger.yahoo.com/features/emoticons/
# https://web.archive.org/web/20090707114539/http://messenger.msn.com/Resource/Emoticons.aspx

EMOTICON_TRIE = to_trie([
    (":-) ", "🙂"), (":) ", "🙂"), (":-)) ", "🙂"), (":)) ", "🙂"), # Smiley
    (":‑D ", "😀"), (":D ", "😀"), (":‑d ", "😀"), (":d ", "😀"), # Grinning face, Open-mouthed
    ("x‑D ", "😆"), ("X‑D ", "😆"), ("xD ", "😆"), ("XD ", "😆"), # Laughing
    (":‑( ", "🙁"), (":( ", "🙁"), # Frowning
    (":\"‑) ", "😂"), (":\") ", "😂"), # Tears of joy
    (">:( ", "😠"), # Angry
    (":-O ", "😮"), (":-o ", "😮"), (":O ", "😮"), (":o ", "😮"), # Surprised
    (":‑3 ", "😸"), (":3 ", "😸"), # Cat face
    (">:‑3 ", "😾"), (">:3 ", "😾"), # Lion face, evil cat
    (":-* ", "😗"), (":* ", "😗"), (":x ", "😗"), # Kissing
    (";-) ", "😉"), (";) ", "😉"), # Winking 
    (";-P ", "😜"), (";P ", "😜"), (";-D ", "😜"), (";D ", "😜"), # Winking mouth open
    (";‑) ", "😉"), (";) ", "😉"), ("*-) ", "😉"), ("*) ", "😉"), # Winking
    (":-P ", "😝"), (":P ", "😝"), (":-p ", "😝"), (":p ", "😝"), # Tongue sticking out,
    (":/ ", "😕"), (":-/ ", "😕"), # Confused
    (":| ", "😐"), (":-| ", "😐"), # Straight face
    (":$ ", "😳"), # Flushed face
    ("://) ", "😔"), # Disappointed
    ("://3 ", "😖"), # Confounded face
    (":-X ", "🤐"), (":X ", "🤐"), (":-# ", "🤐"), (":# ", "🤐"), # Sealed lips
    ("O:-) ", "😇"), ("O:) ", "😇"), # Angel
    (">:‑) ", "😈"), ("3:‑) ", "😈"), ("3:) ", "😈"), # Evil
    ("B-) ", "😎"), ("|:) ", "😎"), # Cool 
    ("<3 ", "❤️"), ("</3 ", "💔"), ("<\3 ", "💔"), # 
    ("(>_<) ", "😣"), (">_< ", "😣"), 
    ("(X_X) ", "😵"), ("(x_x) ", "😵"), ("X_X ", "😵"), ("x_x ", "😵"), # 
    ("(O_O) ", "😳"), ("(o_o) ", "😳"), ("O_O ", "😳"), ("o_o ", "😳"), # 
    ("^^ ", "😄"), ("(^^) ", "😄"), ("^_^ ", "😄"), ("(^_^) ", "😄"), 
    ("^^; ", "😅"), ("(^^;) ", "😅"), ("^_^; ", "😅"), ("(^_^;) ", "😅"), 
])

EMOJI_PROCESSOR = TrieValueProcessor(EMOTICON_TRIE)
EMOJI_PROCESSOR_PIPE = SeqPipeline([EMOJI_PROCESSOR])

def char2emoji(text: str) -> str:
    EMOJI_PROCESSOR_PIPE.init()
    return "".join(EMOJI_PROCESSOR_PIPE.process([*text]))