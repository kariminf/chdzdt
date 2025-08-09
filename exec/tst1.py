import pandas as pd

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dzdt.model.chdzdt_tok import CharTokenizer


char_tokenizer = CharTokenizer(max_position=20)
char_tokenizer.add_charset(33, 126) # basic-latin
char_tokenizer.add_charset(161, 255) # latin1-suppliment
char_tokenizer.add_charset(1536, 1791) # arabic
char_tokenizer.add_charset(11568, 11647) # tifinagh
char_tokenizer.add_charset(592, 687) # IPA extensions (kabyle use this)

print(char_tokenizer.encode_char("❤️"[1]), ord("❤️"[1]))

v = char_tokenizer.size

print(v)

print(char_tokenizer.decode_char(2840))

i=0
for data in pd.read_csv("/home/karim/Data/DZDT/data/2_words/words.csv", sep="\t", chunksize=10000):
    i+= 1
    print(f"Processing chunk {i} ...")
    data["word"] = data["word"].astype(str)

    cc = char_tokenizer.encode_words(data["word"].to_list())
    mm = cc["input_ids"].max()
    if mm >= v:
        print("You ar fuckd", mm, word)
    # for index, row in data.iterrows():
    #     word = row["word"]
    #     cc = char_tokenizer.encode_words([word])
    #     mm = cc["input_ids"].max()
    #     if mm >= v:
    #         print("You ar fuckd", mm, word)
