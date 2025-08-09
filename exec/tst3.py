# from tokenizers import Tokenizer as TokenizerFast


# from transformers import BertTokenizer, BertTokenizerFast
# tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# # print(tokenizer._tokenizer.to_str())

import sys
import os
import timeit
import pandas as pd

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from dzdt.model._char_tokenizer_model import CharDzDtTokenizer
from dzdt.train.tokenizer_trainers import DzDtTokenizerTrainer

from transformers import tokenization_utils_base, configuration_utils, modeling_utils


from transformers import AutoModel


train_corpus = [
    "This is a test",
    "This test is good"
]

# tokenizer.train_new_from_iterator(train_corpus, 100)

# print(tokenizer._tokenizer.to_str())

# # tokenizer2 = TokenizerFast.from_str("{}")

# from torchtext.vocab import build_vocab_from_iterator
# def yield_tokens(it):
#     for s in it:
#         yield s.strip().split()

# vocab = build_vocab_from_iterator(yield_tokens(train_corpus), specials=["<unk>"])


# from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
# tokenizer = Tokenizer(models.WordLevel())
# tokenizer.normalizer = normalizers.NFKC()
# tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
# tokenizer.decoder = decoders.ByteLevel()
# trainer = trainers.WordLevelTrainer(
#     vocab_size=100,
#     # initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
#     min_frequency=2,
#     special_tokens=["<PAD>", "<BOS>", "<EOS>"],
# )

# tokenizer.train_from_iterator(train_corpus, trainer=trainer)

# print(tokenizer.get_vocab())

# PRJPATH    = '/home/karim/Data/DZDT/tokenizer.json'

# tokenizer.save(PRJPATH)


LOCATION = "/home/karim/Data/DZDT/dzdt_test/"

TOKEN_PATH = "/home/karim/Data/DZDT/hug_token.txt"
with open(TOKEN_PATH, "r") as f:
    for l in f:
        hugingface_token = l.rstrip("\n")
        break


tokenizer: CharDzDtTokenizer = CharDzDtTokenizer.from_pretrained("kariminf/tst", token=hugingface_token)

configuration_utils.CONFIG_NAME = "char_config.json"
modeling_utils.CONFIG_NAME = "char_config.json"
modeling_utils.WEIGHTS_NAME = "char_pytorch_model.bin"

model = AutoModel.from_pretrained("kariminf/tst", token='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')


prefixes = ['pre', 'de', 'in']
suffixes = ['ation', 'tion', 'er', 'ed']

tokenizer_trainer = DzDtTokenizerTrainer(prefixes, 
                                         suffixes, 
                                         tokenizer.chartok, 
                                         model,
                                         model_max_length=512)

tokenizer_trainer._process_sentence('prefaction determiner infected')

tokenizer_trainer._process_sentence('prefed termination fecter')

print(tokenizer_trainer.tokenizer.vocab)
print(tokenizer_trainer.stack_tokens)

# print(tokenizer_trainer.tokenizer.get_added_vocab())

# tokenizer_trainer.tokenizer.save_pretrained(LOCATION)

# print(tokenizer_trainer.tokenizer._convert_token_to_id('prefaction', return_tensors="pt"))

print(tokenizer_trainer.tokenizer(['prefaction determiner infected'], 
                                  return_tensors="pt",
                                  truncation=True,
                                  padding="max_length",
                                  max_length=7))

