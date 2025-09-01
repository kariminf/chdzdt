# Test datasets


## 1. Intrinsic quality of word embeddings

- Morphological consistency ("**morph-consist**" folder): Do word embeddings cluster morphologically related forms in close proximity within the embedding space?
- Robustness to orthographic noise ("**ortho-noise**" folder): Are word embeddings resilient to spelling obfuscation and other forms of non-canonical orthography? 
- Interpretability using morphemic probing ("**morpho-lex**" folder): Do word embeddings encode morphemic structure, such that affixes can be linearly separated within the representation space?
- Compositional vector arithmetic ("**morpho-lex**" folder):  Given the embedding of a word (e.g. “unhappi-ness”), does there exist a function that combines the embeddings of its morphological components (e.g. “un”, “happy”, “ness”) to approximate the original embedding?
- Semantic similarity alignment: To what extent do word embeddings reflect human-perceived semantic similarity? The "**Multilingual WordSimilarity-353**" dataset is accessible via [https://github.com/siabar/Multilingual_Wordpairs]() 

## 2. Downstream tasks

- Morphological tagging ("**morph-tag**" folder):  Given an inflected verb, we test whether decoders can recover features such as tense, person, number, gender, or case.
- Part-of-Speech tagging ("**pos-tag**" folder): Given a sentence, find the part-of-speech  of each word.
- Sentiment analyssis ("**sa**" folder): Given a tweet, find its polarity.

# Train dataset