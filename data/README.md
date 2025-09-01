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

## Collecting data

### Collecting YouTube comments

- Get a token here: [https://developers.google.com/youtube/v3/getting-started]()
- Save it into a file. By default it can be saved to '[HOME FOLDER]/.youtube_api_key'
- For each targeted YouTube channel, search for its ID 
- Store all names and IDs in a csv file, separated by a comma (,)
- Download the daily quota (or upgrade for more quota) using this command:
```sh
>> python exec/collect/collect_youtube.py -k <token-file-url> -l <list-of-channels-url> -o <output-folder>
```

### Collecting Wikipedia articles

- If you want to extract all the titles from a Wikipedia page to a file. For example, a category page listing a lot of titles (pages)
```sh
>> python exec/collect/collect_wikipedia.py -p titles -i <wikipedia-page-url> -o <output-file>
```
- If you want to get the names of "good" or "featured" pages in a language such as "en" and store them into a file
```sh
>> python exec/collect/collect_wikipedia.py -p [good|featured] -l <lang> -o <output-file>
```
- If you want to get the names of a certain category in a certain language; with a depth of search (depth=1 means only the articles in this category; depth=2 means the articles of depth=1 plus those of subcategories; etc.)
```sh
>> python exec/collect/collect_wikipedia.py -p category -c <category-name> -l <lang> -d <depth> -o <output-file>
```
- To download the articles of a certain language from a list of titles stored in a file to a folder using a consecutive number to which the start is defined by an offset in case you download many times.
```sh
>> python exec/collect/collect_wikipedia.py -p import -l <lang> -i <file-of-list> -o <output-folder> -s <offset-number>
```

### Collecting Tatoeba

- Just download the data from [https://tatoeba.org/en/downloads]()


## Preprocessing data

### Preprocessing YouTube data

check:
```sh
>>> exec/prepare/youtube.pp.py -h
```

### Preprocessing Wikipedia data

check:
```sh
>>> exec/prepare/wikipedia.pp.py -h
```

### Preprocessing Tatoeba data

check:
```sh
>>> exec/prepare/tatoeba.pp.py -h
```
