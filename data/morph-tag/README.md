# Morphology tagging datasets

Given a conjugated verb, there are many grammatical features such as tense, case, voice, etc. that describe the conjugation.
This is a tabular dataset containg: word \t lemma \t feature-1 \t ... \t feature-n.
It can be used for morphological tagging: given a word, find the value of each feature; e.g. *conjugated can be tense=past*.
Also, it can be used for Morphological (Re-)Inflection: given a lemma and some features' values, generate the inflected word.

## Sources

The raw data was downloaded from [UniMorph](https://unimorph.github.io/) project:
- Arabic: [https://github.com/unimorph/ara]()
- English: [https://github.com/unimorph/eng]()
- French: [https://github.com/unimorph/fra]()

## Data preparation 




## License
All three sources have the same license
[Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0)](https://creativecommons.org/licenses/by-sa/3.0/)
