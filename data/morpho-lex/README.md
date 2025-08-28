# Lexical database for morphological variables


The datasets are used to study words, their roots and affixes.
There are two types of files:
- **affix**: contains a list of words a the most frequent affixes (prefix starts with "-" and a suffix end with it).
The affixes are represented as binary variables (1: exists, 0: does not exist).
- **111**: contains a list of words with just one prefix, one root and one suffix. It can be used to test compositional morphology.

## Sources
- The English words were extracted from [https://github.com/hugomailhot/MorphoLex-en](). 
- The French words were extracted from [https://github.com/hugomailhot/morpholex-fr]().

## License

English and French datasets are licensed the same as the attributed sources above: 
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)