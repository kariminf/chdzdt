# Robustness to orthographic noise datasets


The datasets consist of taboo words in different languages.
There are two types of csv files:
- **cls**: clustering dataset with two columns: word \t cluster. The first word in the cluster is the original one and the rest are its variations (using obfuscation). The idea is to compare these variations to the original one using cosine similarity for example. Also, you can use a clustering algorithm on the words codes to test how well the codes are.
- **noise**: obfuscating one letter. The dataset is composed of three columns: word \t obfus1fix \t obfus1var. obfus1fix is the same word, just a letter is replaced by *. obfus1var is the same word just one letter is replaced by a character used in social media for obfuscation. 


Since the words are too ofensive, we encoded them to prevent direct readability.
To decode them use:
```sh
>> python exec/collect/extract_datasets.py <url-to-encoded-csv>
```

## Sources
- The English taboo words were extracted from [https://github.com/MauriceButler/badwords]() licensed under MIT licence.
Some inflections were removed such as plurals, etc. 
For **cls** dataset, the words were augmented manually, by replacing characters, e.g.: “a” → “@” or “4”, “s” → “5” or “$”, “i” → “!” or “1”, “o” → “0”, “e” → “3”, “t” → “+” or “7”.  
- The French taboo words were extracted from [https://github.com/darwiin/french-badwords-list]() licensed under MIT licence.
- Arabic and Arabizi were compiled by the author.

## License
[Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0)](https://creativecommons.org/licenses/by-sa/3.0/)