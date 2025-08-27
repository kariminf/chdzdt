# Morphology consistency datasets

The datasets are used for clustering taking the form: word \t cluster.
The first word in the cluster is the root and the rest are either inflections or derivations.


## Sources

The raw data was downloaded from [UniMorph](https://unimorph.github.io/) project:
- Arabic: Since the source is under GPL license, we cannot redistribute the files. Instead, we share how to prepare them step by step.
- English and French: [https://github.com/kbatsuren/MorphyNet]() already under CC-BY-SA licence.

## Arabic derivational datasets 

- Download Arramooz: [https://github.com/linuxscout/arramooz]()
- To extract the average derivational dataset used for our experimentation, use at least 15 words per cluster.
Also, we extracted only nouns derivations.
```sh
>> python exec/collect/extract_morpho_var.py -t arramooz -n 14 -f fnouns arramooz/data/arabicdictionary.sqlite ar_deriv_nr_avg15.csv
```
- To extract the min dataset:
```sh
>> python exec/collect/extract_morpho_var.py -t arramooz -n 29 -f fnouns arramooz/data/arabicdictionary.sqlite ar_deriv_nr_min30.csv
```

## Arabic inflectional datasets 
- Download verbs conjugations generated using [Qutrub](https://github.com/linuxscout/qutrub) project from [https://sourceforge.net/projects/qutrub/files/allverbs2.zip/download]()
- To extract the average derivational dataset used for our experimentation, use at least 15 words per cluster.
```sh
>> python exec/collect/extract_morpho_var.py -t qutrub -n 74 allverbs2/allverbs.txt ar_infl_wr_avg75.csv
```
- To extract the min dataset:
```sh
>> python exec/collect/extract_morpho_var.py -t qutrub -n 99 allverbs2/allverbs.txt ar_infl_wr_min100.csv
```



## License
The English and French datasets are distributed as
[Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0)](https://creativecommons.org/licenses/by-sa/3.0/).
They are extracted from [MorphyNet](https://github.com/kbatsuren/MorphyNet) project.
