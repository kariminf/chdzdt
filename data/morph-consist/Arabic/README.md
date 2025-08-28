# Arabic datasets

As the sources are under GPL, it is better to refrain from distributing the extracted datasets in this repo (being under Apache-2 for code and CC-BY-SA for data). 
Instead, these are instructions to extract the exact files we used for our experimentations.

## Arabic derivational datasets 

- Download Arramooz: [https://github.com/linuxscout/arramooz]()
- To extract the average derivational dataset used for our experimentation, use at least 15 words per cluster.
Also, we extracted only nouns derivations.
```sh
>> python exec/collect/extract_morph_forms.py -t arramooz -n 14 -f fnouns arramooz/data/arabicdictionary.sqlite ar_deriv_nr_avg15.csv
```
- To extract the min dataset:
```sh
>> python exec/collect/extract_morph_forms.py -t arramooz -n 29 -f fnouns arramooz/data/arabicdictionary.sqlite ar_deriv_nr_min30.csv
```

## Arabic inflectional datasets 
- Download verbs conjugations generated using [Qutrub](https://github.com/linuxscout/qutrub) project from [https://sourceforge.net/projects/qutrub/files/allverbs2.zip/download]()
- To extract the average derivational dataset used for our experimentation, use at least 15 words per cluster.
```sh
>> python exec/collect/extract_morph_forms.py -t qutrub -n 74 allverbs2/allverbs.txt ar_infl_wr_avg75.csv
```
- To extract the min dataset:
```sh
>> python exec/collect/extract_morph_forms.py -t qutrub -n 99 allverbs2/allverbs.txt ar_infl_wr_min100.csv
```
