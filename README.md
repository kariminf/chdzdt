![chDzDt](logo.png)

Character-based Algerian Dialect Transformer (chDzDT)  is a lightweight transformer model that operates at the character level. 
The model can be described using the symbolic notation $c\cdot h \cdot \frac{\partial z}{\partial t}$, which metaphorically captures its architecture: character-level processing ($c$), deep representations ($h$), and transformation over time ($\frac{\partial z}{\partial t}$).
It is specifically designed to address the challenges posed by morphologically rich, noisy, multilingual, and multi-script dialects such as Algerian Arabic. 
To this end, the model is built to be robust against orthographic variation and morphological obfuscation, which are common in informal text.
chDzDT is intended for word-level encoding based purely on character-level morphological features. 
This makes it well-suited to tasks involving word morphology, such as inflection or derivation. 
In addition, it can serve as a preprocessing module to enhance traditional language models; particularly in handling out-of-vocabulary (OOV) tokens.


**This project is a part of a larger project called DzDT started on 2023. It was separated on August 2025 to handle only the morphology part of the project.**


# Training a chDzDT model

- To specify the configuration of the model, please see this example [config/chdzdt_train/chdzdt_train.json]()
- To train a new model
```sh
>> exec/train/chdzdt.trn.py new <config-file-url>
```
- To resume training a model
```sh
>> exec/train/chdzdt.trn.py resume <config-file-url> <trained-model-folder>
```



# License

Copyright (C) 2025 Abdelkrime Aries

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.