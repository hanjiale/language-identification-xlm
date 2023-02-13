# Language Identification

***4*** labeled samples per language + ***3*** min training = language detector with **99%** F1 😎 


This is an implementation of automatic language identification based on xlm-RoBERTa [1]. We support the following **20** languages:
```bash
Arabic (ar), Bulgarian (bg), German (de), Modern Greek (el), English (en), Spanish (es), French (fr), Hindi (hi), 
Italian (it), Japanese (ja), Dutch (nl), Polish (pl), Portuguese (pt), Russian (ru), Swahili (sw), Thai (th), 
Turkish (tr), Urdu (ur), Vietnamese (vi), and Chinese (zh)
```

## How to use?
##### Install dependencies

- ``python 3.8``
- ``PyTorch 1.11.0``
- ``transformers 4.18.1``
- ``numpy 1.21.5``

Please download our trained model from [here]() and put it under the ``./results/``

Our method can perform sentence-level language identification, here we give an example:
for the document ``./example/example.txt`` with multiple sentences,
```bash
...
綺麗にCDが収納できるからとても良い！
Fonctionne très bien
Love, love, love this! Made cutting my diamonds and triangles a breeze and corners were sharp and precise!
翻译的很差，语句和逻辑不通，耐着性子好几次，实在是读不下去。
...
```
use the following command:
```bash
bash run.sh
```
The predicted file ``./example/example_pred.txt`` will give the predicted language category.
```bash
...
Japanese:綺麗にCDが収納できるからとても良い！
French:Fonctionne très bien
English:Love, love, love this! Made cutting my diamonds and triangles a breeze and corners were sharp and precise!
Chinese:翻译的很差，语句和逻辑不通，耐着性子好几次，实在是读不下去。
...
```
## How to train?
#### Data
We use the [language identification dataset](https://huggingface.co/datasets/papluca/language-identification#additional-information) listed in the huggingface to train and evaluate our model, which is a collection of 90k samples consisting of text passages and corresponding language label. This dataset was created by collecting data from 3 sources: Multilingual Amazon Reviews Corpus, XNLI, and STSb Multi MT.
```bash
'labels': 'fr', 'text': 'Conforme à la description, produit pratique.'
'labels': 'zh', 'text': '有句话说，懂得很多道理，但是仍然过不好这一生。'
'labels': 'en', 'text': 'It was very over priced.'
```

The statistics are listed in the below table:

 | #train | #val | #test |
| -----------  | ------------- | ------------ | 
 | 3,500 x 20 = 70,000 | 500 x 20 = 10,000 | 500 x 20 = 10,000 | 

##### Model

![model](figure/model.pdf)

We provide two methods based on xlm-RoBERTa:
* fine-tuning: a simple classifier on top of xlm-RoBERTa.
* prompt-tuning: 
we add such a templete 
```angular2html
"The language of this sentence is [MASK]"
```  
after the sentence, and predict the language category by generating [MASK] as corresponding language name. 


##### Training
To train our model, use command in the root directory

```bash
bash train.sh
```
The experiments can be conducted on one GPU with 24GB memory.

##### Experiments
We conduct experiments by using *K* labeled instances per language to train and evaluate the model, respectively. The *K*-shot data can be automatically generated using the following command:
```bash
bash generate_k_shot_data.sh
```
The experimental results are shown in the table below.

 |  | *K*=1 | *K*=2 |*K*=4|*K*=8|Full|
| -----------  | ------------- | ------------ | ------------ | ------------ | ------------ | 
 | fine-tuning | 18.6  | 46.9 | 98.0 | 99.3 | 99.6 | 
 | prompt-tuning | 94.9 | 98.4  | 99.3 | 99.5 | 99.7| 

It can be observed that prompt-tuning is more effective in the extremely low resource scenario.

## Future work

Though effective, the method can only detect one language in a sentence. We would like to address the language identification of
codemixed text [2] in our future work. We have a rough idea, as shown in the figure below, which performs token-level classification.

![model](figure/model_codemix.pdf)

Due to the lack of codemixed data, we have not implemented this model.
We find an [off-the-shelf tool](https://github.com/microsoft/CodeMixed-Text-Generator) that helps automatic generation of grammatically valid synthetic codemixed data. Therefore, we can generate data we need and hopefully train the model.

## References
[1] Conneau A, Khandelwal K, Goyal N, et al. [Unsupervised cross-lingual representation learning at scale.](https://aclanthology.org/2020.acl-main.747/)  In Proceedings of ACL, 2020.

[2] Zhang Y, Riesa J, Gillick D, et al. [A fast, compact, accurate model for language identification of codemixed text.](https://aclanthology.org/D18-1030/) In Proceedings of EMNLP, 2018.