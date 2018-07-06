# RAM

A Tensorflow implementation for ["Recurrent Attention Network on Memory for Aspect Sentiment Analysis"](http://www.aclweb.org/anthology/D17-1047) (Peng Chen, EMNLP 2017)

## Quick Start

- Create three empty folders: 'analysis' for saving analyzing results, 'logs' for saving experiment logs and 'models' for saving experiment models 
- Download the 300-dimensional pre-trained word vectors from [Glove](https://nlp.stanford.edu/projects/glove/) and save it in the 'data' folder as 'data/glove.6B.300d.txt'

## Source Code Tree

```
|--- data

|	|--- restaurant

|--- main.py

|--- model.py

|--- utils.py

|--- README.md
```

## Results

| Dataset    | Accuracy |
| ---------- | -------- |
| Laptop     | 74.138   |
| Restaurant | 80.000   |
