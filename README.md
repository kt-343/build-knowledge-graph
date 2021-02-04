#  Create a knowledge graph by extracting the facts: Annotation of relation between Entities in Text
 Create a knowledge graph by extracting the facts: Annotation of relation between Entities in Text


## Requirements

- Python 3 (tested on 3.6.5)
- PyTorch (tested on 1.0.0)
- spacy (tested on 2.2.4)
- spacy large model ( `python -m spacy download en_core_web_lg`)

## Overview
Model: LSTM sequence model with a form of entity position-aware attention for relation extraction.
The combination of better supervised data and a more appropriate high-capacity model enables much better relation extraction performance. But train data is semi supervised. It has only Subject, Object and Relation (facts among them). Hence spacy model is used to generate features: tokens, pos tags, dependency tags, SUB-OBJ positions to find relations between subject and object of given text.
This source code is implementation of research Paper "Position-aware Attention and Supervised Data Improve Slot Filling". Please go through paper for detail.
The major limitation is above paper is based on TACRED dataset created by Stanford Research team in which all labels (ners, pos tags, dependency tags, subject/object tags are annotated). In this, training data is semi supervised where dataset has only subject and object information and their relation.

## Evaluation Results
To evaluate trained model, training data is split into train and dev by 80:20 split.
Results on dev set:
`Precision: 87.996%   Recall: 88.856%   F1 : 88.424%`

## Model and Vocab Files
- Download and extract model files: (https://drive.google.com/file/d/17fF2XqU6nkNktf6gmDGhAuhpCl49qzLi/view?usp=sharing) to `./saved_models/00`
- Download and extract vocab and embeddings files: (https://drive.google.com/file/d/1Z4kZBgA7Mv6WqhhS8qj1bigb7zPvAKU6/view?usp=sharing) to `./vocab`

## Limitations
- Train data is not annotated with NERs. Having NER lables of subject and object would produce better entity relation extraction.
- Relations from sentence to sentence are not formed since the training is sone at sentence level rather than document level. To solve this, Coreference resolution should be implemented.

## Source
Position-aware Attention and Supervised Data Improve Slot Filling (by Yuhao Zhang, Victor Zhong, Danqi Chen, Gabor Angeli, Christopher D. Manning)
https://nlp.stanford.edu/pubs/zhang2017tacred.pdf
