
## Preparation

Download and unzip GloVe embeddings file from the Stanford website (http://nlp.stanford.edu/data/glove.840B.300d.zip)

Build vocabulary and embeddings by running:
```
python prepare_vocab.py
```

This will write vocabulary and word vectors as a numpy matrix into the dir `vocab`

## Training

Train model with:
```
python train.py --data_dir dataset --vocab_dir vocab --id 00
```

Model checkpoints will be saved to `./saved_models/00`

## Evaluation

Run evaluation on the test set with:
```
python eval.py saved_models/00 --dataset test
```

This will use the `best_model.pt` by default. Use `--model checkpoint_epoch_xx.pt` to specify a model checkpoint file.
AEvaluations are saved to `dataset/evaluated.json`
