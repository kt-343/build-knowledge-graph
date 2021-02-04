"""
Prepare vocabulary and initial word vectors.
"""
import io
import json
import pickle
import numpy as np
from collections import Counter

from utils import vocab, constant, helper


# Loads KnowledgeGraph documents
def load_tokens(filename, nlp):
    tokens = []
    reader = io.open(filename, "r", encoding='utf8')
    for line in reader:
        document = json.loads(line)
        for passage in document["passages"]:
            tokens += [token.text.lower() for token in nlp(passage["passageText"])]
    print("{} tokens loaded from {}.".format(len(tokens), filename))
    return tokens


def build_vocab(tokens, glove_vocab, min_freq):
    """ build vocab from tokens and glove words. """
    counter = Counter(t for t in tokens)
    # if min_freq > 0, use min_freq, otherwise keep all glove words
    if min_freq > 0:
        v = sorted([t for t in counter if counter.get(t) >= min_freq], key=counter.get, reverse=True)
    else:
        v = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    # add special tokens and entity mask tokens
    v = constant.VOCAB_PREFIX + v + entity_masks()
    print("vocab built with {}/{} words.".format(len(v), len(counter)))
    return v


def count_oov(tokens, vocab):
    c = Counter(t for t in tokens)
    total = sum(c.values())
    matched = sum(c[t] for t in vocab)
    return total, total - matched


def entity_masks():
    """ Get all entity mask tokens as a list. """
    subj_obj_entities = list(constant.SUB_OBJ_NER_TO_ID.keys())[2:]
    return subj_obj_entities


def prepare_vocab(data_dir, vocab_dir, spacy_model, glove_dir="dataset/glove", wv_file="glove.840B.300d.txt",
                  wv_dim=300, min_freq=0, lower=True):
    # input files
    train_file = data_dir + '/train.json'
    dev_file = data_dir + '/dev.json'
    test_file = data_dir + '/test.json'
    wv_file = glove_dir + '/' + wv_file
    wv_dim = wv_dim

    # output files
    helper.ensure_dir(vocab_dir)
    vocab_file = vocab_dir + '/vocab.pkl'
    emb_file = vocab_dir + '/embedding.npy'

    # load files
    print("loading files...")
    train_tokens = load_tokens(train_file, spacy_model)
    dev_tokens = load_tokens(dev_file)
    test_tokens = load_tokens(test_file, spacy_model)
    if lower:
        train_tokens, test_tokens = [[t.lower() for t in tokens] for tokens in \
                                     (train_tokens, dev_tokens, test_tokens)]

    # load glove
    print("loading glove...")
    glove_vocab = vocab.load_glove_vocab(wv_file, wv_dim)
    print("{} words loaded from glove.".format(len(glove_vocab)))

    print("building vocab...")
    v = build_vocab(train_tokens, glove_vocab, min_freq)

    print("calculating oov...")
    datasets = {'train': train_tokens, 'test': test_tokens}
    for dname, d in datasets.items():
        total, oov = count_oov(d, v)
        print("{} oov: {}/{} ({:.2f}%)".format(dname, oov, total, oov * 100.0 / total))

    print("building embeddings...")
    embedding = vocab.build_embedding(wv_file, v, wv_dim)
    print("embedding size: {} x {}".format(*embedding.shape))

    print("dumping to files...")
    with open(vocab_file, 'wb') as outfile:
        pickle.dump(v, outfile)
    np.save(emb_file, embedding)
    print("all done.")


data_dir = "dataset"
vocab_dir = "vocab"
import spacy

spacy_model = spacy.load("en_core_web_sm")
prepare_vocab(data_dir, vocab_dir, spacy_model, glove_dir="dataset/glove", wv_file="glove.840B.300d.txt",
              wv_dim=300, min_freq=0, lower=True)
