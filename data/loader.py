import json
import io
import random
import torch
import numpy as np

from utils import constant


class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """

    def __init__(self, data_file, batch_size, opt, vocab, spacy_model, evaluation=False):
        self.spacy_model = spacy_model
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        data = get_data(data_file, spacy_model, evaluation=evaluation)
        data = self.preprocess(data, vocab, opt, spacy_model)
        # Do not shuffle for testing
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        id2label = dict([(v, k) for k, v in constant.LABEL_TO_ID.items()])
        class2id = dict([(v, k) for k, v in constant.ID_TO_CLASS.items()])
        self.labels = [class2id[d[-1]] for d in data]
        self.labels = [id2label[id] for id in self.labels]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), data_file))

    # convert data into ids
    def preprocess(self, data, vocab, opt, spacy_model):
        processed = []
        for fact in data:
            doc = spacy_model(fact["passageText"])
            tokens = []
            pos_tags = []
            dep_tags = []
            for token in doc:
                tokens.append(token.text)
                pos_tags.append(token.pos_)
                dep_tags.append(token.dep_)
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            passage_index = fact["passageStart"]
            subject_start_index = fact["subjectStart"] - passage_index
            subject_end_index = fact["subjectEnd"] - passage_index
            object_start_index = fact["objectStart"] - passage_index
            object_end_index = fact["objectEnd"] - passage_index
            subject_span = doc.char_span(subject_start_index, subject_end_index)
            object_span = doc.char_span(object_start_index, object_end_index)
            if subject_span and object_span:
                for subject_token in subject_span:
                    tokens[subject_token.i] = "SUBJ-ENTITY"
                for object_token in object_span:
                    tokens[object_token.i] = "OBJ-ENTITY"
                tokens = map_to_ids(tokens, vocab.word2id)

                pos = map_to_ids(pos_tags, constant.POS_TO_ID)
                dep = map_to_ids(dep_tags, constant.DEP_TO_ID)
                subj_positions = [2 if token == "SUBJ-ENTITY" else 0 for token in tokens]
                obj_positions = [3 if token == "OBJ-ENTITY" else 0 for token in tokens]
                relation = constant.ID_TO_CLASS[int(fact["propertyId"])]
                processed += [(tokens, pos, dep, subj_positions, obj_positions, relation)]
        return processed

    def gold(self):
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        # batches length
        assert len(batch) == 6

        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size)
        masks = torch.eq(words, 0)
        pos = get_long_tensor(batch[1], batch_size)
        dep = get_long_tensor(batch[2], batch_size)
        subj_positions = get_long_tensor(batch[3], batch_size)
        obj_positions = get_long_tensor(batch[4], batch_size)

        realtions = torch.LongTensor(batch[5])

        return (words, masks, pos, dep, subj_positions, obj_positions, realtions, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

# build fact dict from passage text
def get_test_facts(passage, spacy_model, document_id, evaluation=False):
    doc = spacy_model(passage["passageText"])
    passage_start = passage["passageStart"]
    subject_chunks = []
    subject_texts = []
    object_chunks = []
    object_texts = []
    subject_deps = ["nsubj", "nsubjpass"]
    object_deps = ["pobj", "dobj"]  # "attr", "dative", "appos", "compound"
    for chunk in doc.noun_chunks:
        if any(token.dep_ in subject_deps for token in chunk) and (chunk.text.lower() not in subject_texts):
            subject_chunks.append(chunk)
            subject_texts.append(chunk.text.lower())
        elif any(token.dep_ in object_deps for token in chunk) and (chunk.text.lower() not in object_texts):
            object_chunks.append(chunk)
            object_texts.append(chunk.text.lower())
    facts = []
    for subject_chunk in subject_chunks:
        for object_chunk in object_chunks:
            subject_start = subject_chunk[0].idx + passage_start
            subject_end = subject_chunk[-1].idx + len(subject_chunk[-1]) + passage_start
            object_start = object_chunk[0].idx + passage_start
            object_end = object_chunk[-1].idx + len(object_chunk[-1]) + passage_start
            fact_dict = {
                "factId": document_id + ":" + str(subject_start) + ":" + str(subject_end) + ":" + str(
                    object_start) + ":" + str(object_end) + ':1',
                "propertyId": '0',
                "subjectStart": subject_start,
                "subjectEnd": subject_end,
                "subjectText": subject_chunk.text,
                "subjectUri": "",
                "objectStart": object_start,
                "objectEnd": object_end,
                "objectText": object_chunk.text
            }
            facts.append(fact_dict)
            if not evaluation:
                # break if model is training (consider only one "NO_RELATION" for no facts passage
                break
    return facts


# read facts from dataset file for training and testing
def get_data(data_file, spacy_model, evaluation=False):
    reader = io.open(data_file, "r", encoding='utf8')
    data = []
    for line in reader:
        document = json.loads(line)
        document_id = document["documentId"]
        for passage in document["passages"]:
            passage_facts = passage["facts"]
            if not passage_facts:
                passage_facts = get_test_facts(passage, spacy_model, document_id, evaluation=evaluation)
            for d in passage_facts:
                d["passageText"] = passage["passageText"]
                d["passageStart"] = passage["passageStart"]
                d["passageEnd"] = passage["passageEnd"]
            data = data + passage_facts
    return data


# map labels to IDs
def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids


# Convert list of list of tokens to a padded LongTensor.
def get_long_tensor(tokens_list, batch_size):
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens


# Sort all fields by descending order of lens, and return the original indices.
def sort_all(batch, lens):
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]


# Randomly dropout tokens (IDs) and replace them with <UNK> tokens.
def word_dropout(tokens, dropout):
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout else x for x in tokens]
