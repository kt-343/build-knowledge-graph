# functions to build constant file
import io
import json
import random


# read training data
data_path = r"dataset"
reader = io.open(data_path+r'\actual_train.json', "r", encoding='utf8')
lines = []
for line in reader:
    document = json.loads(line)
    lines.append(document)


random.shuffle(lines)
train_length = int(len(lines)*.9)
train_data = lines[:train_length]
dev_data = lines[train_length:]

# split training data into train.json and dev.json: 80:20
with open(data_path+'/train.json', 'w') as fout:
    for line in train_data:
        fout.write(json.dumps(line))
        fout.write('\n')

with open(data_path+'/dev.json', 'w') as fout:
    for line in dev_data:
        fout.write(json.dumps(line))
        fout.write('\n')

# create LAEBEL_TO_ID dict
reader = io.open(data_path+r'\actual_train.json', "r", encoding='utf8')
properties = []
for line in reader:
    document = json.loads(line)
    for passage in document["passages"]:
        properties = properties + passage["exhaustivelyAnnotatedProperties"]

properties = sorted([dict(t) for t in {tuple(d.items()) for d in properties}], key=lambda x: int(x["propertyId"]))
print(properties)
LABEL_TO_ID = {}
for label_dict in properties:
    LABEL_TO_ID[label_dict["propertyName"]] = int(label_dict["propertyId"])
print(LABEL_TO_ID)
ID_TO_CLASS = {}
i = 0
for label, id in LABEL_TO_ID.items():
    ID_TO_CLASS[id] = i
    i = i + 1
print(ID_TO_CLASS)


# spacy tags and deps for features
spacy_tags = {
    "tagger": ["$", "''", ",", "-LRB-", "-RRB-", ".", ":", "ADD", "AFX", "CC", "CD", "DT", "EX", "FW", "HYPH", "IN",
               "JJ", "JJR", "JJS", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB",
               "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$",
               "WRB", "XX", "_SP", "``"],
    "parser": ["ROOT", "acl", "acomp", "advcl", "advmod", "agent", "amod", "appos", "attr", "aux", "auxpass", "case",
               "cc", "ccomp", "compound", "conj", "csubj", "csubjpass", "dative", "dep", "det", "dobj", "expl", "intj",
               "mark", "meta", "neg", "nmod", "npadvmod", "nsubj", "nsubjpass", "nummod", "oprd", "parataxis", "pcomp",
               "pobj", "poss", "preconj", "predet", "prep", "prt", "punct", "quantmod", "relcl", "xcomp"],
}
POS_TO_ID = {}
for index, pos in enumerate(spacy_tags["tagger"]):
    POS_TO_ID[pos] = index + 2
print(POS_TO_ID)


DEP_TO_ID = {}
for index, dep in enumerate(spacy_tags["parser"]):
    DEP_TO_ID[dep] = index + 2
print(DEP_TO_ID)

# all constansts found are defined in constant.py
