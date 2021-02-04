TRAIN_JSON = 'train.json'
DEV_JSON = 'dev.json'
TEST_JSON = 'test.json'

GLOVE_DIR = 'dataset/glove'

EMB_INIT_RANGE = 1.0
MAX_LEN = 100

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

SUB_OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'SUBJ-ENTITY': 2, 'OBJ-ENTITY': 3}

NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'LOCATION': 5, 'DATE': 6, 'NUMBER': 7, 'MISC': 8, 'DURATION': 9, 'MONEY': 10, 'PERCENT': 11, 'ORDINAL': 12, 'TIME': 13, 'SET': 14}

POS_TO_ID_ALL = {PAD_TOKEN: 0, UNK_TOKEN: 1, '$': 2, "''": 3, ',': 4, '-LRB-': 5, '-RRB-': 6, '.': 7, ':': 8, 'ADD': 9, 'AFX': 10, 'CC': 11, 'CD': 12, 'DT': 13, 'EX': 14, 'FW': 15, 'HYPH': 16, 'IN': 17, 'JJ': 18, 'JJR': 19, 'JJS': 20, 'LS': 21, 'MD': 22, 'NFP': 23, 'NN': 24, 'NNP': 25, 'NNPS': 26, 'NNS': 27, 'PDT': 28, 'POS': 29, 'PRP': 30, 'PRP$': 31, 'RB': 32, 'RBR': 33, 'RBS': 34, 'RP': 35, 'SYM': 36, 'TO': 37, 'UH': 38, 'VB': 39, 'VBD': 40, 'VBG': 41, 'VBN': 42, 'VBP': 43, 'VBZ': 44, 'WDT': 45, 'WP': 46, 'WP$': 47, 'WRB': 48, 'XX': 49, '_SP': 50, '``': 51}

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'ADJ': 2, 'ADP': 3, 'ADV': 4, 'AUX': 5, 'CONJ': 6, 'DET': 7, 'INTJ': 8, 'NOUN': 9, 'NUM': 10, 'PART': 11, 'PRON': 12, 'PROPN': 13, 'PUNCT': 14, 'SCONJ': 15, 'SYM': 16, 'VERB': 17, 'X': 18 }

DEP_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'ROOT': 2, 'acl': 3, 'acomp': 4, 'advcl': 5, 'advmod': 6, 'agent': 7, 'amod': 8, 'appos': 9, 'attr': 10, 'aux': 11, 'auxpass': 12, 'case': 13, 'cc': 14, 'ccomp': 15, 'compound': 16, 'conj': 17, 'csubj': 18, 'csubjpass': 19, 'dative': 20, 'dep': 21, 'det': 22, 'dobj': 23, 'expl': 24, 'intj': 25, 'mark': 26, 'meta': 27, 'neg': 28, 'nmod': 29, 'npadvmod': 30, 'nsubj': 31, 'nsubjpass': 32, 'nummod': 33, 'oprd': 34, 'parataxis': 35, 'pcomp': 36, 'pobj': 37, 'poss': 38, 'preconj': 39, 'predet': 40, 'prep': 41, 'prt': 42, 'punct': 43, 'quantmod': 44, 'relcl': 45, 'xcomp': 46}

LABEL_TO_ID = {'NO_RELATION': 0, 'SUBSIDIARY_OF': 1, 'FOUNDED_BY': 2, 'EMPLOYEE_OR_MEMBER_OF': 3, 'CEO': 4, 'DATE_FOUNDED': 5, 'HEADQUARTERS': 6, 'EDUCATED_AT': 9, 'NATIONALITY': 10, 'PLACE_OF_RESIDENCE': 11, 'PLACE_OF_BIRTH': 12, 'DATE_OF_DEATH': 14, 'DATE_OF_BIRTH': 15, 'SPOUSE': 25, 'CHILD_OF': 34, 'POLITICAL_AFFILIATION': 45}
ID_TO_CLASS = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 9: 6, 10: 7, 11: 8, 12: 9, 14: 10, 15: 11, 25: 12, 34: 13, 45: 14, 0: 15}

INFINITY_NUMBER = 1e12
