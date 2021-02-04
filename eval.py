import random
import argparse
import torch
from data.loader import DataLoader, get_data
from model.rnn import SubjectObjectRelationModel
from utils import torch_utils, constant
from utils.vocab import Vocab
import spacy
import io
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='saved_models/00', help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")
parser.add_argument('--out', type=str, default='', help="Save model predictions to this dir.")
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', default=True, action='store_true')
args = parser.parse_args()


torch.manual_seed(args.seed)
random.seed(1234)
# if args.cpu:
#     args.cuda = False
# elif args.cuda:
#     torch.cuda.manual_seed(args.seed)


# forcing to test on CPU
args.cpu = True
args.cuda = False


# load opt
model_file = args.model_dir + '/' + args.model
print(model_file)
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
model = SubjectObjectRelationModel(opt)
model.load(model_file)


# load spacy model
spacy_model = spacy.load("en_core_web_lg")


# load vocab
vocab_file = args.model_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))

batch = DataLoader(data_file, opt['batch_size'], opt, vocab, spacy_model, evaluation=True)

# predict
predictions = []
all_probs = []
for i, b in enumerate(batch):
    preds, probs, _ = model.predict(b)
    predictions += preds
    all_probs += probs

# class to label
class2id = dict([(v, k) for k, v in constant.ID_TO_CLASS.items()])
id2label = dict([(v, k) for k, v in constant.LABEL_TO_ID.items()])
predictions = [class2id[p] for p in predictions]
predictions = [id2label[p] for p in predictions]

data = get_data(data_file, spacy_model, evaluation=True)

prediction_data = []
for index, d in enumerate(data):
    d["propertyName"] = predictions[index]
    # remove no relation entities
    if predictions[index] not in ["NO_RELATION"]:
        d["propertyId"] = str(constant.LABEL_TO_ID[predictions[index]])
        d["factId"] = d["factId"][:-1] + d["propertyId"]
        d["humanReadable"] = "<" + "> <".join([d["subjectText"], d["propertyName"], d["objectText"]]) +">"
        prediction_data.append(d)


# write to evaluated.json
reader = io.open(data_file, "r", encoding='utf8')
documents = []
for line in reader:
    document = json.loads(line)
    document_id = document["documentId"]
    for passage in document["passages"]:
        passage["facts"] = [fact for fact in prediction_data if passage['passageId'] ==
                            (document_id + ":" + str(fact["passageStart"]) + ":" + str(fact["passageEnd"]))]
    documents.append(document)
evaluate_file = opt['data_dir']+'/evaluated.json'
with open(evaluate_file, 'w') as fout:
    print("writing evaluations to {}".format(evaluate_file))
    for line in documents:
        fout.write(json.dumps(line))
        fout.write('\n')
print("evaluated on {} and written to {}".format(data_file, evaluate_file))
