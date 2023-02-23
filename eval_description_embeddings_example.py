import torch
from config import CQDParams, HyperParams, TrainConfig
from util_data import *
from util_data_queries import *
from main import set_logger
from util_models import get_model, load_model
import gensim.downloader


set_logger('', None, True, True)
data_path = "./data/scripts/generated/FB15K-237_dummy_kblrn_desc"
checkpoint_path = './checkpoints_FB15K-237/checkpoint_orig_attr_kblrn_desc'
entity2id = load_mappings_from_file(data_path, "entity")
relation2id = load_mappings_from_file(data_path, "relation")
attribute2id = load_mappings_from_file(data_path, "attr")

train_descs = load_descriptions_from_file(data_path, 'train')
valid_descs = load_descriptions_from_file(data_path, 'valid')
test_descs = load_descriptions_from_file(data_path, 'test')
all_descs = {**train_descs, **valid_descs, **test_descs}

train_config = TrainConfig(data_path, None, checkpoint_path, geo='cqd-complexad')
params = HyperParams(rank=1000)

model = get_model(train_config, params, CQDParams(), nentity=len(entity2id), nrelation=len(relation2id), nattribute=len(attribute2id))
load_model(model, checkpoint_path, train_config.cuda)

word_embeddings_model = gensim.downloader.load('word2vec-google-news-300')

description = 'A university offering a program of computer science'
description = gensim.utils.tokenize(description, lowercase=True)
description = gensim.parsing.preprocessing.remove_stopwords(" ".join(description))
# create single embedding for description
word_embeddings = list()
for word in description.split(' '):
    try:
        word_embeddings.append(word_embeddings_model[word])
    except:
        pass  # skipping words not found in word2vec model
if not word_embeddings:
    print('Unable to compute word embedding')
    exit()
description_embedding = sum(word_embeddings)/len(word_embeddings)

query = [-7, *list(description_embedding), -4]
res = model.forward({name_query_dict['di']: torch.tensor([query])})

print('cosine similarity\tname\t\t\tid\tdescription')
# print closest entity desription embeddings until 5 eval data points are printed
count = 0
for ent in torch.argsort(res[0], descending=True):
    ent = ent.item()
    desc = ''
    if ent in all_descs:
        desc = ' '.join(all_descs[ent].split()[:5])+'...'
    if ent in train_descs:
        print(f'- {res[0][ent].item()}\t{entity2id.inverse[ent]}\t\t{ent}\t{desc}')
        continue
    print(f'{res[0][ent].item()}\t{entity2id.inverse[ent]}\t\t{ent}\t{desc}')
    count += 1
    if count > 4:
        break
print(f"Average cosine similarity to predicted entity descriptions: {torch.mean(res[0]).item()}")
