import torch
from config import CQDParams, HyperParams, TrainConfig
from util_data import (
    denormalize,
    load_data,
    load_descriptions_from_file,
    load_mappings_from_file,
    normalize,
)
from util import name_query_dict, set_logger, parse_idetifier
from util_models import get_model, load_model
import numpy as np
from util_data_queries import *

set_logger("", None, True, True)
data_path = "./data/scripts/generated/FB15K-237_dummy_kblrn"
checkpoint_path = "./checkpoints_FB15K-237/checkpoint_orig_attr_kblrn_desc"  # using model for description
entity2id = load_mappings_from_file(data_path, "entity")
relation2id = load_mappings_from_file(data_path, "relation")
attribute2id = load_mappings_from_file(data_path, "attr")

desc_path = "./data/scripts/generated/FB15K-237_dummy_kblrn_desc"
train_descs = load_descriptions_from_file(desc_path, "train")
valid_descs = load_descriptions_from_file(desc_path, "valid")
test_descs = load_descriptions_from_file(desc_path, "test")
all_descs = {**train_descs, **valid_descs, **test_descs}

_, train_answers, _ = load_data(
    data_path,
    (
        "1p",
        "1ap",
    ),
    "train",
)
_, _, valid_answers = load_data(
    data_path,
    (
        "1p",
        "1ap",
    ),
    "valid",
)
_, _, test_answers = load_data(
    data_path,
    (
        "1p",
        "1ap",
    ),
    "test",
)
all_answers = {**train_answers, **valid_answers, **test_answers}

train_config = TrainConfig(
    data_path, None, checkpoint_path, geo="cqd-complexad", use_attributes=True
)
params = HyperParams(rank=1000)

model = get_model(
    train_config,
    params,
    CQDParams(),
    nentity=len(entity2id),
    nrelation=len(relation2id),
    nattribute=len(attribute2id),
)
load_model(model, checkpoint_path, train_config.cuda)

# train_data_rel, train_data_attr, train_data_desc = load_queries_train(
#         train_config, "train"
#     )

# # calculate stdv from train dataset
# train_attr_tmp = []
# for i in list(train_data_attr[1].values()):
#   train_attr_tmp.append(i.pop())
# train_attr_np = np.asarray(train_attr_tmp)

# stdv = np.std(train_attr_np)
# model.stdv = stdv

age_threshold = normalize(83, 1972, data_path)
attr = attribute2id["/people/person/date_of_birth"]
anchor = entity2id["/m/09c7w0"]  # USA

# rel = relation2id['-/sports/pro_athlete/career_start']
rel = relation2id["-/music/artist/origin"]
query = [anchor, rel, -3, attr, age_threshold, -5]
query = torch.as_tensor(query)
# preds = model({name_query_dict["pai"]: query.unsqueeze(0)})

# score_1,score_2,scores
all_scores,preds = model({name_query_dict["pai"]: query.unsqueeze(0)})

sorted_tuples = sorted(
    enumerate(all_scores), key=lambda x: x[1][2].item(), reverse=True
)


# print('score\tname\t\tid\tTrained Rel.?\tPred. Value\tTrained Value\tDescription')
print(
    "score\t1p_score\tai_score\tname\t\tid\tTrained Rel.?\tPred. Value\tTrained Value\tDescription"
)
count = 0
# for ent in torch.argsort(preds[0], descending=True):
#   ent = ent.item()
#   desc = ''
#   if ent in all_descs:
#       desc = ' '.join(all_descs[ent].split()[:10])+'...'

#   relation_trained = ent in train_answers[(32, (321,))]
#   attribute_value = train_answers[(ent, (-3, 83))]
#   attribute_value = f"{denormalize(83, next(iter(attribute_value)), data_path):.2f}" if attribute_value else '----.--'

#   predicted_birth_date = model.predict_attribute_values(model.ent_embeddings(torch.tensor([ent])), torch.tensor([83])).item()
#   predicted_birth_date = denormalize(83, predicted_birth_date, data_path)
#   print(f"{preds[0][ent].item():.2f}\t{parse_idetifier(entity2id.inverse[ent])}\t{ent}\t{relation_trained}\t\t{predicted_birth_date:.2f}\t\t{attribute_value}\t\t{desc}")

#   count += 1
#   if count > 20:
#       break
    
    
for ent, pred in sorted_tuples:

    desc = ""
    if ent in all_descs:
        desc = " ".join(all_descs[ent].split()[:10]) + "..."

    relation_trained = ent in train_answers[(32, (321,))]
    attribute_value = train_answers[(ent, (-3, 83))]
    attribute_value = (
        f"{denormalize(83, next(iter(attribute_value)), data_path):.2f}"
        if attribute_value
        else "----.--"
    )

    predicted_birth_date = model.predict_attribute_values(
        model.ent_embeddings(torch.tensor([ent])), torch.tensor([83])
    ).item()
    predicted_birth_date = denormalize(83, predicted_birth_date, data_path)
    print(
        f"{pred[2].item():.2f}\t{pred[0].item():.2f}\t{pred[1].item():.2f}\t{parse_idetifier(entity2id.inverse[ent])}\t{ent}\t{relation_trained}\t\t{predicted_birth_date:.2f}\t\t{attribute_value}\t\t{desc}"
    )

    count += 1
    if count > 20:
        break


print()
print(75 * "=" + " Expected Answers " + 75 * "=")
for ent, i in entity2id.items():
    if i in all_answers[(anchor, (rel,))]:
        try:
            attribute_value = all_answers[(i, (-3, attr))]
        except KeyError:
            continue
        if attribute_value and next(iter(attribute_value)) <= age_threshold:
            desc = ""
            if i in all_descs:
                desc = " ".join(all_descs[i].split()[:10]) + "..."
            rel_dataset = "Train"
            if i in test_answers[(anchor, (rel,))]:
                rel_dataset = "Test"
            elif i in valid_answers[(anchor, (rel,))]:
                rel_dataset = "Valid"
            attr_dataset = "Train"
            if test_answers[(i, (-3, attr))]:
                attr_dataset = "Test"
            elif valid_answers[(i, (-3, attr))]:
                attr_dataset = "Valid"
            print(
                f"{ent}\t{i}\t{denormalize(attr, next(iter(attribute_value)), data_path):.2f}\tRelation in {rel_dataset} dataset\tAttribute in {attr_dataset} dataset\t{desc}"
            )
