import torch
from config import CQDParams, HyperParams, TrainConfig
from util_data import denormalize, load_data, load_descriptions_from_file, load_mappings_from_file, normalize
from util import name_query_dict, set_logger, parse_idetifier
from util_models import get_model, load_model
import typing
import re




set_logger('', None, True, True)
data_path = "./data/scripts/generated/FB15K-237_dummy_kblrn"
checkpoint_path = './checkpoints_FB15K-237/checkpoint_orig_attr_kblrn_desc'
entity2id = load_mappings_from_file(data_path, "entity")
relation2id = load_mappings_from_file(data_path, "relation")
attribute2id = load_mappings_from_file(data_path, "attr")

desc_path = "./data/scripts/generated/FB15K-237_dummy_kblrn_desc"
train_descs = load_descriptions_from_file(desc_path, 'train')
valid_descs = load_descriptions_from_file(desc_path, 'valid')
test_descs = load_descriptions_from_file(desc_path, 'test')
all_descs = {**train_descs, **valid_descs, **test_descs}

_, train_answers, _ = load_data(data_path, ('1p', '1ap',), 'train')
_, _, valid_answers = load_data(data_path, ('1p', '1ap',), 'valid')
_, _, test_answers = load_data(data_path, ('1p', '1ap',), 'test')
all_answers = {**train_answers, **valid_answers, **test_answers}

train_config = TrainConfig(data_path, None, checkpoint_path, geo='cqd-complexad', use_attributes=True)
params = HyperParams(rank=1000)

model = get_model(train_config, params, CQDParams(), nentity=len(entity2id), nrelation=len(relation2id), nattribute=len(attribute2id))
load_model(model, checkpoint_path, train_config.cuda)




def example_query():
  answers = []
  counter = 0
  symbol_placeholder_dict = {
  '=': -4,
  '<': -5,
  '>': -6,
}


  input_filter = input('Musicians from the USA are born [>: after, <: before, in: =]')
  while input_filter not in symbol_placeholder_dict:
    input_filter = input('Musicians from the USA are born [>: after, <: before, in: =]')
  # if input_filter not in symbol_placeholder_dict:
  #   raise ValueError('The input operator does not exist.')

  year_regex = re.compile(r"\b\d{4}\b")
  input_year = input('[year]')
  match = year_regex.search(input_year)

  while match==None:
    input_year = input('[year]')
    match = year_regex.search(input_year)
    
  
  anchor = entity2id['/m/09c7w0']  # USA
  age_threshold = normalize(83, int(input_year), data_path)
  attr = attribute2id['/people/person/date_of_birth']
  rel = relation2id['-/music/artist/origin']


  query = [anchor, rel, -3, attr, age_threshold, symbol_placeholder_dict[input_filter]]
  query = torch.as_tensor(query)
  preds = model({name_query_dict['pai']: query.unsqueeze(0)})
  
  for ent in torch.argsort(preds[0], descending=True):
    if counter==10:
      break
    
    ent = ent.item()
    attribute_value = train_answers[(ent, (-3, 83))]
    
    if not attribute_value:
      continue
    
    identifier_str = parse_idetifier(entity2id.inverse[ent])
    predicted_birth_date = model.predict_attribute_values(model.ent_embeddings(torch.tensor([ent])), torch.tensor([83])).item()
    predicted_birth_date = denormalize(83, predicted_birth_date, data_path)
    attribute_value = f"{denormalize(83, next(iter(attribute_value)), data_path):.2f}"
    score = preds[0][ent].item()
    
    temp = (identifier_str.rstrip(), predicted_birth_date, score)
    answers.append(temp)
    print(f"Name: {temp[0]}, Predicted year: {temp[1]:.2f}, Trained Value: {attribute_value}, Score: {temp[2]:.2f}")
    counter+=1
  
  
  return answers



if __name__ == '__main__':
  example_query()
  