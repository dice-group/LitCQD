import pandas as pd
import numpy as np
from .Datasets.KBLRNLiteralsDataset import KBLRNLiteralsDataset
from .Datasets.PickleMappingsDataset import PickleMappingsDataset
# from Datasets.KBLRNLiteralsDataset import KBLRNLiteralsDataset
# from Datasets.PickleMappingsDataset import PickleMappingsDataset
from collections import defaultdict
import itertools
import os
from bidict import bidict

# Define a filename.

def calculate_all_value_std(path):
  cur_file_path = os.path.dirname(__file__)
  
  df = pd.read_csv(path, sep="\t", header=None)

  np_data = df.to_numpy()

  values = np_data[:, 2]
  min_value = min(values)
  max_value = max(values)

  

  _inputDataset = PickleMappingsDataset(f"{cur_file_path}/data/relational/FB15K-237", False)
  _inputDataset.load_data()
  
  _literalsDataset = KBLRNLiteralsDataset(f"{cur_file_path}/data/numeric/KBLRN/")
  # _literalsDataset.attr2id = _inputDataset.rel2id
  _literalsDataset.load_data(_inputDataset.ent2id)
  triples = _literalsDataset._load_triples_from_file()
  _literalsDataset.triples = {"train": triples, "valid": [], "test": []}
  attributes = {triple[1] for triple in itertools.chain.from_iterable(_literalsDataset.triples.values())}
  _literalsDataset.attr2id = bidict({x: i for i, x in enumerate(sorted(attributes))})
  
  def _get_values_per_attribute():
    attr_values = defaultdict(list)
    for triple in itertools.chain.from_iterable(_literalsDataset.triples.values()):
        attr_values[triple[1]].append(triple[2])
    return attr_values
  
  def get_min_max_values_per_attribute():
        # not loaded yet
        attr_values = _get_values_per_attribute()
        min_max_values = dict()
        for i, values in attr_values.items():
          # avoid key error if attribute is not in the dataset
          
          min_max_values[i] = (min(values), max(values))
            
             
        # min_max_values = {
        #     _literalsDataset.attr2id.inverse[i]: (min(values), max(values))
        #     for i, values in attr_values.items() if _literalsDataset.attr2id.get(i)
        # }
        return min_max_values
  def _normalize_min_max(min, max, value):
    if min == max:
        return 1.0
    try:
        normalized_value = (value - min) / (max - min)
    except ZeroDivisionError:
        normalized_value = value
    return normalized_value
  
  def normalize_values():    
      attr_values = get_min_max_values_per_attribute()
      # test
      for _, triples in _literalsDataset.triples.items():
          for i in range(len(triples)):
              value = triples[i][2]
              attr = triples[i][1]
              normalized_value = _normalize_min_max(
                  attr_values[attr][0], attr_values[attr][1], value
              )
              triples[i] = (triples[i][0], triples[i][1], normalized_value)
  
  # normalize the third element in the triples to value in range of [0,1] 
  normalize_values() 
  
  print(_literalsDataset)
  _literalsDataset.triples['train']
  # get all the last elements in the triples of the list _literalsDataset.triples['train']
  all_attr_values = [triple[2] for triple in _literalsDataset.triples['train']]
  # convert the list to numpy array
  all_attr_values_np = np.asarray(all_attr_values)
  all_values_std = np.std(all_attr_values_np)
        
  # # _literalsDataset.load_data()
  # # _map_entities_to_ids(_inputDataset.ent2id,_literalsDataset.triples)
  # # print(_literalsDataset)
  


  # attr_values = defaultdict(list)
  
  # for triple in itertools.chain.from_iterable(_literalsDataset.triples.values()):
  #     attr_values[triple[1]].append(triple[2])

  # min_max_values = {
  #     _literalsDataset.attr2id.inverse[i]: (min(values), max(values))
  #     for i, values in attr_values.items()
  # }




  # # Normalize values
  # for _, triples in _literalsDataset.triples.items():
  #     for i in range(len(triples)):
  #         value = triples[i][2]
  #         attr = _literalsDataset.attr2id.inverse[triples[i][1]]
  #         normalized_value = _normalize_min_max(
  #             min_max_values[attr][0], min_max_values[attr][1], value
  #         )
  #         triples[i] = (triples[i][0], triples[i][1], normalized_value)

  # all_attr_values = []

  # for triple in _literalsDataset.triples["train"]:
  #     all_attr_values.append(triple[2])

  # all_attr_values_np = np.asarray(all_attr_values)

  # all_values_std = np.std(all_attr_values_np)

  return all_values_std


# path = "./data/numeric/KBLRN/FB15K_NumericalTriples.txt"

# print(calculate_all_value_std(path))