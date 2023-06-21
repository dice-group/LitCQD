import logging

import numpy as np

from config import HyperParams, TrainConfig
from util_data import load_data
from dataloader import CQDTrainDataset, DescriptionsDataset, DescriptionsDatasetJointly, TestDataset, TrainDataset, AttributeDataset
from util import flatten_query, name_query_dict

from torch.utils.data import DataLoader


def load_queries_train(train_config: TrainConfig, name='train',not_flatten = False):
    if train_config.train_data_type.name == 'triples':
    # if train_config.train_data_type == 'triples':
        train_tasks = ('1p', '1ap', )
    elif train_config.train_data_type.name == 'queries':
    # elif train_config.train_data_type == 'queries':
        train_tasks = ('1p', '2p', '3p', '2i', '3i',)
    if train_config.use_attributes and '1ap' not in train_tasks:
        train_tasks = train_tasks + ('1ap',)
    elif not train_config.use_attributes and '1ap' in train_tasks:
        train_tasks = tuple(x for x in train_tasks if x != '1ap')

    if train_config.use_descriptions:
        train_tasks = train_tasks + ('1dp',)

    train_queries, train_answers, train_answers_hard = load_data(train_config.data_path, train_tasks, name)
    if name == 'valid':
        train_answers = train_answers_hard

    if train_config.use_attributes:
        train_queries_attr = {k: v for k, v in train_queries.items() if k == name_query_dict['1ap']}
        train_answers_attr = {q: a for q, a in train_answers.items() if q in train_queries_attr[name_query_dict['1ap']]}
        del train_queries[name_query_dict["1ap"]]
        train_answers = {q: a for q, a in train_answers.items() if q not in train_answers_attr}
    else:
        train_queries_attr = dict()
        train_answers_attr = dict()

    if train_config.use_descriptions:
        train_queries_desc = {k: v for k, v in train_queries.items() if k == name_query_dict['1dp']}
        train_answers_desc = {q: a for q, a in train_answers.items() if q in train_queries_desc[name_query_dict['1dp']]}
        del train_queries[name_query_dict["1dp"]]
        train_answers = {q: a for q, a in train_answers.items() if q not in train_answers_desc}
    else:
        train_queries_desc = dict()
        train_answers_desc = dict()
    if not_flatten:
      return (train_queries, train_answers), (train_queries_attr, train_answers_attr), (train_queries_desc, train_answers_desc)  
    
    train_queries = flatten_query(train_queries)
    return (train_queries, train_answers), (train_queries_attr, train_answers_attr), (train_queries_desc, train_answers_desc)


def get_train_dataloader(train_dataset, train_attr_dataset, train_desc_dataset, batch_size, use_attributes, use_descriptions, cpu_num=0, seed=0):
    # cpu_num is set to 0
    def _init_fn(worker_id):
        np.random.seed(seed)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cpu_num,
        collate_fn=type(train_dataset).collate_fn,
        worker_init_fn=_init_fn
    )

    # last batch in epoch may be smaller than batch size
    nbatches = (len(train_dataset)//batch_size) + (len(train_dataset) % batch_size > 0)

    if use_attributes:
        attr_batch_size = (len(train_attr_dataset) // nbatches) + (len(train_attr_dataset) % nbatches > 0)
        logging.info('attribute batch size: %d' % attr_batch_size)
        train_attr_dataloader = DataLoader(
            train_attr_dataset,
            batch_size=attr_batch_size,
            shuffle=True,
            num_workers=cpu_num,
            worker_init_fn=_init_fn
        )
    else:
        train_attr_dataloader = None

    if use_descriptions:
        desc_batch_size = (len(train_desc_dataset) // nbatches) + (len(train_desc_dataset) % nbatches > 0)
        train_desc_dataloader = DataLoader(
            dataset=train_desc_dataset,
            batch_size=desc_batch_size,
            shuffle=True,
            num_workers=cpu_num,
            collate_fn=type(train_desc_dataset).collate_fn,
            worker_init_fn=_init_fn
        )
    else:
        train_desc_dataloader = None

    return train_dataloader, train_attr_dataloader, train_desc_dataloader


def get_dataset_train(queries, answers, train_config: TrainConfig, nentity, nrelation, params: HyperParams):
    if train_config.train_data_type.name == 'triples':
        datasetClass = CQDTrainDataset
    elif train_config.train_data_type.name == 'queries':
        datasetClass = TrainDataset

    return datasetClass(queries, nentity, nrelation, params.negative_sample_size, answers)


def get_dataset_train_attr(queries, answers, nentity, params: HyperParams):
    if name_query_dict['1ap'] not in queries:
        return None
    return AttributeDataset(queries[name_query_dict['1ap']], answers, nentity, params.negative_attr_sample_size)


def get_dataset_train_desc(queries, answers, jointly=False):
    if name_query_dict['1dp'] not in queries:
        return None
    if jointly:
        return DescriptionsDatasetJointly(queries[name_query_dict['1dp']], answers)
    return DescriptionsDataset(queries[name_query_dict['1dp']], answers)


def get_dataset_eval(queries):
    return TestDataset(queries)


def load_queries_eval(data_path, tasks, name='valid', not_flatten=False):
    if not tasks:
        tasks = ('1p',)
    queries, easy_answers, hard_answers = load_data(data_path, tasks, name)
    if not_flatten:
      return queries, easy_answers, hard_answers  
    
    queries = flatten_query(queries)
    return queries, easy_answers, hard_answers


def get_eval_dataloader(dataset: TestDataset, batch_size, cpu_num):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=cpu_num,
        collate_fn=type(dataset).collate_fn
    )



  