import csv
import logging
import statistics

from bidict import bidict
from util import query_name_dict, name_query_dict
from collections import defaultdict
import os
import pickle


def load_data(data_path, tasks, type):
    '''
    Load queries and remove queries not in tasks
    '''
    if type == 'train':
        queries = pickle.load(open(os.path.join(data_path, "train-queries.pkl"), 'rb'))
        answers_easy = pickle.load(open(os.path.join(data_path, "train-answers.pkl"), 'rb'))
        answers_hard = defaultdict(set)
    elif type == 'valid':
        queries = pickle.load(open(os.path.join(data_path, "valid-queries.pkl"), 'rb'))
        answers_hard = pickle.load(open(os.path.join(data_path, "valid-hard-answers.pkl"), 'rb'))
        answers_easy = pickle.load(open(os.path.join(data_path, "valid-easy-answers.pkl"), 'rb'))
    elif type == 'test':
        queries = pickle.load(open(os.path.join(data_path, "test-queries.pkl"), 'rb'))
        answers_hard = pickle.load(open(os.path.join(data_path, "test-hard-answers.pkl"), 'rb'))
        answers_easy = pickle.load(open(os.path.join(data_path, "test-easy-answers.pkl"), 'rb'))

    # remove query structures not in tasks
    for task in list(queries.keys()):
        if task not in query_name_dict or query_name_dict[task] not in tasks:
            del queries[task]

    for qs in tasks:
        try:
            logging.info(type+': '+qs+": "+str(len(queries[name_query_dict[qs]])))
        except:
            logging.warn(type+': '+qs+": not in pkl file")

    return queries, answers_easy, answers_hard


def load_attr_exists_data_dummy(data_path, mode='valid'):
    '''
    Load queries to evaluate relations to the attr_exists dummy entity.
    (e, r_a, 14505) ~ 14505 dummy entity AND
    (14505, r_a_inv, e)
    '''
    queries = pickle.load(open(os.path.join(data_path, mode+"-attr-exists-queries.pkl"), 'rb'))
    answers_easy = pickle.load(open(os.path.join(data_path, mode+"-attr-exists-answers.pkl"), 'rb'))
    return queries, defaultdict(set), answers_easy


def load_attr_exists_data(args):
    '''
    Load queries to evaluate if an entity has an attribute.
    '''
    queries = pickle.load(open(os.path.join(args.data_path, "train-queries.pkl"), 'rb'))

    result = dict()
    result[name_query_dict['attr_exists']] = set()
    result_answers = defaultdict(set)
    for query in queries[name_query_dict['1ap']]:
        q = (query[1][1],)
        result[name_query_dict['attr_exists']].add(q)
        result_answers[q].add(query[0])
    return result, defaultdict(set), result_answers


def get_all_entity_descriptions(data_path):
    queries, answers, _ = load_data(data_path, ('1dp',), 'train')
    descriptions = dict()
    for q in queries[name_query_dict['1dp']]:
        descriptions[q[0]] = list(answers[q])
    return descriptions


def load_stats(data_path):
    try:
        nentity = 0
        nrelation = 0
        nattributes = 0
        with open(os.path.join(data_path, "entity2id.txt"), 'r') as f:
            nentity = int(f.readline())
        with open(os.path.join(data_path, "relation2id.txt"), 'r') as f:
            nrelation = int(f.readline())
        try:
            with open(os.path.join(data_path, "attr2id.txt"), 'r') as f:
                nattributes = int(f.readline())
        except FileNotFoundError:
            nattributes = 0
        return nentity, nrelation, nattributes
    except FileNotFoundError:
        nentity = len(pickle.load(open(os.path.join(data_path, 'ent2id.pkl'), 'rb')))
        nrelation = len(pickle.load(open(os.path.join(data_path, 'rel2id.pkl'), 'rb')))
        return nentity, nrelation, 0


def load_mappings_from_file(path, name):
    mapping = bidict()
    with open(os.path.join(path, f"{name}2id.txt"), "r") as file:
        reader = csv.DictReader(file, delimiter='\t', fieldnames=("name", "id"))
        next(reader)
        for row in reader:
            mapping[row["name"]] = int(row["id"])
    return mapping


def load_descriptions_from_file(path, name):
    mapping = bidict()
    with open(os.path.join(path, f"desc_{name}2id.txt"), "r") as file:
        reader = csv.DictReader(file, delimiter='\t', fieldnames=("id", "desc"))
        next(reader)
        for row in reader:
            mapping[int(row["id"])] = row["desc"]
    return mapping


def get_all_attribute_values(path):
    """
    Get all values for each attribute.
    """
    attr_values = dict()
    with open(os.path.join(path, "attr_train2id.txt"), "r") as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)
        for row in reader:
            if int(row[1]) in attr_values:
                attr_values[int(row[1])].append(float(row[2]))
            else:
                attr_values[int(row[1])] = [float(row[2])]

    with open(os.path.join(path, "attr_valid2id.txt"), "r") as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)
        for row in reader:
            if int(row[1]) in attr_values:
                attr_values[int(row[1])].append(float(row[2]))
            else:
                attr_values[int(row[1])] = [float(row[2])]

    with open(os.path.join(path, "attr_test2id.txt"), "r") as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)
        for row in reader:
            if int(row[1]) in attr_values:
                attr_values[int(row[1])].append(float(row[2]))
            else:
                attr_values[int(row[1])] = [float(row[2])]
    return attr_values


def get_mads(attr_values):
    """
    Return mean average deviations for a given dict of attribute values.
    """
    mads = dict()
    for attr, values in dict(sorted(attr_values.items())).items():
        try:
            mads[attr] = sum([abs(statistics.mean(values)-v) for v in values])/len(values)
        except:
            mads[attr] = 1.0e-10

        if mads[attr] == 0.0:
            mads[attr] = 1.0e-10
    return mads


def denormalize(attribute, value, data_path):
    with open(os.path.join(data_path, "attr2id_min_max.txt"), "r") as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)
        for row in reader:
            if int(row[1]) == attribute:
                return value*(float(row[3])-float(row[2]))+float(row[2])


def normalize(attribute, value, data_path):
    with open(os.path.join(data_path, "attr2id_min_max.txt"), "r") as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)
        for row in reader:
            if int(row[1]) == attribute:
                return (value-float(row[2]))/(float(row[3])-float(row[2]))
