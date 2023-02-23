#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random

import numpy as np
import torch

from torch.utils.data import Dataset
from util import list2tuple, tuple2list, flatten
from typing import DefaultDict


class TestDataset(Dataset):
    def __init__(self, queries):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]
        return flatten(query), query, query_structure

    @staticmethod
    def collate_fn(data):
        query = [_[0] for _ in data]
        query_unflatten = [_[1] for _ in data]
        query_structure = [_[2] for _ in data]
        return query, query_unflatten, query_structure


class TrainDataset(Dataset):
    def __init__(self, queries, nentity, nrelation, negative_sample_size, answer):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.count = self.count_frequency(queries, answer)
        self.answer = answer

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]
        tail = np.random.choice(list(self.answer[query]))
        subsampling_weight = self.count[query]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            mask = np.in1d(
                negative_sample,
                self.answer[query],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor([tail])
        return positive_sample, negative_sample, subsampling_weight, flatten(query), query_structure

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.cat([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        query = [_[3] for _ in data]
        query_structure = [_[4] for _ in data]
        return positive_sample, negative_sample, subsample_weight, query, query_structure

    @staticmethod
    def count_frequency(queries, answer, start=4):
        count = {}
        for query, qtype in queries:
            count[query] = start + len(answer[query])
        return count


class SingledirectionalOneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0

    def __next__(self):
        self.step += 1
        data = next(self.iterator)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data


class CQDTrainDataset(Dataset):
    """
    Implemenation of CQD not using queries, but triples.
    """

    def __init__(self, queries, nentity, nrelation, negative_sample_size, answer):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.answer = answer

        self.qa_lst = []
        for q, qs in queries:
            for a in self.answer[q]:
                qa_entry = (qs, q, a)
                self.qa_lst += [qa_entry]

        self.qa_len = len(self.qa_lst)

    def __len__(self):
        return self.qa_len

    def __getitem__(self, idx):
        query = self.qa_lst[idx][1]
        query_structure = self.qa_lst[idx][0]
        tail = self.qa_lst[idx][2]
        subsampling_weight = torch.tensor([1.0])

        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_sample,
                self.answer[query],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        if self.negative_sample_size > 0:
            negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
            negative_sample = torch.from_numpy(negative_sample)
        else:
            negative_sample = torch.LongTensor(negative_sample_list)
        positive_sample = torch.LongTensor([tail])
        return positive_sample, negative_sample, subsampling_weight, flatten(query), query_structure

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.cat([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        query = [_[3] for _ in data]
        query_structure = [_[4] for _ in data]
        return positive_sample, negative_sample, subsample_weight, query, query_structure


class EvalTripleDataset(Dataset):
    """
    Used to evaluate link prediction task.
    """

    def __init__(self, queries, answer):
        # queries is a list of 1p queries
        self.queries = queries
        self.answer = answer

        self.triples = list()
        for q in queries:
            for a in self.answer[q]:
                self.triples += [(q[0], q[1][0], a)]

        self.num_triples = len(self.triples)

    def __len__(self):
        return self.num_triples

    def __getitem__(self, idx):
        triple = self.triples[idx]
        return torch.LongTensor([triple[0]]), torch.LongTensor([triple[1]]), torch.LongTensor([triple[2]])

    @staticmethod
    def collate_fn(data):
        head = torch.cat([_[0] for _ in data], dim=0)
        rel = torch.cat([_[1] for _ in data], dim=0)
        tail = torch.cat([_[2] for _ in data], dim=0)
        return head, rel, tail


class DescriptionsDatasetJointly(Dataset):
    # Dataset for 1dp queries only; for model jointly learning word embeddigns
    def __init__(self, queries, answers):
        self.qa_lst = list()
        for query in queries:
            a = list(answers[query])
            # always return 20 keywords
            q, r = divmod(20, len(a))
            a = q * a + a[:r]
            self.qa_lst.append((flatten(query), a))

    def __len__(self):
        return len(self.qa_lst)

    def __getitem__(self, idx):
        query, answers = self.qa_lst[idx]
        return torch.LongTensor([query[0]]), torch.as_tensor(answers)

    @staticmethod
    def collate_fn(data):
        e = torch.cat([_[0] for _ in data], dim=0)
        answers = torch.stack([_[1] for _ in data], dim=0)
        return e, answers


class DescriptionsDataset(Dataset):
    # Dataset for 1dp queries only
    def __init__(self, queries, answers):
        self.qa_lst = list()
        for query in queries:
            answer = next(iter(answers[query]))  # 1 answer expected
            self.qa_lst.append((flatten(query), answer))

    def __len__(self):
        return len(self.qa_lst)

    def __getitem__(self, idx):
        query, answer = self.qa_lst[idx]
        return torch.LongTensor([query[0]]), torch.as_tensor(answer)

    @staticmethod
    def collate_fn(data):
        e = torch.cat([_[0] for _ in data], dim=0)
        vector = torch.stack([_[1] for _ in data], dim=0)
        return e, vector


class AttributeDataset(Dataset):
    # Dataset for 1ap queries only
    def __init__(self, queries, answers, nentity, neg_sample_size=0):
        self.neg_sample_size = neg_sample_size
        self.nentity = nentity
        self.qa_lst = list()
        for query in queries:
            answer = next(iter(answers[query]))  # 1 answer expected
            self.qa_lst.append((flatten(query), answer))

        self.ent_by_attr = DefaultDict(set)
        for q, _ in self.qa_lst:
            self.ent_by_attr[q[-1]].add(q[0])
        self.ent_with_attr = tuple({a for b in self.ent_by_attr.values() for a in b})

    def __len__(self):
        return len(self.qa_lst)

    def _sample_negative(self, e, a):
        """Sample random entity without that attribute value."""
        while True:
            ent = random.choice(self.ent_with_attr)
            if ent not in self.ent_by_attr[a]:
                yield (ent, a, 0.5)

    def __getitem__(self, idx):
        query, answer = self.qa_lst[idx]
        e, a, v = [], [], []
        e.append(query[0])
        a.append(query[-1])
        v.append(answer)
        negative_sampler = self._sample_negative(e[0], a[0])
        for _ in range(self.neg_sample_size):
            e_n, a_n, v_n = next(negative_sampler)
            e.append(e_n)
            a.append(a_n)
            v.append(v_n)
        return torch.LongTensor(e), torch.LongTensor(a), torch.FloatTensor(v)
