from collections import defaultdict
import itertools
import os
import csv

from bidict import bidict
from Datasets.Dataset import Dataset


class DefaultDataset(Dataset):
    """
    Basic dataset structure with train/valid/test.txt files and (ent1, rel, ent2) structure.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_triples_from_file(self, name):
        with open(os.path.join(self.path, f"{name}.txt"), "r") as file:
            reader = csv.reader(file, delimiter='\t')
            if self.header:
                next(reader)  # skip header
            return list(reader)

    def _load_data(self):
        triples = {name: self._load_triples_from_file(name) for name in ('train', 'valid', 'test')}
        entities = set()
        relations = set()
        for ent1, rel, ent2 in itertools.chain.from_iterable(triples.values()):
            entities.add(ent1)
            entities.add(ent2)
            relations.add(rel)
        self.ent2id = bidict({x: i for i, x in enumerate(sorted(list(entities)))})
        self.rel2id = bidict({x: i for i, x in enumerate(sorted(list(relations)))})
        self.triples = defaultdict(list)
        for name, rows in triples.items():
            for row in rows:
                self.triples[name].append((self.ent2id[row[0]], self.rel2id[row[1]], self.ent2id[row[2]]))
