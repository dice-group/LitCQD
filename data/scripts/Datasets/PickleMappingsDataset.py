import os
import csv
from .Dataset import Dataset
from bidict import bidict
import pickle


class PickleMappingsDataset(Dataset):
    """
    Basic dataset structure with train/valid/test.txt files and (ent1, rel, ent2) structure.
    These files already contain the ids of the entities and relations. Their mappins are stored using pickle.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_triples_from_file(self, name):
        with open(os.path.join(self.path, f"{name}.txt"), "r") as file:
            reader = csv.reader(file, delimiter='\t')
            if self.header:
                next(reader)  # skip header
            return [(int(row[0]), int(row[1]), int(row[2])) for row in reader]

    def _load_mappings_from_pickle(self, filename):
        return bidict(pickle.load(open(os.path.join(self.path, f"{filename}.pkl"), "rb")))

    def _load_data(self):
        self.triples = {name: self._load_triples_from_file(name) for name in ('train', 'valid', 'test')}
        self.ent2id = self._load_mappings_from_pickle("ent2id")
        self.rel2id = self._load_mappings_from_pickle("rel2id")
