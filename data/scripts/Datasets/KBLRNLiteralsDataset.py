from bidict import bidict
import itertools
import os
import csv

from Datasets.LiteralsDataset import LiteralsDataset


class KBLRNLiteralsDataset(LiteralsDataset):
    """
    For literal data provided by KBLRN.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_triples_from_file(self):
        with open(os.path.join(self.path, f"FB15K_NumericalTriples.txt"), "r") as file:
            reader = csv.reader(file, delimiter='\t')
            triples = list()
            for row in reader:
                triples.append((row[0], row[1].replace('>', '').replace('<http://rdf.freebase.com/ns', '').replace('.', '/'), float(row[2])))
            return triples

    def _load_data(self):
        self.triples = {"train": self._load_triples_from_file(), "valid": [], "test": []}
        attributes = {triple[1] for triple in itertools.chain.from_iterable(self.triples.values())}
        self.attr2id = bidict({x: i for i, x in enumerate(sorted(attributes))})
