from bidict import bidict
import itertools
import os
import csv

from .LiteralsDataset import LiteralsDataset


class LiteralELiteralsDataset(LiteralsDataset):
    """
    For literal data provided by LiteralE.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_triples_from_file(self):
        with open(os.path.join(self.path, f"numerical_literals.txt"), "r") as file:
            reader = csv.reader(file, delimiter='\t')
            triples = list()
            for row in reader:
                triples.append((row[0], row[1].replace('http://rdf.freebase.com/ns', '').replace('.', '/'), float(row[2])))
        return triples

    def _load_data(self):
        self.triples = {"train": self._load_triples_from_file(), "valid": [], "test": []}
        attributes = {triple[1] for triple in itertools.chain.from_iterable(self.triples.values())}
        self.attr2id = bidict({x: i for i, x in enumerate(sorted(attributes))})
