from bidict import bidict
import itertools
import os
import csv
import re
from datetime import datetime

from Datasets.LiteralsDataset import LiteralsDataset


class LitWDLiteralsDataset(LiteralsDataset):
    """
    For literal data part of the LiterallyWikiData benchmark.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_triples_from_file(self):
        with open(os.path.join(self.path, f"numeric_literals.txt"), "r") as file:
            reader = csv.reader(file, delimiter='\t')
            triples = list()
            for row in reader:
                value = re.match(r'\+?(.*?)\^', row[2]).group(1)
                try:
                    value = float(value)
                except ValueError:
                    tmp = datetime.strptime("1961-01-01T00:00:00Z", "%Y-%m-%dT%H:%M:%S%z")
                    value = tmp.year + (tmp.timetuple().tm_yday-1)/366
                triples.append((row[0], row[1], value))
            return triples

    def _load_data(self):
        self.triples = {"train": self._load_triples_from_file(), "valid": [], "test": []}
        attributes = {triple[1] for triple in itertools.chain.from_iterable(self.triples.values())}
        self.attr2id = bidict({x: i for i, x in enumerate(sorted(attributes))})
