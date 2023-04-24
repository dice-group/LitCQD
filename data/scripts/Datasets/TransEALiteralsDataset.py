from bidict import bidict
import os
import csv

from Datasets.LiteralsDataset import LiteralsDataset


class TransEALiteralsDataset(LiteralsDataset):
    """
    For literal data provided by TransEA. Attribute values are normalized using min-max normalization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_triples_from_file(self, ent2id: bidict, attr2id: bidict):
        with open(os.path.join(self.path, "attrm2id.txt"), "r") as file:
            reader = csv.reader(file, delimiter='\t')
            next(reader)  # skip header
            return [(ent2id.inverse[int(row[0])], attr2id.inverse[int(row[1])], float(row[2])) for row in reader]

    def _load_mappings_from_file(self, name):
        mapping = bidict()
        with open(os.path.join(self.path, f"{name}2id.txt"), "r") as file:
            reader = csv.DictReader(file, delimiter='\t', fieldnames=("name", "id"))
            next(reader)
            for row in reader:
                mapping[row["name"]] = int(row["id"])
        return mapping

    def _load_data(self):
        self.attr2id = self._load_mappings_from_file("attr")
        ent2id_attr = self._load_mappings_from_file("entity")
        self.triples = {"train": self._load_triples_from_file(ent2id_attr, self.attr2id), "valid": [], "test": []}
