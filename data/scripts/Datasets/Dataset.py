import itertools
from bidict import bidict

from Datasets.DatasetProperty import DatasetProperty
from Datasets.LiteralsDataset import LiteralsDataset


class Dataset(object):
    def __init__(self, path, header=True):
        self.path = path
        self.header = header
        self.triples = None
        self.ent2id = None
        self.rel2id = None

    triples = DatasetProperty("triples")
    ent2id = DatasetProperty("ent2id")
    rel2id = DatasetProperty("rel2id")

    def __str__(self):
        res = f"{type(self).__name__}:\n"
        for name, triples in self.triples.items():
            res += f"\t{name} triples: {len(triples)}\n"
        res += f"\tentities: {len(self.ent2id)}\n\trelations: {len(self.rel2id)}"
        return res

    def _validate_data(self):
        entities = [{triple[0], triple[2]} for triple in itertools.chain.from_iterable(self.triples.values())]
        entities = set(itertools.chain.from_iterable(entities))
        relations = {triple[1] for triple in itertools.chain.from_iterable(self.triples.values())}

        if len(entities) != len(self.ent2id):
            print(f"{len(entities) - len(self.ent2id)} entity ids are not used")
        if len(relations) != len(self.rel2id):
            print(f"{len(relations) - len(self.rel2id)} relation ids are not used")

    def _load_data(self):
        raise NotImplementedError()

    def load_data(self):
        self._load_data()
        self._validate_data()

    def add_inverse_relations(self):
        relations = list(self.rel2id.keys())
        self.rel2id = bidict()
        i = 0
        while i < len(relations):
            self.rel2id['+'+relations[i]] = 2*i
            self.rel2id['-'+relations[i]] = 2*i + 1
            i += 1

        for data in self.triples.values():
            triples = [(ent1, 2*rel, ent2) for (ent1, rel, ent2) in data]
            inverse_triples = [(ent2, 2*rel+1, ent1) for (ent1, rel, ent2) in data]
            data[:] = itertools.chain.from_iterable(zip(triples, inverse_triples))

    def remove_inverse_relations(self):
        raise NotImplementedError()

    def add_attribute_exists_relations(self, literalsDataset: LiteralsDataset, has_inverse: bool):
        """
        Add a relation for each attribute, an entity /attribute/exists and triples:
        /attribute/exists, r_a, e for each e, r_a if e,r_a\in triples
        If has_inverse, add inverse as well. It may also be added by setting add_inverse in the config file.
        """
        multiplier = 2 if has_inverse else 1
        attr_exists_ent_id = len(self.ent2id)
        self.ent2id['/attribute/exists'] = attr_exists_ent_id
        offset = len(self.rel2id)
        for name, id in literalsDataset.attr2id.items():
            if has_inverse:
                self.rel2id['+'+name] = 2*id + offset
                self.rel2id['-'+name] = 2*id + 1 + offset
            else:
                self.rel2id[name] = id + offset

        for name, triples in self.triples.items():
            attr_exists_triple = set()
            for ent, attr, _ in literalsDataset.triples[name]:
                attr_exists_triple.add((attr_exists_ent_id, multiplier*attr+offset, ent))
                if has_inverse:
                    attr_exists_triple.add((ent, multiplier*attr+1+offset, attr_exists_ent_id))
            triples.extend(list(attr_exists_triple))
