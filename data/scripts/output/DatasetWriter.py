from collections import defaultdict
from typing import Union
from Datasets.Dataset import Dataset
import os
from Datasets.DescriptionsDataset import DescriptionsDataset
from Datasets.DescriptionsDatasetJointly import DescriptionsDatasetJointly

from Datasets.LiteralsDataset import LiteralsDataset
from Datasets.DescriptionsDatasetJointly_GoogleNews import DescriptionsDatasetJointly_GoogleNews
from Datasets.DescriptionsDataset_GoogleNews import DescriptionsDataset_GoogleNews


class DatasetWriter(object):
    def __init__(self, path, postfix="2id"):
        self.path = path
        self.postfix = postfix

    def _write_descriptions(self, name, triples):
        raise NotImplementedError()

    def _write_triples_attr(self, name, triples):
        raise NotImplementedError()

    def _write_triples(self, name, triples):
        raise NotImplementedError()

    def _write_mapping(self, name, mapping):
        raise NotImplementedError()

    def _write_min_max_values(self, dataset: LiteralsDataset):
        raise NotImplementedError()

    def write_triples(self, dataset: Union[Dataset, LiteralsDataset]):
        func = self._write_triples
        if isinstance(dataset, LiteralsDataset):
            func = self._write_triples_attr
        elif isinstance(dataset, DescriptionsDataset):
            func = self._write_descriptions
        elif isinstance(dataset, DescriptionsDatasetJointly):
            func = self._write_descriptions_jointly(dataset.word2id.inverse)
        for name in ("train", "valid", "test"):
            func(name, dataset.triples[name])
        print(f"Done writing triples to {self.path}.")

    def write_mappings(self, dataset: Union[Dataset, LiteralsDataset]):
        if isinstance(dataset, DescriptionsDataset) or isinstance(dataset, DescriptionsDataset_GoogleNews):
            return
        elif isinstance(dataset, DescriptionsDatasetJointly) or isinstance(dataset, DescriptionsDatasetJointly_GoogleNews):
            self._write_mapping("word", dataset.word2id)
        elif isinstance(dataset, Dataset):
            self._write_mapping("relation", dataset.rel2id)
            self._write_mapping("entity", dataset.ent2id)
        else:
            self._write_mapping("attr", dataset.attr2id)
            self._write_min_max_values(dataset)
        print(f"Done writing mappings to {self.path}.")

    def write(self, dataset: Union[Dataset, LiteralsDataset]):
        self.write_triples(dataset)
        self.write_mappings(dataset)


class DummyDatasetWriter(DatasetWriter):
    def __init__(self, path, postfix="2id"):
        super().__init__(path, postfix=postfix)

    def _write_descriptions(self, name, triples):
        return

    def _write_descriptions_jointly(self, name, triples):
        return

    def _write_triples_attr(self, name, triples):
        return

    def _write_triples(self, name, triples):
        return

    def _write_mapping(self, name, mapping):
        return

    def _write_min_max_values(self, dataset: LiteralsDataset):
        return


class ThesisDatasetWriter(DatasetWriter):
    def __init__(self, path, postfix="2id"):
        super().__init__(path, postfix=postfix)

    def _write_descriptions(self, name, triples):
        with open(os.path.join(self.path, f"desc_{name}{self.postfix}.txt"), "w") as f:
            f.write(f"{len(triples)}\n")
            for ent, _, desc in triples:
                f.write(f"{ent}\t{' '.join(desc)}\n")

    def _write_descriptions_jointly(self, id2word):
        def write(name, triples):
            ent2words = defaultdict(set)
            for ent, _, word in triples:
                ent2words[ent].add(word)
            with open(os.path.join(self.path, f"desc_{name}{self.postfix}.txt"), "w") as f:
                f.write(f"{len(triples)}\n")
                for ent, words in ent2words.items():
                    f.write(f"{ent}\t{' '.join([id2word[word] for word in words])}\n")
        return write

    def _write_triples_attr(self, name, triples):
        with open(os.path.join(self.path, f"attr_{name}{self.postfix}.txt"), "w") as f:
            f.write(f"{len(triples)}\n")
            for ent, attr, val in triples:
                f.write(f"{ent}\t{attr}\t{val}\n")

    def _write_triples(self, name, triples):
        with open(os.path.join(self.path, f"{name}{self.postfix}.txt"), "w") as f:
            f.write(f"{len(triples)}\n")
            for ent1, rel, ent2 in triples:
                f.write(f"{ent1}\t{ent2}\t{rel}\n")

    def _write_mapping(self, name, mapping):
        with open(os.path.join(self.path, f"{name}{self.postfix}.txt"), "w") as f:
            f.write(f"{len(mapping)}\n")
            for id, item in dict(sorted(mapping.inverse.items())).items():
                f.write(f"{item}\t{id}\n")

    def _write_min_max_values(self, dataset: LiteralsDataset):
        try:
            min_max_values = dataset.get_min_max_values_per_attribute()
        except:
            # attribute values are not normalized; nothing to write
            return
        with open(os.path.join(self.path, f"attr{self.postfix}_min_max.txt"), "w") as f:
            f.write(f"{len(dataset.attr2id)}\n")
            for id, item in dict(sorted(dataset.attr2id.inverse.items())).items():
                f.write(f"{item}\t{id}\t{min_max_values[item][0]}\t{min_max_values[item][1]}\n")


class PickleDatasetWriter(DatasetWriter):
    def __init__(self, path, postfix="2id"):
        super().__init__(path, postfix=postfix)

    def _write_mapping(self, name, mapping):
        import pickle
        with open(os.path.join(self.path, f"{name[:3]}{self.postfix}.pkl"), "wb") as f:
            pickle.dump({v: k for k, v in mapping.items()}, f)

    def _write_descriptions(self, name, triples):
        raise NotImplementedError("Descriptions are not stored using pickle")

    def _write_descriptions_joinlty(self, name, triples):
        raise NotImplementedError("Descriptions are not stored using pickle")

    def _write_triples_attr(self, name, triples):
        raise NotImplementedError("Triples are not stored using pickle")

    def _write_triples(self, name, triples):
        raise NotImplementedError("Triples are not stored using pickle")

    def _write_min_max_values(self, name, triples):
        raise NotImplementedError("Attribute values are not stored using pickle")
