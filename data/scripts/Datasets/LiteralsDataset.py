from collections import defaultdict
import itertools
import random
from typing import Tuple
import statistics

from .DatasetProperty import DatasetProperty

random.seed(0)


class LiteralsDataset(object):
    def __init__(self, path, triples=None, attr2id=None):
        self.path = path
        self.triples = triples
        self.attr2id = attr2id
        self.min_max_values = None

    triples = DatasetProperty("triples")
    ent2id = DatasetProperty("ent2id")
    attr2id = DatasetProperty("attr2id")
    min_max_values = DatasetProperty("min_max_values")

    def __str__(self):
        res = f"{type(self).__name__}:\n"
        for name, triples in self.triples.items():
            res += f"\t{name} triples: {len(triples)}\n"
        res += f"\tentities: {len(self._get_attr_values_per_entity())}\n\tattributes: {len(self.attr2id)}"
        attr_values = self._get_values_per_attribute()
        all_values = list(itertools.chain.from_iterable(attr_values.values()))
        res += f"\n\tunique values: {len(set(all_values))}"
        mean = statistics.mean(all_values)
        meandev = sum([abs(mean - v) for v in all_values]) / len(all_values)
        res += f"\n\tmean: {mean:.3f}"
        res += f"\n\tMAD: {meandev:.3f}"
        means = list()
        madevs = list()
        for _, values in attr_values.items():
            means.append(statistics.mean(values))
            madevs.append(sum([abs(means[-1] - v) for v in values]) / len(values))
        res += f"\n\tmean per attribute: {statistics.mean(means):.3f}"
        res += f"\n\tMAD per attribute: {statistics.mean(madevs):.3f}"

        return res

    def _load_data(self):
        raise NotImplementedError()

    def load_data(self, ent2id):
        self._load_data()
        self._map_entities_to_ids(ent2id)
        self._remove_unused_attributes()
        self._map_attributes_to_ids()
        self._remove_duplicate_triples()

    def _map_entities_to_ids(self, ent2id):
        skipped = defaultdict(int)
        for _, rows in self.triples.items():
            for i in range(len(rows) - 1, -1, -1):
                row = rows[i]
                if row[0] not in ent2id:
                    skipped[row[0]] += 1
                    del rows[i]
                    continue
                rows[i] = (ent2id[row[0]], row[1], row[2])
        if skipped:
            print(
                f"Skipped {sum(skipped.values())} triples with attr data for {len(skipped)} entities that are not part of the dataset."
            )

    def _map_attributes_to_ids(self):
        for _, triples in self.triples.items():
            triples[:] = [(e, self.attr2id[a], v) for e, a, v in triples]

    def _remove_unused_attributes(self):
        num_attr = len(self.attr2id)
        attr_count = defaultdict(int)
        for triple in itertools.chain.from_iterable(self.triples.values()):
            attr_count[self.attr2id[triple[1]]] += 1
        diff = num_attr - len(attr_count)
        if diff > 0:
            i, cur = 0, 0
            while i < num_attr:
                if i not in attr_count:
                    cur += 1
                    self.attr2id.inverse.pop(i)
                elif cur > 0:
                    val = self.attr2id.inverse[i]
                    self.attr2id[val] = i - cur
                i += 1
            print(
                f"Removed mapping for {diff} attributes as they are not part of the dataset."
            )

    def _remove_duplicate_triples(self):
        assert (
            not self.triples["test"] and not self.triples["valid"]
        )  # works only before splitting data
        values = defaultdict(set)
        for triple in self.triples["train"]:
            values[(triple[0], triple[1])].add(triple[2])
        mean_values = {k: sum(v) / len(v) for k, v in values.items()}
        len_old = len(self.triples["train"])
        self.triples["train"] = [(e, a, mean_values[(e, a)]) for e, a in values.keys()]

        diff = len_old - len(self.triples["train"])
        if diff > 0:
            print(
                f"Skipped {diff} triples as their entity-attribute pairs have more than one value. Mean values are used instead."
            )

    def _remove_duplicate_triples_inplace(self):
        """unused. for troubleshooting"""
        values_per_tuple = defaultdict(set)
        for ent, attr, value in itertools.chain.from_iterable(self.triples.values()):
            values_per_tuple[(ent, attr)].add(value)

        values_per_tuple = {
            x: [sum(values) / len(values), *values]
            for x, values in values_per_tuple.items()
        }

        for _, triples in self.triples.items():
            for i in range(len(triples) - 1, -1, -1):
                ent, attr, _ = triples[i]
                values = values_per_tuple[(ent, attr)]
                if len(values) > 2:
                    del triples[i]
                    values_per_tuple[(ent, attr)].pop()
                elif len(values) == 2:
                    triples[i] == (ent, attr, values[0])

    def split_eval_data(self, name="test", test_size=1000):
        triples = list(self.triples["train"])
        test_indices = random.sample(range(1, len(triples)), test_size)
        self.triples["train"] = [
            triple for i, triple in enumerate(triples) if i not in test_indices
        ]
        self.triples[name] = [triples[i] for i in test_indices]

    def _get_attr_values_per_entity(self):
        attr_values = defaultdict(list)
        for triple in itertools.chain.from_iterable(self.triples.values()):
            attr_values[triple[0]].append((triple[1], triple[2]))
        return attr_values

    def _get_values_per_attribute(self):
        attr_values = defaultdict(list)
        for triple in itertools.chain.from_iterable(self.triples.values()):
            attr_values[triple[1]].append(triple[2])
        return attr_values

    
    # def _get_overall_std(self):
      
    #   import numpy as np
      
      
    #   attr_values = self._get_values_per_attribute()
    #   all_values = []
    #   for v in attr_values.values():
    #     all_values.extend(v)
      
      
      
    #   stdv = np.std(all_values)
      
    #   return stdv
      
    
    
    def get_min_max_values_per_attribute(self):
        try:
            return self.min_max_values
        except:
            # not loaded yet
            attr_values = self._get_values_per_attribute()
            
            self.min_max_values = {
                self.attr2id.inverse[i]: (min(values), max(values))
                for i, values in attr_values.items()
            }
            return self.min_max_values

    def _normalize_min_max(self, min, max, value):
        if min == max:
            return 1.0
        try:
            normalized_value = (value - min) / (max - min)
        except ZeroDivisionError:
            normalized_value = value
        return normalized_value

    def _denormalize_min_max(self, min, max, value):
        return value * (max - min) + min

    def normalize_value_std(self):
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        
        attr_values = self._get_values_per_attribute()
        mean_and_stdv = dict()
        for i, value in attr_values.items():
            scaler = StandardScaler()
            np_val = np.array([value])
            trans_val = np.transpose(np_val)
            scaler.fit(trans_val)
            mean = scaler.mean_
            stdv = scaler.scale_ 
            # res[self.attr2id.inverse[i]] = (
            #     np.transpose(trans_data).reshape((len(value))).tolist()
            # )
            mean_and_stdv[self.attr2id.inverse[i]] = (mean, stdv)
            
        for _, triples in self.triples.items():
            for i in range(len(triples)):
              value = triples[i][2]
              attr = self.attr2id.inverse[triples[i][1]]
              normalized_value = (value - mean_and_stdv[attr][0])/mean_and_stdv[attr][1]
              triples[i] = (triples[i][0], triples[i][1], normalized_value[0])
        
        
        # print(self.triples)

    def normalize_values(self, use_std = False):
        
        if use_std:
          self.normalize_value_std()
          return
        
        attr_values = self.get_min_max_values_per_attribute()
        # test
        for _, triples in self.triples.items():
            for i in range(len(triples)):
                value = triples[i][2]
                attr = self.attr2id.inverse[triples[i][1]]
                normalized_value = self._normalize_min_max(
                    attr_values[attr][0], attr_values[attr][1], value
                )
                triples[i] = (triples[i][0], triples[i][1], normalized_value)

    def _denormalize_values(self, datasets: Tuple["LiteralsDataset", ...]):
        """
        Literal data is normalized. To denormalize, other literal datasets with unnormalized values are required.
        Triples that could not be denormalized are deleted.
        Not used.
        """
        denormalized_min_max_values = defaultdict(tuple)
        for attr_values_dict in [
            dataset.get_min_max_values_per_attribute() for dataset in datasets
        ]:
            for attr, values in attr_values_dict.items():
                if attr not in denormalized_min_max_values:
                    denormalized_min_max_values[attr] = values
                else:
                    denormalized_min_max_values[attr] = (
                        min(denormalized_min_max_values[attr][0], values[0]),
                        max(denormalized_min_max_values[attr][1], values[1]),
                    )

        skipped = defaultdict(int)
        for _, triples in self.triples.items():
            for i in range(len(triples) - 1, -1, -1):
                attr = self.attr2id.inverse[triples[i][1]]
                value = triples[i][2]
                try:
                    denormalized_value = self._denormalize_min_max(
                        denormalized_min_max_values[attr][0],
                        denormalized_min_max_values[attr][1],
                        value,
                    )
                    triples[i] = (triples[i][0], triples[i][1], denormalized_value)
                except IndexError:
                    skipped[attr] += 1
                    del triples[i]

        print(
            f"Skipped {sum(skipped.values())} triples with attr data because the values could not be denormalized using the provided datasets."
        )
        print(f"The following {len(skipped)} attributes are affected:")
        for i, attr in sorted([(self.attr2id[attr], attr) for attr in skipped.keys()]):
            print(i, attr)
