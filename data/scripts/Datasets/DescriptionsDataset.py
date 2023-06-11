import time
import gensim.downloader
import gensim
from .DatasetProperty import DatasetProperty
from collections import defaultdict
import csv
import os
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short
import random
random.seed(0)


class DescriptionsDataset(object):
    """
    triples contains pre-processed entity descriptions.
    vectors contains connections from entities to the vector representing their descriptions.
    word vectors are trained using word2vec.
    """

    def __init__(self, path, triples=None, vectors=None):
        self.path = path
        self.triples = triples
        self.vectors = vectors

    triples = DatasetProperty("triples")
    vectors = DatasetProperty("vectors")

    def __str__(self):
        res = f"{type(self).__name__}:"
        for name, triples in self.triples.items():
            res += f"\n\t{name} entities: {len(triples)}"
        return res

    def _load_data(self):
        triples = list()
        with open(os.path.join(self.path, 'text_literals.txt'), "r") as file:
            reader = csv.reader(file, delimiter='\t')
            for entity, _, description in reader:
                triples.append((entity, '/description', description))
        self.triples = {"train": triples, "valid": [], "test": []}

    def load_data(self, ent2id):
        self._load_data()
        self._map_entities_to_ids(ent2id)
        self._remove_duplicate_triples()
        self._compute_vectors()

    def _train_model(self, corpus):
        print('Training word2vec model...', end='\t', flush=True)
        start_time = time.time()
        model = Word2Vec(sentences=corpus, workers=1, seed=0, vector_size=100)
        model.save(os.path.join(self.path, 'word2vec_model'))
        # explictly free memory
        word_vectors = model.wv
        del model
        print(" {:2f} seconds".format(time.time() - start_time))
        return word_vectors

    def _map_entities_to_ids(self, ent2id):
        skipped = defaultdict(int)
        for _, rows in self.triples.items():
            for i in range(len(rows)-1, -1, -1):
                row = rows[i]
                if row[0] not in ent2id:
                    skipped[row[0]] += 1
                    del rows[i]
                    continue
                rows[i] = (ent2id[row[0]], row[1], row[2])
        if skipped:
            print(f"Skipped {sum(skipped.values())} triples with attr data for {len(skipped)} entities that are not part of the dataset.")

    def _remove_duplicate_triples(self):
        assert not self.triples['test'] and not self.triples['valid']  # works only before splitting data
        descriptions = defaultdict(set)
        for triple in self.triples['train']:
            descriptions[(triple[0], triple[1])].add(triple[2])
        longest_descriptions = {k: max(v, key=len) for k, v in descriptions.items()}
        len_old = len(self.triples["train"])
        self.triples['train'] = [(e, d, longest_descriptions[(e, d)]) for e, d in descriptions.keys()]

        diff = len_old - len(self.triples["train"])
        if diff > 0:
            print(f"Skipped {diff} triples as some entities have multiple textual descriptions. The longest description is used instead.")

    def _compute_vectors(self):
        # preprocess textual descriptions
        for i in range(len(self.triples['train'])):
            ent, rel, desc = self.triples['train'][i]
            processed_desc = preprocess_string(desc, filters=[lambda x: x.lower(), strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short])
            self.triples['train'][i] = (ent, rel, processed_desc)
        descriptions = [x[2] for x in self.triples['train']]

        model = self._train_model(descriptions)
        vectors = defaultdict(list)
        for ent, _, description in self.triples['train']:
            # x create single embedding for description
            word_embeddings = list()
            for word in description:
                try:
                    word_embeddings.append(model[word])
                except:
                    pass  # skipping words not found in word2vec model
            if word_embeddings:
                vectors['train'].append((ent,  sum(word_embeddings)/len(word_embeddings)))
            else:
                print(f'Unable to compute word embedding for entity {ent}. Exiting')
                exit()
        self.vectors = vectors

    def split_eval_data(self, name="test", test_size=1000):
        triples = list(self.triples["train"])
        vectors = list(self.vectors["train"])
        test_indices = random.sample(range(1, len(triples)), test_size)
        self.triples["train"] = [triple for i, triple in enumerate(triples) if i not in test_indices]
        self.vectors["train"] = [triple for i, triple in enumerate(vectors) if i not in test_indices]
        self.triples[name] = [triples[i] for i in test_indices]
        self.vectors[name] = [vectors[i] for i in test_indices]
