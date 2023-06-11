from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short
from gensim.models import Word2Vec
import itertools
import time
from bidict import bidict
from .DatasetProperty import DatasetProperty
import csv
import os

from gensim import corpora
from gensim.models import TfidfModel
import pandas as pd
import random
random.seed(0)


class DescriptionsDatasetJointly(object):
    """
    Extracts the top 20 keywords from each entity description using tf-idf.
    triples contain all connections from entities to these 20 keywords.
    """

    def __init__(self, path, name_path, triples=None):
        self.path = path
        self.name_path = name_path
        self.triples = triples
        self.word2id = None

    triples = DatasetProperty("triples")
    word2id = DatasetProperty("word2id")

    def __str__(self):
        res = f"{type(self).__name__}:"
        for name, triples in self.triples.items():
            res += f"\n\t{name} triples: {len(triples)}"
        res += f"\n\tentities: {len(set([x[0] for x in itertools.chain.from_iterable(self.triples.values())]))}"
        res += f'\n\tvocab size: {len(self.word2id)}'
        return res

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

    def _load_descriptions(self):
        ent2name = dict()
        names = list()
        with open(os.path.join(self.name_path, 'ent2name.txt'), "r") as file:
            reader = pd.read_csv(file, delimiter='\s+', names=['ent', 'name'], header=None)
            for _, row in reader.iterrows():
                if row['ent'] not in ent2name:
                    ent2name[row['ent']] = len(names)
                    names.append(row['name'])

        ent2desc = dict()
        descriptions = list()
        with open(os.path.join(self.path, 'text_literals.txt'), "r") as file:
            reader = csv.reader(file, delimiter='\t')
            for entity, _, description in reader:
                if entity not in ent2desc:
                    ent2desc[entity] = len(descriptions)
                    descriptions.append(description)

        # Use name if description is unavailable
        for ent, i in ent2name.items():
            if ent not in ent2desc:
                ent2desc[ent] = len(descriptions)
                descriptions.append(names[i])
        return descriptions, ent2desc

    def _preprocess_descriptions(self, descriptions):
        # preprocess textual descriptions
        descriptions = [preprocess_string(desc, filters=[lambda x: x.lower(), strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short]) for desc in descriptions]
        model = self._train_model(descriptions)
        word2id = {x: i for i, x in enumerate(model.index_to_key)}
        # remove words not part of vocab and map to ids
        for i in range(len(descriptions)):
            descriptions[i] = [word for word in descriptions[i] if word in word2id]
        return descriptions

    def _extract_top_k_keywords(self, descriptions, k=20):
        dct = corpora.Dictionary(descriptions)
        corpus = [dct.doc2bow(desc) for desc in descriptions]
        model = TfidfModel(corpus)

        for i in range(len(descriptions)):
            top_k = list()
            top_words = [dct[j] for j, _ in sorted(model[corpus[i]], key=lambda x: x[1], reverse=True)]
            j = 0
            while j < k and j < len(top_words):
                top_k.append(top_words[j])
                j += 1
            descriptions[i] = top_k

        # map words to ids
        unique_words = set()
        for desc in descriptions:
            unique_words.update(desc)

        word2id = {word: i for i, word in enumerate(unique_words)}
        for i in range(len(descriptions)):
            descriptions[i] = [word2id[word] for word in descriptions[i]]
        return descriptions, word2id

    def _create_triples(self, descriptions, ent2desc, ent2id):
        triples = list()
        for ent, id in ent2id.items():
            if ent in ent2desc:
                for word in descriptions[ent2desc[ent]]:
                    triples.append((id, '/description', word))
        self.triples = {"train": triples, "valid": [], "test": []}

    def load_data(self, ent2id):
        descriptions, ent2desc = self._load_descriptions()
        descriptions = self._preprocess_descriptions(descriptions)
        descriptions, word2id = self._extract_top_k_keywords(descriptions)
        self.word2id = bidict(word2id)
        self._create_triples(descriptions, ent2desc, ent2id)

    def split_eval_data(self, name="test", test_size=1000):
        # split by entities
        triples = list(self.triples["train"])
        entities = {x[0] for x in triples}
        test_indices = random.sample(range(1, len(entities)), test_size)
        self.triples["train"] = [triple for triple in triples if triple[0] not in test_indices]
        self.triples[name] = [triple for triple in triples if triple[0] in test_indices]
