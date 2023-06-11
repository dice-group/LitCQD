from collections import defaultdict
from copy import deepcopy
import rdflib
import pickle
import os
import csv
import time
import random
import argparse
from bidict import bidict
import itertools

random.seed(0)

query_name_dict = {('e', ('r',)): '1p',
                   ('e', ('ap', 'a')): '1ap',
                   ('e', ('r', 'r')): '2p',
                   (('e', ('r',)), ('ap', 'a')): '2ap',
                   ('e', ('r', 'r', 'r',)): '3p',
                   (('e', ('r',)), ('e', ('r',))): '2i',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                   ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                   (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                   (('ap', 'a'), ('v', 'f')): 'ai',
                   ((('ap', 'a'), ('v', 'f')), (('ap', 'a'), ('v', 'f'))): '2ai',
                   (('e', ('r')), (('ap', 'a'), ('v', 'f'))): 'pai',
                   ((('ap', 'a'), ('v', 'f')), ('r')): 'aip',
                   (('e', ('r',)), ('e', ('r',)), ('u',)): '2u',
                   ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up',
                   ((('ap', 'a'), ('v', 'f')), (('ap', 'a'), ('v', 'f')), ('u',)): 'au',
                   }
name_query_dict = {value: key for key, value in query_name_dict.items()}

symbol_placeholder_dict = {
    'u': -1,
    'ap': -3,
    '=': -4,
    '<': -5,
    '>': -6,
}
placeholder_symbol_dict = {value: key for key, value in symbol_placeholder_dict.items()}


def list2tuple(l):
    return tuple(list2tuple(x) if type(x) == list else x for x in l)


def tuple2list(t):
    return list(tuple2list(x) if type(x) == tuple else x for x in t)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Create queries for Q2B',
        usage='query-genration.py [<args>] [-h | --help]'
    )

    parser.add_argument('--data_path', type=str, default=None, help="KG data path")

    return parser.parse_args(args)


class AnswersContainer(object):
    """Contains answers to 1p/1ap queries"""

    def __init__(self, all_answers, queries) -> None:
        if not queries or not all_answers:
            self._answers_rel = dict()
            self._answers_attr = dict()
        else:
            self._answers_rel = {q: a for q, a in all_answers.items() if q in queries[name_query_dict['1p']]}
            self._answers_attr = {q: a for q, a in all_answers.items() if q in queries[name_query_dict['1ap']]}
        self._all_answers = self._answers_rel.copy()
        self._all_answers.update(self._answers_attr)
        self._objects_rel = list(itertools.chain.from_iterable(self._answers_rel.values()))
        self._objects_attr = list(itertools.chain.from_iterable(self._answers_attr.values()))
        self._triples_by_obj_rel = self._get_triples_by_obj_dict(self._answers_rel)
        self._triples_by_obj_attr = self._get_triples_by_obj_dict(self._answers_attr)
        self._triples_by_subj_rel = self._get_triples_by_subj_dict(self._answers_rel)
        self._triples_by_subj_attr = self._get_triples_by_subj_dict(self._answers_attr)

    def _get_triples_by_obj_dict(self, all_answers):
        triples_by_obj = defaultdict(list)
        for q, answers in all_answers.items():
            for answer in answers:
                triples_by_obj[answer].append(q)
        return triples_by_obj

    def _get_triples_by_subj_dict(self, all_answers):
        triples_by_subj = defaultdict(list)
        for q, _ in all_answers.items():
            triples_by_subj[q[0]].append(q)
        return triples_by_subj

    def get(self, query, attr=False):
        return self._answers_attr.get(query, set()) if attr else self._answers_rel.get(query, set())

    def get_objects(self, attr=False):
        return self._objects_attr if attr else self._objects_rel

    def get_triples_by_obj(self, obj, attr=False):
        return self._triples_by_obj_attr[obj] if attr else self._triples_by_obj_rel[obj]

    def get_triples_by_subj(self, subj, attr=False):
        return self._triples_by_subj_attr[subj] if attr else self._triples_by_subj_rel[subj]


class QueryGeneratorGeneric(object):
    def __init__(self, ent2id: bidict, rel2id: bidict):
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.query_names = list(name_query_dict.keys())
        e = 'e'
        r = 'r'
        u = 'u'
        ap = 'ap'
        a = 'a'
        f = 'f'
        v = 'v'
        self.query_structures = [
            [e, [r]],
            [e, [ap, a]],
            [e, [r, r]],
            [[e, [r]], [ap, a]],
            [e, [r, r, r]],
            [[e, [r]], [e, [r]]],
            [[e, [r]], [e, [r]], [e, [r]]],
            [[[e, [r]], [e, [r]]], [r]],
            [[e, [r, r]], [e, [r]]],
            [[ap, a], [v, f]],
            [[[ap, a], [v, f]], [[ap, a], [v, f]]],
            [[e, [r]], [[ap, a], [v, f]]],
            [[[ap, a], [v, f]], [r]],
            [[e, [r]], [e, [r]], [u]],
            [[[e, [r]], [e, [r]], [u]], [r]],
            [[[ap, a], [v, f]], [[ap, a], [v, f]], [u]],
        ]

    def list2tuple(l):
        return tuple(list2tuple(x) if type(x) == list else x for x in l)

    def _is_inverse(self, rel, rel2):
        """The way the relations were created, the inverse of a relation appears at the next index."""
        if rel == rel2+1 and rel2 % 2 == 0:
            return True
        if rel2 == rel+1 and rel % 2 == 0:
            return True
        return False

    def _objects_gen(self, objects_list):
        """Yields endless objects"""
        random.shuffle(objects_list)
        objects = (x for x in objects_list)
        while True:
            res = next(objects)
            if res:
                yield res
            else:
                random.shuffle(objects_list)
                objects = (x for x in objects_list)

    def create_queries(self, queries, all_answers, limit, query_structure_name, answers_base=None, queries_base=None):
        """
        Wrapper for the generate query/answers method.
        limit defines how many queries are supposed to be created.
        answers_base are the answers created for training/validation.
        """
        answer_is_attr_val = query_structure_name in ("1ap", "2ap")
        answers_container = AnswersContainer(all_answers, queries)
        base_container = AnswersContainer(answers_base, queries_base)

        def query_exists(x): return x in queries[name_query_dict[query_structure_name]]
        query_structure = self.query_structures[self.query_names.index(query_structure_name)]

        gen = self._objects_gen(answers_container.get_objects(answer_is_attr_val))
        while len(queries[name_query_dict[query_structure_name]]) < limit:
            query_structure_result = deepcopy(query_structure)
            found = self._generate_query(query_exists, answers_container, next(gen), query_structure_result)
            if found:
                query = list2tuple(query_structure_result)
                query_answers = self._get_answers(answers_container, query)
                # skip queries that can be answered with base graph (g_{train} or g_{val})
                if answers_base:
                    try:
                        query_answers_base = self._get_answers(base_container, query)
                    except KeyError:
                        continue
                    if query_answers == query_answers_base:
                        continue
                queries[name_query_dict[query_structure_name]].add(query)
                all_answers[query] = query_answers

    def _find_variables_prediction(self, query_exists, answers_container: AnswersContainer, answer, rels, pos):
        # rels = ["r","r"] e.g.
        # pos starts with len(rels)-1 and goes to 0
        possible_queries = answers_container.get_triples_by_obj(answer)
        for _ in range(min(len(possible_queries), 4)):
            query = random.choice(possible_queries)
            if query[0] not in answers_container.get_objects():
                # subject of query is never an object
                continue
            if pos == 0:
                if not query_exists(query):
                    if len(rels) == pos + 1 or not self._is_inverse(query[1][0], rels[pos+1]):
                        rels[pos] = query[1][0]
                        return query[0]
            else:
                rels[pos] = query[1][0]
                return self._find_variables_prediction(query_exists, answers_container, query[0], rels, pos-1)
        return None

    def _eval_restriction(self, restriction, value):
        return lambda x: restriction == '=' and value == x or restriction == '<' and value >= x or restriction == '>' and value <= x

    def _generate_query(self, query_exists, answers_container: AnswersContainer, answer, query_structure):
        (('e', ('ap', 'a')), ('v', ('f',)))  # first choose random restriction with an existing value, then set first part of query to a query with that same value as an answer
        (('ap', 'a'), ('v', 'f'))  # first choose random value and restriction, then choose random attribute with that value
        if not [x for x in query_structure[-1] if x != "r"]:
            # only relations at query_structure[-1]
            rels = query_structure[-1]
            val = self._find_variables_prediction(query_exists, answers_container, answer, rels, len(rels)-1)
            if not val:
                return None
            if query_structure[0] == "e":
                query_structure[0] = val
                return True
            else:
                # (...), ('r', ...)
                return self._generate_query(query_exists, answers_container, val, query_structure[0])
        elif query_structure[1][-1] == 'a':
            query = random.choice(answers_container.get_triples_by_obj(answer, attr=True))
            ent = query[0]
            query_structure[1][-1] = query[1][1]
            query_structure[1][-2] = symbol_placeholder_dict['ap']
            if query_structure[0] == 'e':
                query_structure[0] = ent
                return True
            else:
                return self._generate_query(query_exists, answers_container, ent, query_structure[0])
        elif query_structure[1][0] == 'v':
            # ('ap', 'a'), ('v', 'f')
            queries = answers_container.get_triples_by_subj(answer, attr=True)
            if not queries:
                return None
            query = random.choice(queries)
            query_structure[0] = query[1]
            value = next(iter(answers_container.get(query, attr=True)))

            restriction = random.choice(('=', '<', '>'))
            query_structure[1][1] = symbol_placeholder_dict[restriction]

            meets_restriction = self._eval_restriction(restriction, value)
            values = [val for val in answers_container.get_objects(attr=True) if meets_restriction(val)]
            if not values:
                return None
            query_structure[1][0] = random.choice(values)
            return True
        else:
            for i in range(len(query_structure)):
                if query_structure[i][0] == 'u':
                    query_structure[i][0] = symbol_placeholder_dict['u']
                    continue
                found = self._generate_query(query_exists, answers_container, answer, query_structure[i])
                if not found:
                    return None
            if len(query_structure) != len(set(list2tuple(query_structure))):
                return None
            return True

    def _get_answers(self, answers_container, query):
        (('e', ('r')), ('ap', 'a'))
        query = tuple2list(query)
        if not [x for x in query[-1] if type(x) == list or x in symbol_placeholder_dict.values()]:
            # ..., ('r'*x) only relations at query_structure[-1]
            if type(query[0]) == int:
                tmp = {query[0]}
            else:
                tmp = self._get_answers(answers_container, query[0])
            for i in range(len(query[-1])):
                answers = set()
                for entity in tmp:
                    answers = answers.union(answers_container.get((entity, (query[-1][i],))))
                tmp = answers
            return answers
        elif query[1][0] == symbol_placeholder_dict['ap']:
            # ..., ('ap', 'a')
            answers = set()
            for ent in self._get_answers(answers_container, query[0]):
                answers = answers.union(answers_container.get((ent, (query[1][0], query[1][1],)), attr=True))
            return answers
        elif query[1][1] in (symbol_placeholder_dict['<'], symbol_placeholder_dict['='], symbol_placeholder_dict['>']):
            # ('ap', 'a'), ('v', 'f')
            meets_restriction = self._eval_restriction(placeholder_symbol_dict[query[1][1]], query[1][0])
            # all entities with an attribute value meeting the restriction
            return set(itertools.chain.from_iterable([answers_container.get_triples_by_obj(v, attr=True)[0] for v in answers_container.get_objects(attr=True) if meets_restriction(v)]))
        else:
            # union or intersection
            answers = self._get_answers(answers_container, query[0])
            union_flag = False
            if query[-1][0] == symbol_placeholder_dict['u']:
                union_flag = True
            for i in range(1, len(query)):
                if not union_flag:
                    answers = answers.intersection(self._get_answers(answers_container, query[i]))
                else:
                    if i == len(query) - 1:
                        continue
                    answers = answers.union(self._get_answers(answers_container, query[i]))
            return answers


class QueryGenerator(object):
    def __init__(self, output_path, print_debug=True, data_path=None, ent2id=None, rel2id=None, attr2id=None, triples=None, triples_attr=None, do_valid=True, do_test=True):
        self.output_path = output_path
        self.print_debug = print_debug

        self.ent2id, self.rel2id = self._load_mappings(data_path, ent2id, rel2id)
        self.attr2id = self._load_attr_mappings(data_path, attr2id)

        self.graph_train = self._load_data(data_path, 'train', triples["train"] if triples else None)
        self.graph_attr_train = self._load_attr_data(data_path, 'train', triples_attr["train"] if triples_attr else None)

        self.do_valid = do_valid
        self.do_test = do_test
        if self.do_valid or self.do_test:
            self.graph_valid = self._load_data(data_path, 'valid', triples["valid"] if triples else None)
            self.graph_attr_valid = self._load_attr_data(data_path, 'valid', triples_attr["valid"] if triples_attr else None)
        if self.do_test:
            self.graph_test = self._load_data(data_path, 'test', triples["test"] if triples else None)
            self.graph_attr_test = self._load_attr_data(data_path, 'test', triples_attr["test"] if triples_attr else None)

    def _load_mappings(self, data_path, ent2id=None, rel2id=None):
        print("loading mappings...", end='\t\t', flush=True)
        start_time = time.time()
        # Map ids to URIRefs for rdflib support
        prefix = "http://example.org"
        if ent2id and rel2id:
            ent2id = bidict({rdflib.URIRef(prefix+ent): i for ent, i in ent2id.items()})
            rel2id = bidict({rdflib.URIRef(prefix+'/'+rel): i for rel, i in rel2id.items()})
        else:
            ent2id, rel2id = bidict(), bidict()

            if os.path.isfile(os.path.join(data_path, "ent2id.pkl")) and os.path.isfile(os.path.join(data_path, "rel2id.pkl")):
                ent2id_pickle = pickle.load(open(os.path.join(data_path, "ent2id.pkl"), 'rb'))
                rel2id_pickle = pickle.load(open(os.path.join(data_path, "rel2id.pkl"), 'rb'))

                for k, v in ent2id_pickle.items():
                    ent2id.inverse[v] = rdflib.URIRef(prefix+k)
                for k, v in rel2id_pickle.items():
                    rel2id.inverse[v] = rdflib.URIRef(prefix+'/'+k)
            else:
                with open(os.path.join(data_path, "entity2id.txt"), "r") as file:
                    reader = csv.DictReader(file, delimiter='\t', fieldnames=("name", "id"))
                    next(reader)
                    for row in reader:
                        ent2id.inverse[int(row["id"])] = rdflib.URIRef(prefix+row["name"])

                with open(os.path.join(data_path, "relation2id.txt"), "r") as file:
                    reader = csv.DictReader(file, delimiter='\t', fieldnames=("name", "id"))
                    next(reader)
                    for row in reader:
                        rel2id.inverse[int(row["id"])] = rdflib.URIRef(prefix+'/'+row["name"])

        print(" {:2f} seconds".format(time.time() - start_time))
        return ent2id, rel2id

    def _load_attr_mappings(self, data_path, attr2id=None):
        print("loading attribute mappings...", end='\t', flush=True)
        start_time = time.time()
        # Map ids to URIRefs for rdflib support
        prefix = "http://example.org"
        if attr2id:
            attr2id = bidict({rdflib.URIRef(prefix+attr): i for attr, i in attr2id.items()})
        else:
            attr2id = bidict()
            with open(os.path.join(data_path, "attr2id.txt"), "r") as file:
                reader = csv.DictReader(file, delimiter='\t', fieldnames=("name", "id"))
                next(reader)
                for row in reader:
                    attr2id.inverse[int(row["id"])] = rdflib.URIRef(prefix+row["name"])

        print(" {:2f} seconds".format(time.time() - start_time))
        return attr2id

    def _load_data(self, data_path, name, triples=None):
        print(f"loading {name} data...", end='\t\t', flush=True)
        start_time = time.time()

        if triples is None:
            triples = list()
            with open(os.path.join(data_path, name+".txt")) as csvfile:
                for row in csv.reader(csvfile, delimiter='\t'):
                    triples.append((int(row[0]), int(row[1]), int(row[2])))

        g = rdflib.Graph()
        for ent1, rel, ent2 in triples:
            g.add((self.ent2id.inverse[ent1], self.rel2id.inverse[rel], self.ent2id.inverse[ent2]))

        print(" {:2f} seconds".format(time.time() - start_time))
        return g

    def _load_attr_data(self, data_path, name, triples=None):
        print(f"loading {name} attribute data...", end='\t', flush=True)
        start_time = time.time()

        if triples is None:
            triples = list()
            with open(os.path.join(data_path, f"attrm2id_{name}.txt")) as csvfile:
                reader = csv.reader(csvfile, delimiter='\t')
                next(reader)
                for row in reader:
                    triples.append((int(row[0]), int(row[1]), float(row[2])))

        g = rdflib.Graph()
        for ent, attr, val in triples:
            g.add((self.ent2id.inverse[ent], self.attr2id.inverse[attr], rdflib.Literal(val)))

        print(" {:2f} seconds".format(time.time() - start_time))
        return g

    def _store_stats(self, num_ent, num_rel):
        with open(os.path.join(self.output_path, "stats.txt"), "w") as f:
            f.write("numentity: %d\nnumrelations: %d" % num_ent, num_rel)

    def _print_debuginfo(self, name, queries, answers):
        count = 0
        for q in queries[name_query_dict[name]]:
            count += len(answers[q])
        print(f"{name}: {len(queries[name_query_dict[name]])} queries; {count} answers; Average number of answers: ", end='')
        if len(queries[name_query_dict[name]]) > 0:
            print(count / len(queries[name_query_dict[name]]))
        else:
            print(0)

    def _generate_1p_queries(self, graph, queries, answers):
        """Create all possible queries. Limit is ignored"""
        for s, p, o in graph.triples((None, None, None)):
            query = (self.ent2id[s], (self.rel2id[p],))
            queries[name_query_dict["1p"]].add(query)
            if query in answers:
                answers[query].add(self.ent2id[o])
            else:
                answers[query] = {self.ent2id[o]}

    def _generate_1ap_queries(self, graph, queries, answers):
        """Create all possible queries. Limit is ignored"""
        for s, p, o in graph.triples((None, None, None)):
            query = (self.ent2id[s], (symbol_placeholder_dict['ap'], self.attr2id[p]))
            queries[name_query_dict["1ap"]].add(query)
            if query in answers:
                answers[query].add(o.toPython())
            else:
                answers[query] = {o.toPython()}

    def _create_train_queries(self):
        graph = self.graph_train
        graph_attr = self.graph_attr_train
        queries = dict()
        answers = dict()
        query_names_train = ("1p", "1ap")
        for name in query_names_train:
            queries[name_query_dict[name]] = set()

        print(f"train: Creating 1p queries...", end='\t', flush=True)
        start_time = time.time()
        self._generate_1p_queries(graph, queries, answers)
        print(" {:2f} seconds".format(time.time() - start_time))

        print(f"train: Creating 1ap queries...", end='\t', flush=True)
        start_time = time.time()
        self._generate_1ap_queries(graph_attr, queries, answers)
        print(" {:2f} seconds".format(time.time() - start_time))

        if self.print_debug:
            for name in query_names_train:
                self._print_debuginfo(name, queries, answers)

        return queries, answers

    def _create_eval_queries(self, type, answers_base, queries_base):
        if type == 'valid':
            graph_eval = self.graph_valid
            graph_full = self.graph_valid + self.graph_train
            graph_attr_eval = self.graph_attr_valid
            graph_attr_full = self.graph_attr_valid + self.graph_attr_train
        elif type == 'test':
            graph_eval = self.graph_test
            graph_full = self.graph_test + self.graph_valid + self.graph_train
            graph_attr_eval = self.graph_attr_test
            graph_attr_full = self.graph_attr_test + self.graph_attr_valid + self.graph_attr_train
        queries_eval = dict()
        answers_eval_hard = dict()
        answers_eval_easy = dict()
        answers_eval_full = dict()
        complex_query_names = [x for x in name_query_dict.keys() if not x.startswith('1')]
        for name in name_query_dict.keys():
            queries_eval[name_query_dict[name]] = set()

        generator = QueryGeneratorGeneric(self.ent2id, self.rel2id)

        # Generate simple queries using the full graph
        def generate_simple(name, func, graph):
            print(f"{type}: Creating {name} queries...", end='\t', flush=True)
            start_time = time.time()
            func(graph, queries_eval, answers_eval_full)
            print(" {:2f} seconds".format(time.time() - start_time))

        generate_simple('1p', self._generate_1p_queries, graph_full)
        generate_simple('1ap', self._generate_1ap_queries, graph_attr_full)

        # Generate complex queries using the full graph
        def generate_complex(name, limit):
            print(f"{type}: Creating {name} queries...", end='\t', flush=True)
            start_time = time.time()
            generator.create_queries(queries_eval, answers_eval_full, limit, name, answers_base, queries_base)
            print(" {:2f} seconds".format(time.time() - start_time))

        for name in complex_query_names:
            limit = 5000
            if 'a' in name:
                limit = 1000
            generate_complex(name, limit)

        print(f"{type}: Computing easy & hard answers...")

        # Get answers using G_{train} or G_{train}+G_{valid} (easy answers)
        for name in ('1p', '1ap'):
            for query in queries_eval[name_query_dict[name]]:
                answers_eval_easy[query] = answers_base.get(query, set())
        easy_answers_container = AnswersContainer(answers_eval_easy, queries_eval)
        for name in complex_query_names:
            for query in queries_eval[name_query_dict[name]]:
                answers_eval_easy[query] = generator._get_answers(easy_answers_container, query)

        # Generate 1p/1ap queries using eval graph only
        queries_eval[name_query_dict["1p"]] = set()
        queries_eval[name_query_dict["1ap"]] = set()
        self._generate_1p_queries(graph_eval, queries_eval, answers_eval_hard)
        self._generate_1ap_queries(graph_attr_eval, queries_eval, answers_eval_hard)

        # Compute hard answers for complex queries
        for name in complex_query_names:
            for query in queries_eval[name_query_dict[name]]:
                answers_eval_hard[query] = answers_eval_full[query].difference(answers_eval_easy[query])

        if self.print_debug:
            for name in name_query_dict.keys():
                print('-'*0+"easy"+70*'-'+"easy")
                self._print_debuginfo(name, queries_eval, answers_eval_easy)
                print('-'*0+"hard"+70*'-'+"hard")
                self._print_debuginfo(name, queries_eval, answers_eval_hard)

        return queries_eval, answers_eval_full, answers_eval_easy, answers_eval_hard

    def create_queries(self):
        queries_train, answers_train = self._create_train_queries()

        if self.do_valid:
            queries_val, answers_val_full, answers_val_easy, answers_val_hard = self._create_eval_queries("valid", answers_train, queries_train)
            queries_val_train = queries_train.copy()
            queries_val_train.update(queries_val)
        else:
            answers_val_full = answers_train
            queries_val_train = queries_train
        if self.do_test:
            queries_test, answers_test_full, answers_test_easy, answers_test_hard = self._create_eval_queries("test", answers_val_full, queries_val_train)

        def test():
            def do(name, queries, answers):
                print(f"{name} queries with...")
                print(f"1     answer:  {len([q for q in queries[name_query_dict[name]] if len(answers[q]) == 1])}")
                print(f"<=3   answers: {len([q for q in queries[name_query_dict[name]] if len(answers[q]) in range(2,4)])}")
                print(f"<=10  answers: {len([q for q in queries[name_query_dict[name]] if len(answers[q]) in range(4, 11)])}")
                print(f"<=50  answers: {len([q for q in queries[name_query_dict[name]] if len(answers[q]) in range(11, 51)])}")
                print(f"<=100 answers: {len([q for q in queries[name_query_dict[name]] if len(answers[q]) in range(51, 101)])}")
                print(f">100  answers: {len([q for q in queries[name_query_dict[name]] if len(answers[q]) > 100])}")
                print("==========================")

            print("Getting paper queries...")
            queries = pickle.load(open(os.path.join(self.output_path, "train-queries.pkl"), 'rb'))
            answers = pickle.load(open(os.path.join(self.output_path, "train-answers.pkl"), 'rb'))

            for name in ("1p", "2p", "3p", "2i", "3i"):
                self._print_debuginfo(name, queries, answers)

            exit(0)
            for name in ("1p", "2p", "3p", "2i", "3i"):
                print("paper's ", end='')
                do(name, queries, answers)
                do(name, queries_train, answers_train)
        # test()

        print("Storing queries...")
        pickle.dump(queries_train, open(os.path.join(self.output_path, "train-queries.pkl"), 'wb'))
        pickle.dump(answers_train, open(os.path.join(self.output_path, "train-answers.pkl"), 'wb'))
        if self.do_valid:
            pickle.dump(queries_val, open(os.path.join(self.output_path, "valid-queries.pkl"), 'wb'))
            pickle.dump(answers_val_hard, open(os.path.join(self.output_path, "valid-hard-answers.pkl"), 'wb'))
            pickle.dump(answers_val_easy, open(os.path.join(self.output_path, "valid-easy-answers.pkl"), 'wb'))
        if self.do_test:
            pickle.dump(queries_test, open(os.path.join(self.output_path, "test-queries.pkl"), 'wb'))
            pickle.dump(answers_test_hard, open(os.path.join(self.output_path, "test-hard-answers.pkl"), 'wb'))
            pickle.dump(answers_test_easy, open(os.path.join(self.output_path, "test-easy-answers.pkl"), 'wb'))

        num_ent = len(self.ent2id)  # +1
        num_rel = len(self.rel2id)  # +1
        # self._store_stats(num_ent, num_rel)
        return num_ent, num_rel, queries_train, answers_train, queries_val, answers_val_hard, answers_val_easy, queries_test, answers_test_hard, answers_test_easy


def main(args):
    QueryGenerator(args.data_path).create_queries()


if __name__ == '__main__':
    main(parse_args())
