from collections import defaultdict
from copy import deepcopy
import pickle
import os
import statistics
import time
import random
import itertools
import numpy

random.seed(0)

query_name_dict = {('e', ('r',)): '1p',
                   ('e', ('ap', 'a')): '1ap',
                   ('e', ('dp',)): '1dp',
                   ('e', ('r', 'r')): '2p',
                   (('e', ('r',)), ('ap', 'a')): '2ap',
                   ('e', ('r', 'r', 'r',)): '3p',
                   (('e', ('r', 'r',)), ('ap', 'a')): '3ap',
                   (('e', ('r',)), ('e', ('r',))): '2i',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                   ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                   (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                   (('dp',), ('dv', '=')): 'di',
                   (('ap', 'a'), ('v', 'f')): 'ai',
                   (('ap', 'a'), ('v', '=')): 'ai-eq',
                   (('ap', 'a'), ('v', '<')): 'ai-lt',
                   (('ap', 'a'), ('v', '>')): 'ai-gt',
                   ((('ap', 'a'), ('v', 'f')), (('ap', 'a'), ('v', 'f'))): '2ai',
                   (('e', ('r',)), (('ap', 'a'), ('v', 'f'))): 'pai',
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
    'dp': -7,
    'dv': -8,
}
placeholder_symbol_dict = {value: key for key, value in symbol_placeholder_dict.items()}


def list2tuple(l):
    return tuple(list2tuple(x) if type(x) == list else x for x in l)


def tuple2list(t):
    return list(tuple2list(x) if type(x) == tuple else x for x in t)


class AnswersContainer(object):
    """Contains 1p/1ap triples"""

    def __init__(self, triples_rel, triples_attr, triples_desc=None) -> None:
        self.t_rel, self.t_attr = triples_rel, triples_attr
        self.triples_rel, self.triples_rel_inv = self._construct_graph(triples_rel)
        self.triples_attr, self.triples_attr_inv = self._construct_graph(triples_attr)
        triples_desc = triples_desc if triples_desc else list()
        self.triples_desc, self.triples_desc_inv = self._construct_graph(triples_desc)

    def _construct_graph(self, triples):
        ent_in, ent_out = dict(), dict()
        for s, p, o in triples:
            if s not in ent_out:
                ent_out[s] = dict()
            if o not in ent_in:
                ent_in[o] = dict()
            if p in ent_out[s]:
                ent_out[s][p].add(o)
            else:
                ent_out[s][p] = {o}
            if p in ent_in[o]:
                ent_in[o][p].add(s)
            else:
                ent_in[o][p] = {s}
        return ent_out, ent_in


class QueryGeneratorGeneric(object):
    def __init__(self, attr_values: dict()):
        self.attr_values = attr_values
        self.query_names = list(name_query_dict.keys())
        e = 'e'
        r = 'r'
        u = 'u'
        ap = 'ap'
        a = 'a'
        dp = 'dp'
        d = 'd'
        f = 'f'
        eq = '='
        lt = '<'
        gt = '>'
        v = 'v'
        self.query_structures = [
            [e, [r]],
            [e, [ap, a]],
            [e, [dp, d]],
            [e, [r, r]],
            [[e, [r]], [ap, a]],
            [e, [r, r, r]],
            [[e, [r, r]], [ap, a]],
            [[e, [r]], [e, [r]]],
            [[e, [r]], [e, [r]], [e, [r]]],
            [[[e, [r]], [e, [r]]], [r]],
            [[e, [r, r]], [e, [r]]],
            [[dp], [[e, [dp]], eq]],
            [[ap, a], [v, f]],
            [[ap, a], [v, eq]],
            [[ap, a], [v, lt]],
            [[ap, a], [v, gt]],
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
        return rel != rel2 and rel // 2 == rel2 // 2

    def _objects_gen(self, objects_list):
        """Yields endless objects"""
        random.shuffle(objects_list)
        objects = (x for x in objects_list)
        while True:
            try:
                yield next(objects)
            except StopIteration:
                random.shuffle(objects_list)
                objects = (x for x in objects_list)

    def _objects_gen_orig(self, objects_list):
        """yielding random entity as done in the q2b script"""
        while True:
            yield random.sample(objects_list, 1)[0]

    def create_queries(self, limit, query_structure_name, answers_container: AnswersContainer, base_container: AnswersContainer):
        """
        Wrapper for the generate query/answers method.
        limit defines how many queries are supposed to be created.
        answers_base are the answers created for training/validation.
        """
        answer_is_attr_val = query_structure_name in ("1ap", "2ap", "3ap")

        query_structure = self.query_structures[self.query_names.index(query_structure_name)]

        queries = set()
        answers = dict()
        answers_easy = dict()

        # sample including duplicates -> prefer answers occuring more often
        # gen = self._objects_gen([t[2] for t in answers_container.t_attr] if answer_is_attr_val else [t[2] for t in answers_container.t_rel])
        gen = self._objects_gen(list(answers_container.triples_attr_inv.keys()) if answer_is_attr_val else list(answers_container.triples_rel_inv.keys()))
        tries_since_last_found = 0
        while len(queries) < limit:
            # TODO: it seems like that the amount of generated queries of "<" and ">" cannot meet the given limit of more than 100
            if tries_since_last_found > 10000:
                break  # unable to find as many queries as the given limit
            tries_since_last_found += 1
            query_structure_result = deepcopy(query_structure)
            found = self._generate_query(answers_container, next(gen), query_structure_result)
            if not found:
                continue
            query = list2tuple(query_structure_result)
            if query in queries:
                continue
            query_answers = self.get_answers(answers_container, query)
            if not query_answers:
                # no answers found
                continue
            # skip queries that can be answered with base graph (g_{train} or g_{val})
            query_answers_base = self.get_answers(base_container, query)
            if query_answers == query_answers_base:
                continue
            queries.add(query)
            answers[query] = query_answers
            answers_easy[query] = query_answers_base
            tries_since_last_found = 0
        return queries, answers, answers_easy

    def _find_variables_prediction(self, answers_container: AnswersContainer, answer, rels, pos):
        # rels = ["r","r"] e.g.
        # pos starts with len(rels) and goes to 0
        if pos == 0:
            return answer

        found = False
        for _ in range(40):
            relations = list()
            for k, values in answers_container.triples_rel_inv[answer].items():
                for v in values:
                    relations.append((k, v))
            r = random.sample(relations, 1)[0][0]
            if len(rels) != pos and self._is_inverse(r, rels[pos]):
                continue
            found = True
            break
        if not found:
            return None
        s = random.sample(answers_container.triples_rel_inv[answer][r], 1)[0]
        rels[pos-1] = r
        return self._find_variables_prediction(answers_container, s, rels, pos-1)

    def _eval_restriction(self, restriction, value, stdev=0):
        def fun(x):
            if restriction == '=':
                return value >= (x - stdev) and value <= (x + stdev)
            elif restriction == '<':
                return value >= x
            elif restriction == '>':
                return value <= x
        return fun

    def _generate_query(self, answers_container: AnswersContainer, answer, query_structure):
        if not [x for x in query_structure[-1] if x != "r"]:
            # only relations at query_structure[-1]
            rels = query_structure[-1]
            val = self._find_variables_prediction(answers_container, answer, rels, len(rels))
            if not val:
                return None
            if query_structure[0] == "e":
                query_structure[0] = val
                return True
            else:
                # (...), ('r', ...)
                return self._generate_query(answers_container, val, query_structure[0])
        elif query_structure[1][-1] == 'a':
            # ..., ('ap', 'a')
            a = random.sample(answers_container.triples_attr_inv[answer].keys(), 1)[0]
            e = next(iter(answers_container.triples_attr_inv[answer][a]))
            query_structure[1][-1] = a
            query_structure[1][-2] = symbol_placeholder_dict['ap']
            if query_structure[0] == 'e':
                query_structure[0] = e
                return True
            else:
                return self._generate_query(answers_container, e, query_structure[0])
        elif query_structure[1][0] == 'v':
            # ('ap', 'a'), ('v', 'f')
            # the queries of smaller, equal, bigger restrictions

            if query_structure[1][1] == 'f':
                restriction = random.choice(('=', '<', '>'))
            else:
                restriction = query_structure[1][1]
            query_structure[1][1] = symbol_placeholder_dict[restriction]

            if restriction == '=':
                try:
                    a = random.sample(answers_container.triples_attr[answer].keys(), 1)[0]
                except KeyError:
                    return None  # entity has no attributes
                v = next(iter(answers_container.triples_attr[answer][a]))
                # query_structure[0] = (symbol_placeholder_dict['ap'], a)
                query_structure[1][0] = v
            else:
                use_mean_value = True
                if use_mean_value:
                    a = random.sample(self.attr_values.keys(), 1)[0]
                    query_structure[1][0] = sum(self.attr_values[a]) / len(self.attr_values[a])
                else:
                    try:
                        a = random.sample(answers_container.triples_attr[answer].keys(), 1)[0]
                    except KeyError:
                        return None  # entity has no attributes
                    v = next(iter(answers_container.triples_attr[answer][a]))
                    meets_restriction = self._eval_restriction(restriction, v)
                    possible_values = [val for val, attr_ent in answers_container.triples_attr_inv.items() if a in attr_ent.keys()]
                    values = [val for val in possible_values if meets_restriction(val)]
                    try:
                        query_structure[1][0] = random.choice(values)
                    except ValueError:
                        return None  # no value meets restriction

            query_structure[0] = (symbol_placeholder_dict['ap'], a)
            return True
        else:
            for i in range(len(query_structure)):
                if query_structure[i][0] == 'u':
                    query_structure[i][0] = symbol_placeholder_dict['u']
                    continue
                found = self._generate_query(answers_container, answer, query_structure[i])
                if not found:
                    return None
            if len(query_structure) != len(set(list2tuple(query_structure))):
                return None
            return True

    def get_answers(self, answers_container: AnswersContainer, query):
        query = tuple2list(query)
        if not [x for x in query[-1] if type(x) == list or x in symbol_placeholder_dict.values()]:
            # ..., ('r'*x) only relations at query_structure[-1]
            if type(query[0]) == int:
                tmp = {query[0]}
            else:
                tmp = self.get_answers(answers_container, query[0])
            for i in range(len(query[-1])):
                answers = set()
                for entity in tmp:
                    try:
                        answers = answers.union(answers_container.triples_rel[entity][query[-1][i]])
                    except KeyError:
                        pass
                tmp = answers
            return answers
        elif query[1][0] == symbol_placeholder_dict['ap']:
            # ..., ('ap', 'a')
            if type(query[0]) == int:
                # ('e', ('ap', 'a'))
                try:
                    return answers_container.triples_attr[query[0]][query[1][1]]
                except KeyError:
                    return set()
            # (...), ('ap', 'a')
            answers = set()
            for ent in self.get_answers(answers_container, query[0]):
                try:
                    answers = answers.union(answers_container.triples_attr[ent][query[1][1]])
                except KeyError:
                    pass
            return answers
        elif query[1][1] in (symbol_placeholder_dict['<'], symbol_placeholder_dict['='], symbol_placeholder_dict['>']):
            # ('ap', 'a'), ('v', 'f')
            stdev = 0
            if query[1][1] == symbol_placeholder_dict['=']:
                attr_values = self.attr_values[query[0][1]]
                if len(attr_values) > 1:
                    stdev = statistics.stdev(attr_values)
            meets_restriction = self._eval_restriction(placeholder_symbol_dict[query[1][1]], query[1][0], stdev)
            answers = set()
            for ent, attr_value_dict in answers_container.triples_attr.items():
                for attr, value in attr_value_dict.items():
                    if attr == query[0][1] and meets_restriction(next(iter(value))):
                        answers.add(ent)
            return answers
        else:
            # union or intersection
            answers = self.get_answers(answers_container, query[0])
            union_flag = False
            if query[-1][0] == symbol_placeholder_dict['u']:
                union_flag = True
            for i in range(1, len(query)):
                if not union_flag:
                    answers = answers.intersection(self.get_answers(answers_container, query[i]))
                else:
                    if i == len(query) - 1:
                        continue
                    answers = answers.union(self.get_answers(answers_container, query[i]))
            return answers


class QueryGenerator(object):
    def __init__(self, output_path, print_debug=True, triples=None, triples_attr=None, do_valid=True, do_test=True, attr_exists_threshold=None, complex_train_queries=False, descriptions=None, desc_jointly=False):
        self.output_path = output_path
        self.print_debug = print_debug

        self.triples = triples
        self.triples_attr = triples_attr

        self.do_valid = do_valid
        self.do_test = do_test

        self.attr_exists_threshold = attr_exists_threshold

        self.complex_train_queries = complex_train_queries

        self.descriptions = descriptions
        self.desc_jointly = desc_jointly

    def _print_debuginfo(self, all_queries, answers_easy, answers_hard=None):
        print("type\t (#queries): ", end='')
        if answers_hard:
            print("\t#easy \t\t#hard")
        else:
            print("\t#answers")
        for query_type, name in query_name_dict.items():
            queries = set()
            try:
                queries = all_queries[query_type]
            except KeyError:
                pass
            count_easy = 0
            count_hard = 0
            for query in queries:
                count_easy += len(answers_easy[query])
                if answers_hard:
                    count_hard += len(answers_hard[query])
            print(f"{name}\t ({len(queries):,}): \t{count_easy:,}", end='')
            if answers_hard:
                print(f"  \t{count_hard:,}")
            else:
                print()

    def _generate_1p_queries(self, triples, queries, answers):
        """Create all possible 1p queries"""
        for s, p, o in triples:
            query = (s, (p,))
            queries[name_query_dict["1p"]].add(query)
            answers[query].add(o)

    def _generate_1ap_queries(self, triples, queries, answers):
        """Create all possible 1ap queries"""
        for s, p, o in triples:
            query = (s, (symbol_placeholder_dict['ap'], p))
            queries[name_query_dict["1ap"]].add(query)
            answers[query].add(o)

    def _generate_1dp_queries(self, descriptions, queries, answers):
        if self.desc_jointly:
            return self._generate_1dp_queries_jointly(descriptions, queries, answers)
        """Create all possible 1dp queries"""
        for entity, description in descriptions:
            query = (entity, (symbol_placeholder_dict['dp'],))
            queries[name_query_dict["1dp"]].add(query)
            answers[query] = {tuple(description)}

    def _generate_1dp_queries_jointly(self, triples, queries, answers):
        """Create all possible 1dp queries"""
        for entity, _, word in triples:
            query = (entity, (symbol_placeholder_dict['dp'],))
            if query in queries[name_query_dict["1dp"]]:
                answers[query].add(word)
            else:
                queries[name_query_dict["1dp"]].add(query)
                answers[query] = {word}

    def _generate_di_queries_jointly(self, triples, queries, answers):
        # eval 1dp queries have been generated already
        eval_descriptions = [x for x in triples if (x[0], (symbol_placeholder_dict['dp'],)) in queries[name_query_dict['1dp']]]
        container_eval = AnswersContainer(list(), list(), eval_descriptions)
        container_full = AnswersContainer(list(), list(), triples)
        entities = container_eval.triples_desc.keys()
        for ent in entities:
            words = container_eval.triples_desc[ent]['/description']
            query = ((symbol_placeholder_dict['dp'],), (tuple(words), symbol_placeholder_dict['=']))
            queries[name_query_dict['di']].add(query)
            answers_query = list()
            for word in words:
                answers_query.append(container_full.triples_desc_inv[word]['/description'])
            counts = defaultdict(int)
            for sub_answers in answers_query:
                for ent in sub_answers:
                    counts[ent] += 1
            answers_query = {ent for ent, count in counts.items() if count >= len(words)/2}
            answers[query] = answers_query

    def _generate_di_queries(self, descriptions, queries, answers):
        if self.desc_jointly:
            return self._generate_di_queries_jointly(descriptions, queries, answers)

        def cosine_similarity(x, y):
            return numpy.dot(x, y)/(numpy.linalg.norm(x)*numpy.linalg.norm(y))

        # eval 1dp queries have been generated already
        eval_descriptions = [x for x in descriptions if (x[0], (symbol_placeholder_dict['dp'],)) in queries[name_query_dict['1dp']]]
        for _, vector in eval_descriptions:
            query_answers = set()
            for i in range(len(descriptions)):
                ent_i, vector_i = descriptions[i]
                similarity = cosine_similarity(vector, vector_i)
                if similarity > 0.9:
                    query_answers.add(ent_i)
            if not query_answers:
                continue
            query = ((symbol_placeholder_dict['dp'],), (tuple(vector), symbol_placeholder_dict['=']))
            queries[name_query_dict['di']].add(query)
            answers[query] = query_answers

    def _create_train_queries(self):
        queries = dict()
        answers = defaultdict(set)
        query_names_train = ("1p", "1ap", "1dp")
        for name in query_names_train:
            queries[name_query_dict[name]] = set()

        def generate_simple(name, func, triples):
            print(f"train: Creating {name} queries...", end='\t', flush=True)
            start_time = time.time()
            func(triples, queries, answers)
            print(" {:2f} seconds".format(time.time() - start_time))

        generate_simple('1p', self._generate_1p_queries, self.triples['train'])
        generate_simple('1ap', self._generate_1ap_queries, self.triples_attr['train'])
        generate_simple('1dp', self._generate_1dp_queries, self.descriptions['train'])

        if self.complex_train_queries:
            generator = QueryGeneratorGeneric(self.attr_values)

            answers_container = AnswersContainer(self.triples['train'], self.triples_attr['train'])
            base_container = AnswersContainer(list(), list())

            def generate_complex(name, limit):
                print(f"train: Creating {name} queries...", end='\t', flush=True)
                start_time = time.time()
                queries_tmp, answers_tmp, _ = generator.create_queries(limit, name, answers_container, base_container=base_container)
                queries[name_query_dict[name]] = queries_tmp
                answers.update(answers_tmp)
                print(" {:2f} seconds".format(time.time() - start_time))

            attr_values = set()
            for q in queries[name_query_dict['1ap']]:
                for a in answers[q]:
                    attr_values.add(a)
            # limit = min(len(queries[name_query_dict['1ap']]), len(attr_values)//2)
            limit = 100
            for name in ('ai', 'ai-eq', 'ai-lt', 'ai-gt'):
                generate_complex(name, limit)

            limit = len(queries[name_query_dict['1p']])
            for name in ('2p', '3p', '2i', '3i'):
                generate_complex(name, limit)

        if self.print_debug:
            self._print_debuginfo(queries, answers)

        return queries, answers

    def _create_eval_queries(self, type):
        if type == 'valid':
            triples_base = self.triples['train']
            triples_eval = self.triples['valid']
            triples_attr_base = self.triples_attr['train']
            triples_attr_eval = self.triples_attr['valid']
            triples_desc_base = self.descriptions['train']
            triples_desc_eval = self.descriptions['valid']
        elif type == 'test':
            triples_base = self.triples['train'] + self.triples['valid']
            triples_eval = self.triples['test']
            triples_attr_base = self.triples_attr['train'] + self.triples_attr['valid']
            triples_attr_eval = self.triples_attr['test']
            triples_desc_base = self.descriptions['train'] + self.descriptions['valid']
            triples_desc_eval = self.descriptions['test']
        triples_full = triples_base + triples_eval
        triples_attr_full = triples_attr_base + triples_attr_eval
        triples_desc_full = triples_desc_base + triples_desc_eval

        # todo: need to take a deep look at this
        # why dataset_eval is not used? 
        dataset_eval = AnswersContainer(triples_eval, triples_attr_eval, triples_desc_eval if self.desc_jointly else list())
        dataset_base = AnswersContainer(triples_base, triples_attr_base, triples_desc_base if self.desc_jointly else list())
        dataset_full = AnswersContainer(triples_full, triples_attr_full, triples_desc_full if self.desc_jointly else list())
        queries_eval = dict()
        answers_eval_hard = defaultdict(set)
        answers_eval_easy = defaultdict(set)
        answers_eval_full = defaultdict(set)
        complex_query_names = [x for x in name_query_dict.keys() if not x.startswith('1') and x != 'di']
        for name in name_query_dict.keys():
            queries_eval[name_query_dict[name]] = set()

        generator = QueryGeneratorGeneric(self.attr_values)

        # Generate simple queries using the full graph
        def generate_simple(name, func, triples, answers):
            print(f"{type}: Creating {name} queries...", end='\t', flush=True)
            start_time = time.time()
            func(triples, queries_eval, answers)
            print(" {:2f} seconds".format(time.time() - start_time))

        # TODO: 1p and 1ap will be first created 
        generate_simple('1p', self._generate_1p_queries, triples_full, answers_eval_full)
        generate_simple('1ap', self._generate_1ap_queries, triples_attr_full, answers_eval_full)
        
        for name in ('1p', '1ap'):
            for query in list(queries_eval[name_query_dict[name]]):
                answers_easy = generator.get_answers(dataset_base, query)
                answers_hard = answers_eval_full[query].difference(answers_easy)
                if not answers_hard:
                    del answers_eval_full[query]
                    queries_eval[name_query_dict[name]].remove(query)
                else:
                    answers_eval_easy[query] = answers_easy
                    answers_eval_hard[query] = answers_hard

        if self.desc_jointly:
            # Description triples contains ids of words instead of a description embedding
            generate_simple('1dp', self._generate_1dp_queries, triples_desc_full, answers_eval_full)
            for query in list(queries_eval[name_query_dict['1dp']]):
                try:
                    answers_easy = dataset_base.triples_desc[query[0]]['/description']
                except KeyError:
                    answers_easy = {}
                answers_hard = answers_eval_full[query].difference(answers_easy)

                if not answers_hard:
                    del answers_eval_full[query]
                    queries_eval[name_query_dict['1dp']].remove(query)
                else:
                    answers_eval_easy[query] = answers_easy
                    answers_eval_hard[query] = answers_hard
            generate_simple('di', self._generate_di_queries, triples_desc_full, answers_eval_full)
            for query in list(queries_eval[name_query_dict['di']]):
                answers_easy = list()
                for word in query[1][0]:
                    try:
                        answers_easy.append(dataset_base.triples_desc_inv[word]['/description'])
                    except KeyError:
                        continue

                counts = defaultdict(int)
                for sub_answers in answers_easy:
                    for ent in sub_answers:
                        counts[ent] += 1
                answers_easy = {ent for ent, count in counts.items() if count >= len(query[1][0])/2}

                answers_hard = answers_eval_full[query].difference(answers_easy)

                if not answers_hard:
                    del answers_eval_full[query]
                    queries_eval[name_query_dict['di']].remove(query)
                else:
                    answers_eval_easy[query] = answers_easy
                    answers_eval_hard[query] = answers_hard
        else:
            generate_simple('1dp', self._generate_1dp_queries, triples_desc_eval, answers_eval_full)
            generate_simple('di', self._generate_di_queries, triples_desc_full, answers_eval_full)
            for query in queries_eval[name_query_dict['1dp']]:
                # no easy answers possible for description prediction
                answers_eval_hard[query] = answers_eval_full[query]
            for query in queries_eval[name_query_dict['di']]:
                # Only entity description of eval dataset are hard answers
                for answer in answers_eval_full[query]:
                    if (answer, (symbol_placeholder_dict['dp'],)) in queries_eval[name_query_dict['1dp']]:
                        answers_eval_hard[query].add(answer)
                    else:
                        answers_eval_easy[query].add(answer)
        # Generate complex queries using the full graph

        def generate_complex(name, limit):
            print(f"{type}: Creating {name} queries...", end='\t', flush=True)
            start_time = time.time()
            queries, answers, answers_easy = generator.create_queries(limit, name, dataset_full, dataset_base)
            queries_eval[name_query_dict[name]] = queries
            answers_eval_full.update(answers)
            answers_eval_easy.update(answers_easy)
            print(" {:2f} seconds".format(time.time() - start_time))

        # TODO: CODE FOR DEFINE THE LIMIT OF THE COMPLEX QUERIES
        for name in complex_query_names:
            limit = 5000
            # if 'a' in name:
            #     limit = 100
            # if name.startswith('ai-'):
            #     limit = 100
            generate_complex(name, limit)

        print(f"{type}: Computing easy & hard answers...")

        # Compute hard answers for complex queries
        for name in complex_query_names:
            for query in queries_eval[name_query_dict[name]]:
                answers_eval_hard[query] = answers_eval_full[query].difference(answers_eval_easy[query])
                if len(answers_eval_hard[query]) == 0:
                    print(query, answers_eval_easy[query], answers_eval_full[query])
                    exit()

        if self.print_debug:
            self._print_debuginfo(queries_eval, answers_eval_easy, answers_eval_hard)

        return queries_eval, answers_eval_full, answers_eval_easy, answers_eval_hard

    def _create_attr_exists_queries(self):
        # Skip all evaluation triples with a relation id above the given threshold
        # used to skip attr_exists relations during evaluation
        for name in ("train", "valid", "test"):
            triples = list()
            for i in range(len(self.triples[name])-1, -1, -1):
                if self.triples[name][i][1] >= self.attr_exists_threshold:
                    triples.append(self.triples[name][i])
                    del self.triples[name][i]

            queries = dict()
            answers = defaultdict(set)
            queries[name_query_dict['1p']] = set()
            self._generate_1p_queries(triples, queries, answers)
            pickle.dump(queries, open(os.path.join(self.output_path, name + "-attr-exists-queries.pkl"), 'wb'))
            pickle.dump(answers, open(os.path.join(self.output_path, name + "-attr-exists-answers.pkl"), 'wb'))

    def _get_values_per_attribute(self):
        attr_values = defaultdict(list)
        for triple in itertools.chain.from_iterable(self.triples_attr.values()):
            attr_values[triple[1]].append(triple[2])
        return attr_values

    def create_queries(self):
        self.attr_values = self._get_values_per_attribute()
        queries_train, answers_train = self._create_train_queries()
        print("Storing train queries...")
        pickle.dump(queries_train, open(os.path.join(self.output_path, "train-queries.pkl"), 'wb'))
        pickle.dump(answers_train, open(os.path.join(self.output_path, "train-answers.pkl"), 'wb'))

        if self.attr_exists_threshold:
            print(f"Creating attr_exists queries...", end='\t', flush=True)
            start_time = time.time()
            self._create_attr_exists_queries()
            print(" {:2f} seconds".format(time.time() - start_time))

        if self.do_valid:
            queries_val, answers_val_full, answers_val_easy, answers_val_hard = self._create_eval_queries("valid")
        if self.do_test:
            queries_test, answers_test_full, answers_test_easy, answers_test_hard = self._create_eval_queries("test")

        print("Storing eval queries...")
        if self.do_valid:
            pickle.dump(queries_val, open(os.path.join(self.output_path, "valid-queries.pkl"), 'wb'))
            pickle.dump(answers_val_hard, open(os.path.join(self.output_path, "valid-hard-answers.pkl"), 'wb'))
            pickle.dump(answers_val_easy, open(os.path.join(self.output_path, "valid-easy-answers.pkl"), 'wb'))
        if self.do_test:
            pickle.dump(queries_test, open(os.path.join(self.output_path, "test-queries.pkl"), 'wb'))
            pickle.dump(answers_test_hard, open(os.path.join(self.output_path, "test-hard-answers.pkl"), 'wb'))
            pickle.dump(answers_test_easy, open(os.path.join(self.output_path, "test-easy-answers.pkl"), 'wb'))

        return queries_train, answers_train, queries_val, answers_val_hard, answers_val_easy, queries_test, answers_test_hard, answers_test_easy
