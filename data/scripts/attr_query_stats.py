import pickle
import statistics

path = './generated/FB15K-237_dummy_kblrn/'

name = "test"

test_queries = pickle.load(open(f"{path}{name}-queries.pkl", 'rb'))
hard_answers = pickle.load(open(f"{path}{name}-hard-answers.pkl", 'rb'))

for query_type, queries in test_queries.items():
    if 'ap' in query_type[-1]:
        attr_values = list()
        for query in queries:
            attr_values.append(sum(hard_answers[query])/len(hard_answers[query]))
        print(f'{query_type}: Mean: {sum(attr_values)/len(attr_values)}')
        print(f'{query_type}: SD: {statistics.stdev(attr_values)}')
        print(f'{query_type}: Random Guesser MAE: {sum([abs(0.5-v) for v in attr_values])/len(attr_values)}')
        print(f'{query_type}: Random Guesser MSE: {sum([(0.5-v)**2 for v in attr_values])/len(attr_values)}')

    if query_type == (('ap', 'a'), ('v', '<')):
        attr_values = list()
        for query in queries:
            attr_values.append(query[-1][0])
        print(f'{query_type}: Mean: {sum(attr_values)/len(attr_values)}')

    if query_type == (('ap', 'a'), ('v', '>')):
        attr_values = list()
        for query in queries:
            attr_values.append(query[-1][0])
        print(f'{query_type}: Mean: {sum(attr_values)/len(attr_values)}')
