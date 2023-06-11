from config import TrainConfig
from models import CQDComplExDJointly
from models.CQDBaseModel import CQDBaseModel
import torch
import collections
import torch.nn.functional as F

from tqdm import tqdm
from util import flatten, name_query_dict


class Tester(object):
    def __init__(self, model, data_loader, use_gpu):
        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu

    def test_mean_attr_pred(self, nentity, nattr, train_config: TrainConfig):
        self.model.eval()
        query_structure = name_query_dict['1ap']

        # construct queries for each entity for each attribute
        queries_unflatten = list()
        for ent in range(nentity):
            for attr in range(nattr):
                queries_unflatten.append((ent, (-3, attr)))

        queries = [flatten(q) for q in queries_unflatten]

        with torch.no_grad():
            queries_tensor = torch.FloatTensor(queries)
            if train_config.cuda:
                queries_tensor = queries_tensor.cuda()

            predictions = list()
            i = 0
            while i < queries_tensor.shape[0]:
                tmp = min(i+train_config.test_batch_size, queries_tensor.shape[0])
                batch_queries_dict = {query_structure: queries_tensor[i:tmp]}

                negative_logit = self.model(batch_queries_dict)

                for idx in range(batch_queries_dict[query_structure].shape[0]):
                    predictions.append(abs(negative_logit[idx]).item())
                i = tmp

            return sum(predictions)/len(predictions)

    def test_attributes(self, queries_unflatten, hard_answers, train_config: TrainConfig):
        self.model.eval()
        query_structure = name_query_dict['1ap']
        queries = [flatten(q) for q in queries_unflatten]

        with torch.no_grad():
            queries_tensor = torch.FloatTensor(queries)
            if train_config.cuda:
                queries_tensor = queries_tensor.cuda()

            mae_per_attribute = collections.defaultdict(list)
            mse_per_attribute = collections.defaultdict(list)
            i = 0
            while i < queries_tensor.shape[0]:
                tmp = min(i+train_config.test_batch_size, queries_tensor.shape[0])
                batch_queries_dict = {query_structure: queries_tensor[i:tmp]}

                negative_logit = self.model(batch_queries_dict)

                for idx in range(batch_queries_dict[query_structure].shape[0]):
                    query = queries_unflatten[idx+i]
                    answer_mean = sum(hard_answers[query])/len(hard_answers[query])
                    prediction_mean = torch.mean(negative_logit[idx]).item()
                    mae = abs(answer_mean - prediction_mean)
                    mse = (answer_mean - prediction_mean)**2
                    mae_per_attribute[query[1][1]].append(mae)
                    mse_per_attribute[query[1][1]].append(mse)

                i = tmp

            return mae_per_attribute, mse_per_attribute

    def test_relations(self, queries_unflatten, hard_answers, easy_answers, train_config: TrainConfig):
        self.model.eval()
        query_structure = name_query_dict['1p']
        queries = [flatten(q) for q in queries_unflatten]

        with torch.no_grad():
            queries_tensor = torch.FloatTensor(queries)
            if train_config.cuda:
                queries_tensor = queries_tensor.cuda()

            mrr_per_relation = collections.defaultdict(list)
            mr_per_relation = collections.defaultdict(list)
            hits10_per_relation = collections.defaultdict(list)
            batch = 0
            while batch < queries_tensor.shape[0]:
                tmp = min(batch+train_config.test_batch_size, queries_tensor.shape[0])
                batch_queries_dict = {query_structure: queries_tensor[batch:tmp]}

                negative_logit = self.model(batch_queries_dict)

                negative_logit = torch.stack(negative_logit, dim=0)
                argsort = torch.argsort(negative_logit, dim=-1, descending=True)
                ranking = argsort.clone().to(torch.float)
                scatter_src = torch.arange(self.model.nentity).to(torch.float).repeat(argsort.shape[0], 1)
                if train_config.cuda:
                    scatter_src = scatter_src.cuda()
                # achieve the ranking (positions) of all entities for all queries in the batch
                # [B, N]
                ranking = ranking.scatter_(1, argsort, scatter_src)

                for idx in range(batch_queries_dict[query_structure].shape[0]):
                    query = queries_unflatten[idx+batch]

                    hard_answer = hard_answers[query]
                    easy_answer = easy_answers[query]
                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)
                    assert len(hard_answer.intersection(easy_answer)) == 0
                    # positions in the ranking (of all entities) for easy and hard answers
                    cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                    # sort by position in the ranking; indices for (easy + hard) answers
                    cur_ranking, indices = torch.sort(cur_ranking)
                    # indices with hard answers only
                    masks = indices >= num_easy
                    if train_config.cuda:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float)
                    # Reduce ranking for each answer entity by the amount of (easy+hard) answers appearing before it
                    # cur_ranking now ignores other correct answers
                    cur_ranking = cur_ranking - answer_list + 1
                    # only take indices that belong to the hard answers
                    cur_ranking = cur_ranking[masks]

                    mr = torch.mean(cur_ranking).item()
                    mrr = torch.mean(1./cur_ranking).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                    mr_per_relation[query[1][0]].append(mr)
                    mrr_per_relation[query[1][0]].append(mrr)
                    hits10_per_relation[query[1][0]].append(h10)

                batch = tmp
            return mrr_per_relation, mr_per_relation, hits10_per_relation

    def run_link_prediction(self, easy_answers, hard_answers, train_config: TrainConfig, query_name_dict):
        self.model.eval()
        train_config.print_on_screen = True

        step = 0
        logs = collections.defaultdict(list)
        device = torch.device("cuda:0" if self.use_gpu else "cpu")

        with torch.no_grad():
            progress_bar = tqdm(self.data_loader, disable=not train_config.print_on_screen)
            for head, rel, tail in progress_bar:

                head = head.to(device=device)
                rel = rel.to(device=device)
                tail = tail.to(device=device)

                triples = torch.cat((head.unsqueeze(1), rel.unsqueeze(1), tail.unsqueeze(1)), -1)
                (scores_o, scores_s), _ = self.model.score_candidates(triples)

                # use queries for tail prediction
                #queries_tail = torch.as_tensor([[h, r] for h, r in zip(head, rel)], device=device)
                #score = self.model({query_structure: queries_tail})
                #scores_o = scores_s = torch.stack(score, dim=0)

                for scores, name in zip((scores_o, scores_s), ('right', 'left')):
                    ### Hits@i EVALUATION ###
                    # Evaluate remaining queries with H@i metrics
                    argsort = torch.argsort(scores, dim=-1, descending=True)
                    ranking = argsort.clone().to(torch.float)
                    scatter_src = torch.arange(self.model.nentity).to(torch.float).repeat(argsort.shape[0], 1)
                    if train_config.cuda:
                        scatter_src = scatter_src.cuda()
                    # achieve the ranking (positions) of all entities for all queries in the batch
                    # [B, N]
                    ranking = ranking.scatter_(1, argsort, scatter_src)

                    for i in range(head.shape[0]):
                        query = (head[i].item(), (rel[i].item(),))
                        if name == 'left':
                            # Use inverse relation to get easy/hard answers
                            if rel[i].item() % 2 == 0:
                                query = (tail[i].item(), (rel[i].item()+1,))
                            else:
                                query = (tail[i].item(), (rel[i].item()-1,))

                        # ignoring inverse relations for tail prediction:
                        # else:
                        #    if rel[i].item() % 2 != 0:
                        #        continue

                        progress_bar.set_description(f"Evaluating link prediction")
                        hard_answer = hard_answers[query]
                        easy_answer = easy_answers[query]
                        num_hard = len(hard_answer)
                        num_easy = len(easy_answer)
                        assert len(hard_answer.intersection(easy_answer)) == 0
                        # positions in the ranking (of all entities) for easy and hard answers
                        cur_ranking = ranking[i, list(easy_answer) + list(hard_answer)]
                        # sort by position in the ranking; indices for (easy + hard) answers
                        cur_ranking, indices = torch.sort(cur_ranking)
                        # indices with hard answers only
                        masks = indices >= num_easy
                        if self.use_gpu:
                            answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                        else:
                            answer_list = torch.arange(num_hard + num_easy).to(torch.float)

                        # Reduce ranking for each answer entity by the amount of (easy+hard) answers appearing before it
                        # cur_ranking now ignores other correct answers
                        cur_ranking = cur_ranking - answer_list + 1
                        # only take indices that belong to the hard answers
                        cur_ranking = cur_ranking[masks]

                        mrr = torch.mean(1./cur_ranking).item()
                        h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                        h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                        h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                        logs[name].append({
                            f'MRR_{name}': mrr,
                            f'HITS1_{name}': h1,
                            f'HITS3_{name}': h3,
                            f'HITS10_{name}': h10,
                            f'num_hard_answer_{name}': num_hard,
                        })

                    step += 1

        metrics = collections.defaultdict(int)
        for name in logs.keys():
            for metric in logs[name][0].keys():
                if 'num_hard_answer' in metric:
                    continue
                metrics[metric] = sum([log[metric] for log in logs[name]])/len(logs[name])
            metrics['num_triples_'+name] = len(logs[name])

        return metrics

    def test_step(self, easy_answers, hard_answers, args: TrainConfig, query_name_dict):
        self.model.eval()
        args.print_on_screen = False

        step = 0
        logs = collections.defaultdict(list)

        requires_grad = isinstance(self.model, CQDBaseModel) and self.model.method == 'continuous'

        # with torch.no_grad():
        with torch.set_grad_enabled(requires_grad):
            progress_bar = tqdm(self.data_loader, disable=not args.print_on_screen)
            for queries, queries_unflatten, query_structures in progress_bar:
                batch_queries_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                for query_structure in batch_queries_dict:
                    if hasattr(self.model, 'desc_jointly') and self.model.desc_jointly and query_structure == name_query_dict['di']:
                        for i in range(len(batch_queries_dict[query_structure])):
                            # always return 20 keywords
                            keywords = batch_queries_dict[query_structure][i][1:-1]
                            q, r = divmod(20, len(keywords))
                            batch_queries_dict[query_structure][i][1:-1] = q * keywords + keywords[:r]
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.FloatTensor(batch_queries_dict[query_structure]).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.FloatTensor(batch_queries_dict[query_structure])
                # TODO:take a look of return value whether it is a'
                negative_logit = self.model(batch_queries_dict) # forward function of the model

                ### MEAN ABSOLUTE ERROR EVALUATION ###
                attr_query_structures = {i: x for i, x in enumerate(query_structures) if x[-1][0] == 'ap'}

                error_per_attribute = collections.defaultdict(list)
                for idx, query_structure in attr_query_structures.items():
                    progress_bar.set_description(f"Evaluating {query_name_dict[query_structure]} queries")
                    query = queries_unflatten[idx]

                    answer_mean = sum(hard_answers[query])/len(hard_answers[query])
                    prediction_mean = torch.mean(negative_logit[idx]).item()

                    mae = abs(answer_mean - prediction_mean)
                    mse = (answer_mean - prediction_mean)**2
                    error_per_attribute[query[1][1]].append(mae)
                    logs[query_structure].append({
                        'MAE': mae,
                        'MSE': mse,
                        'RMSE': mse,
                    })

                for idx in sorted(attr_query_structures.keys(), reverse=True):
                    del negative_logit[idx]
                    del query_structures[idx]
                    del queries_unflatten[idx]

                ### Cosine Similarity Evaluation ###
                cosine_sim_qs = {i: x for i, x in enumerate(query_structures) if x == name_query_dict['1dp']}
                for idx, qs in cosine_sim_qs.items():
                    progress_bar.set_description(f"Evaluating {query_name_dict[qs]} queries")
                    query = queries_unflatten[idx]

                    predicted_vector = negative_logit[idx]
                    try:
                        expected_vector = torch.as_tensor(next(iter(hard_answers[query])), device=predicted_vector.device)
                    except StopIteration:
                        # Only used to evaluate on training dataset (debugging)
                        expected_vector = torch.as_tensor(next(iter(easy_answers[query])), device=predicted_vector.device)

                    if type(self.model) == CQDComplExDJointly:
                        if hard_answers[query]:
                            expected_vector = torch.mean(self.model.word_embeddings(torch.as_tensor(list(hard_answers[query]), device=predicted_vector.device)), dim=0)
                        else:
                            expected_vector = torch.mean(self.model.word_embeddings(torch.as_tensor(list(easy_answers[query]), device=predicted_vector.device)), dim=0)

                    sim = F.cosine_similarity(expected_vector, predicted_vector, dim=0)
                    logs[qs].append({
                        'cos_sim': sim,
                    })

                for idx in sorted(cosine_sim_qs.keys(), reverse=True):
                    del negative_logit[idx]
                    del query_structures[idx]
                    del queries_unflatten[idx]

                ### ACCURACY EVALUATION ###
                if False:
                    attr_restriction_query_structures = {i: x for i, x in enumerate(query_structures) if x == (
                        ('ap', 'a'), ('v', 'f')) or x[0] == (('ap', 'a'), ('v', 'f')) and x[1] == (('ap', 'a'), ('v', 'f')) or x == ('a',)}

                    for idx, query_structure in attr_restriction_query_structures.items():
                        progress_bar.set_description(f"Evaluating {query_name_dict[query_structure]} queries")
                        query = queries_unflatten[idx]
                        answers = hard_answers[query]
                        correct = 0
                        for answer in answers:
                            if negative_logit[idx][answer] > .5:
                                correct += 1

                        logs[query_structure].append({
                            'accuracy': correct / len(answers),
                            'num_answers': negative_logit[idx].count_nonzero(),
                        })

                    for idx in sorted(attr_restriction_query_structures.keys(), reverse=True):
                        del negative_logit[idx]
                        del query_structures[idx]
                        del queries_unflatten[idx]

                ### Hits@i EVALUATION ###
                # Evaluate remaining queries with H@i metrics
                if negative_logit:
                    negative_logit = torch.stack(negative_logit, dim=0)
                    argsort = torch.argsort(negative_logit, dim=-1, descending=True)
                    ranking = argsort.clone().to(torch.float)
                    scatter_src = torch.arange(self.model.nentity).to(torch.float).repeat(argsort.shape[0], 1)
                    if args.cuda:
                        scatter_src = scatter_src.cuda()
                    # achieve the ranking (positions) of all entities for all queries in the batch
                    # [B, N]
                    ranking = ranking.scatter_(1, argsort, scatter_src)

                    for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
                        progress_bar.set_description(f"Evaluating {query_name_dict[query_structure]} queries")
                        hard_answer = hard_answers[query]
                        easy_answer = easy_answers[query]
                        num_hard = len(hard_answer)
                        num_easy = len(easy_answer)
                        assert len(hard_answer.intersection(easy_answer)) == 0
                        # positions in the ranking (of all entities) for easy and hard answers
                        cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                        # sort by position in the ranking; indices for (easy + hard) answers
                        cur_ranking, indices = torch.sort(cur_ranking)
                        # indices with hard answers only
                        masks = indices >= num_easy
                        if args.cuda:
                            answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                        else:
                            answer_list = torch.arange(num_hard + num_easy).to(torch.float)
                        # Reduce ranking for each answer entity by the amount of (easy+hard) answers appearing before it
                        # cur_ranking now ignores other correct answers
                        cur_ranking = cur_ranking - answer_list + 1
                        # only take indices that belong to the hard answers
                        cur_ranking = cur_ranking[masks]

                        mrr = torch.mean(1./cur_ranking).item()
                        h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                        h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                        h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                        logs[query_structure].append({
                            'MRR': mrr,
                            'HITS1': h1,
                            'HITS3': h3,
                            'HITS10': h10,
                            'num_hard_answer': num_hard,
                        })

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue

                if metric == 'RMSE':
                    metrics[query_structure][metric] = (sum([log[metric] for log in logs[query_structure]])/len(logs[query_structure]))**0.5
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]])/len(logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics
