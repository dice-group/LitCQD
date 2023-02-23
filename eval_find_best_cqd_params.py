import copy
from enum import Enum
import itertools
import math
import torch
from Tester import Tester
from config import CQDParams, HyperParams, TrainConfig, parse_args
from models import CQDBaseModel
from util_data import *
from util_data_queries import get_dataset_eval, get_eval_dataloader, load_queries_eval
from main import evaluate, test_model
from util_models import get_model, load_model
from util import set_logger


def find_best_cqd_params(train_config: TrainConfig, hyperparams: HyperParams):
    cqd_params = CQDParams()
    norms = Enum('t-norm', list(CQDBaseModel.NORMS))
    types = Enum('type', ['continuous', 'discrete'])

    nentity, nrelation, nattribute = load_stats(train_config.data_path)

    num_bound_vars = {x: 1 for x in name_query_dict.keys()}
    for t in ('3p', '3ap', '2ai', 'aip', 'au'):
        num_bound_vars[t] += 1
    attr_pred_tasks = ('1ap', '2ap', '3ap',)

    evaluated_params = defaultdict(list)
    metrics = defaultdict(list)

    # Add random attribute embeddings to evalute model trained without attribute data on attribute queries
    add_attributes = False
    if add_attributes:
        nentity -= 1
        nrelation = 474

    types = Enum('type', ['discrete'])
    tasks = ('1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2u', 'up',)
    if train_config.use_attributes:
        tasks = ('1ap', '2ap', '3ap', 'ai-lt', 'ai-eq', 'ai-gt', '2ai', 'aip', 'pai', 'au',)

    for task in tasks:
        queries, easy_answers, hard_answers = load_queries_eval(train_config.data_path, (task,), 'valid')
        dataset = get_dataset_eval(queries)
        dataloader = get_eval_dataloader(dataset, train_config.test_batch_size, train_config.cpu_num)

        for norm, cqd_type in itertools.product(norms, types):
            cqd_params.cqd_t_norm = norm
            cqd_params.cqd_type = cqd_type
            base_options = (2,)
            if cqd_type.name == 'discrete':
                base_options = range(2, 9)
            for base in base_options:
                cqd_params.cqd_k = 2 ** base
                # num_bound^k retained variables max -> ram usage increased by 2^(num_bound*log_2(k)-2)
                dataloader = get_eval_dataloader(dataset, max(1, train_config.test_batch_size // 2**(num_bound_vars[task]*(base-2))), train_config.cpu_num)
                print(cqd_params)
                model = get_model(train_config, hyperparams, cqd_params, nentity, nrelation, nattribute)
                checkpoint = os.path.join(train_config.checkpoint_path, 'checkpoint')
                model_state, _ = torch.load(checkpoint, map_location=torch.device('cuda:0') if train_config.cuda else torch.device('cpu'))
                remove_attribute_exists = False
                if model_state['ent_embeddings.weight'].shape[0] == model.ent_embeddings.weight.shape[0] + 1:
                    # Remove dummy entity and relations from checkpoint
                    remove_attribute_exists = True
                if remove_attribute_exists:
                    model_state['ent_embeddings.weight'] = model_state['ent_embeddings.weight'][:-1]
                    model_state['rel_embeddings.weight'] = model_state['rel_embeddings.weight'][:474]
                    if 'description_embeddings.weight' in model_state:
                        model_state['description_embeddings.weight'] = model_state['description_embeddings.weight'][:-1]
                    if 'attr_embeddings.weight' in model_state:
                        del model_state['attr_embeddings.weight']
                        del model_state['b.weight']
                if add_attributes:
                    model_state['attr_embeddings.weight'] = model_state['rel_embeddings.weight'][:nattribute]
                    model_state['b.weight'] = model_state['rel_embeddings.weight'][:nattribute, :2]

                model.load_state_dict(model_state)
                tester = Tester(model, dataloader, train_config.cuda)
                try:
                    res = evaluate(tester, easy_answers, hard_answers, train_config, query_name_dict, 'Valid', train_config.train_times)
                except:
                    res = {task+'_MAE': 1.0, task+'_MRR': 0.0}  # co-op unable to find solution
                if task in attr_pred_tasks:
                    metrics[task].append(res[task+'_MAE'])
                else:
                    metrics[task].append(res[task+'_MRR'])
                evaluated_params[task].append(copy.deepcopy(cqd_params))
                print('-'*150)

                # skip higher beam sizes k (bases) if higher values do not result in better metrics
                #skip_next = False
                skip_next = True
                if len(metrics[task]) < 2:
                    skip_next = False
                elif task not in attr_pred_tasks and metrics[task][-1] > metrics[task][-2]:
                    skip_next = False
                elif task in attr_pred_tasks and metrics[task][-1] < metrics[task][-2]:
                    skip_next = False
                if skip_next:
                    break

    best_configs = dict()
    for task in tasks:
        if task in attr_pred_tasks:
            best_config = evaluated_params[task][min(enumerate(metrics[task]), key=lambda x:x[1])[0]]
        else:
            best_config = evaluated_params[task][max(enumerate(metrics[task]), key=lambda x:x[1])[0]]
        best_configs[task] = best_config
        print(f'Best config for task {task}:')
        print(best_config)
        print()

    print('Evaluating on test dataset...')
    batch_size_tmp = train_config.test_batch_size
    for task in tasks:
        model = get_model(train_config, hyperparams, best_configs[task], nentity, nrelation, nattribute)
        checkpoint = os.path.join(train_config.checkpoint_path, 'checkpoint')
        model_state, _ = torch.load(checkpoint, map_location=torch.device('cuda:0') if train_config.cuda else torch.device('cpu'))
        remove_attribute_exists = False
        if model_state['ent_embeddings.weight'].shape[0] == model.ent_embeddings.weight.shape[0] + 1:
            # Remove dummy entity and relations from checkpoint
            remove_attribute_exists = True
        if remove_attribute_exists:
            model_state['ent_embeddings.weight'] = model_state['ent_embeddings.weight'][:-1]
            model_state['rel_embeddings.weight'] = model_state['rel_embeddings.weight'][:474]
            if 'description_embeddings.weight' in model_state:
                model_state['description_embeddings.weight'] = model_state['description_embeddings.weight'][:-1]
            if 'attr_embeddings.weight' in model_state:
                del model_state['attr_embeddings.weight']
                del model_state['b.weight']
        if add_attributes:
            model_state['attr_embeddings.weight'] = model_state['rel_embeddings.weight'][:nattribute]
            model_state['b.weight'] = model_state['rel_embeddings.weight'][:nattribute, :2]
        model.load_state_dict(model_state)
        train_config.test_batch_size = max(1, int(batch_size_tmp // (2**(num_bound_vars[t]*(math.log(best_configs[task].cqd_k, 2)-2)))))
        test_model(model, train_config, 'Test', (task,))


def main(args):
    train_config: TrainConfig = args.train_config
    params: HyperParams = args.hyperparams
    set_logger('', None, True, True)
    find_best_cqd_params(train_config, params)


if __name__ == '__main__':
    main(parse_args())
