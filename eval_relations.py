from Tester import Tester
from config import CQDParams, HyperParams, TrainConfig, parse_args
from util_data import *
from util_data_queries import *
from main import set_logger
from util_models import get_model, load_model


def eval_relations(model, train_config: TrainConfig, hyperparams: HyperParams, mode='Test'):
    queries, easy_answers, hard_answers = load_data(train_config.data_path, ('1p',), mode.lower())
    dataset = get_dataset_eval(queries)
    dataloader = get_eval_dataloader(dataset, hyperparams.batch_size, train_config.cpu_num)
    tester = Tester(model, dataloader, train_config.cuda)
    mrr_per_relation, mr_per_relation, hits10_per_relation = tester.test_relations(list(queries[name_query_dict['1p']]), hard_answers, easy_answers, train_config)

    mrr_per_relation_mean = {k: sum(v)/len(v) for k, v in mrr_per_relation.items()}
    mr_per_relation_mean = {k: sum(v)/len(v) for k, v in mr_per_relation.items()}
    hits10_per_relation_mean = {k: sum(v)/len(v) for k, v in hits10_per_relation.items()}

    mrr_per_relation_mean = dict(sorted(mrr_per_relation_mean.items(), key=lambda x: x[1], reverse=True))

    print('Id\tMRR\t\t\tMR\t\t\tH@10\t\t\tcount')
    for rel, mrr in mrr_per_relation_mean.items():
        print(f"{rel}\t{mrr:.18f}\t{mr_per_relation_mean[rel]:.18f}\t{hits10_per_relation_mean[rel]:.18f}\t{len(mrr_per_relation[rel])}")

    print(f'Unique relations in {mode} dataset: {len(mrr_per_relation)}')


def main(args):
    train_config: TrainConfig = args.train_config
    cqd_params: CQDParams = args.cqd_params
    params: HyperParams = args.hyperparams
    set_logger('', None, True, True)
    model = get_model(train_config, params, cqd_params, *load_stats(train_config.data_path))
    load_model(model, train_config.checkpoint_path, train_config.cuda)
    eval_relations(model, train_config, params)


if __name__ == '__main__':
    main(parse_args())
