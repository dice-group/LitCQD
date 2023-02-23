import torch
from Tester import Tester
from config import CQDParams, HyperParams, TrainConfig, parse_args
from dataloader import EvalTripleDataset
from util_data import *
from util_data_queries import *
from main import set_logger
from util_models import get_model, load_model
from util import log_metrics


def eval_link_prediction(model, train_config: TrainConfig, params: HyperParams):
    queries, easy_answers, hard_answers = load_data(train_config.data_path, ('1p',), 'test')

    dataset = EvalTripleDataset(queries[name_query_dict['1p']], hard_answers)
    dataloader, _, _ = get_train_dataloader(dataset, None, None, params.batch_size, False, False, seed=train_config.seed)

    tester = Tester(model, dataloader, train_config.cuda)
    metrics = tester.run_link_prediction(easy_answers, hard_answers, train_config, query_name_dict)

    log_metrics("Link Prediction", 0, metrics)
    average_metrics = defaultdict(float)

    for metric in metrics:
        average_metrics[metric.split('_')[0]] += metrics[metric]
    for metric in average_metrics:
        average_metrics[metric] /= 2
    log_metrics('Link Prediction average', 0, average_metrics)


def main(args):
    train_config: TrainConfig = args.train_config
    cqd_params: CQDParams = args.cqd_params
    params: HyperParams = args.hyperparams
    set_logger('', None, True, True)
    model = get_model(train_config, params, cqd_params, *load_stats(train_config.data_path))
    load_model(model, train_config.checkpoint_path, train_config.cuda)
    eval_link_prediction(model, train_config, params)


if __name__ == '__main__':
    main(parse_args())
