from Tester import Tester
from config import CQDParams, HyperParams, TrainConfig, parse_args
from util_data import *
from util_data_queries import *
from main import evaluate, set_logger
from util_models import get_model, load_model


def eval_attr_exists_relations(model, train_config: TrainConfig, mode='Test'):
    queries, easy_answers, hard_answers = load_attr_exists_data_dummy(train_config.data_path, mode.lower())
    queries = flatten_query(queries)
    dataset = get_dataset_eval(queries)
    dataloader = get_eval_dataloader(dataset, train_config.test_batch_size, train_config.cpu_num)
    tester = Tester(model, dataloader, train_config.cuda)
    return evaluate(tester, easy_answers, hard_answers, train_config, query_name_dict, mode, train_config.train_times)


def main(args):
    train_config: TrainConfig = args.train_config
    cqd_params: CQDParams = args.cqd_params
    params: HyperParams = args.hyperparams
    set_logger('', None, True, True)
    model = get_model(train_config, params, cqd_params, *load_stats(train_config.data_path))
    load_model(model, train_config.checkpoint_path, train_config.cuda)
    eval_attr_exists_relations(model, train_config)


if __name__ == '__main__':
    main(parse_args())
