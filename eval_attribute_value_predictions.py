import itertools
from Tester import Tester
from config import CQDParams, HyperParams, TrainConfig, parse_args
from util import set_logger
from util_data import *
from util_data_queries import get_dataset_eval, get_eval_dataloader
from util_models import get_model, load_model


def eval_attributes(model, train_config: TrainConfig, hyperparams: HyperParams, mode='Test', skip_threshold=10):
    queries, easy_answers, hard_answers = load_data(train_config.data_path, ('1ap',), mode.lower())
    if not hard_answers:
        # there are no hard answers in training dataset
        hard_answers = easy_answers
    dataset = get_dataset_eval(queries)
    dataloader = get_eval_dataloader(dataset, hyperparams.batch_size, train_config.cpu_num)
    tester = Tester(model, dataloader, train_config.cuda)
    mae_per_attribute, mse_per_attribute = tester.test_attributes(list(queries[name_query_dict['1ap']]), hard_answers, train_config)
    mae_per_attribute = {k: v for k, v in mae_per_attribute.items() if len(v) >= skip_threshold}

    all_values = get_all_attribute_values(train_config.data_path)
    mads = get_mads(all_values)

    print('='*50+' lowest MAE '+50*'=')
    print(f'id\tMAE\t\t\tcount')
    mae_per_attribute_mean = {k: sum(v)/len(v) for k, v in mae_per_attribute.items()}
    mae_per_attribute_mean = dict(sorted(mae_per_attribute_mean.items(), key=lambda x: x[1]))
    for attr, value in mae_per_attribute_mean.items():
        print(f"{attr}\t{value:.18f}\t{len(mae_per_attribute[attr])}")

    print('='*50+' MAE relative to MAD '+50*'=')
    print(f'id\tcount\tMAE/MAD\t\t\tMAE\t\t\tMAD\t\t\tMean')
    relative_error = {attr: mae_per_attribute_mean[attr]/mad for attr, mad in mads.items() if attr in mae_per_attribute}
    relative_error = dict(sorted(relative_error.items(), key=lambda x: x[1], reverse=False))
    for attr, value in relative_error.items():
        print(f"{attr}\t{len(mae_per_attribute[attr])}\t{value}\t{mae_per_attribute_mean[attr]}\t{mads[attr]}\t{statistics.mean(all_values[attr])}")

    all_maes = list(itertools.chain.from_iterable(mae_per_attribute.values()))
    print(f'Average MAE: {sum(all_maes)/len(all_maes)}')
    print(f'Average MAE per attribute: {sum(mae_per_attribute_mean.values())/len(mae_per_attribute_mean)}')
    print(f'Unique attributes in {mode} dataset: {len(mae_per_attribute)}')
    print(f'average MAE/MAD ratio: {sum(relative_error.values())/len(relative_error)}')


def main(args):
    set_logger('', None, True, True)
    train_config: TrainConfig = args.train_config
    cqd_params: CQDParams = args.cqd_params
    params: HyperParams = args.hyperparams
    model = get_model(train_config, params, cqd_params, *load_stats(train_config.data_path))
    load_model(model, train_config.checkpoint_path, train_config.cuda)
    eval_attributes(model, train_config, params)


if __name__ == '__main__':
    main(parse_args())
