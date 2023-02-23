from Tester import Tester
from config import CQDParams, HyperParams, TrainConfig, parse_args
from util_data import *
from util_data_queries import *
from main import set_logger
from util_models import get_model, load_model


def eval_mean_attr_pred(model, train_config: TrainConfig, nentity, nattr):
    tester = Tester(model, None, train_config.cuda)
    average_prediction = tester.test_mean_attr_pred(nentity, nattr, train_config)
    print('Average predicted attribute value for all attributes and entities:')
    print(average_prediction)


def main(args):
    train_config: TrainConfig = args.train_config
    cqd_params: CQDParams = args.cqd_params
    params: HyperParams = args.hyperparams
    set_logger('', None, True, True)
    nentity, nrelation, nattribute = load_stats(train_config.data_path)
    model = get_model(train_config, params, cqd_params, nentity, nrelation, nattribute)
    load_model(model, train_config.checkpoint_path, train_config.cuda)
    eval_mean_attr_pred(model, train_config, nentity, nattribute)


if __name__ == '__main__':
    main(parse_args())
