import math
from config import CQDParams, HyperParams, TrainConfig, parse_args
from util_data import *
from util_data_queries import *
from main import set_logger, test_model
from util_models import get_model, load_model

num_bound_vars = {x: 1 for x in name_query_dict.keys()}
for t in ('3p', '3ap', '2ai', 'aip', 'au'):
    num_bound_vars[t] += 1


def eval(model, train_config, cqd_params, query_type, test_batch_size):
    model.k = cqd_params.cqd_k
    model.t_norm_name = cqd_params.cqd_t_norm.name
    beam_sizes = {
        '2p': 256,
        '3p': 4,
        'ip': 256,
        'pi': 256,
        'up': 256,
        '2ap': 32,
        '3ap': 32,
        'aip': 256,
    }
    if any(dataset in train_config.checkpoint_path.lower() for dataset in ('kblrn', 'literale')):
        beam_sizes['2p'] = 128
    norms = {
        '2u': 'min',
        'up': 'min',
        '2ai': 'min',
        'au': 'min',
    }
    if query_type in beam_sizes:
        model.k = beam_sizes[query_type]
    if query_type in norms:
        model.t_norm_name = norms[query_type]

    train_config.test_batch_size = max(1, int(test_batch_size // (2**(num_bound_vars[t]*(math.log(model.k, 2)-2)))))
    test_model(model, train_config, 'Test', tasks=(query_type,))


def main(args):
    set_logger('', None, True, True)
    train_config: TrainConfig = args.train_config
    params: HyperParams = args.hyperparams
    cqd_params: CQDParams = args.cqd_params
    model = get_model(train_config, params, cqd_params, *load_stats(train_config.data_path))
    load_model(model, train_config.checkpoint_path, train_config.cuda)
    tasks = ('1p',  '2p', '3p', '2i', '3i', 'ip', 'pi', '2u', 'up',)
    if train_config.use_attributes:
        tasks = ('1ap', '2ap', '3ap', 'ai-lt', 'ai-eq', 'ai-gt', '2ai', 'aip', 'pai', 'au',)
    if train_config.use_descriptions:
        tasks = ('1dp', 'di',)
    for query_type in tasks:
        eval(model, train_config, cqd_params, query_type, train_config.test_batch_size)


if __name__ == '__main__':
    main(parse_args())
