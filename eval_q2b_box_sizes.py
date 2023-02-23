import torch
from Tester import Tester
from config import CQDParams, HyperParams, TrainConfig, parse_args
from util_data import *
from util_data_queries import get_dataset_eval, get_eval_dataloader
from util_models import get_model, load_model


def eval_q2b_box_sizes(model, train_config: TrainConfig, hyperparams: HyperParams, skip_threshold=10):
    if hasattr(model, 'offset_attr_embeddings'):
        all_values = get_all_attribute_values(train_config.data_path)
        mads = get_mads(all_values)

        queries, easy_answers, hard_answers = load_data(train_config.data_path, ('1ap',), 'test')
        dataset = get_dataset_eval(queries)
        dataloader = get_eval_dataloader(dataset, hyperparams.batch_size, train_config.cpu_num)
        tester = Tester(model, dataloader, train_config.cuda)
        mae_per_attribute, mse_per_attribute = tester.test_attributes(list(queries[name_query_dict['1ap']]), hard_answers, train_config)
        mae_per_attribute_mean = {k: sum(v)/len(v) for k, v in mae_per_attribute.items()}

        sizes_attr = dict()
        for i, attr in enumerate(model.offset_attr_embeddings.weight):
            size = torch.norm(attr, p=1, dim=-1).item()
            sizes_attr[i] = size

        print(f'id\tBox size\tMean\t\t\tMAD\t\t\t\tMAE\t\t\t\tcount')
        # Sort by MAE/MAD ratio
        for i, size in dict(sorted(sizes_attr.items(), key=lambda x: mads[x[0]]/(mae_per_attribute_mean[x[0]] if x[0] in mae_per_attribute_mean else 1.0), reverse=True)).items():
            if len(mae_per_attribute[i]) < skip_threshold:
                continue
            print(
                f'{i}\t{size:.5f}\t{statistics.mean(all_values[i]):.18f}\t{mads[i]:.18f}\t\t{(mae_per_attribute_mean[i]if i in mae_per_attribute_mean else 0.0 ):.18f}\t\t{len(mae_per_attribute[i])}')
        print(f'Average attribute box size: {(sum(sizes_attr.values())/len(sizes_attr))}')

    # Relational box sizes:
    sizes = list()
    for rel in model.offset_embedding.weight:
        sizes.append(torch.norm(rel, p=1, dim=-1).item())
    print(f'Average relation box size: {(sum(sizes)/len(sizes))}')
    print(f'Max relation box size: {max(sizes)}')
    print(f'Min relation box size: {min(sizes)}')


def main(args):
    train_config: TrainConfig = args.train_config
    cqd_params: CQDParams = args.cqd_params
    params: HyperParams = args.hyperparams
    model = get_model(train_config, params, cqd_params, *load_stats(train_config.data_path))
    load_model(model, train_config.checkpoint_path, train_config.cuda)
    eval_q2b_box_sizes(model, train_config, params)


if __name__ == '__main__':
    main(parse_args())
