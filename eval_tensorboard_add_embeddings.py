import torch
from Tester import Tester
from config import CQDParams, HyperParams, TrainConfig, parse_args
from util_data import *
from util_data_queries import *
from main import set_logger
from util_models import get_model, load_model
from tensorboardX import SummaryWriter


def add_tensorboard_embeddings_simple(model, writer: SummaryWriter, epoch, id2ent, id2rel, id2attr):
    """
    Add the embeddings to Tensorboard.
    """
    writer.add_embedding(model.ent_embeddings.weight, list(id2ent.values()), tag='entities', global_step=epoch)
    writer.add_embedding(model.rel_embeddings.weight, list(id2rel.values()), tag='relations', global_step=epoch)
    writer.add_embedding(model.attr_embeddings.weight, list(id2attr.values()), tag='attributes', global_step=epoch)


def add_tensorboard_embeddings(model, writer: SummaryWriter, epoch, id2ent, id2rel, id2attr):
    """
    Add histograms representing the distribution of predicted values per attribute.
    Add embeddings based on the predicted attribute values, the attribute existence scores, and both.
    """
    attr_pred_and_exists_scores = torch.empty((len(id2attr), len(id2ent)))
    for id, attr in id2attr.items():
        filters = torch.tensor([[1., -4]])
        tmp = model.score_attribute_restriction(filters, torch.tensor([id]))(None).squeeze(0)
        writer.add_histogram(attr+'_pred_and_exists', tmp)
        attr_pred_and_exists_scores[id] = tmp
    attr_pred_and_exists_scores = attr_pred_and_exists_scores.transpose(0, 1)
    writer.add_embedding(attr_pred_and_exists_scores, list(id2ent.values()), tag='attr_pred_and_exists_scores', global_step=epoch)
    del attr_pred_and_exists_scores
    attr_exists_scores = torch.empty((len(id2attr), len(id2ent)))
    for id, attr in id2attr.items():
        tmp = model.score_attribute_exists(torch.tensor([id]))(None)
        writer.add_histogram(attr+'_exists', tmp)
        attr_exists_scores[id] = tmp
    attr_exists_scores = attr_exists_scores.transpose(0, 1)
    writer.add_embedding(attr_exists_scores, list(id2ent.values()), tag='attr_exists_scores', global_step=epoch)
    del attr_exists_scores

    attr_predictions = torch.empty((len(id2attr), len(id2ent)))
    for id, attr in id2attr.items():
        tmp = model.predict_attribute_values(model.ent_embeddings.weight, torch.tensor([id]).repeat(len(id2ent)))
        writer.add_histogram(attr, tmp)
        attr_predictions[id] = tmp
    attr_predictions = attr_predictions.transpose(0, 1)
    writer.add_embedding(attr_predictions, list(id2ent.values()), tag='attr_predictions', global_step=epoch)


def main(args):
    train_config: TrainConfig = args.train_config
    cqd_params: CQDParams = args.cqd_params
    params: HyperParams = args.hyperparams
    set_logger('', None, True, True)
    entity2id = load_mappings_from_file(train_config.data_path, "entity")
    relation2id = load_mappings_from_file(train_config.data_path, "relation")
    attribute2id = load_mappings_from_file(train_config.data_path, "attr")
    model = get_model(train_config, params, cqd_params, len(entity2id), len(relation2id), len(attribute2id))
    load_model(model, train_config.checkpoint_path, train_config.cuda)
    writer = SummaryWriter()
    add_tensorboard_embeddings_simple(model, writer, 100, entity2id.inverse, relation2id.inverse, attribute2id.inverse)
    add_tensorboard_embeddings(model, writer, 100, entity2id.inverse, relation2id.inverse, attribute2id.inverse)
    print('Done')
    print('Run "tensorboard --logdir runs" to start tensorboard')


if __name__ == '__main__':
    main(parse_args())
