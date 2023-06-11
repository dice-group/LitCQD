from dataclasses import dataclass
from typing import Optional
from enum import Enum
from enum import Enum
from typing import Optional

from simple_parsing.helpers.fields import field
from models import CQDBaseModel

from simple_parsing import ArgumentParser


# @dataclass
# class TrainConfig:
#     """General configurations for training"""
#     # Path to queries
#     data_path: str ='data/FB15k-237-q2b'
#     # data_path: str ='data/scripts/generated/LitWD1K'
#     # Output path for checkpoint and logs
#     save_path: Optional[str] = None
#     # path for loading checkpoints
#     checkpoint_path: Optional[str] = None
#     # checkpoint_path: Optional[str] = 'checkpoints_FB15K-237/checkpoint_orig_no_attr'
#     # the model to be trained
#     # geo: Enum('geo', ['cqd-transea', 'cqd-transeadistmult', 'cqd-transeacomplex', 'cqd-transra', 'cqd-mtkgnn', 'cqd-distmulta', 'cqd-complex', 'cqd-complexa', 'cqd-complexa-weighted', 'cqd-complexad', 'cqd-complexd', 'cqd-complexd-jointly', 'cqd-complex-simple',
#     #                   'cqd-transcomplexa', 'cqd-transcomplexdice', 'q2b', 'gqe', 'random_guesser']) ='cqd-complexa'
#     geo: Enum('geo', ['cqd-transea', 'cqd-transeadistmult', 'cqd-transeacomplex', 'cqd-transra', 'cqd-mtkgnn', 'cqd-distmulta', 'cqd-complex', 'cqd-complexa', 'cqd-complexa-weighted', 'cqd-complexad', 'cqd-complexd', 'cqd-complexd-jointly', 'cqd-complex-simple',
#                       'cqd-transcomplexa', 'cqd-transcomplexdice', 'q2b', 'gqe', 'random_guesser']) ='cqd-complexa'
#     # loss function of the relational part
#     loss: Enum('loss', ["margin", "ce", "q2b"]) = 'ce'
#     # How many epochs the model is trained for
#     train_times: int = 3
#     # Evaluate validation queries every xx epochs
#     valid_epochs: int = 3
#     # How many workers pytorch uses to load data
#     cpu_num: int = 1
#     # random seed applied globally
#     seed: int = 0
#     # use GPU
#     cuda: bool = False
#     # use attribute data
#     use_attributes: bool = False
#     use_descriptions: bool = False
#     # train using triples and the cqd dataloader or use queries with a subsampling weight
#     train_data_type: Enum('train_data_type', ['queries', 'triples']) = 'triples'
#     # valid/test batch size
#     test_batch_size: int = 100
#     # tune hyperparameters using ray tune
#     do_tune: bool = False
#     # do_tune: bool = True
#     do_train: bool = False
#     do_test: bool = False
#     # evaluate on train queries aswell
#     eval_on_train: bool = False
#     # evaluate on simple (1-hop) queries only
#     simple_eval: bool = False
#     # embedding dimension of the word embeddings
#     word_emb_dim: int = 3    #00
#     use_modulus: bool = False

# @dataclass
# class CQDParams:
#     """Params to configure the CQD framework"""
#     # Optimization algorithm used to answer complex queries
#     cqd_type: Enum('type', ['continuous', 'discrete']) = 'discrete'
#     # t-norm used to compute conjunctions and disjunctions (t-co-norm)
#     cqd_t_norm: Enum('t-norm', list(CQDBaseModel.NORMS)) = 'prod'
#     # How many samples are retained for each step in the discrete optimization algorithm
#     cqd_k: int = 4


# @dataclass
# class HyperParams:
#     """Hyperparameter"""
#     # hidden dim; embedding dimension
#     rank: int = 1
#     # batch size during training
#     batch_size: int = 16 #1024
#     # loss function of the attribute part
#     attr_loss: Enum('attr_loss', ['mae', 'mse']) = 'mae'
#     # learning rate
#     learning_rate: float = field(0.1, alias='-lr')
#     # learning rate for attribute embeddings only
#     learning_rate_attr: Optional[float] = field(0.1, alias='-lr_attr')
#     # negative entities samples per query
#     negative_sample_size: int = field(1, alias='-n')
#     # negative attribute values sampled per query
#     negative_attr_sample_size: int = field(0, alias='-na')
#     # regularization weight for N3 regularization
#     reg_weight: float = 0
#     # L2 regularization weight for entity embeddings
#     reg_weight_ent: float = 0
#     # L2 regularization weight for relation embeddings
#     reg_weight_rel: float = 0
#     # L2 regularization weight for attribute embeddings
#     reg_weight_attr: float = 0
#     # Determines which fraction of the loss makes up the attribute loss
#     alpha: float = 0.5
#     # optimizer
#     optimizer: Enum('optimizer', ['adam', 'adagrad', 'sgd']) = 'adagrad'
#     # Number of epochs with no improvement after which learning rate will be reduced
#     scheduler_patience: float = 5
#     # Factor by which the learning rate will be reduced
#     scheduler_factor: float = 0.95
#     # Threshold for measuring the new optimum, to only focus on significant changes
#     scheduler_threshold: float = 0.01
#     # Required for a margin-based loss function
#     margin: float = field(2.0, alias='-g')
#     # p_norm for TransE
#     p_norm: int = 2
#     # apply sigmoid on the attribute value predictions
#     do_sigmoid: bool = False
#     # rank for transr
#     rank_attr: int = 50
#     # how to represent description embeddings
#     desc_emb: Enum('desc_emb', ['1-layer', '2-layer', 'gate']) = '1-layer'
#     # Use modules for attribute value prediction instead of mean
#     use_modulus: bool = False


# set all the value of bool to false at default
@dataclass
class TrainConfig:
    """General configurations for training"""

    # Path to queries
    data_path: str = "data/scripts/generated/FB15K-237_dummy_kblrn"
    # data_path: str = "data/FB15k-237-q2b"
    
   
    # Output path for checkpoint and logs
    # save_path: Optional[str] = './ablation_models/no_exists_scores/'
    save_path: Optional[str] = 'checkpoints_FB15K-237/demo'
    # path for loading checkpoints
    # checkpoint_path: Optional[str] = None
    checkpoint_path: Optional[str] = 'checkpoints_FB15K-237/checkpoint_orig_attr_kblrn'
    # the model to be trained
    geo: Enum(
        "geo",
        [
            "cqd-transea",
            "cqd-transeadistmult",
            "cqd-transeacomplex",
            "cqd-transra",
            "cqd-mtkgnn",
            "cqd-distmulta",
            "cqd-complex",
            "cqd-complexa",
            "cqd-complexa-weighted",
            "cqd-complexad",
            "cqd-complexd",
            "cqd-complexd-jointly",
            "cqd-complex-simple",
            "cqd-transcomplexa",
            "cqd-transcomplexdice",
            "q2b",
            "gqe",
            "random_guesser",
        ],
    ) = "cqd-complexa"
    # loss function of the relational part
    loss: Enum("loss", ["margin", "ce", "q2b"]) = "ce"
    # How many epochs the model is trained for
    train_times: int = 100
    # Evaluate validation queries every xx epochs
    valid_epochs: int = 1 #10
    # How many workers pytorch uses to load data
    cpu_num: int = 13
    # random seed applied globally
    seed: int = 0
    # use GPU
    cuda: bool = False
    # use attribute data
    use_attributes: bool = False
    use_descriptions: bool = False
    # train using triples and the cqd dataloader or use queries with a subsampling weight
    train_data_type: Enum("train_data_type", ["queries", "triples"]) = "triples"
    # valid/test batch size
    test_batch_size: int = 1024  #100
    # tune hyperparameters using ray tune
    do_tune: bool = False
    do_train: bool = False
    do_test: bool = False
    # evaluate on train queries aswell
    eval_on_train: bool = False
    # evaluate on simple (1-hop) queries only
    simple_eval: bool = False
    # embedding dimension of the word embeddings
    word_emb_dim: int = 300
    # create latex table
    to_latex: bool = False


@dataclass
class CQDParams:
    """Params to configure the CQD framework"""

    # Optimization algorithm used to answer complex queries
    cqd_type: Enum("type", ["continuous", "discrete"]) = "discrete"
    # t-norm used to compute conjunctions and disjunctions (t-co-norm)
    cqd_t_norm: Enum("t-norm", list(CQDBaseModel.NORMS)) = "prod"
    # How many samples are retained for each step in the discrete optimization algorithm
    cqd_k: int = 4

@dataclass
class HyperParams:
    """Hyperparameter"""

    # hidden dim; embedding dimension
    rank: int = 1000
    # batch size during training
    batch_size: int = 1024
    # loss function of the attribute part
    attr_loss: Enum("attr_loss", ["mae", "mse"]) = "mae"
    # learning rate
    learning_rate: float = field(0.1, alias="-lr")
    # learning rate for attribute embeddings only
    learning_rate_attr: Optional[float] = field(0.1, alias="-lr_attr")
    # negative entities samples per query
    negative_sample_size: int = field(1, alias="-n")
    # negative attribute values sampled per query
    negative_attr_sample_size: int = field(0, alias="-na")
    # regularization weight for N3 regularization
    reg_weight: float = 0
    # L2 regularization weight for entity embeddings
    reg_weight_ent: float = 0
    # L2 regularization weight for relation embeddings
    reg_weight_rel: float = 0
    # L2 regularization weight for attribute embeddings
    reg_weight_attr: float = 0
    # Determines which fraction of the loss makes up the attribute loss
    alpha: float = 0.5 #0.3
    beta:float = 1.0
    # optimizer
    optimizer: Enum("optimizer", ["adam", "adagrad", "sgd"]) = "adagrad"
    # Number of epochs with no improvement after which learning rate will be reduced
    scheduler_patience: float = 5
    # Factor by which the learning rate will be reduced
    scheduler_factor: float = 0.95
    # Threshold for measuring the new optimum, to only focus on significant changes
    scheduler_threshold: float = 0.01
    # Required for a margin-based loss function
    margin: float = field(2.0, alias="-g")
    # p_norm for TransE
    p_norm: int = 2
    # apply sigmoid on the attribute value predictions
    do_sigmoid: bool = True   #False
    # rank for transr
    rank_attr: int = 50
    # how to represent description embeddings
    desc_emb: Enum("desc_emb", ["1-layer", "2-layer", "gate"]) = "1-layer"
    # Use modules for attribute value prediction instead of mean
    use_modulus: bool = False


def parse_args(args=None):
    parser = ArgumentParser(
        description="Training and Testing Knowledge Graph Embedding Models",
        usage="train.py [<args>] [-h | --help]",
    )
    parser.add_argument("--print_on_screen", action="store_true")
    parser.add_argument("--dataloader_type", default="cpp", choices=["cpp", "python"])
    parser.add_arguments(TrainConfig, dest="train_config")
    parser.add_arguments(HyperParams, dest="hyperparams")
    parser.add_arguments(CQDParams, dest="cqd_params")
    return parser.parse_args(args)