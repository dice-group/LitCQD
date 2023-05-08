from enum import Enum
from ray.tune.progress_reporter import CLIReporter

from Loss import MAELoss, MSELoss, CELoss, MRLoss, Q2BLoss
from config import parse_args, CQDParams, HyperParams, TrainConfig
import json

import logging
import os

import torch

from Trainer import Trainer
from Tester import Tester

from tensorboardX import SummaryWriter

from collections import defaultdict
from util_models import get_model, load_model
from util import log_metrics, parse_time, set_global_seed, query_name_dict, set_logger
from util_data import *
from util_data_queries import *


import dataclasses
import ray
from ray import tune
import numpy as np
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

writer = None


def create_latex_table(results, model_name):
    import jinja2

    latex_jinja_env = jinja2.Environment(
        block_start_string="\BLOCK{",
        block_end_string="}",
        variable_start_string="\VAR{",
        variable_end_string="}",
        comment_start_string="\  # {",
        comment_end_string="}",
        line_statement_prefix="%-",
        line_comment_prefix="%  # ",
        trim_blocks=True,
        autoescape=False,
        loader=jinja2.FileSystemLoader(os.path.abspath(".")),
    )
    template = latex_jinja_env.get_template("metrics_template.tex")
    query_structures = [query_name_dict[x] for x in results.keys()]
    metrics = [x.keys() for x in results.values()]

    values_by_metric = defaultdict(list)
    for metrics_dict in results.values():
      
        for metric, value in metrics_dict.items():
          
          values_by_metric[metric.replace("_", "\_")].append(
                f"{value:.3}" if type(value) == float else value
            )
    # values = [f'{x["HITS3"]:.3}' for x in results.values()]
    # values = {metric: f'{value:.3}' for metric, value in values.items()}

    document = template.render(
        model_name=model_name,
        query_structures=query_structures,
        metrics=values_by_metric,
    )
    print(document)


def tensorboard_write_loss(loss, rel_loss, attr_loss, epoch):
    writer.add_scalar("Train_Loss", loss, epoch)
    writer.add_scalar("Train_Loss_Relations", rel_loss, epoch)
    writer.add_scalar("Train_Loss_Attributes", attr_loss, epoch)


def evaluate(
    tester,
    tp_answers,
    fn_answers,
    train_config: TrainConfig,
    query_name_dict,
    mode,
    epoch,
):
    """
    Evaluate queries in dataloader
    """
    global writer
    average_metrics = defaultdict(float)
    average_metrics_attr = defaultdict(float)
    all_metrics = defaultdict(float)

    metrics = tester.test_step(tp_answers, fn_answers, train_config, query_name_dict)
    num_query_structures = 0
    num_query_structures_attr = 0
    table = None
    
    
    if train_config.geo.name =='q2b' and train_config.to_latex:
      import util
      table = util.create_latex_table(train_config)

    
    for query_structure in metrics:
        if "ME" in metrics[query_structure]:
            # ignore attr pred metrics for average calculation
            num_query_structures -= 1
        log_metrics(
            mode + " " + query_name_dict[query_structure],
            epoch,
            metrics[query_structure],
        )
        for metric in metrics[query_structure]:
            if writer:
                writer.add_scalar(
                    "_".join([mode, query_name_dict[query_structure], metric]),
                    metrics[query_structure][metric],
                    epoch,
                )
            all_metrics["_".join([query_name_dict[query_structure], metric])] = metrics[
                query_structure
            ][metric]
            if metric not in ("num_queries"):
                if metric in ("MAE", "MSE", "RMSE"):
                    average_metrics_attr[metric] += metrics[query_structure][metric]
                elif metric not in ("cos_sim",):
                    average_metrics[metric] += metrics[query_structure][metric]
        
        if train_config.to_latex and train_config.geo.name =='q2b':
          from util import create_table_col
          table = create_table_col(query_name_dict[query_structure],metrics[query_structure],table)
                    
           
        if query_name_dict[query_structure].endswith("ap"):
            num_query_structures_attr += 1
        elif query_name_dict[query_structure] != "1dp":
            num_query_structures += 1

    if table and train_config.geo.name=='q2b':
      from util import store_latex
      store_latex(table, train_config)
    
    for metric in average_metrics:
        average_metrics[metric] /= num_query_structures
        if writer:
            writer.add_scalar(
                "_".join([mode, "average", metric]), average_metrics[metric], epoch
            )
        all_metrics["_".join(["average", metric])] = average_metrics[metric]
    for metric in average_metrics_attr:
        average_metrics_attr[metric] /= num_query_structures_attr
        if writer:
            writer.add_scalar(
                "_".join([mode, "average", metric]), average_metrics_attr[metric], epoch
            )
        all_metrics["_".join(["average", metric])] = average_metrics_attr[metric]
    log_metrics("%s average" % mode, epoch, average_metrics)
    log_metrics("%s average" % mode, epoch, average_metrics_attr)

    if False and mode.lower() == "test":
        create_latex_table(metrics, train_config.geo.name)
   
    
    return all_metrics


def test_model(
    model,
    train_config: TrainConfig,
    mode="Test",
    tasks=(
        "1p",
        "1ap",
    ),
):
    if not train_config.use_attributes:
        tasks = tuple(x for x in tasks if "a" not in x)
    if not train_config.use_descriptions:
        tasks = tuple(x for x in tasks if "d" not in x)
    queries, easy_answers, hard_answers = load_queries_eval(
        train_config.data_path, tasks, mode.lower()
    )
    dataset = get_dataset_eval(queries)
    dataloader = get_eval_dataloader(
        dataset, train_config.test_batch_size, train_config.cpu_num
    )
    
    
    
    if not tasks[0] in "1ap" and not tasks[0] in "2ap" and not tasks[0] in "3ap":
      attr_values = defaultdict(list)
      train_queries = dataset.queries
      for query in train_queries:
        attr_values[query[0][0][1]].append(query[0][1][0])
        
      model.attr_values = attr_values
    
    
    
    tester = Tester(model, dataloader, train_config.cuda)
    
    metrics = evaluate(
        tester,
        easy_answers,
        hard_answers,
        train_config,
        query_name_dict,
        mode,
        train_config.train_times,
    )
    
    
    
    # from util import create_latex_table
    
    # create_latex_table(train_config,tasks,model,None,metrics=metrics)
    
    
    return metrics
    
    # return evaluate(
    #     tester,
    #     easy_answers,
    #     hard_answers,
    #     train_config,
    #     query_name_dict,
    #     mode,
    #     train_config.train_times,
    # )


def train(train_config: TrainConfig, cqd_params: CQDParams):
    def train_ray(
        config,
        nentity,
        nrelation,
        nattribute,
        train_data_rel,
        train_data_attr,
        train_data_desc,
        valid_loss_data_rel,
        valid_loss_data_attr,
        valid_queries,
        valid_answers_easy,
        valid_answers_hard,
        eval_train_queries,
        eval_train_answers,
        checkpoint_dir=None,
    ):

        set_global_seed(train_config.seed)
        params = HyperParams(**config) # config parameters are passed here
        dataloader_type = "python"

        if not train_config.use_attributes:
            params.alpha = 0

        loss = train_config.loss.name
        if loss == "margin":
            rel_loss = MRLoss(params.margin, params.negative_sample_size)
        elif loss == "ce":
            rel_loss = CELoss(params.negative_sample_size)
        elif loss == "q2b":
            rel_loss = Q2BLoss(params.margin, params.negative_sample_size)

        optimizer = params.optimizer.name
        name_to_optimizer = {
            "adam": torch.optim.Adam,
            "adagrad": torch.optim.Adagrad,
            "sgd": torch.optim.SGD,
        }
        assert optimizer in name_to_optimizer
        OptimizerClass = name_to_optimizer[optimizer]

        train_dataset = get_dataset_train(
            *train_data_rel, train_config, nentity, nrelation, params
        )
        
        
        
        train_dataset_attr = get_dataset_train_attr(*train_data_attr, nentity, params)
        train_dataset_desc = get_dataset_train_desc(
            *train_data_desc, train_config.geo.name == "cqd-complexd-jointly"
        )

        (
            train_dataloader,
            train_dataloader_attr,
            train_dataloader_desc,
        ) = get_train_dataloader(
            train_dataset,
            train_dataset_attr,
            train_dataset_desc,
            params.batch_size,
            train_config.use_attributes,
            train_config.use_descriptions,
            train_config.seed,
        )

        attr_loss = None
        attr_loss_param = params.attr_loss
        # if type(attr_loss_param) != str:
        #     # bug
        #     attr_loss_param = attr_loss_param.name
        
        if train_config.use_attributes and dataloader_type == "python":
            if attr_loss_param == "mae":
                attr_loss = MAELoss(params.negative_attr_sample_size)
            elif attr_loss_param == "mse":
                attr_loss = MSELoss(params.negative_attr_sample_size)

        model = get_model(
            train_config, params, cqd_params, nentity, nrelation, nattribute
        )

        learning_rate = params.learning_rate
        learning_rate_attr = params.learning_rate_attr
        
        trainer = Trainer(
            model,
            train_dataloader,
            dataloader_type,
            train_config.cuda,
            learning_rate,
            learning_rate_attr,
            "./tmp",
            train_config.train_times,
            OptimizerClass,
            rel_loss,
            attr_loss,
            params.alpha,
            params.beta,
            train_dataloader_attr,
            train_dataloader_desc,
            params.negative_attr_sample_size,
            params.reg_weight_ent,
            params.reg_weight_rel,
            params.reg_weight_attr,
            params.scheduler_patience,
            params.scheduler_factor,
            params.scheduler_threshold,
        )

        logging.info("-------------------------------" * 3)
        logging.info("Geo: %s" % train_config.geo)
        logging.info("Data Path: %s" % checkpoint_dir)
        logging.info("#entity: %d" % nentity)
        logging.info("#relation: %d" % nrelation)
        logging.info("#attributes: %d" % nattribute)
        logging.info("batch size: %d" % params.batch_size)

        # used to compute loss on validation set
        valid_dataset = get_dataset_train(
            *valid_loss_data_rel, train_config, nentity, nrelation, params
        )
        valid_dataset_attr = get_dataset_train_attr(
            *valid_loss_data_attr, nentity, params
        )
        valid_dataloader, valid_attr_dataloader, _ = get_train_dataloader(
            valid_dataset,
            valid_dataset_attr,
            None,
            params.batch_size,
            train_config.use_attributes,
            False,
        )

        # used to compute other metrics on validation set
        valid_dataset_eval = get_dataset_eval(valid_queries)
        valid_dataloader_eval = get_eval_dataloader(
            valid_dataset_eval, train_config.test_batch_size, train_config.cpu_num
        )
        validator = Tester(model, valid_dataloader_eval, train_config.cuda)

        if eval_train_queries:
            eval_train_answers_easy = defaultdict(set)
            eval_train_answers_hard = eval_train_answers
            valid_dataset_eval = get_dataset_eval(eval_train_queries)
            valid_dataloader_eval = get_eval_dataloader(
                valid_dataset_eval, train_config.test_batch_size, train_config.cpu_num
            )
            train_validator = Tester(model, valid_dataloader_eval, train_config.cuda)

        logging.info("Model Parameter Configuration:")
        num_params = 0
        for name, param in model.named_parameters():
            logging.info(
                "Parameter %s: %s, require_grad = %s"
                % (name, str(param.size()), str(param.requires_grad))
            )
            if param.requires_grad:
                num_params += np.prod(param.size())
        logging.info("Parameter Number: %d" % num_params)

        if checkpoint_dir:
            checkpoint = os.path.join(checkpoint_dir, "checkpoint")
            model_state, optimizer_state = torch.load(checkpoint)
            model.load_state_dict(model_state)
            trainer.optimizer.load_state_dict(optimizer_state)
        else:
            logging.info("Ramdomly Initializing %s Model..." % train_config.geo.name)

        if eval_train_queries:

            def eval_fn(epoch):
                train_metrics = evaluate(
                    train_validator,
                    eval_train_answers_easy,
                    eval_train_answers_hard,
                    train_config,
                    query_name_dict,
                    "Train",
                    epoch,
                )
                valid_metrics = evaluate(
                    validator,
                    valid_answers_easy,
                    valid_answers_hard,
                    train_config,
                    query_name_dict,
                    "Valid",
                    epoch,
                )
                train_metrics = {"train_" + k: v for k, v in train_metrics.items()}
                valid_metrics.update(train_metrics)
                return valid_metrics

        else:

            def eval_fn(epoch):
                return evaluate(
                    validator,
                    valid_answers_easy,
                    valid_answers_hard,
                    train_config,
                    query_name_dict,
                    "Valid",
                    epoch,
                )

        print("Training starts...")
        if train_config.do_tune:
            if train_config.geo.name in ("cqd-complexd", "cqd-complexad"):
                trainer.train_ray_desc(eval_fn, train_config.valid_epochs)
            else:
                trainer.train_ray(
                    eval_fn,
                    valid_dataloader,
                    valid_attr_dataloader,
                    train_config.valid_epochs,
                )
        else:
            trainer.train(eval_fn, tensorboard_write_loss, train_config.valid_epochs)
            
            torch.save(
                (model.state_dict(), trainer.optimizer.state_dict()),
                os.path.join(train_config.save_path, "checkpoint"),
            )

    return train_ray


def run_tune(
    train_config: TrainConfig,
    cqd_params: CQDParams,
    params: HyperParams,
    # nentity,
    # nrelation,
    # nattribute,
    **data,
):
    ray.init(num_gpus=1)

    if train_config.geo.name in ("cqd-complexd", "cqd-complexad"):
        # Models using description embeddings

        params.desc_emb = tune.grid_search(["1-layer", "2-layer", "gate"])

        search_alg = BasicVariantGenerator()
        reporter = CLIReporter()
        reporter.add_metric_column("mrr")
        reporter.add_metric_column("cos_sim")
        reporter.add_metric_column("train_cos_sim")
        reporter.add_metric_column("di_mrr")
        reporter.add_metric_column("loss")
        reporter.add_metric_column("valid_loss")
        

        result = tune.run(
            tune.with_parameters(
                train(train_config, cqd_params),
                # nentity=nentity,
                # nrelation=nrelation,
                # nattribute=nattribute,
                **data,
            ),
            config=dataclasses.asdict(
                params
            ),  # convert parameters to dict (argument config in the train!!!)
            metric="mrr",
            mode="max",
            num_samples = 10,
            # num_samples=20,
            # training has to be done on the same device for reproducibility; randomness not guaranteed on different devices; also not guaranteed at 100% utilization
            resources_per_trial={"gpu": 0.25, "cpu": 1},
            search_alg=search_alg,
            keep_checkpoints_num=1,
            progress_reporter=reporter,
            fail_fast=False,
            max_failures=2,
            # name='train_ray_2022-01-17_09-56-26',
            # resume=True,
        )
        best_trial = result.get_best_trial("mrr", "max", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final loss: {}".format(best_trial.last_result["loss"]))
        print(
            "Best trial final valid loss: {}".format(
                best_trial.last_result["valid_loss"]
            )
        )
        print(
            "Best trial final validation mrr: {}".format(best_trial.last_result["mrr"])
        )
        print(
            "Best trial final validation cos_sim: {}".format(
                best_trial.last_result["cos_sim"]
            )
        )
        print(
            "Best trial final train cos_sim: {}".format(
                best_trial.last_result["train_cos_sim"]
            )
        )
        print(
            "Best trial final validation di MRR: {}".format(
                best_trial.last_result["di_mrr"]
            )
        )

        best_params = HyperParams(**best_trial.config)
        best_trained_model = get_model(
            train_config, best_params, cqd_params, data['nentity'], data['nrelation'], data['nattribute']
        )
        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, _ = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
        best_trained_model.load_state_dict(model_state)
        tasks = ("1p", "1ap", "1dp", "di")
        test_model(best_trained_model, train_config, "Valid", tasks=tasks)
        test_model(best_trained_model, train_config, "Test", tasks=tasks)
    else:

        def custom_asdict_factory(data):
            def convert_value(obj):
                if isinstance(obj, Enum):
                    return obj.name
                return obj

            return dict((k, convert_value(v)) for k, v in data)

        # current_best_params = [dataclasses.asdict(params)]  # , dict_factory=custom_asdict_factory)]
        # params.batch_size = tune.choice([100, 500, 1000])
        # params.learning_rate = tune.loguniform(1e-3, 1)
        # params.learning_rate_attr = tune.loguniform(1e-3, 1)
        # params.rank_attr = tune.grid_search([20, 50])
        # params.learning_rate = tune.grid_search([0.1, 0.01,0.001])
        # params.rank = tune.grid_search([128, 256, 512, 1024])
        params.rank = tune.grid_search([200, 500, 1000])
        params.alpha = tune.grid_search([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        # params.p_norm = tune.grid_search([1, 2])
        # params.attr_loss = tune.grid_search(["mae", "mse"])
        # params.do_sigmoid = tune.grid_search([True, False])
        
        search_alg = BasicVariantGenerator()
        # search_alg = BasicVariantGenerator(max_concurrent=2)
        # search_alg = HyperOptSearch(points_to_evaluate=current_best_params)
        # search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)
        # scheduler = AsyncHyperBandScheduler(grace_period=1, reduction_factor=3)
        reporter = CLIReporter()
        reporter.add_metric_column("score")
        reporter.add_metric_column("mrr")
        reporter.add_metric_column("mae")
        reporter.add_metric_column("mse")
        reporter.add_metric_column("loss")
        reporter.add_metric_column("rel_loss")
        reporter.add_metric_column("attribute_loss")
        
        # test
        # _train = train(train_config, cqd_params)
        # _train(config=dataclasses.asdict(
        #         params
        #     ),nentity=nentity,nrelation=nrelation,
        #         nattribute=nattribute,**data)
        
        # exit(1)
        
        result = tune.run(
            tune.with_parameters(
                train(train_config, cqd_params),
                # nentity=nentity,
                # nrelation=nrelation,
                # nattribute=nattribute,
                **data,
            ),
            config=dataclasses.asdict(params),
            metric="score",
            mode="max",
            num_samples=10,
            # training has to be done on the same device for reproducibility; randomness not guaranteed on different devices; also not guaranteed at 100% utilization
            resources_per_trial={"gpu": 0.25, "cpu": 1},
            search_alg=search_alg,
            keep_checkpoints_num=1,
            progress_reporter=reporter,
            fail_fast=False,
            max_failures=2,
            
            # name='train_ray_2022-01-17_09-56-26',
            # resume=True,
            # scheduler=scheduler
        )

        best_trial = result.get_best_trial("loss", "min", "last")
        print("Best trial (loss) config: {}".format(best_trial.config))
        print(
            "Best trial (loss) final validation loss: {}".format(
                best_trial.last_result["loss"]
            )
        )
        print(
            "Best trial (loss) final validation attr_loss: {}".format(
                best_trial.last_result["attribute_loss"]
            )
        )
        print(
            "Best trial (loss) final validation rel_loss: {}".format(
                best_trial.last_result["rel_loss"]
            )
        )
        print(
            "Best trial (loss) final validation score: {}".format(
                best_trial.last_result["score"]
            )
        )
        print(
            "Best trial (loss) final validation mrr: {}".format(
                best_trial.last_result["mrr"]
            )
        )
        print(
            "Best trial (loss) final validation mae: {}".format(
                best_trial.last_result["mae"]
            )
        )

        best_params = HyperParams(**best_trial.config)
        best_trained_model = get_model(
            train_config, best_params, cqd_params, data['nentity'], data['nrelation'], data['nattribute']
        )
        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, _ = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
        best_trained_model.load_state_dict(model_state)
        test_model(best_trained_model, train_config, "Valid")
        test_model(best_trained_model, train_config)

        best_trial = result.get_best_trial("mrr", "max", "last")
        print("Best trial (mrr) config: {}".format(best_trial.config))
        print(
            "Best trial (mrr) final validation loss: {}".format(
                best_trial.last_result["loss"]
            )
        )
        print(
            "Best trial (mrr) final validation attr_loss: {}".format(
                best_trial.last_result["attribute_loss"]
            )
        )
        print(
            "Best trial (mrr) final validation rel_loss: {}".format(
                best_trial.last_result["rel_loss"]
            )
        )
        print(
            "Best trial (mrr) final validation score: {}".format(
                best_trial.last_result["score"]
            )
        )
        print(
            "Best trial (mrr) final validation mrr: {}".format(
                best_trial.last_result["mrr"]
            )
        )
        print(
            "Best trial (mrr) final validation mae: {}".format(
                best_trial.last_result["mae"]
            )
        )

        best_params = HyperParams(**best_trial.config)
        best_trained_model = get_model(
            train_config, best_params, cqd_params, data['nentity'], data['nrelation'], data['nattribute']
        )
        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, _ = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
        best_trained_model.load_state_dict(model_state)
        test_model(best_trained_model, train_config, "Valid")
        test_model(best_trained_model, train_config)


def new_train(
    train_config: TrainConfig,
    cqd_params: CQDParams,
    params: HyperParams,
    nentity,
    nrelation,
    nattribute,
    **data,
):
    # TODO: train without ray tuning
    # ignore tensorboard
    # ignore ray tune
    # dataloader uses 10 works???
    """
    1. preparation phase for trainer
     - get loss function, optimizer, dataloader, train_dataset, lr
     - get model
    2. initialize trainer
    3. things to do with validation set, logging and so on

    4. train the models without ray tune (using train() in Trainer.py)
    """
    config = dataclasses.asdict(params)

    # get the model
    set_global_seed(train_config.seed)
    params = HyperParams(**config)
    dataloader_type = "python"

    if not train_config.use_attributes:
        params.alpha = 0

    # loss function
    loss = train_config.loss.name
    if loss == "margin":
        rel_loss = MRLoss(params.margin, params.negative_sample_size)
    elif loss == "ce":
        rel_loss = CELoss(params.negative_sample_size)
    elif loss == "q2b":
        rel_loss = Q2BLoss(params.margin, params.negative_sample_size)

    # optimizer
    optimizer = params.optimizer.name
    name_to_optimizer = {
        "adam": torch.optim.Adam,
        "adagrad": torch.optim.Adagrad,
        "sgd": torch.optim.SGD,
    }
    assert optimizer in name_to_optimizer
    OptimizerClass = name_to_optimizer[optimizer]

    # dataset
    train_data_rel = data["train_data_rel"]
    train_data_attr = data["train_data_attr"]
    train_data_desc = data["train_data_desc"]

    train_dataset = get_dataset_train(
        *train_data_rel, train_config, nentity, nrelation, params
    )
    train_dataset_attr = get_dataset_train_attr(*train_data_attr, nentity, params)
    train_dataset_desc = get_dataset_train_desc(
        *train_data_desc, train_config.geo.name == "cqd-complexd-jointly"
    )

    (
        train_dataloader,
        train_dataloader_attr,
        train_dataloader_desc,
    ) = get_train_dataloader(
        train_dataset,
        train_dataset_attr,
        train_dataset_desc,
        params.batch_size,
        train_config.use_attributes,
        train_config.use_descriptions,
        train_config.seed,
    )

    attr_loss = None
    attr_loss_param = params.attr_loss
    if type(attr_loss_param) != str:
        attr_loss_param = attr_loss_param.name
    if train_config.use_attributes and dataloader_type == "python":
        if attr_loss_param == "mae":
            attr_loss = MAELoss(params.negative_attr_sample_size)
        elif attr_loss_param == "mse":
            attr_loss = MSELoss(params.negative_attr_sample_size)

    # initialize model
    model = get_model(train_config, params, cqd_params, nentity, nrelation, nattribute)
    
    
    train_queries = data['train_data_attr'][1]
    attr_values = defaultdict(list)
    for key,value in train_queries.items():
      attr_values[key[1][1]].append(next(iter(value)))
    
    model.attr_values = attr_values
      
    
    

    learning_rate = params.learning_rate
    learning_rate_attr = params.learning_rate_attr

    # initialize trainer
    trainer = Trainer(
        model,
        train_dataloader,
        dataloader_type,
        train_config.cuda,
        learning_rate,
        learning_rate_attr,
        "./tmp",
        train_config.train_times,
        OptimizerClass,
        rel_loss,
        attr_loss,
        params.alpha,
        params.beta,
        train_dataloader_attr,
        train_dataloader_desc,
        params.negative_attr_sample_size,
        params.reg_weight_ent,
        params.reg_weight_rel,
        params.reg_weight_attr,
        params.scheduler_patience,
        params.scheduler_factor,
        params.scheduler_threshold,
    )

    # used to compute loss on validation set
    valid_loss_data_rel = data["valid_loss_data_rel"]
    valid_loss_data_attr = data["valid_loss_data_attr"]
    valid_queries = data["valid_queries"]

    valid_dataset = get_dataset_train(
        *valid_loss_data_rel, train_config, nentity, nrelation, params
    )
    valid_dataset_attr = get_dataset_train_attr(*valid_loss_data_attr, nentity, params)
    valid_dataloader, valid_attr_dataloader, _ = get_train_dataloader(
        valid_dataset,
        valid_dataset_attr,
        None,
        params.batch_size,
        train_config.use_attributes,
        False,
    )

    # used to compute other metrics on validation set
    valid_dataset_eval = get_dataset_eval(valid_queries)
    valid_dataloader_eval = get_eval_dataloader(
        valid_dataset_eval, train_config.test_batch_size, train_config.cpu_num
    )
    validator = Tester(model, valid_dataloader_eval, train_config.cuda)

    valid_answers_easy = data["valid_answers_easy"]
    valid_answers_hard = data["valid_answers_hard"]
    # currently just test one of eval_fn
    # def eval_fn(epoch):
    #     return evaluate(validator, valid_answers_easy, valid_answers_hard, train_config, query_name_dict,'Valid', epoch)

    # trainer.train(tensorboard_write_loss, train_config.valid_epochs, eval_fn=None)
    trainer.train(None, eval_fn=None, eval_epochs=train_config.valid_epochs)
    
    torch.save(
        (model.state_dict(), trainer.optimizer.state_dict()),
        os.path.join(train_config.save_path, "checkpoint"),
    )

    
    
# def train_ray_test():
#     test = train()
    
    
    
    


def main(args):
    global writer
    # (1) Fixing the seed
    set_global_seed(args.train_config.seed)

    ##### Create logs/output folder #####
    if args.train_config.save_path is None:
        args.train_config.save_path = os.path.join(
            "logs",
            args.train_config.data_path.split("/")[-1],
            args.train_config.geo.name,
        )
        if args.train_config.checkpoint_path is not None:
            args.train_config.save_path = args.train_config.checkpoint_path
        else:
            args.train_config.save_path = os.path.join(
                args.train_config.save_path, parse_time()
            )

    if not os.path.exists(args.train_config.save_path):
        os.makedirs(args.train_config.save_path)
    # (2) Logging starts
    print("logging to", args.train_config.save_path)
    set_logger(
        args.train_config.save_path, args.train_config.do_train, args.print_on_screen
    )

    ##### Write config to log folder ######
    if args.train_config.do_train:
        # @TODO create a method for this operation and move out from the main
        argparse_dict = vars(args)
        with open(
            os.path.join(args.train_config.save_path, "config.json"), "w"
        ) as fjson:

            class EnhancedJSONEncoder(json.JSONEncoder):
                def default(self, o):
                    if dataclasses.is_dataclass(o):
                        return dataclasses.asdict(o)
                    if isinstance(o, Enum):
                        return o.name
                    return super().default(o)

            json.dump(argparse_dict, fjson, cls=EnhancedJSONEncoder)
        del argparse_dict

    ##### Create TensorBoard Writer #####
    if args.train_config.do_tune:
        writer = None
    elif (
        not args.train_config.do_train
    ):  # if not training, then create tensorboard files in some tmp location
        writer = SummaryWriter("./logs-debug/unused-tb")
    else:
        writer = SummaryWriter(args.train_config.save_path)

    train_config: TrainConfig = args.train_config
    cqd_params: CQDParams = args.cqd_params
    params: HyperParams = args.hyperparams

    nentity, nrelation, nattribute = load_stats(train_config.data_path)
    if train_config.do_tune or train_config.do_train:
        print("Loading queries for the training...")
        train_data_rel, train_data_attr, train_data_desc = load_queries_train(
            train_config, "train"
        )
        print("Loading queries for the valid...")
        valid_loss_data_rel, valid_loss_data_attr, _ = load_queries_train(
            train_config, "valid"
        )
        valid_queries, valid_answers_easy, valid_answers_hard = load_queries_eval(
            train_config.data_path,
            ("1p", "1ap", "1dp", "di")
            if train_config.use_attributes
            else ("1p", "1dp", "di"),
            "valid",
        )
        if train_config.eval_on_train and "q2b" not in train_config.geo.name.lower():
            eval_train_queries, eval_train_answers, _ = load_queries_eval(
                train_config.data_path,
                (
                    "1dp",
                    "1ap",
                )
                if train_config.use_attributes
                else ("1dp",),
                "train",
            )
        else:
            # It takes too long for query2box to answer all train queries
            eval_train_queries, eval_train_answers = None, None
    
    # CD: do_tune doesn' work
    if train_config.do_tune:
        run_tune(
            train_config,
            cqd_params,
            params,
            # nentity,
            # nrelation,
            # nattribute,
            nentity=nentity,
            nrelation=nrelation,
            nattribute=nattribute,
            train_data_rel=train_data_rel,
            train_data_attr=train_data_attr,
            train_data_desc=train_data_desc,
            valid_loss_data_rel=valid_loss_data_rel,
            valid_loss_data_attr=valid_loss_data_attr,
            valid_queries=valid_queries,
            valid_answers_easy=valid_answers_easy,
            valid_answers_hard=valid_answers_hard,
            eval_train_queries=eval_train_queries,
            eval_train_answers=eval_train_answers,
        )
    else:
        # print(train_config.do_train)
        if train_config.do_train:
            
            logging.info("Training starts...")
            new_train(
                train_config,
                cqd_params,
                params,
                nentity,
                nrelation,
                nattribute,
                train_data_rel=train_data_rel,
                train_data_attr=train_data_attr,
                train_data_desc=train_data_desc,
                valid_loss_data_rel=valid_loss_data_rel,
                valid_loss_data_attr=valid_loss_data_attr,
                valid_queries=valid_queries,
                valid_answers_easy=valid_answers_easy,
                valid_answers_hard=valid_answers_hard,
                eval_train_queries=eval_train_queries,
                eval_train_answers=eval_train_answers,
            )
            logging.info("Training finished!!")

        if train_config.do_test:
            model = get_model(
                train_config, params, cqd_params, nentity, nrelation, nattribute
            )
            load_model(model, train_config.checkpoint_path, train_config.cuda)

            tasks = (
                "1p",
                "1ap",
                "2ap",
                "3ap",
                "ai-lt",
                "1dp",
                "di",
            )
            if not train_config.use_attributes and not train_config.use_descriptions:
                tasks = (
                    "1p",
                    "2p",
                    "3p",
                    "2i",
                    "3i",
                    "ip",
                    "pi",
                    "2u",
                    "up",
                )

            if train_config.simple_eval:
                tasks = ("1p",)
                if train_config.use_attributes:
                    tasks += ("1ap",)
            test_model(model, train_config, "Test", tasks=tasks)

    writer.close()


if __name__ == "__main__":
    main(parse_args())
