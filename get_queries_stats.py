from config import parse_args, CQDParams, HyperParams, TrainConfig
from util_data_queries import load_queries_eval, load_queries_train


def main(args):
  
  train_config: TrainConfig = args.train_config


  # the summary of the queires of the training 
  # not all the queries will be used, because the training is only in respect to specific query types
  print("Loading queries for the training...")
  # queries with or without attributes for the train dataset
  print('train dataset:')
  train_data_rel, train_data_attr, train_data_desc = load_queries_train(
            train_config, "train",not_flatten=True
        )
  for k,v in train_data_rel[0].items():
    print(f'{k}: {len(v)}')

  for k,v in train_data_attr[0].items():
    print(f'{k}: {len(v)}')
  
  print('valiadation dataset of loss:')
  # queries with or without attributes for the validation dataset during training
  valid_loss_data_rel, valid_loss_data_attr, _ = load_queries_train(
            train_config, "valid",not_flatten=True
        )
            
  for k,v in valid_loss_data_rel[0].items():
      print(f'{k}: {len(v)}')

  for k,v in valid_loss_data_attr[0].items():
      print(f'{k}: {len(v)}')


  print('validation dataset of queries:')
  valid_queries, valid_answers_easy, valid_answers_hard = load_queries_eval(
            train_config.data_path,
            ("1p", "1ap", "1dp", "di")
            if train_config.use_attributes
            else ("1p", "1dp", "di"),
            "valid",not_flatten=True
        )
  
  
  for k,v in valid_queries.items():
      print(f'{k}: {len(v)}')


  print('evaluation dataset of queires:')
  eval_train_queries, eval_train_answers, _ = load_queries_eval(
                train_config.data_path,
                (
                    "1dp",
                    "1ap",
                )
                if train_config.use_attributes
                else ("1dp",),
                "train",not_flatten=True
            )

  for k,v in eval_train_queries.items():
      print(f'{k}: {len(v)}')



  print('testing dataset: ')
  
  eval_train_queries, eval_train_answers, _ = load_queries_eval(
                  train_config.data_path,
                  (
                      "1dp",
                      "1ap",
                  )
                  if train_config.use_attributes
                  else ("1dp",),
                  "train",not_flatten=True
              )
  
  
  for k,v in eval_train_queries.items():
      print(f'{k}: {len(v)}')
  
  ####################################
  # the summary of the queries of the evaluation
  
  
  if train_config.use_attributes:
    print('queries of testing data with attributes: ')
    tasks = ('1ap', '2ap', '3ap', 'ai-lt', 'ai-eq', 'ai-gt', '2ai', 'aip', 'pai', 'au',)
    attr_eval_dict = dict()
    
    for task in tasks:
      tmp_task = (task,)
      queries, easy_answers, hard_answers = load_queries_eval(
            train_config.data_path, tmp_task, "test", not_flatten=True
        )
      attr_eval_dict[list(queries.keys())[0]] = queries[list(queries.keys())[0]]
  else:
    print('queries of testing data without attributes: ')

    tasks = ('1p',  '2p', '3p', '2i', '3i', 'ip', 'pi', '2u', 'up',)
    
    attr_eval_dict = dict()
    
    for task in tasks:
      tmp_task = (task,)
      queries, easy_answers, hard_answers = load_queries_eval(
            train_config.data_path, tmp_task, "test", not_flatten=True
        )
      attr_eval_dict[list(queries.keys())[0]] = queries[list(queries.keys())[0]]
  

  
  
  for k in attr_eval_dict.keys():
    print(f'{k}: {len(attr_eval_dict[k])}')


  
      
if __name__ == "__main__":
    main(parse_args())