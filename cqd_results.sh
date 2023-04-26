# Evaluate performance for complex queries
python3 eval_cqd.py --cuda --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_no_attr --data_path data/FB15k-237-q2b --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen
python3 eval_cqd.py --cuda --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_literale --data_path data/FB15k-237-q2b --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen
python3 eval_cqd.py --cuda --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_transea --data_path data/FB15k-237-q2b --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen
python3 eval_cqd.py --cuda --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn --data_path data/FB15k-237-q2b --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen
# Evaluate performance for complex attribute queries
python3 eval_cqd.py --cuda --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen

# The results for the different baselines can be achieved by uncommenting the relevant parts in the score_attribute_restriction method of CQDBaseModel.py or setting no_filter_scores/no_exists_scores to True in _score_attribute_restriction.
# The runtime may be evaluated by uncommenting the two lines at the end and beginng of the forward method of CQDBaseModel.py

# Reproducing the results for the example query containing an attribute filtering expression
python3 eval_attribute_filtering_example.py

# Evaluate predicted attribute values for each attribute
python3 eval_attribute_value_predictions.py --cuda  --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen --cqd_type discrete --cqd_t_norm prod

# Evaluate the models utilzing descriptions (Table 5.31):
python3 eval_cqd.py --cuda --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_no_attr_desc --data_path data/FB15k-237-q2b --rank 1000 --geo cqd-complexd --test_batch_size 1024 --print_on_screen
python3 eval_cqd.py --cuda --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_no_attr_desc_trained --data_path data/FB15k-237-q2b --rank 1000 --geo cqd-complexd --test_batch_size 1024 --print_on_screen
python3 eval_cqd.py --cuda --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn_desc --data_path data/FB15k-237-q2b --rank 1000 --geo cqd-complexad --test_batch_size 1024 --print_on_screen

# Evalute queries including descriptions
python3 eval_cqd.py --cuda --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_no_attr_desc --data_path data/scripts/generated/FB15K-237_kblrn_desc --use_descriptions --rank 1000 --geo cqd-complexd --test_batch_size 1024 --print_on_screen

# Reproducing the results fot the example query containing a desription filtering expression
python3 eval_description_embeddings_example.py



# (litcqd) renzhong@litcqd:~/LitCQD$ ./cqd_results.sh 
# Traceback (most recent call last):
#   File "eval_cqd.py", line 60, in <module>
#     main(parse_args())
#   File "eval_cqd.py", line 48, in main
#     model = get_model(train_config, params, cqd_params, *load_stats(train_config.data_path))
#   File "/home/renzhong/LitCQD/util_models.py", line 174, in get_model
#     model = model.cuda()
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 637, in cuda
#     return self._apply(lambda t: t.cuda(device))
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 530, in _apply
#     module._apply(fn)
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 552, in _apply
#     param_applied = fn(param)
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 637, in <lambda>
#     return self._apply(lambda t: t.cuda(device))
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/cuda/__init__.py", line 172, in _lazy_init
#     torch._C._cuda_init()
# RuntimeError: No HIP GPUs are available
# Traceback (most recent call last):
#   File "eval_cqd.py", line 60, in <module>
#     main(parse_args())
#   File "eval_cqd.py", line 48, in main
#     model = get_model(train_config, params, cqd_params, *load_stats(train_config.data_path))
#   File "/home/renzhong/LitCQD/util_models.py", line 174, in get_model
#     model = model.cuda()
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 637, in cuda
#     return self._apply(lambda t: t.cuda(device))
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 530, in _apply
#     module._apply(fn)
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 552, in _apply
#     param_applied = fn(param)
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 637, in <lambda>
#     return self._apply(lambda t: t.cuda(device))
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/cuda/__init__.py", line 172, in _lazy_init
#     torch._C._cuda_init()
# RuntimeError: No HIP GPUs are available
# Traceback (most recent call last):
#   File "eval_cqd.py", line 60, in <module>
#     main(parse_args())
#   File "eval_cqd.py", line 48, in main
#     model = get_model(train_config, params, cqd_params, *load_stats(train_config.data_path))
#   File "/home/renzhong/LitCQD/util_models.py", line 174, in get_model
#     model = model.cuda()
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 637, in cuda
#     return self._apply(lambda t: t.cuda(device))
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 530, in _apply
#     module._apply(fn)
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 552, in _apply
#     param_applied = fn(param)
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 637, in <lambda>
#     return self._apply(lambda t: t.cuda(device))
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/cuda/__init__.py", line 172, in _lazy_init
#     torch._C._cuda_init()
# RuntimeError: No HIP GPUs are available
# Traceback (most recent call last):
#   File "eval_cqd.py", line 60, in <module>
#     main(parse_args())
#   File "eval_cqd.py", line 48, in main
#     model = get_model(train_config, params, cqd_params, *load_stats(train_config.data_path))
#   File "/home/renzhong/LitCQD/util_models.py", line 174, in get_model
#     model = model.cuda()
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 637, in cuda
#     return self._apply(lambda t: t.cuda(device))
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 530, in _apply
#     module._apply(fn)
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 552, in _apply
#     param_applied = fn(param)
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 637, in <lambda>
#     return self._apply(lambda t: t.cuda(device))
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/cuda/__init__.py", line 172, in _lazy_init
#     torch._C._cuda_init()
# RuntimeError: No HIP GPUs are available
# Traceback (most recent call last):
#   File "eval_cqd.py", line 60, in <module>
#     main(parse_args())
#   File "eval_cqd.py", line 48, in main
#     model = get_model(train_config, params, cqd_params, *load_stats(train_config.data_path))
#   File "/home/renzhong/LitCQD/util_models.py", line 174, in get_model
#     model = model.cuda()
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 637, in cuda
#     return self._apply(lambda t: t.cuda(device))
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 530, in _apply
#     module._apply(fn)
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 552, in _apply
#     param_applied = fn(param)
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 637, in <lambda>
#     return self._apply(lambda t: t.cuda(device))
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/cuda/__init__.py", line 172, in _lazy_init
#     torch._C._cuda_init()
# RuntimeError: No HIP GPUs are available
# 2023-03-08 20:08:42 INFO     train: 1p: 173033
# 2023-03-08 20:08:42 INFO     train: 1ap: 23229
# 2023-03-08 20:08:46 INFO     valid: 1p: 20101
# 2023-03-08 20:08:46 INFO     valid: 1ap: 3000
# 2023-03-08 20:08:56 INFO     test: 1p: 22812
# 2023-03-08 20:08:56 INFO     test: 1ap: 3000
# Traceback (most recent call last):
#   File "eval_attribute_filtering_example.py", line 29, in <module>
#     load_model(model, checkpoint_path, train_config.cuda)
#   File "/home/renzhong/LitCQD/util_models.py", line 206, in load_model
#     model.load_state_dict(model_state)
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1406, in load_state_dict
#     raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
# RuntimeError: Error(s) in loading state_dict for CQDComplExAD:
#         size mismatch for description_net.weight: copying a param with shape torch.Size([300, 2000]) from checkpoint, the shape in current model is torch.Size([3, 2000]).
#         size mismatch for description_net.bias: copying a param with shape torch.Size([300]) from checkpoint, the shape in current model is torch.Size([3]).
# Traceback (most recent call last):
#   File "eval_attribute_value_predictions.py", line 56, in <module>
#     main(parse_args())
#   File "eval_attribute_value_predictions.py", line 50, in main
#     model = get_model(train_config, params, cqd_params, *load_stats(train_config.data_path))
#   File "/home/renzhong/LitCQD/util_models.py", line 174, in get_model
#     model = model.cuda()
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 637, in cuda
#     return self._apply(lambda t: t.cuda(device))
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 530, in _apply
#     module._apply(fn)
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 552, in _apply
#     param_applied = fn(param)
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 637, in <lambda>
#     return self._apply(lambda t: t.cuda(device))
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/cuda/__init__.py", line 172, in _lazy_init
#     torch._C._cuda_init()
# RuntimeError: No HIP GPUs are available
# Traceback (most recent call last):
#   File "eval_cqd.py", line 60, in <module>
#     main(parse_args())
#   File "eval_cqd.py", line 48, in main
#     model = get_model(train_config, params, cqd_params, *load_stats(train_config.data_path))
#   File "/home/renzhong/LitCQD/util_models.py", line 174, in get_model
#     model = model.cuda()
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 637, in cuda
#     return self._apply(lambda t: t.cuda(device))
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 530, in _apply
#     module._apply(fn)
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 552, in _apply
#     param_applied = fn(param)
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 637, in <lambda>
#     return self._apply(lambda t: t.cuda(device))
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/cuda/__init__.py", line 172, in _lazy_init
#     torch._C._cuda_init()
# RuntimeError: No HIP GPUs are available
# Traceback (most recent call last):
#   File "eval_cqd.py", line 60, in <module>
#     main(parse_args())
#   File "eval_cqd.py", line 48, in main
#     model = get_model(train_config, params, cqd_params, *load_stats(train_config.data_path))
#   File "/home/renzhong/LitCQD/util_models.py", line 174, in get_model
#     model = model.cuda()
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 637, in cuda
#     return self._apply(lambda t: t.cuda(device))
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 530, in _apply
#     module._apply(fn)
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 552, in _apply
#     param_applied = fn(param)
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 637, in <lambda>
#     return self._apply(lambda t: t.cuda(device))
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/cuda/__init__.py", line 172, in _lazy_init
#     torch._C._cuda_init()
# RuntimeError: No HIP GPUs are available
# Traceback (most recent call last):
#   File "eval_cqd.py", line 60, in <module>
#     main(parse_args())
#   File "eval_cqd.py", line 48, in main
#     model = get_model(train_config, params, cqd_params, *load_stats(train_config.data_path))
#   File "/home/renzhong/LitCQD/util_models.py", line 174, in get_model
#     model = model.cuda()
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 637, in cuda
#     return self._apply(lambda t: t.cuda(device))
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 530, in _apply
#     module._apply(fn)
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 552, in _apply
#     param_applied = fn(param)
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 637, in <lambda>
#     return self._apply(lambda t: t.cuda(device))
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/cuda/__init__.py", line 172, in _lazy_init
#     torch._C._cuda_init()
# RuntimeError: No HIP GPUs are available
# Traceback (most recent call last):
#   File "eval_cqd.py", line 60, in <module>
#     main(parse_args())
#   File "eval_cqd.py", line 48, in main
#     model = get_model(train_config, params, cqd_params, *load_stats(train_config.data_path))
#   File "/home/renzhong/LitCQD/util_models.py", line 174, in get_model
#     model = model.cuda()
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 637, in cuda
#     return self._apply(lambda t: t.cuda(device))
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 530, in _apply
#     module._apply(fn)
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 552, in _apply
#     param_applied = fn(param)
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 637, in <lambda>
#     return self._apply(lambda t: t.cuda(device))
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/cuda/__init__.py", line 172, in _lazy_init
#     torch._C._cuda_init()
# RuntimeError: No HIP GPUs are available
# Traceback (most recent call last):
#   File "eval_description_embeddings_example.py", line 26, in <module>
#     load_model(model, checkpoint_path, train_config.cuda)
#   File "/home/renzhong/LitCQD/util_models.py", line 206, in load_model
#     model.load_state_dict(model_state)
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1406, in load_state_dict
#     raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
# RuntimeError: Error(s) in loading state_dict for CQDComplExAD:
#         size mismatch for description_net.weight: copying a param with shape torch.Size([300, 2000]) from checkpoint, the shape in current model is torch.Size([3, 2000]).
#         size mismatch for description_net.bias: copying a param with shape torch.Size([300]) from checkpoint, the shape in current model is torch.Size([3]).