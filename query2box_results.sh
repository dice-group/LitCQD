# Evaluate box sizes
# python3 eval_q2b_box_sizes.py --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn_q2b --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 400 --geo q2b --test_batch_size 100 --print_on_screen

# (litcqd) renzhong@litcqd:~/LitCQD$ ./query2box_results.sh 
# id      Box size        Mean                    MAD                             MAE                             count
# 102     2425.04810      0.009945867869614907    0.009384399356686115            0.001754575344952599            26
# 95      25.49397        0.968500226019753607    0.035066834951511777            0.012004767475043455            35
# 67      8232.63770      0.002807417382245438    0.005109476441458810            0.002119988745126038            258
# 44      22.82104        0.800509173447539135    0.146324880393196438            0.082095356582270060            204
# 65      50.35764        0.758737134397224566    0.067603222514195510            0.038857924590402278            328
# 66      50.19565        0.391950301104341037    0.170324870300690129            0.106489373983895949            316
# 101     7504.41699      0.001909407559453922    0.003073877646567583            0.002164861922205476            204
# 75      89.64756        0.869215686274509847    0.054386774317744406            0.039568665985702030            30
# 63      21.73389        0.947918446086584043    0.037214237378787533            0.028279850150392322            87
# 105     1072.33191      0.049687480016010879    0.075025577246085579            0.057383300750900687            32
# 83      94.67579        0.974880328158290621    0.012499264639729793            0.009864764082013517            407
# 12      27.89830        0.713239247311827973    0.153791623309053188            0.122755544053183716            24
# 103     33.01576        0.714880373666432645    0.181040964478098171            0.147364145326799117            32
# 85      74.59190        0.332658272394535603    0.131174846866843026            0.107284354158975567            20
# 79      28.05922        0.870606806783493337    0.077302010369565868            0.063894204980095418            124
# 104     184.89937       0.080965701374114935    0.061024866187841705            0.051750044575285967            36
# 92      68.54289        0.697937308216904584    0.116846485162309374            0.104227298710138833            88
# 60      36.42636        0.330340579259903622    0.250966974068104642            0.228820175977454326            24
# 82      23.02396        0.977142676848678393    0.022298289208545333            0.020442916039038358            102
# 84      75.33191        0.497450722348949936    0.097893341517829219            0.093223308384027856            307
# 11      23.22722        0.667085714285714282    0.197992228571428536            0.204851033744357841            30
# 61      33.54718        0.479567207657095318    0.245938718577087351            0.266256252905067881            24
# 94      24.99562        0.969249438936926055    0.033033411444057421            0.037169108365826736            27
# 106     91.50652        0.114420803782505909    0.089090086011770098            0.109463619536314269            26
# 7       71.51011        0.862255238213472830    0.067037708057993320            0.085798367160574249            48
# Average attribute box size: 254.33885770051376
# Average relation box size: 214.76249259168452
# Max relation box size: 402.7469177246094
# Min relation box size: 71.48856353759766


# Eval performance on complex queries
python3 main.py --cuda --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_literale_q2b --data_path data/FB15k-237-q2b --rank 400 --geo q2b --test_batch_size 10 --print_on_screen
python3 main.py --cuda --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_transea_q2b --data_path data/FB15k-237-q2b --rank 400 --geo q2b --test_batch_size 10 --print_on_screen
python3 main.py --cuda --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn_q2b --data_path data/FB15k-237-q2b --rank 400 --geo q2b --test_batch_size 10 --print_on_screen

# Eval performance on complex attribute quries
python3 main.py --cuda --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn_q2b --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 400 --geo q2b --test_batch_size 10 --print_on_screen


# (litcqd) renzhong@litcqd:~/LitCQD$ ./query2box_results.sh 
# logging to Experiments
# Loading queries for the training...
# 2023-03-08 20:06:58,596 INFO     train: 1p: 149689
# Loading queries for the valid...
# 2023-03-08 20:07:02,442 INFO     valid: 1p: 20101
# 2023-03-08 20:07:06,540 INFO     valid: 1p: 20101
# 2023-03-08 20:07:06,541 INFO     valid: 1dp: 0
# 2023-03-08 20:07:06,541 INFO     valid: di: 0
# 2023-03-08 20:07:06,551 INFO     Training starts...
# Traceback (most recent call last):
#   File "main.py", line 654, in <module>
#     main(parse_args())
#   File "main.py", line 618, in main
#     new_train(train_config,
#   File "main.py", line 489, in new_train
#     model = get_model(train_config, params, cqd_params, nentity, nrelation, nattribute)
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
# logging to Experiments
# Loading queries for the training...
# 2023-03-08 20:07:40,640 INFO     train: 1p: 149689
# Loading queries for the valid...
# 2023-03-08 20:07:44,540 INFO     valid: 1p: 20101
# 2023-03-08 20:07:48,685 INFO     valid: 1p: 20101
# 2023-03-08 20:07:48,686 INFO     valid: 1dp: 0
# 2023-03-08 20:07:48,686 INFO     valid: di: 0
# 2023-03-08 20:07:48,697 INFO     Training starts...
# Traceback (most recent call last):
#   File "main.py", line 654, in <module>
#     main(parse_args())
#   File "main.py", line 618, in main
#     new_train(train_config,
#   File "main.py", line 489, in new_train
#     model = get_model(train_config, params, cqd_params, nentity, nrelation, nattribute)
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
# logging to Experiments
# Loading queries for the training...
# 2023-03-08 20:08:23,027 INFO     train: 1p: 149689
# Loading queries for the valid...
# 2023-03-08 20:08:26,976 INFO     valid: 1p: 20101
# 2023-03-08 20:08:31,195 INFO     valid: 1p: 20101
# 2023-03-08 20:08:31,196 INFO     valid: 1dp: 0
# 2023-03-08 20:08:31,196 INFO     valid: di: 0
# 2023-03-08 20:08:31,207 INFO     Training starts...
# Traceback (most recent call last):
#   File "main.py", line 654, in <module>
#     main(parse_args())
#   File "main.py", line 618, in main
#     new_train(train_config,
#   File "main.py", line 489, in new_train
#     model = get_model(train_config, params, cqd_params, nentity, nrelation, nattribute)
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
# logging to Experiments
# Loading queries for the training...
# 2023-03-08 20:09:33,229 INFO     train: 1p: 173033
# 2023-03-08 20:09:33,229 INFO     train: 1ap: 23229
# Loading queries for the valid...
# 2023-03-08 20:09:43,821 INFO     valid: 1p: 20101
# 2023-03-08 20:09:43,822 INFO     valid: 1ap: 3000
# 2023-03-08 20:09:47,649 INFO     valid: 1p: 20101
# 2023-03-08 20:09:47,649 INFO     valid: 1ap: 3000
# 2023-03-08 20:09:47,650 WARNING  valid: 1dp: not in pkl file
# 2023-03-08 20:09:47,650 WARNING  valid: di: not in pkl file
# 2023-03-08 20:09:47,667 INFO     Training starts...
# 2023-03-08 20:09:48,130 INFO     attribute batch size: 2
# Traceback (most recent call last):
#   File "main.py", line 654, in <module>
#     main(parse_args())
#   File "main.py", line 618, in main
#     new_train(train_config,
#   File "main.py", line 489, in new_train
#     model = get_model(train_config, params, cqd_params, nentity, nrelation, nattribute)
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




# python3 main.py --cuda --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_no_attr_q2b/ --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 400 --geo q2b --test_batch_size 10 --print_on_screen


python3 main.py --cuda --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn_q2b/ --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 400 --geo q2b --test_batch_size 10 --print_on_screen