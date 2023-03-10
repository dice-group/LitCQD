# Results for TransE with and without facts representing the extence of an attribute:
# python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transe --data_path data/scripts/generated/LitWD1K --rank 1024 --p_norm 2 --geo cqd-transea --print_on_screen


# (litcqd) renzhong@litcqd:~/LitCQD$ ./attribute_prediction_results.sh 
# logging to Experiments
# Loading queries for the training...
# 2023-03-08 19:33:55,379 INFO     train: 1p: 9030
# Loading queries for the valid...
# 2023-03-08 19:33:56,612 INFO     valid: 1p: 1951
# 2023-03-08 19:33:57,759 INFO     valid: 1p: 1951
# 2023-03-08 19:33:57,759 WARNING  valid: 1dp: not in pkl file
# 2023-03-08 19:33:57,760 WARNING  valid: di: not in pkl file
# 2023-03-08 19:33:57,760 INFO     Training starts...
# /home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/utils/data/dataloader.py:478: UserWarning: This DataLoader will create 10 worker processes in total. Our suggested max number of worker in current system is 8, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
#   warnings.warn(_create_warning_msg(
#   0%|                                                                                                                 | 0/2 [00:00<?, ?it/s]./attribute_prediction_results.sh: line 2: 1742955 Aborted                 (core dumped) python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transe --data_path data/scripts/generated/LitWD1K --rank 1024 --p_norm 2 --geo cqd-transea --print_on_screen






# python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transe_dummy --data_path data/scripts/generated/LitWD1K_dummy --rank 1024 --p_norm 2 --geo cqd-transea --print_on_screen

# # TransEA
# python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_best_mae --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --rank 128 --p_norm 1 --geo cqd-transea --print_on_screen
# python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_best_mrr --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --rank 1024 --p_norm 2 --geo cqd-transea --print_on_screen

# # TransEA + Sigmoid
# python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_sigmoid_best_mae --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --do_sigmoid --rank 1024 --p_norm 2 --geo cqd-transea --print_on_screen
# python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_sigmoid_best_mrr --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --do_sigmoid --rank 1024 --p_norm 2 --geo cqd-transea --print_on_screen

# # TransEA + TransR
# python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_transr_best_mae --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --rank 256 --rank_attr 20 --p_norm 2 --geo cqd-transra --print_on_screen
# python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_transr_best_mrr --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --rank 1024 --rank_attr 50 --p_norm 2 --geo cqd-transra --print_on_screen

# # TransE + MTKGNN
# python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transe_mtkgnn_best_mae --data_path data/scripts/generated/LitWD1K_dummy --rank 1024 --p_norm 2 --use_attributes --geo cqd-mtkgnn --print_on_screen
# python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transe_mtkgnn_best_mrr --data_path data/scripts/generated/LitWD1K_dummy --rank 1024 --p_norm 2 --use_attributes --geo cqd-mtkgnn --print_on_screen

# # TransEA + ComplEx (Mean)
# python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_complex_mean_best_mae --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --do_sigmoid --rank 1024 --geo cqd-transeacomplex --print_on_screen
# python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_complex_mean_best_mrr --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --do_sigmoid --rank 1024 --geo cqd-transeacomplex --print_on_screen

# # TransEA + ComplEx (Modulus)
# python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_complex_modulus_best_mae --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --do_sigmoid --use_modulus --rank 1024 --geo cqd-transeacomplex --print_on_screen
# python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_complex_modulus_best_mrr --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --do_sigmoid --use_modulus --rank 1024 --geo cqd-transeacomplex --print_on_screen

# # TransEA + TransComplEx
# python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transcomplex_best_mae --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --do_sigmoid --rank 1024 --p_norm 2 --geo cqd-transcomplexa --print_on_screen
# python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transcomplex_best_mrr --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --do_sigmoid --rank 1024 --p_norm 2 --geo cqd-transcomplexa --print_on_screen

# # TransEA + DistMult
# python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_distmult_best_mae --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --do_sigmoid --rank 256  --geo cqd-transeadistmult --print_on_screen
# python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_distmult_best_mrr --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --do_sigmoid --rank 512  --geo cqd-transeadistmult --print_on_screen

# Evaluate average attribute prediction value of TransEA for different dimensions:
python3 eval_mean_attribute_value_prediction.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_best_mrr --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --rank 1024 --p_norm 2 --geo cqd-transea --print_on_screen
python3 eval_mean_attribute_value_prediction.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_best_mrr_rank_512 --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --rank 512 --p_norm 2 --geo cqd-transea --print_on_screen
python3 eval_mean_attribute_value_prediction.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_best_mrr_rank_256 --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --rank 256 --p_norm 2 --geo cqd-transea --print_on_screen
python3 eval_mean_attribute_value_prediction.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_best_mrr_rank_128 --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --rank 128 --p_norm 1 --geo cqd-transea --print_on_screen
# And their performance when answering 1p and 1ap queries:
# python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_best_mrr --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --rank 1024 --p_norm 2 --geo cqd-transea --print_on_screen
# python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_best_mrr_rank_512 --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --rank 512 --p_norm 2 --geo cqd-transea --print_on_screen
# python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_best_mrr_rank_256 --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --rank 256 --p_norm 2 --geo cqd-transea --print_on_screen
# python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_best_mrr_rank_128 --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --rank 128 --p_norm 1 --geo cqd-transea --print_on_screen
