# Results for TransE with and without facts representing the extence of an attribute:
python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transe --data_path data/scripts/generated/LitWD1K --rank 1024 --p_norm 2 --geo cqd-transea --print_on_screen
python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transe_dummy --data_path data/scripts/generated/LitWD1K_dummy --rank 1024 --p_norm 2 --geo cqd-transea --print_on_screen

# TransEA
python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_best_mae --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --rank 128 --p_norm 1 --geo cqd-transea --print_on_screen
python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_best_mrr --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --rank 1024 --p_norm 2 --geo cqd-transea --print_on_screen

# TransEA + Sigmoid
python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_sigmoid_best_mae --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --do_sigmoid --rank 1024 --p_norm 2 --geo cqd-transea --print_on_screen
python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_sigmoid_best_mrr --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --do_sigmoid --rank 1024 --p_norm 2 --geo cqd-transea --print_on_screen

# TransEA + TransR
python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_transr_best_mae --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --rank 256 --rank_attr 20 --p_norm 2 --geo cqd-transra --print_on_screen
python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_transr_best_mrr --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --rank 1024 --rank_attr 50 --p_norm 2 --geo cqd-transra --print_on_screen

# TransE + MTKGNN
python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transe_mtkgnn_best_mae --data_path data/scripts/generated/LitWD1K_dummy --rank 1024 --p_norm 2 --use_attributes --geo cqd-mtkgnn --print_on_screen
python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transe_mtkgnn_best_mrr --data_path data/scripts/generated/LitWD1K_dummy --rank 1024 --p_norm 2 --use_attributes --geo cqd-mtkgnn --print_on_screen

# TransEA + ComplEx (Mean)
python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_complex_mean_best_mae --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --do_sigmoid --rank 1024 --geo cqd-transeacomplex --print_on_screen
python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_complex_mean_best_mrr --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --do_sigmoid --rank 1024 --geo cqd-transeacomplex --print_on_screen

# TransEA + ComplEx (Modulus)
python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_complex_modulus_best_mae --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --do_sigmoid --use_modulus --rank 1024 --geo cqd-transeacomplex --print_on_screen
python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_complex_modulus_best_mrr --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --do_sigmoid --use_modulus --rank 1024 --geo cqd-transeacomplex --print_on_screen

# TransEA + TransComplEx
python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transcomplex_best_mae --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --do_sigmoid --rank 1024 --p_norm 2 --geo cqd-transcomplexa --print_on_screen
python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transcomplex_best_mrr --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --do_sigmoid --rank 1024 --p_norm 2 --geo cqd-transcomplexa --print_on_screen

# TransEA + DistMult
python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_distmult_best_mae --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --do_sigmoid --rank 256  --geo cqd-transeadistmult --print_on_screen
python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_distmult_best_mrr --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --do_sigmoid --rank 512  --geo cqd-transeadistmult --print_on_screen

# Evaluate average attribute prediction value of TransEA for different dimensions:
python3 eval_mean_attribute_value_prediction.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_best_mrr --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --rank 1024 --p_norm 2 --geo cqd-transea --print_on_screen
python3 eval_mean_attribute_value_prediction.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_best_mrr_rank_512 --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --rank 512 --p_norm 2 --geo cqd-transea --print_on_screen
python3 eval_mean_attribute_value_prediction.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_best_mrr_rank_256 --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --rank 256 --p_norm 2 --geo cqd-transea --print_on_screen
python3 eval_mean_attribute_value_prediction.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_best_mrr_rank_128 --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --rank 128 --p_norm 1 --geo cqd-transea --print_on_screen
# And their performance when answering 1p and 1ap queries:
python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_best_mrr --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --rank 1024 --p_norm 2 --geo cqd-transea --print_on_screen
python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_best_mrr_rank_512 --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --rank 512 --p_norm 2 --geo cqd-transea --print_on_screen
python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_best_mrr_rank_256 --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --rank 256 --p_norm 2 --geo cqd-transea --print_on_screen
python3 main.py --do_test --simple_eval --checkpoint_path checkpoints_litwd1k/checkpoint_transea_best_mrr_rank_128 --data_path data/scripts/generated/LitWD1K_dummy --use_attributes --rank 128 --p_norm 1 --geo cqd-transea --print_on_screen
