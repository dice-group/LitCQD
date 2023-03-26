#!/bin/bash

# Table 2
# CQD
python3 ../eval_cqd.py --cuda --checkpoint_path check_point/FB15k-237-q2b/cqd-complexa/2023.03.24-15:38:21/ --data_path ../data/FB15k-237-q2b --rank 1000 --test_batch_size 1024 --print_on_screen --train_times 100 --geo cqd-complexa --to_latex 2>&1 | tee FB15k-237-q2b.log

#CQDLite
python3 ../eval_cqd.py --cuda --checkpoint_path check_point/FB15K-237_dummy_transea/cqd-complexa/2023.03.19-17:17:40/ --data_path ../data/FB15k-237-q2b --rank 1000 --test_batch_size 1024 --print_on_screen --train_times 100 --geo cqd-complexa --to_latex 2>&1 | tee FB15K-237_dummy_transea.log

python3 ../eval_cqd.py --cuda --checkpoint_path check_point/FB15K-237_dummy_literale/cqd-complexa/2023.03.19-16:33:01/ --data_path ../data/FB15k-237-q2b --rank 1000 --test_batch_size 1024 --print_on_screen --train_times 100 --geo cqd-complexa --to_latex 2>&1 | tee FB15K-237_dummy_literale.log

python3 ../eval_cqd.py --cuda --checkpoint_path check_point/FB15K-237_dummy_kblrn/cqd-complexa/2023.03.24-16:13:24/ --data_path ../data/FB15k-237-q2b --rank 1000 --test_batch_size 1024 --print_on_screen --train_times 100 --geo cqd-complexa --to_latex 2>&1 | tee FB15K-237_dummy_kblrn.log

# Query2Box with attribute (kblrn)
python3 ../main.py --cuda --do_test --checkpoint_path check_point/FB15K-237_dummy_kblrn/q2b/2023.03.24-23:43:42/ --data_path ../data/FB15k-237-q2b --rank 400 --geo q2b --test_batch_size 10 --print_on_screen --to_latex


origin


# Table 3, 4(not support to create latext table)

# Query2Box
python3 ../main.py --cuda --do_test --checkpoint_path check_point/FB15K-237_dummy_kblrn/q2b/2023.03.24-23:43:42/ --data_path ../data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 400 --geo q2b --test_batch_size 10 --print_on_screen



# table 4 (use the model with alpha 0.5; model with alpha 0.5 doesnt work for queries invloving attribute filtering)
python3 ../eval_cqd.py --do_test --checkpoint_path check_point/old_trained/FB15K-237_dummy_kblrn/2023.03.19-18:04:34/ --data_path ../data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen


# litcqd
python3 ../eval_cqd.py --cuda --do_test --checkpoint_path check_point/old_trained/FB15K-237_dummy_kblrn/2023.03.19-18:04:34/ --data_path ../data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen

