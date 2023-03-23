#!/bin/bash

# python3 ../eval_cqd.py --cuda --do_test --checkpoint_path Experiments --data_path ../data/FB15k-237-q2b --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen

# python3 ../eval_cqd.py --cuda --do_test --checkpoint_path ../checkpoints_FB15K-237/checkpoint_orig_attr_literale --data_path ../data/FB15k-237-q2b --rank 1000 --test_batch_size 1024 --print_on_screen --train_times 100 



python3 ../eval_cqd.py --cuda --checkpoint_path logs/FB15k-237-q2b/cqd-complexa/2023.03.19-15:57:49/ --data_path ../data/FB15k-237-q2b --rank 1000 --test_batch_size 1024 --print_on_screen --train_times 100 --geo cqd-complexa 2>&1 | tee FB15k-237-q2b.log

python3 ../eval_cqd.py --cuda --checkpoint_path logs/FB15K-237_dummy_transea/cqd-complexa/2023.03.19-17:17:40/ --data_path ../data/FB15k-237-q2b --rank 1000 --test_batch_size 1024 --print_on_screen --train_times 100 --geo cqd-complexa 2>&1 | tee FB15K-237_dummy_transea.log

python3 ../eval_cqd.py --cuda --checkpoint_path logs/FB15K-237_dummy_literale/cqd-complexa/2023.03.19-16:33:01/ --data_path ../data/FB15k-237-q2b --rank 1000 --test_batch_size 1024 --print_on_screen --train_times 100 --geo cqd-complexa 2>&1 | tee FB15K-237_dummy_literale.log

python3 ../eval_cqd.py --cuda --checkpoint_path logs/FB15K-237_dummy_kblrn/cqd-complexa/2023.03.19-18:04:34/ --data_path ../data/FB15k-237-q2b --rank 1000 --test_batch_size 1024 --print_on_screen --train_times 100 --geo cqd-complexa 2>&1 | tee FB15K-237_dummy_kblrn.log


# table 4
python3 ../eval_cqd.py --cuda --do_test --checkpoint_path logs/FB15K-237_dummy_kblrn/cqd-complexa/2023.03.19-18:04:34/ --data_path ../data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen

