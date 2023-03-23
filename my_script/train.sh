#!/bin/bash

# training model with cqd-complexa without attributes
python3 ../main.py --cuda --do_train --data_path ../data/FB15k-237-q2b -n 0 --rank 1000 -lr 0.1 --geo cqd-complexa --batch_size 1024 --test_batch_size 100 --train_times 100 --valid_epochs 10 --print_on_screen

# currently no validation during the traning, so the valid_epochs can be deleted from command line paramters
# training the attribute model with cqd-complexa with different attribute datasets

python3 ../main.py --cuda --do_train --data_path ../data/scripts/generated/FB15K-237_dummy_literale --use_attributes -n 0 --rank 1000 -lr 0.1 --attr_loss mae --alpha 0.5 --geo cqd-complexa --batch_size 1024 --test_batch_size 100 --train_times 100 --valid_epochs 10 --print_on_screen

python3 ../main.py --cuda --do_train --data_path ../data/scripts/generated/FB15K-237_dummy_transea --use_attributes -n 0 --rank 1000 -lr 0.1 --attr_loss mae --alpha 0.5 --geo cqd-complexa --batch_size 1024 --test_batch_size 100 --train_times 100 --valid_epochs 10 --print_on_screen

python3 ../main.py --cuda --do_train --data_path ../data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes -n 0 --rank 1000 -lr 0.1 --attr_loss mae --alpha 0.5 --geo cqd-complexa --batch_size 1024 --test_batch_size 100 --train_times 100 --valid_epochs 10 --print_on_screen





