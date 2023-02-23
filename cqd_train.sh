# Train the CDD model without attributes
python3 main.py --cuda --do_train --do_test --data_path data/FB15k-237-q2b -n 0 --rank 1000 -lr 0.1 --geo cqd-complexa --batch_size 1024 --test_batch_size 100 --train_times 100 --valid_epochs 10 --print_on_screen

# Train the CQD-based approach using attributes from the different datasets
python3 main.py --cuda --do_train --do_test --data_path data/scripts/generated/FB15K-237_dummy_literale --use_attributes -n 0 --rank 1000 -lr 0.1 --attr_loss mae --alpha 0.5 --geo cqd-complexa --batch_size 1024 --test_batch_size 100 --train_times 100 --valid_epochs 10 --print_on_screen
python3 main.py --cuda --do_train --do_test --data_path data/scripts/generated/FB15K-237_dummy_transea --use_attributes -n 0 --rank 1000 -lr 0.1 --attr_loss mae --alpha 0.5 --geo cqd-complexa --batch_size 1024 --test_batch_size 100 --train_times 100 --valid_epochs 10 --print_on_screen
python3 main.py --cuda --do_train --do_test --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes -n 0 --rank 1000 -lr 0.1 --attr_loss mae --alpha 0.5 --geo cqd-complexa --batch_size 1024 --test_batch_size 100 --train_times 100 --valid_epochs 10 --print_on_screen

# Train the CQD-based approach using desription embeddings
# using word embeddings based on Google News
python3 main.py --cuda --do_train --do_test --desc_emb 1-layer --word_emb_dim 300 --data_path data/scripts/generated/FB15K-237_kblrn_desc --use_descriptions -n 0 --rank 1000 -lr 0.1 --alpha 0.5 --geo cqd-complexd --batch_size 1024 --test_batch_size 1024 --train_times 100 --valid_epochs 5 --print_on_screen
# or self-trained word embeddings
python3 main.py --cuda --do_train --do_test --desc_emb 1-layer --word_emb_dim 100 --data_path data/scripts/generated/FB15K-237_kblrn_desc_trained --use_descriptions -n 0 --rank 1000 -lr 0.1 --alpha 0.5 --geo cqd-complexd --batch_size 1024 --test_batch_size 1024 --train_times 100 --valid_epochs 5 --print_on_screen

# Train the CQD-based approach using both, attributes and descriptions
python3 main.py --cuda --do_train --do_test --desc_emb 1-layer --data_path data/scripts/generated/FB15K-237_dummy_kblrn_desc --use_descriptions --use_attributes -n 0 --rank 1000 -lr 0.1 --alpha 0.5 --geo cqd-complexad --batch_size 1024 --test_batch_size 1024 --train_times 100 --valid_epochs 10 --print_on_screen
