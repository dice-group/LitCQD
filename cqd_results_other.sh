# Add embeddings to Tensorboard
python3 eval_tensorboard_add_embeddings.py --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 1000 --geo cqd-complexa

# Find best CQD params
python3 eval_find_best_cqd_params.py --cuda --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn --data_path data/FB15k-237-q2b --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen
python3 eval_find_best_cqd_params.py --cuda --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen

# Evaluate on Link Prediction Task
python3 eval_link_prediction.py --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn --data_path data/scripts/generated/FB15K-237_dummy_kblrn --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen

# Eval 1p performance for attribute exists facts
python3 eval_attribute_exists_facts.py --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen

# Eval 1p performance for each relation
python3 eval_relations.py --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn --data_path data/scripts/generated/FB15K-237_dummy_kblrn --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen
