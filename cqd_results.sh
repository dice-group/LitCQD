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
