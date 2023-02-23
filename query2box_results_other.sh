# Eval 1p performance for each relation
python3 eval_relations.py --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn_q2b --data_path data/scripts/generated/FB15K-237_dummy_kblrn --rank 400 --geo q2b --test_batch_size 100 --print_on_screen

# Eval 1p performance for attribute exists facts
python3 eval_attribute_exists_facts.py --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn_q2b --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 400 --geo q2b --test_batch_size 100 --print_on_screen
