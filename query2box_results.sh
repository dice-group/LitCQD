# Evaluate box sizes
python3 eval_q2b_box_sizes.py --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn_q2b --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 400 --geo q2b --test_batch_size 100 --print_on_screen

# Eval performance on complex queries
python3 main.py --cuda --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_literale_q2b --data_path data/FB15k-237-q2b --rank 400 --geo q2b --test_batch_size 10 --print_on_screen
python3 main.py --cuda --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_transea_q2b --data_path data/FB15k-237-q2b --rank 400 --geo q2b --test_batch_size 10 --print_on_screen
python3 main.py --cuda --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn_q2b --data_path data/FB15k-237-q2b --rank 400 --geo q2b --test_batch_size 10 --print_on_screen

# Eval performance on complex attribute quries
python3 main.py --cuda --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn_q2b --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 400 --geo q2b --test_batch_size 10 --print_on_screen
