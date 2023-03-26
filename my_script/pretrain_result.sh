
# Table 2

# CQD
python3 ../eval_cqd.py --cuda --do_test --to_latex --checkpoint_path ../checkpoints_FB15K-237/checkpoint_orig_no_attr --data_path ../data/FB15k-237-q2b --rank 1000 --geo cqd-complexa --test_batch_size 1024 --train_times 100 --print_on_screen #2>&1 | tee FB15k-237-q2b.log

# CQDLite
# python3 ../eval_cqd.py --cuda --do_test --checkpoint_path ../checkpoints_FB15K-237/checkpoint_orig_attr_literale --data_path ../data/FB15k-237-q2b --rank 1000 --geo cqd-complexa --test_batch_size 1024 --train_times 100 --print_on_screen 2>&1 | tee FB15K-237_dummy_literale.log

# python3 ../eval_cqd.py --cuda --do_test --checkpoint_path ../checkpoints_FB15K-237/checkpoint_orig_attr_transea --data_path ../data/FB15k-237-q2b --rank 1000 --geo cqd-complexa --test_batch_size 1024 --train_times 100 --print_on_screen 2>&1 | tee FB15K-237_dummy_transea.log

python3 ../eval_cqd.py --cuda --do_test --to_latex --checkpoint_path ../checkpoints_FB15K-237/checkpoint_orig_attr_kblrn --data_path ../data/FB15k-237-q2b --rank 1000 --geo cqd-complexa --test_batch_size 1024 --train_times 100 --print_on_screen #2>&1 | tee FB15K-237_dummy_kblrn.log



# query2box without attribute
python3 ../main.py --cuda --do_test --to_latex --checkpoint_path ../checkpoints_FB15K-237/checkpoint_orig_no_attr_q2b --data_path ../data/FB15k-237-q2b --rank 400 --geo q2b --test_batch_size 10 --print_on_screen
# best reault of quer2box with attribute and dataset kblrn
python3 ../main.py --cuda --do_test --to_latex --checkpoint_path ../checkpoints_FB15K-237/checkpoint_orig_attr_kblrn_q2b --data_path ../data/FB15k-237-q2b --rank 400 --geo q2b --test_batch_size 10 --print_on_screen







# Table 3, 4(not support to create latext table)

# litcqd used attribute
python3 ../eval_cqd.py --do_test --checkpoint_path ../checkpoints_FB15K-237/checkpoint_orig_attr_kblrn --data_path ../data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen

# query2box used attribute with kblrn 
python3 ../main.py --cuda --do_test --checkpoint_path ../checkpoints_FB15K-237/checkpoint_orig_attr_kblrn_q2b/ --data_path ../data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 400 --geo q2b --test_batch_size 10 --print_on_screen

