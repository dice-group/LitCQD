
# train the model
python3 main.py --cuda --do_train --do_test --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes -n 0 --rank 1000 -lr 0.1 --attr_loss mae --alpha 0.5 --geo cqd-complexa --batch_size 1024 --test_batch_size 100 --train_times 100 --valid_epochs 10 --print_on_screen


# results
# table 2
python3 eval_cqd.py --cuda --do_test --checkpoint_path checkpoints_FB15K-237/submitted_paper --data_path data/FB15k-237-q2b --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen
# table 3, 4
python3 eval_cqd.py --cuda --do_test --checkpoint_path checkpoints_FB15K-237/submitted_paper --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen

# to perform the ablation study of table 3, uncomment the relevant parts in the score_attribute_restriction method of CQDBaseModel.py or setting no_filter_scores/no_exists_scores to True in _score_attribute_restriction.


python3 main.py --cuda --do_train --do_test --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes -n 0 --rank 1000 -lr 0.1 --attr_loss mae --alpha 0.5 --geo cqd-complexa --batch_size 1024 --test_batch_size 100 --train_times 100 --valid_epochs 10 --print_on_screen


# experiments
python3 eval_cqd.py --do_test --checkpoint_path checkpoints_FB15K-237/submitted_paper --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen

python3 eval_cqd.py --do_test --checkpoint_path checkpoints_FB15K-237/submitted_paper --data_path data/scripts/generated/FB15K-237_dummy_kblrn_std_demo2 --use_attributes --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen
