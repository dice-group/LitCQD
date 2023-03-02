# Evaluation script for LitCQD: Multi-Hop Reasoning in Knowledge Graphs with Literals
# Evaluate performance for complex queries
# To reproduce the results reported in Table 2.
# CQD
python3 eval_cqd.py --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_no_attr --data_path data/FB15k-237-q2b --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen

2023-02-22 08:54:35 INFO     test: 1p: 22812
2023-02-22 08:55:07 INFO     Test 1p MRR at epoch 100: 0.454033
2023-02-22 08:55:07 INFO     Test 1p HITS1 at epoch 100: 0.354408
2023-02-22 08:55:07 INFO     Test 1p HITS3 at epoch 100: 0.497937
2023-02-22 08:55:07 INFO     Test 1p HITS10 at epoch 100: 0.656162
2023-02-22 08:55:07 INFO     Test 1p num_queries at epoch 100: 22812.000000
2023-02-22 08:55:07 INFO     Test average MRR at epoch 100: 0.454033
2023-02-22 08:55:07 INFO     Test average HITS1 at epoch 100: 0.354408
2023-02-22 08:55:07 INFO     Test average HITS3 at epoch 100: 0.497937
2023-02-22 08:55:07 INFO     Test average HITS10 at epoch 100: 0.656162
2023-02-22 08:55:09 INFO     test: 2p: 5000
2023-02-22 08:57:37 INFO     Test 2p MRR at epoch 100: 0.275003
2023-02-22 08:57:37 INFO     Test 2p HITS1 at epoch 100: 0.198299
2023-02-22 08:57:37 INFO     Test 2p HITS3 at epoch 100: 0.297106
2023-02-22 08:57:37 INFO     Test 2p HITS10 at epoch 100: 0.422059
2023-02-22 08:57:37 INFO     Test 2p num_queries at epoch 100: 5000.000000
2023-02-22 08:57:37 INFO     Test average MRR at epoch 100: 0.275003
2023-02-22 08:57:37 INFO     Test average HITS1 at epoch 100: 0.198299
2023-02-22 08:57:37 INFO     Test average HITS3 at epoch 100: 0.297106
2023-02-22 08:57:37 INFO     Test average HITS10 at epoch 100: 0.422059
2023-02-22 08:57:39 INFO     test: 3p: 5000
2023-02-22 08:58:56 INFO     Test 3p MRR at epoch 100: 0.196629
2023-02-22 08:58:56 INFO     Test 3p HITS1 at epoch 100: 0.137420
2023-02-22 08:58:56 INFO     Test 3p HITS3 at epoch 100: 0.208355
2023-02-22 08:58:56 INFO     Test 3p HITS10 at epoch 100: 0.312464
2023-02-22 08:58:56 INFO     Test 3p num_queries at epoch 100: 5000.000000
2023-02-22 08:58:56 INFO     Test average MRR at epoch 100: 0.196629
2023-02-22 08:58:56 INFO     Test average HITS1 at epoch 100: 0.137420
2023-02-22 08:58:56 INFO     Test average HITS3 at epoch 100: 0.208355
2023-02-22 08:58:56 INFO     Test average HITS10 at epoch 100: 0.312464
2023-02-22 08:58:58 INFO     test: 2i: 5000
2023-02-22 09:00:37 INFO     Test 2i MRR at epoch 100: 0.338923
2023-02-22 09:00:37 INFO     Test 2i HITS1 at epoch 100: 0.234751
2023-02-22 09:00:37 INFO     Test 2i HITS3 at epoch 100: 0.379752
2023-02-22 09:00:37 INFO     Test 2i HITS10 at epoch 100: 0.550612
2023-02-22 09:00:37 INFO     Test 2i num_queries at epoch 100: 5000.000000
2023-02-22 09:00:37 INFO     Test average MRR at epoch 100: 0.338923
2023-02-22 09:00:37 INFO     Test average HITS1 at epoch 100: 0.234751
2023-02-22 09:00:37 INFO     Test average HITS3 at epoch 100: 0.379752
2023-02-22 09:00:37 INFO     Test average HITS10 at epoch 100: 0.550612
2023-02-22 09:00:39 INFO     test: 3i: 5000
2023-02-22 09:02:44 INFO     Test 3i MRR at epoch 100: 0.457353
2023-02-22 09:02:44 INFO     Test 3i HITS1 at epoch 100: 0.354467
2023-02-22 09:02:44 INFO     Test 3i HITS3 at epoch 100: 0.507940
2023-02-22 09:02:44 INFO     Test 3i HITS10 at epoch 100: 0.656467
2023-02-22 09:02:44 INFO     Test 3i num_queries at epoch 100: 5000.000000
2023-02-22 09:02:44 INFO     Test average MRR at epoch 100: 0.457353
2023-02-22 09:02:44 INFO     Test average HITS1 at epoch 100: 0.354467
2023-02-22 09:02:44 INFO     Test average HITS3 at epoch 100: 0.507940
2023-02-22 09:02:44 INFO     Test average HITS10 at epoch 100: 0.656467
2023-02-22 09:02:46 INFO     test: ip: 5000
2023-02-22 09:06:06 INFO     Test ip MRR at epoch 100: 0.188486
2023-02-22 09:06:06 INFO     Test ip HITS1 at epoch 100: 0.129657
2023-02-22 09:06:06 INFO     Test ip HITS3 at epoch 100: 0.194541
2023-02-22 09:06:06 INFO     Test ip HITS10 at epoch 100: 0.305378
2023-02-22 09:06:06 INFO     Test ip num_queries at epoch 100: 5000.000000
2023-02-22 09:06:06 INFO     Test average MRR at epoch 100: 0.188486
2023-02-22 09:06:06 INFO     Test average HITS1 at epoch 100: 0.129657
2023-02-22 09:06:06 INFO     Test average HITS3 at epoch 100: 0.194541
2023-02-22 09:06:06 INFO     Test average HITS10 at epoch 100: 0.305378
2023-02-22 09:06:08 INFO     test: pi: 5000
2023-02-22 09:09:27 INFO     Test pi MRR at epoch 100: 0.266995
2023-02-22 09:09:27 INFO     Test pi HITS1 at epoch 100: 0.185956
2023-02-22 09:09:27 INFO     Test pi HITS3 at epoch 100: 0.290269
2023-02-22 09:09:27 INFO     Test pi HITS10 at epoch 100: 0.425156
2023-02-22 09:09:27 INFO     Test pi num_queries at epoch 100: 5000.000000
2023-02-22 09:09:27 INFO     Test average MRR at epoch 100: 0.266995
2023-02-22 09:09:27 INFO     Test average HITS1 at epoch 100: 0.185956
2023-02-22 09:09:27 INFO     Test average HITS3 at epoch 100: 0.290269
2023-02-22 09:09:27 INFO     Test average HITS10 at epoch 100: 0.425156
2023-02-22 09:09:29 INFO     test: 2u: 5000
2023-02-22 09:11:05 INFO     Test 2u MRR at epoch 100: 0.260840
2023-02-22 09:11:05 INFO     Test 2u HITS1 at epoch 100: 0.164962
2023-02-22 09:11:05 INFO     Test 2u HITS3 at epoch 100: 0.287322
2023-02-22 09:11:05 INFO     Test 2u HITS10 at epoch 100: 0.464745
2023-02-22 09:11:05 INFO     Test 2u num_queries at epoch 100: 5000.000000
2023-02-22 09:11:05 INFO     Test average MRR at epoch 100: 0.260840
2023-02-22 09:11:05 INFO     Test average HITS1 at epoch 100: 0.164962
2023-02-22 09:11:05 INFO     Test average HITS3 at epoch 100: 0.287322
2023-02-22 09:11:05 INFO     Test average HITS10 at epoch 100: 0.464745
2023-02-22 09:11:07 INFO     test: up: 5000
2023-02-22 09:14:28 INFO     Test up MRR at epoch 100: 0.212770
2023-02-22 09:14:28 INFO     Test up HITS1 at epoch 100: 0.135727
2023-02-22 09:14:28 INFO     Test up HITS3 at epoch 100: 0.231448
2023-02-22 09:14:28 INFO     Test up HITS10 at epoch 100: 0.368384
2023-02-22 09:14:28 INFO     Test up num_queries at epoch 100: 5000.000000
2023-02-22 09:14:28 INFO     Test average MRR at epoch 100: 0.212770
2023-02-22 09:14:28 INFO     Test average HITS1 at epoch 100: 0.135727
2023-02-22 09:14:28 INFO     Test average HITS3 at epoch 100: 0.231448
2023-02-22 09:14:28 INFO     Test average HITS10 at epoch 100: 0.368384


# LitCQD
python3 eval_cqd.py --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn --data_path data/FB15k-237-q2b --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen
(litcqd) cdemir@raki:~/Multi-Hop-Reasoning-in-Knowledge-Graphs-with-Literals$ python3 eval_cqd.py --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen
2023-02-22 09:20:12 INFO     test: 1ap: 3000
2023-02-22 09:20:12 INFO     Test 1ap MAE at epoch 100: 0.049066
2023-02-22 09:20:12 INFO     Test 1ap MSE at epoch 100: 0.010391
2023-02-22 09:20:12 INFO     Test 1ap RMSE at epoch 100: 0.101938
2023-02-22 09:20:12 INFO     Test 1ap num_queries at epoch 100: 3000.000000
2023-02-22 09:20:12 INFO     Test average MAE at epoch 100: 0.049066
2023-02-22 09:20:12 INFO     Test average MSE at epoch 100: 0.010391
2023-02-22 09:20:12 INFO     Test average RMSE at epoch 100: 0.101938
2023-02-22 09:20:15 INFO     test: 2ap: 100
2023-02-22 09:20:15 INFO     Test 2ap MAE at epoch 100: 0.031695
2023-02-22 09:20:15 INFO     Test 2ap MSE at epoch 100: 0.003925
2023-02-22 09:20:15 INFO     Test 2ap RMSE at epoch 100: 0.062648
2023-02-22 09:20:15 INFO     Test 2ap num_queries at epoch 100: 100.000000
2023-02-22 09:20:15 INFO     Test average MAE at epoch 100: 0.031695
2023-02-22 09:20:15 INFO     Test average MSE at epoch 100: 0.003925
2023-02-22 09:20:15 INFO     Test average RMSE at epoch 100: 0.062648
2023-02-22 09:20:18 INFO     test: 3ap: 100
2023-02-22 09:20:20 INFO     Test 3ap MAE at epoch 100: 0.040821
2023-02-22 09:20:20 INFO     Test 3ap MSE at epoch 100: 0.006615
2023-02-22 09:20:20 INFO     Test 3ap RMSE at epoch 100: 0.081335
2023-02-22 09:20:20 INFO     Test 3ap num_queries at epoch 100: 100.000000
2023-02-22 09:20:20 INFO     Test average MAE at epoch 100: 0.040821
2023-02-22 09:20:20 INFO     Test average MSE at epoch 100: 0.006615
2023-02-22 09:20:20 INFO     Test average RMSE at epoch 100: 0.081335
2023-02-22 09:20:23 INFO     test: ai-lt: 66
2023-02-22 09:20:26 INFO     Test ai-lt MRR at epoch 100: 0.315298
2023-02-22 09:20:26 INFO     Test ai-lt HITS1 at epoch 100: 0.218398
2023-02-22 09:20:26 INFO     Test ai-lt HITS3 at epoch 100: 0.372291
2023-02-22 09:20:26 INFO     Test ai-lt HITS10 at epoch 100: 0.516620
2023-02-22 09:20:26 INFO     Test ai-lt num_queries at epoch 100: 66.000000
2023-02-22 09:20:26 INFO     Test average MRR at epoch 100: 0.315298
2023-02-22 09:20:26 INFO     Test average HITS1 at epoch 100: 0.218398
2023-02-22 09:20:26 INFO     Test average HITS3 at epoch 100: 0.372291
2023-02-22 09:20:26 INFO     Test average HITS10 at epoch 100: 0.516620
2023-02-22 09:20:29 INFO     test: ai-eq: 100
2023-02-22 09:20:34 INFO     Test ai-eq MRR at epoch 100: 0.190988
2023-02-22 09:20:34 INFO     Test ai-eq HITS1 at epoch 100: 0.100243
2023-02-22 09:20:34 INFO     Test ai-eq HITS3 at epoch 100: 0.228257
2023-02-22 09:20:34 INFO     Test ai-eq HITS10 at epoch 100: 0.352451
2023-02-22 09:20:34 INFO     Test ai-eq num_queries at epoch 100: 100.000000
2023-02-22 09:20:34 INFO     Test average MRR at epoch 100: 0.190988
2023-02-22 09:20:34 INFO     Test average HITS1 at epoch 100: 0.100243
2023-02-22 09:20:34 INFO     Test average HITS3 at epoch 100: 0.228257
2023-02-22 09:20:34 INFO     Test average HITS10 at epoch 100: 0.352451
2023-02-22 09:20:36 INFO     test: ai-gt: 64
2023-02-22 09:20:39 INFO     Test ai-gt MRR at epoch 100: 0.242383
2023-02-22 09:20:39 INFO     Test ai-gt HITS1 at epoch 100: 0.150710
2023-02-22 09:20:39 INFO     Test ai-gt HITS3 at epoch 100: 0.278370
2023-02-22 09:20:39 INFO     Test ai-gt HITS10 at epoch 100: 0.433298
2023-02-22 09:20:39 INFO     Test ai-gt num_queries at epoch 100: 64.000000
2023-02-22 09:20:39 INFO     Test average MRR at epoch 100: 0.242383
2023-02-22 09:20:39 INFO     Test average HITS1 at epoch 100: 0.150710
2023-02-22 09:20:39 INFO     Test average HITS3 at epoch 100: 0.278370
2023-02-22 09:20:39 INFO     Test average HITS10 at epoch 100: 0.433298
2023-02-22 09:20:42 INFO     test: 2ai: 100
2023-02-22 09:20:51 INFO     Test 2ai MRR at epoch 100: 0.227936
2023-02-22 09:20:51 INFO     Test 2ai HITS1 at epoch 100: 0.154531
2023-02-22 09:20:51 INFO     Test 2ai HITS3 at epoch 100: 0.250330
2023-02-22 09:20:51 INFO     Test 2ai HITS10 at epoch 100: 0.377042
2023-02-22 09:20:51 INFO     Test 2ai num_queries at epoch 100: 100.000000
2023-02-22 09:20:51 INFO     Test average MRR at epoch 100: 0.227936
2023-02-22 09:20:51 INFO     Test average HITS1 at epoch 100: 0.154531
2023-02-22 09:20:51 INFO     Test average HITS3 at epoch 100: 0.250330
2023-02-22 09:20:51 INFO     Test average HITS10 at epoch 100: 0.377042
2023-02-22 09:20:54 INFO     test: aip: 100
2023-02-22 09:21:02 INFO     Test aip MRR at epoch 100: 0.111507
2023-02-22 09:21:02 INFO     Test aip HITS1 at epoch 100: 0.077597
2023-02-22 09:21:02 INFO     Test aip HITS3 at epoch 100: 0.095817
2023-02-22 09:21:02 INFO     Test aip HITS10 at epoch 100: 0.200431
2023-02-22 09:21:02 INFO     Test aip num_queries at epoch 100: 100.000000
2023-02-22 09:21:02 INFO     Test average MRR at epoch 100: 0.111507
2023-02-22 09:21:02 INFO     Test average HITS1 at epoch 100: 0.077597
2023-02-22 09:21:02 INFO     Test average HITS3 at epoch 100: 0.095817
2023-02-22 09:21:02 INFO     Test average HITS10 at epoch 100: 0.200431
2023-02-22 09:21:05 INFO     test: pai: 100
2023-02-22 09:21:11 INFO     Test pai MRR at epoch 100: 0.336157
2023-02-22 09:21:11 INFO     Test pai HITS1 at epoch 100: 0.245919
2023-02-22 09:21:11 INFO     Test pai HITS3 at epoch 100: 0.368868
2023-02-22 09:21:11 INFO     Test pai HITS10 at epoch 100: 0.511210
2023-02-22 09:21:11 INFO     Test pai num_queries at epoch 100: 100.000000
2023-02-22 09:21:11 INFO     Test average MRR at epoch 100: 0.336157
2023-02-22 09:21:11 INFO     Test average HITS1 at epoch 100: 0.245919
2023-02-22 09:21:11 INFO     Test average HITS3 at epoch 100: 0.368868
2023-02-22 09:21:11 INFO     Test average HITS10 at epoch 100: 0.511210
2023-02-22 09:21:13 INFO     test: au: 100
2023-02-22 09:21:23 INFO     Test au MRR at epoch 100: 0.171622
2023-02-22 09:21:23 INFO     Test au HITS1 at epoch 100: 0.097019
2023-02-22 09:21:23 INFO     Test au HITS3 at epoch 100: 0.183570
2023-02-22 09:21:23 INFO     Test au HITS10 at epoch 100: 0.329039
2023-02-22 09:21:23 INFO     Test au num_queries at epoch 100: 100.000000
2023-02-22 09:21:23 INFO     Test average MRR at epoch 100: 0.171622
2023-02-22 09:21:23 INFO     Test average HITS1 at epoch 100: 0.097019
2023-02-22 09:21:23 INFO     Test average HITS3 at epoch 100: 0.183570
2023-02-22 09:21:23 INFO     Test average HITS10 at epoch 100: 0.329039

# To reproduce the results reported in Table 3 and Table 4.
python3 eval_cqd.py --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen
(litcqd) cdemir@raki:~/Multi-Hop-Reasoning-in-Knowledge-Graphs-with-Literals$ python3 eval_cqd.py --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen
2023-02-22 09:27:34 INFO     test: 1ap: 3000
2023-02-22 09:27:34 INFO     Test 1ap MAE at epoch 100: 0.049066
2023-02-22 09:27:34 INFO     Test 1ap MSE at epoch 100: 0.010391
2023-02-22 09:27:34 INFO     Test 1ap RMSE at epoch 100: 0.101938
2023-02-22 09:27:34 INFO     Test 1ap num_queries at epoch 100: 3000.000000
2023-02-22 09:27:34 INFO     Test average MAE at epoch 100: 0.049066
2023-02-22 09:27:34 INFO     Test average MSE at epoch 100: 0.010391
2023-02-22 09:27:34 INFO     Test average RMSE at epoch 100: 0.101938
2023-02-22 09:27:36 INFO     test: 2ap: 100
2023-02-22 09:27:37 INFO     Test 2ap MAE at epoch 100: 0.031695
2023-02-22 09:27:37 INFO     Test 2ap MSE at epoch 100: 0.003925
2023-02-22 09:27:37 INFO     Test 2ap RMSE at epoch 100: 0.062648
2023-02-22 09:27:37 INFO     Test 2ap num_queries at epoch 100: 100.000000
2023-02-22 09:27:37 INFO     Test average MAE at epoch 100: 0.031695
2023-02-22 09:27:37 INFO     Test average MSE at epoch 100: 0.003925
2023-02-22 09:27:37 INFO     Test average RMSE at epoch 100: 0.062648
2023-02-22 09:27:39 INFO     test: 3ap: 100
2023-02-22 09:27:42 INFO     Test 3ap MAE at epoch 100: 0.040821
2023-02-22 09:27:42 INFO     Test 3ap MSE at epoch 100: 0.006615
2023-02-22 09:27:42 INFO     Test 3ap RMSE at epoch 100: 0.081335
2023-02-22 09:27:42 INFO     Test 3ap num_queries at epoch 100: 100.000000
2023-02-22 09:27:42 INFO     Test average MAE at epoch 100: 0.040821
2023-02-22 09:27:42 INFO     Test average MSE at epoch 100: 0.006615
2023-02-22 09:27:42 INFO     Test average RMSE at epoch 100: 0.081335
2023-02-22 09:27:44 INFO     test: ai-lt: 66
2023-02-22 09:27:48 INFO     Test ai-lt MRR at epoch 100: 0.315298
2023-02-22 09:27:48 INFO     Test ai-lt HITS1 at epoch 100: 0.218398
2023-02-22 09:27:48 INFO     Test ai-lt HITS3 at epoch 100: 0.372291
2023-02-22 09:27:48 INFO     Test ai-lt HITS10 at epoch 100: 0.516620
2023-02-22 09:27:48 INFO     Test ai-lt num_queries at epoch 100: 66.000000
2023-02-22 09:27:48 INFO     Test average MRR at epoch 100: 0.315298
2023-02-22 09:27:48 INFO     Test average HITS1 at epoch 100: 0.218398
2023-02-22 09:27:48 INFO     Test average HITS3 at epoch 100: 0.372291
2023-02-22 09:27:48 INFO     Test average HITS10 at epoch 100: 0.516620
2023-02-22 09:27:50 INFO     test: ai-eq: 100
2023-02-22 09:27:55 INFO     Test ai-eq MRR at epoch 100: 0.190988
2023-02-22 09:27:55 INFO     Test ai-eq HITS1 at epoch 100: 0.100243
2023-02-22 09:27:55 INFO     Test ai-eq HITS3 at epoch 100: 0.228257
2023-02-22 09:27:55 INFO     Test ai-eq HITS10 at epoch 100: 0.352451
2023-02-22 09:27:55 INFO     Test ai-eq num_queries at epoch 100: 100.000000
2023-02-22 09:27:55 INFO     Test average MRR at epoch 100: 0.190988
2023-02-22 09:27:55 INFO     Test average HITS1 at epoch 100: 0.100243
2023-02-22 09:27:55 INFO     Test average HITS3 at epoch 100: 0.228257
2023-02-22 09:27:55 INFO     Test average HITS10 at epoch 100: 0.352451
2023-02-22 09:27:58 INFO     test: ai-gt: 64
2023-02-22 09:28:01 INFO     Test ai-gt MRR at epoch 100: 0.242383
2023-02-22 09:28:01 INFO     Test ai-gt HITS1 at epoch 100: 0.150710
2023-02-22 09:28:01 INFO     Test ai-gt HITS3 at epoch 100: 0.278370
2023-02-22 09:28:01 INFO     Test ai-gt HITS10 at epoch 100: 0.433298
2023-02-22 09:28:01 INFO     Test ai-gt num_queries at epoch 100: 64.000000
2023-02-22 09:28:01 INFO     Test average MRR at epoch 100: 0.242383
2023-02-22 09:28:01 INFO     Test average HITS1 at epoch 100: 0.150710
2023-02-22 09:28:01 INFO     Test average HITS3 at epoch 100: 0.278370
2023-02-22 09:28:01 INFO     Test average HITS10 at epoch 100: 0.433298
2023-02-22 09:28:04 INFO     test: 2ai: 100
2023-02-22 09:28:13 INFO     Test 2ai MRR at epoch 100: 0.227936
2023-02-22 09:28:13 INFO     Test 2ai HITS1 at epoch 100: 0.154531
2023-02-22 09:28:13 INFO     Test 2ai HITS3 at epoch 100: 0.250330
2023-02-22 09:28:13 INFO     Test 2ai HITS10 at epoch 100: 0.377042
2023-02-22 09:28:13 INFO     Test 2ai num_queries at epoch 100: 100.000000
2023-02-22 09:28:13 INFO     Test average MRR at epoch 100: 0.227936
2023-02-22 09:28:13 INFO     Test average HITS1 at epoch 100: 0.154531
2023-02-22 09:28:13 INFO     Test average HITS3 at epoch 100: 0.250330
2023-02-22 09:28:13 INFO     Test average HITS10 at epoch 100: 0.377042
2023-02-22 09:28:16 INFO     test: aip: 100
2023-02-22 09:28:24 INFO     Test aip MRR at epoch 100: 0.111507
2023-02-22 09:28:24 INFO     Test aip HITS1 at epoch 100: 0.077597
2023-02-22 09:28:24 INFO     Test aip HITS3 at epoch 100: 0.095817
2023-02-22 09:28:24 INFO     Test aip HITS10 at epoch 100: 0.200431
2023-02-22 09:28:24 INFO     Test aip num_queries at epoch 100: 100.000000
2023-02-22 09:28:24 INFO     Test average MRR at epoch 100: 0.111507
2023-02-22 09:28:24 INFO     Test average HITS1 at epoch 100: 0.077597
2023-02-22 09:28:24 INFO     Test average HITS3 at epoch 100: 0.095817
2023-02-22 09:28:24 INFO     Test average HITS10 at epoch 100: 0.200431
2023-02-22 09:28:27 INFO     test: pai: 100
2023-02-22 09:28:33 INFO     Test pai MRR at epoch 100: 0.336157
2023-02-22 09:28:33 INFO     Test pai HITS1 at epoch 100: 0.245919
2023-02-22 09:28:33 INFO     Test pai HITS3 at epoch 100: 0.368868
2023-02-22 09:28:33 INFO     Test pai HITS10 at epoch 100: 0.511210
2023-02-22 09:28:33 INFO     Test pai num_queries at epoch 100: 100.000000
2023-02-22 09:28:33 INFO     Test average MRR at epoch 100: 0.336157
2023-02-22 09:28:33 INFO     Test average HITS1 at epoch 100: 0.245919
2023-02-22 09:28:33 INFO     Test average HITS3 at epoch 100: 0.368868
2023-02-22 09:28:33 INFO     Test average HITS10 at epoch 100: 0.511210
2023-02-22 09:28:35 INFO     test: au: 100
2023-02-22 09:28:45 INFO     Test au MRR at epoch 100: 0.171622
2023-02-22 09:28:45 INFO     Test au HITS1 at epoch 100: 0.097019
2023-02-22 09:28:45 INFO     Test au HITS3 at epoch 100: 0.183570
2023-02-22 09:28:45 INFO     Test au HITS10 at epoch 100: 0.329039
2023-02-22 09:28:45 INFO     Test au num_queries at epoch 100: 100.000000
2023-02-22 09:28:45 INFO     Test average MRR at epoch 100: 0.171622
2023-02-22 09:28:45 INFO     Test average HITS1 at epoch 100: 0.097019
2023-02-22 09:28:45 INFO     Test average HITS3 at epoch 100: 0.183570
2023-02-22 09:28:45 INFO     Test average HITS10 at epoch 100: 0.329039

