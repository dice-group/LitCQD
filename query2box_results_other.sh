# Eval 1p performance for each relation
python3 eval_relations.py --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn_q2b --rank 400 --geo q2b --test_batch_size 100 --print_on_screen

# Eval 1p performance for attribute exists facts
python3 eval_attribute_exists_facts.py --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn_q2b --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 400 --geo q2b --test_batch_size 100 --print_on_screen


# (litcqd) renzhong@litcqd:~/LitCQD$ ./query2box_results_other.sh 
# Traceback (most recent call last):
#   File "eval_relations.py", line 40, in <module>
#     main(parse_args())
#   File "eval_relations.py", line 35, in main
#     load_model(model, train_config.checkpoint_path, train_config.cuda)
#   File "/home/renzhong/LitCQD/util_models.py", line 206, in load_model
#     model.load_state_dict(model_state)
#   File "/home/renzhong/.conda/envs/litcqd/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1406, in load_state_dict
#     raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
# RuntimeError: Error(s) in loading state_dict for Query2Box:
#         size mismatch for ent_embeddings.weight: copying a param with shape torch.Size([14506, 400]) from checkpoint, the shape in current model is torch.Size([1533, 400]).
#         size mismatch for rel_embeddings.weight: copying a param with shape torch.Size([704, 400]) from checkpoint, the shape in current model is torch.Size([94, 400]).
#         size mismatch for offset_embedding.weight: copying a param with shape torch.Size([704, 400]) from checkpoint, the shape in current model is torch.Size([94, 400]).