# The hyperparameters to tune need to be uncommented in the ray_tune function in main.py

# TransE with and without facts representing the extence of an attribute:
python3 main.py --cuda --do_tune --data_path data/scripts/generated/LitWD1K -n 10 --rank 0 --loss ce --train_times 100 --optimizer adagrad --reg_weight_ent 1.3e-7 --reg_weight_rel 3.7e-18 --geo cqd-transea --batch_size 1024 --test_batch_size 1000 --valid_epochs 100 --print_on_screen
python3 main.py --cuda --do_tune --data_path data/scripts/generated/LitWD1K_dummy -n 10 --rank 0 --loss ce --train_times 100 --optimizer adagrad --reg_weight_ent 1.3e-7 --reg_weight_rel 3.7e-18 --geo cqd-transea --batch_size 1024 --test_batch_size 1000 --valid_epochs 100 --print_on_screen

# TransEA
python3 main.py --cuda --do_tune --data_path data/scripts/generated/LitWD1K_dummy -n 10 --rank 0 --use_attributes --loss ce --train_times 100 --optimizer adagrad --reg_weight_ent 1.3e-7 --reg_weight_rel 3.7e-18 --reg_weight_attr 1.0e-5 --geo cqd-transea --test_batch_size 1000 --batch_size 1024 --valid_epochs 100 --print_on_screen 

# TransE + MTKGNN
python3 main.py --cuda --do_tune --data_path data/scripts/generated/LitWD1K_dummy -n 10 --rank 0 --use_attributes --loss ce --train_times 100 --optimizer adagrad --reg_weight_ent 1.3e-7 --reg_weight_rel 3.7e-18 --reg_weight_attr 1.0e-5 --geo cqd-mtkgnn --test_batch_size 1000 --batch_size 1024 --valid_epochs 100 --print_on_screen

# TransEA + TransComplEx
python3 main.py --cuda --do_tune --data_path data/scripts/generated/LitWD1K_dummy -n 10 --rank 0 --use_attributes --loss ce --train_times 100 --optimizer adagrad --reg_weight_ent 1.3e-7 --reg_weight_rel 3.7e-18 --reg_weight_attr 1.0e-5 --geo cqd-transcomplexa --test_batch_size 1000 --batch_size 1024 --valid_epochs 100 --print_on_screen --do_sigmoid

# TransEA + DistMult
python3 main.py --cuda --do_tune --data_path data/scripts/generated/LitWD1K_dummy -n 10 --rank 0 --use_attributes --loss ce --train_times 100 --optimizer adagrad --reg_weight_ent 1.3e-7 --reg_weight_rel 3.7e-18 --reg_weight_attr 1.0e-5 --geo cqd-transeadistmult --test_batch_size 1000 --batch_size 1024 --valid_epochs 100 --print_on_screen --do_sigmoid

# TransEA + ComplEx
python3 main.py --cuda --do_tune --data_path data/scripts/generated/LitWD1K_dummy -n 10 --rank 0 --use_attributes --loss ce --train_times 100 --optimizer adagrad --reg_weight_ent 1.3e-7 --reg_weight_rel 3.7e-18 --reg_weight_attr 1.0e-5 --geo cqd-transeacomplex --test_batch_size 1000 --batch_size 1024 --valid_epochs 100 --print_on_screen --do_sigmoid

# TransEA + TransR
python3 main.py --cuda --do_tune --data_path data/scripts/generated/LitWD1K_dummy -n 10 --rank 0 --use_attributes --loss ce --train_times 100 --optimizer adagrad --reg_weight_ent 1.3e-7 --reg_weight_rel 3.7e-18 --reg_weight_attr 1.0e-5 --geo cqd-transra --test_batch_size 1000 --batch_size 1024 --valid_epochs 100 --print_on_screen
