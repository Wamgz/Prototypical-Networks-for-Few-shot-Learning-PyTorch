# coding=UTF-8
python ../train.py \
--dataset_root ~/WZ/data \
--epochs 1000 \
--model_name vit \
--dataset_name miniImagenet \
--classes_per_it_tr 5 \
--num_support_tr 5 \
--num_query_tr 15 \
--classes_per_it_val 5 \
--num_support_val 5 \
--num_query_val 15 \
--height 96 \
--width 96 \
--iterations 1000 \
--learning_rate 0.001 \
--balance_scale 0.1 \
--use_join_loss false \
--cuda 2 \
--use_aux_loss \
--comment "加上aux loss"