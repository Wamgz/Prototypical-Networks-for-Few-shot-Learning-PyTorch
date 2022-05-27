# coding=UTF-8
python ../train.py \
--dataset_root ~/WZ/data \
--epochs 3000 \
--model_name vit \
--dataset_name miniImagenet \
--classes_per_it_tr 20 \
--num_support_tr 5 \
--num_query_tr 15 \
--classes_per_it_val 16 \
--num_support_val 5 \
--num_query_val 15 \
--height 96 \
--width 96 \
--weight_decay 0.001 \
--iterations 100 \
--cuda 1 \
