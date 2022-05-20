python train.py \
--dataset_root ~/WZ/data \
--model_name cnn \
--epochs 1000 \
--dataset_name stanfordCars \
--classes_per_it_tr 20 \
--num_support_tr 5 \
--num_query_tr 15 \
--classes_per_it_val 16 \
--num_support_val 5 \
--num_query_val 15 \
--height 128 \
--width 128 \
--cuda cuda:0