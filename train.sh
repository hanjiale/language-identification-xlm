export CUDA_VISIBLE_DEVICES=0
for k in 1 2 4 8
do
  for seed in 13 21 42 87 100
  do
python3 code/train.py \
--data_dir ./data/k-shot/$k-$seed \
--output_dir ./results \
--tuning_type pt \
--model_name_or_path xlm-roberta-base \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 8 \
--gradient_accumulation_steps 1 \
--max_seq_length 512 \
--learning_rate 2e-5 \
--num_train_epochs 20 \
--weight_decay 1e-2 \
--adam_epsilon 1e-6
done
done

python3 code/train.py \
--data_dir ./data \
--output_dir ./results \
--tuning_type pt \
--model_name_or_path xlm-roberta-base \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 8 \
--gradient_accumulation_steps 1 \
--max_seq_length 512 \
--learning_rate 2e-5 \
--num_train_epochs 2 \
--weight_decay 1e-2 \
--adam_epsilon 1e-6



