export CUDA_VISIBLE_DEVICES=1
python3 code/inference.py --data_dir ./example --name example.txt \
--tuning_type pt \
--per_gpu_eval_batch_size 1 \
--output_dir ./results \
--model_name_or_path xlm-roberta-base