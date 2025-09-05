#!/bin/bash
source .venv/bin/activate
cd transformers/examples/pytorch/text-classification/

pip install datasets
export TASK_NAME=qqp

output_dir="ds_results"

num_gpus=1

batch_size=16

python -m torch.distributed.launch --nproc_per_node=${num_gpus} \
  run_glue.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --max_seq_length 256 \
  --warmup_steps 500 \
  --per_device_train_batch_size ${batch_size} \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir $output_dir \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --logging_dir $output_dir
