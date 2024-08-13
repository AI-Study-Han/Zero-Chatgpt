



lr=1e-4
block_size=1024

per_device_train_batch_size=24
gradient_accumulation_steps=1
model_path=./model
train_dataset_file=./sft.jsonl
log_file=./log/sft.log
output_dir=./output
deepspeed_config_file=./ds_config.json
random_seed=42
torchrun --nnodes 1 --nproc_per_node 2 sft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_path ${model_path} \
    --train_dataset_file ${train_dataset_file} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --do_train \
    --bf16 True\
    --torch_dtype bfloat16 \
    --seed ${random_seed} \
    --num_train_epochs 3 \
    --logging_strategy steps \
    --logging_steps 100 \
    --log_file ${log_file} \
    --logging_first_step True \
    --adam_beta1 0.9 \
    --adam_beta1 0.95 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --save_steps 0.01 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --block_size ${block_size} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --use_device cuda \
    --use_compile False \