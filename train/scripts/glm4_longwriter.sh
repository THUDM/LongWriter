export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32'

deepspeed --include localhost:0,1,2,3,4,5,6,7 main.py \
--model_name_or_path THUDM/glm-4-9b \
--train_file ./data/glm4/longwriter \
--output_dir ./output/glm4/longwriter \
--num_train_epochs 4 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--save_strategy "steps" \
--save_steps 400 \
--save_total_limit 10 \
--preprocessing_num_workers 64 \
--learning_rate 1e-5 \
--weight_decay 0.1 \
--warmup_ratio 0.03 \
--lr_scheduler_type cosine \
--logging_dir "./logs/" \
--deepspeed ds_config/stage3.json \
--bf16 \
--gradient_checkpointing 1 \
--adam_beta1 0.9 \
--adam_beta2 0.95 \
--report_to "wandb" \
--run_name "glm4_longwriter" \
--logging_steps 1 \
--batch_method "pack" \
--pack_loss \
