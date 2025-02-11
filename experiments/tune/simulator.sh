# The `input_template` has no space at the end because the example starts with a <|sep|> token.

source /root/neurondb/.venv/bin/activate

deepspeed --module openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset kh4dien/explainer-gemma-2_simulator-qwen2.5 \
   --input_key explanation \
   --output_key example \
   --input_template $'## Description: {}\n## Input:' \
   --train_batch_size 128 \
   --micro_train_batch_size 32 \
   --max_samples 500000 \
   --pretrain Qwen/Qwen2.5-7B-Instruct \
   --save_path /root/checkpoint/qwen2.5-7b-sim \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 0 \
   --max_epochs 3 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --use_wandb f4f8426bb398048c9b50d2235c42346015f6e743 \
   --lora_rank 32 \
   --target_modules "preset"
