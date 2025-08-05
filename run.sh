#!/bin/bash

# Detecta o número de GPUs disponíveis
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

# Cria configuração apropriada para o número de GPUs
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Detectado $NUM_GPUS GPUs. Configurando para MULTI_GPU com fp16..."
    ACCELERATE_ARGS="--multi_gpu --mixed_precision fp16 --num_processes $NUM_GPUS --num_machines 1 --machine_rank 0 --same_network --rdzv_backend static --gpu_ids all"
else
    echo "Uma GPU detectada. Configurando para modo padrão com fp16..."
    ACCELERATE_ARGS="--mixed_precision fp16 --num_processes 1 --num_machines 1 --machine_rank 0"
fi

# Loga no wandb
wandb login 040dc38adce9abb4e0206b4885c087efe0d85ffd && \


# Inicia o treinamento com o Accelerate
accelerate launch $ACCELERATE_ARGS F5-TTS/src/f5_tts/train/train.py --config-name common-voice_v1_Base.yaml

accelerate launch \
  --mixed_precision=fp16 
  /workspace/F5-TTS/src/f5_tts/train/finetune_cli.py \ 
  --exp_name F5TTS_v1_Base \ 
  --learning_rate 7.5e-05 \
  --batch_size_per_gpu 4 \
  --batch_size_type frame \
  --max_samples 4 \
  --grad_accumulation_steps 1 \
  --max_grad_norm 1 \
  --epochs 32 \
  --num_warmup_updates 537 \
  --save_per_updates 500 \
  --keep_last_n_checkpoints 3 \
  --last_per_updates 100 \
  --dataset_name cv_pt-br \
  --tokenizer pinyin \
  --logger wandb \
  --log_samples