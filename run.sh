#!/bin/bash

echo "Iniciando o script de treinamento..."
echo "-------------------------------"
nvidia-smi
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Detectado $NUM_GPUS GPUs. Configurando para MULTI_GPU com fp16..."
    ACCELERATE_ARGS="--multi_gpu --mixed_precision fp16 --num_processes $NUM_GPUS --num_machines 1 --machine_rank 0 --same_network --rdzv_backend static --gpu_ids all"
else
    echo "Uma GPU detectada. Configurando para modo padrão com fp16..."
    ACCELERATE_ARGS="--mixed_precision fp16 --num_processes 1 --num_machines 1 --machine_rank 0"
fi

echo "-------------------------------"
echo "Preparando os dados..."
echo "-------------------------------"

mkdir -p /workspace/F5-TTS/data/entoa_prosodic_pinyin

tar xvf /workspace/wavs.tar -C /workspace/F5-TTS/data/entoa_prosodic_pinyin/

python /workspace/F5-TTS/src/f5_tts/train/datasets/prepare_csv_wavs.py /workspace/F5-TTS/data/entoa_prosodic_pinyin /workspace/F5-TTS/data/entoa_prosodic_pinyin

echo "-------------------------------"
echo "Iniciando o treinamento..."
echo "-------------------------------"

wandb login 040dc38adce9abb4e0206b4885c087efe0d85ffd && \
accelerate launch $ACCELERATE_ARGS /workspace/F5-TTS/src/f5_tts/train/finetune_cli.py \
  --exp_name F5TTS_v1_Base \
  --learning_rate 7.5e-05  \
  --batch_size_per_gpu 20 \
  --batch_size_type sample \
  --max_samples 64 \
  --grad_accumulation_steps 1 \
  --max_grad_norm 1 \
  --epochs 800 \
  --num_warmup_updates 1688 \
  --save_per_updates 5000 \
  --keep_last_n_checkpoints 2 \
  --last_per_updates 5000 \
  --dataset_name entoa_prosodic \
  --finetune \
  --pretrain /workspace/F5-TTS/ckpts/model_last.safetensors  \
  --tokenizer pinyin \
  --logger wandb \
  --log_samples

  
  #--finetune \
  #--pretrain /workspace/F5-TTS/ckpts/model_last.pt  \
