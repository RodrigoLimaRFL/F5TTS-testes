## Instruções para finetuning

- Clone os repositórios do F5-TTS, F5-TTS-pt-br e ENTOA-TTS
```
git clone https://github.com/SWivid/F5-TTS
git clone https://huggingface.co/firstpixel/F5-TTS-pt-br
git clone https://huggingface.co/datasets/nilc-nlp/NURC-SP_ENTOA_TTS
```

- Copie os arquivos seguintes para as pastas adequadas
```
cp F5TTS_v1_Base.yaml F5-TTS/src/f5_tts/configs/F5TTS_v1_Base.yaml
cp trainer.py F5-TTS/src/f5_tts/model/trainer.py
```

- Descompacte NURC-SP_ENTOA_TTS/prosodic/audios.tar.gz em um diretório NURC-SP_ENTOA_TTS/prosodic/wavs
- Pegue 'path' e 'text' de NURC-SP_ENTOA_TTS/prosodic/train.csv (segmentação prosódica) e crie um arquivo NURC-SP_ENTOA_TTS/prosodic/metadata.csv com os campos 'audio_file' e 'text' e separador '|' (para segmentação automática, pegue de NURC-SP_ENTOA_TTS/automatic/train.csv )
- Compile o container
```
docker build . -t [YOUR_TAG_HERE]
```
- Rode o container
```
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm [YOUR_TAG_HERE]
```

## Descrição

Primeiro, o F5-TTS foi treinado com o dataset CML_TTS por 800 épocas (treino-cml_tts/7.5e-05 lr/model_last.pt).

Depois, após transformar o checkpoint em .safetensors, foi finetunado com o dataset ENTOA_TTS. Foram treinados dois checkpoints:

1- Um treinado com o subset 'automatic', feito com segmentação automática do whisper-large-v2.

2- Um treinado com o subset 'prosodic', feito com segmentação prosódica feita manualmente.

O link para os checkpoints é [{link}](https://huggingface.co/RodrigoLimaRFL/f5tts-pt-br)