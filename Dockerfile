FROM nvcr.io/nvidia/pytorch:24.12-py3

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    && apt-get install -y python3 \
    && apt-get install -y python3-pip \
    && apt-get install -y python3-pip \
    && apt-get install -y git \
    && apt-get install -y ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY F5-TTS /workspace/F5-TTS

RUN pip install --upgrade torchvision

RUN pip install -e /workspace/F5-TTS

COPY F5-TTS-pt-br /workspace/F5-TTS-pt-br
COPY NURC-SP_ENTOA_TTS/prosodic /workspace/NURC-SP_ENTOA_TTS/prosodic
COPY NURC-SP_ENTOA_TTS/test /workspace/NURC-SP_ENTOA_TTS/test

RUN python /workspace/F5-TTS/src/f5_tts/train/datasets/prepare_csv_wavs.py /workspace/NURC-SP_ENTOA_TTS/prosodic /workspace/F5-TTS/data/ENTOA_TTS_pinyin

RUN python /workspace/F5-TTS/src/f5_tts/train/datasets/prepare_csv_wavs.py NURC-SP_ENTOA_TTS/prosodic /workspace/F5-TTS/data/ENTOA_TTS

COPY run.sh /workspace/run.sh

CMD ["sh", "/workspace/run.sh"]