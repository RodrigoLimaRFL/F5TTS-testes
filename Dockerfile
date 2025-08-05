FROM nvcr.io/nvidia/pytorch:24.12-py3

WORKDIR /workspace

COPY F5-TTS /workspace/F5-TTS

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install --upgrade torchvision

RUN mkdir -p /workspace/F5-TTS/ckpts

RUN pip install -e /workspace/F5-TTS

#COPY common-voice /workspace/common-voice

#COPY common-voice_v1_Base.yaml /workspace/F5-TTS/ckpts/common-voice_v1_Base.yaml

COPY run.sh /workspace/F5-TTS/ckpts/run.sh

CMD ["sh", "/workspace/F5-TTS/ckpts/run.sh"]