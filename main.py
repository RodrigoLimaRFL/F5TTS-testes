import sys
sys.path.append('./F5-TTS-pt-br')

from AgentF5TTSChunk import AgentF5TTS

agent = AgentF5TTS(
    ckpt_file="./F5-TTS/ckgs/pt-br/model_last.safetensors",
    vocoder_name="vocos",
    delay=0,
    device="cpu"
)

print("Agent initialized.")