import os
import json
import shutil
from torchaudio.info import info
from torchaudio.backend.utils import get_audio_backend
from torchaudio import load
from datasets.arrow_writer import ArrowWriter
from f5_tts.utils.text.preprocess import convert_char_to_pinyin, clear_text
from tqdm import tqdm

def get_audio_duration(file_audio):
    return info(file_audio).num_frames / info(file_audio).sample_rate

def get_correct_audio_path(name_audio, path_wavs):
    for ext in [".mp3", ".wav"]:
        path = os.path.join(path_wavs, name_audio + ext)
        if os.path.isfile(path):
            return path
    return os.path.join(path_wavs, name_audio)  # fallback

def format_seconds_to_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"

def create_metadata(name_project, ch_tokenizer=True):
    path_data = "/workspace/F5-TTS/data"
    path_project = os.path.join(path_data, name_project)
    path_project_wavs = os.path.join(path_project, "wavs")
    file_metadata = os.path.join(path_project, "metadata.csv")
    file_raw = os.path.join(path_project, "raw.arrow")
    file_duration = os.path.join(path_project, "duration.json")
    file_vocab = os.path.join(path_project, "vocab.txt")

    if not os.path.isfile(file_metadata):
        raise FileNotFoundError("metadata.csv not found at " + file_metadata)

    with open(file_metadata, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()

    result = []
    duration_list = []
    text_vocab_set = set()
    total_duration = 0
    error_files = []

    for line in tqdm(lines, desc="Parsing metadata"):
        parts = line.split("|")
        if len(parts) != 2:
            continue

        name_audio, text = parts
        file_audio = get_correct_audio_path(name_audio, path_project_wavs)

        if not os.path.isfile(file_audio):
            error_files.append([file_audio, "not found"])
            continue

        try:
            duration = get_audio_duration(file_audio)
        except Exception as e:
            error_files.append([file_audio, "duration error"])
            continue

        if duration < 1 or duration > 30:
            reason = "duration < 1" if duration < 1 else "duration > 30"
            error_files.append([file_audio, reason])
            continue

        if len(text.strip()) < 3:
            error_files.append([file_audio, "text too short"])
            continue

        text = clear_text(text)
        text = convert_char_to_pinyin([text], polyphone=True)[0]

        result.append({"audio_path": file_audio, "text": text, "duration": duration})
        duration_list.append(duration)
        total_duration += duration

        if ch_tokenizer:
            text_vocab_set.update(list(text))

    # Write Arrow file
    with ArrowWriter(path=file_raw, writer_batch_size=1) as writer:
        for item in tqdm(result, desc="Writing .arrow"):
            writer.write(item)

    # Write durations
    with open(file_duration, "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False, indent=2)

    # Handle vocab
    if ch_tokenizer:
        with open(file_vocab, "w", encoding="utf-8") as f:
            for vocab in sorted(text_vocab_set):
                f.write(vocab + "\n")
    else:
        file_vocab_finetune = os.path.join(path_data, "Emilia_ZH_EN_pinyin/vocab.txt")
        if not os.path.isfile(file_vocab_finetune):
            raise FileNotFoundError("Fallback vocab file not found.")
        shutil.copy2(file_vocab_finetune, file_vocab)

    # Summary
    print("\n✅ Preprocessing complete.")
    print(f"Samples         : {len(result)}")
    print(f"Total duration  : {format_seconds_to_hms(total_duration)}")
    print(f"Min duration    : {round(min(duration_list), 2)} sec")
    print(f"Max duration    : {round(max(duration_list), 2)} sec")
    print(f"Saved arrow     : {file_raw}")
    print(f"Saved durations : {file_duration}")
    print(f"Saved vocab     : {file_vocab}")
    print(f"Tokenizer mode  : {'character' if ch_tokenizer else 'pretrained'}")

    if error_files:
        print("\n⚠️  Skipped files:")
        for item in error_files:
            print(" - ", " = ".join(item))

if __name__ == "__main__":
    create_metadata("cv_pt-br_pinyin", ch_tokenizer=True)
