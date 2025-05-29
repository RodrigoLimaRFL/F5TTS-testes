import pandas as pd

ORIGINAL_CSV_PATH = './NURC-SP_ENTOA_TTS/prosodic/train.csv'
ADAPTED_CSV_PATH = 'metadata.csv'
PATH_TO_FILES = 'wavs/'

original_df = pd.read_csv(ORIGINAL_CSV_PATH)
rows = []

for index, row in original_df.iterrows():
    audio_file = PATH_TO_FILES + row['path']
    text = row['text']
    rows.append({
        'audio_file': audio_file,
        'text': text,
    })

adapted_df = pd.DataFrame(rows)
adapted_df.to_csv(ADAPTED_CSV_PATH, sep='|', index=False)
