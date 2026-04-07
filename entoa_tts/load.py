from datasets import load_dataset, get_dataset_config_names
import pandas as pd
import os
import soundfile as sf

# Choose dataset
dataset_name = "nilc-nlp/NURC-SP_ENTOA_TTS"

# Get all configs
configs = get_dataset_config_names(dataset_name)

# Base export folder
base_output = "exported_datasets"

for config in configs:
    if config != 'automatic':
        continue
    print(f"Processing config: {config}")
    
    # Load full dataset (all splits)
    dataset_dict = load_dataset(dataset_name, config)
    
    # Create folder for this config
    config_dir = os.path.join(base_output, config)
    os.makedirs(config_dir, exist_ok=True)
    
    for split, dataset in dataset_dict.items():
        print(f"  Split: {split}")
        
        # Create folder for this split
        split_dir = os.path.join(config_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        rows = []
        
        for i, example in enumerate(dataset):
            audio = example["audio"]
            rel_path = example["path"]
            
            # full audio path = split_dir + rel_path
            full_path = os.path.join(split_dir, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Save audio
            sf.write(full_path, audio["array"], audio["sampling_rate"])
            
            # metadata (exclude raw audio)
            row = {k: v for k, v in example.items() if k != "audio"}
            row["path"] = rel_path  # keep relative to split folder
            rows.append(row)
        
        # Save metadata CSV in split folder
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(split_dir, "metadata.csv"), index=False)
