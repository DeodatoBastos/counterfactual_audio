import pandas as pd
import os
import json

dataset_mappings = [
#     {
#         "json_path": ".data/counterfactual-audio/audiocaps-train-counterfactual.json",
#         "audio_dir": "./data/AUDIOCAPS/audio/train"
#     },
    {
        "json_path": "./data/counterfactual-audio/clotho-development-counterfactual.json",
        "audio_dir": "./data/CLOTHO_v2.1/clotho_audio_files/dev"
    },
    {
        "json_path": "./data/counterfactual-audio/clotho-validation-counterfactual.json",
        "audio_dir": "./data/CLOTHO_v2.1/clotho_audio_files/val"
    },
    {
        "json_path": "./data/counterfactual-audio/macs-counterfactual.json",
        "audio_dir": "./data/MACS/audio/"
    }
]

all_data = []

for mapping in dataset_mappings:
    json_path = mapping["json_path"]
    audio_dir = mapping["audio_dir"]

    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found. Skipping.")
        continue

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        if 'path' in item:
            file_name = os.path.basename(item['path'])
        else:
            continue

        audio_path = os.path.join(audio_dir, file_name)

        # Datasets like Clotho contain multiple captions per audio file.
        # We need to flatten them so each (caption, counterfactual) pair gets its own row.
        captions = item.get('captions', [])
        counterfactuals = item.get('captions_counterfactual', [])

        # Ensure we only iterate up to the matching number of pairs available
        num_pairs = min(len(captions), len(counterfactuals))

        for i in range(num_pairs):
            all_data.append({
                'audio_path': audio_path,
                'caption': captions[i],
                'counterfactual': counterfactuals[i]
            })

if all_data:
    author_data = pd.DataFrame(all_data)
    author_data.to_csv("./data/metadata.csv", index=False)
    print(f"Successfully generated metadata.csv with {len(author_data)} pairs.")
else:
    print("No data was processed. Please check the JSON paths and formats.")


csv_path = "./data/CLOTHO_v2.1/clotho_csv_files/clotho_captions_evaluation.csv"
audio_dir = "./data/CLOTHO_v2.1/clotho_audio_files/eval"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    all_eval_data = []

    for _, row in df.iterrows():
        audio_path = os.path.join(audio_dir, row['file_name'])

        # Each audio file has 5 ground-truth captions
        for i in range(1, 6):
            caption = row.get(f'caption_{i}')
            if pd.notna(caption):
                all_eval_data.append({
                    'audio_path': audio_path,
                    'caption': caption,
                    # We pass the factual caption as a dummy counterfactual 
                    # so the CLIP tokenizer in dataset.py doesn't throw an error.
                    # The train.py evaluation loop ignores this column completely.
                    'counterfactual': caption 
                })

    eval_df = pd.DataFrame(all_eval_data)
    eval_df.to_csv("./data/clotho_eval_metadata.csv", index=False)
    print(f"Successfully generated clotho_eval_metadata.csv with {len(eval_df)} pairs.")
else:
    print(f"Could not find {csv_path}. Did you run the 'eval' download in Step 2?")