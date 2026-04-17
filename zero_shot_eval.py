"""
Zero-shot evaluation for ESC-50 and UrbanSound8K datasets.
It is necessary to have a trained model and the two datasets before running this code.
"""

import os
import torch
import torch.nn.functional as F
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm
from models import AudioTextCounterfactualModel

def load_model_weights(checkpoint_path, device):
    """Loads weights into the model, handling potential key mismatches."""
    model = AudioTextCounterfactualModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle state_dict key prefixing if necessary
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('base.') for k in state_dict.keys()):
        state_dict = {k.replace('base.', ''): v for k, v in state_dict.items()}
    
    model.audio_encoder.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess_audio(file_path, target_sr=32000, duration=10):
    """Loads, resamples, and pads/truncates audio to a fixed duration."""
    waveform, sr = torchaudio.load(file_path)
    
    # To Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        
    # Resample
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    
    # Pad or Truncate
    max_samples = target_sr * duration
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]
    else:
        waveform = F.pad(waveform, (0, max_samples - waveform.shape[1]))
        
    return waveform

def run_zero_shot_eval_esc50(checkpoint_path, data_home, device):
    print(f"\n--- Zero-Shot Evaluation: ESC-50 (5-Fold) ---")
    model = load_model_weights(checkpoint_path, device)

    labels = ["dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow",
              "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds", "water_drops",
              "wind", "pouring_water", "toilet_flush", "thunderstorm", "crying_baby", "sneezing",
              "clapping", "breathing", "coughing", "footsteps", "laughing", "brushing_teeth",
              "snoring", "drinking_sipping", "door_wood_knock", "mouse_click", "keyboard_typing",
              "door_wood_creaks", "can_opening", "washing_machine", "vacuum_cleaner",
              "clock_tick", "glass_breaking", "helicopter", "chainsaw", "siren", "car_horn",
              "engine", "train", "church_bells", "airplane", "fireworks", "hand_saw"]

    with torch.no_grad():
        label_embeddings = model.encode_text(labels, device)

    df = pd.read_csv(os.path.join(data_home, "meta", "esc50.csv"))
    audio_dir = os.path.join(data_home, "audio")
    fold_accuracies = []

    for fold in range(1, 6):
        fold_df = df[df['fold'] == fold]
        correct = 0
        
        print(f"Evaluating Fold {fold}...")
        for _, row in tqdm(fold_df.iterrows(), total=len(fold_df), leave=False):
            file_path = os.path.join(audio_dir, row['filename'])
            waveform = preprocess_audio(file_path).to(device)

            with torch.no_grad():
                audio_embed = model.encode_audio(waveform)
                sims = torch.matmul(audio_embed, label_embeddings.T)
                
            prediction = labels[torch.argmax(sims).item()].replace("_", " ")
            actual = row['category'].replace("_", " ")
            
            if prediction == actual:
                correct += 1

        acc = correct / len(fold_df)
        fold_accuracies.append(acc)
        print(f"Fold {fold} Accuracy: {acc:.4f}")

    print_final_results("ESC-50", fold_accuracies)
    return np.mean(fold_accuracies), np.std(fold_accuracies)

def run_zero_shot_eval_us8k(checkpoint_path, data_home, device):
    print(f"\n--- Zero-Shot Evaluation: UrbanSound8K (10-Fold) ---")
    model = load_model_weights(checkpoint_path, device)

    labels = ["air conditioner", "car horn", "children playing", "dog bark",
              "drilling", "engine idling", "gunshot", "jackhammer", "siren", "street music"]

    with torch.no_grad():
        label_embeddings = model.encode_text(labels, device)

    df = pd.read_csv(os.path.join(data_home, "metadata", "UrbanSound8K.csv"))
    audio_dir = os.path.join(data_home, "audio")
    fold_accuracies = []

    for fold in range(1, 11):
        fold_df = df[df['fold'] == fold]
        correct = 0

        print(f"Evaluating Fold {fold}...")
        for _, row in tqdm(fold_df.iterrows(), total=len(fold_df), leave=False):
            file_path = os.path.join(audio_dir, f"fold{fold}", row['slice_file_name'])
            waveform = preprocess_audio(file_path).to(device)

            with torch.no_grad():
                audio_embed = model.encode_audio(waveform)
                sims = torch.matmul(audio_embed, label_embeddings.T)

            prediction = labels[torch.argmax(sims).item()].replace(" ", "_").lower()
            actual = row['class'].lower()

            if prediction == actual:
                correct += 1

        acc = correct / len(fold_df)
        fold_accuracies.append(acc)
        print(f"Fold {fold} Accuracy: {acc:.4f}")

    print_final_results("UrbanSound8K", fold_accuracies)
    return np.mean(fold_accuracies), np.std(fold_accuracies)

def print_final_results(name, accuracies):
    print(f"\n" + "="*40)
    print(f"Final average {name}: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
    print("="*40 + "\n")

if __name__ == "__main__":
    # Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT = "path/to/your/model_checkpoint.pth"
    DATA_ESC = "path/to/esc50_folder"
    DATA_US8K = "path/to/urbansound8k_folder"

    # Execution
    print("Starting Zero-Shot Evaluation Pipeline...")
    
    # ESC-50
    run_zero_shot_eval_esc50(CKPT, DATA_ESC, DEVICE)
    
    # UrbanSound8K
    run_zero_shot_eval_us8k(CKPT, DATA_US8K, DEVICE)
