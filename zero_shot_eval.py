import os
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from models import AudioTextCounterfactualModel

def get_labels(dataset_name):
    if dataset_name == 'urbansound8k':
        return ["air conditioner", "car horn", "children playing", "dog bark", 
                "drilling", "engine idling", "gunshot", "jackhammer", "siren", "street music"]
    elif dataset_name == 'esc50':
        return ["dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow",
                "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds", "water_drops", 
                "wind", "pouring_water", "toilet_flush", "thunderstorm", "crying_baby", "sneezing", 
                "clapping", "breathing", "coughing", "footsteps", "laughing", "brushing_teeth", 
                "snoring", "drinking_sipping", "door_wood_knock", "mouse_click", "keyboard_typing", 
                "door_wood_creaks", "can_opening", "washing_machine", "vacuum_cleaner", 
                "clock_tick", "glass_breaking", "helicopter", "chainsaw", "siren", "car_horn", 
                "engine", "train", "church_bells", "airplane", "fireworks", "hand_saw"]
    return []

def run_zero_shot_eval(dataset_name, checkpoint_path, data_home, device):
    print(f"\n--- Évaluation Zero-Shot : {dataset_name} ---")
    
    # Charger le modèle
    model = AudioTextCounterfactualModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.audio_encoder.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    labels = get_labels(dataset_name)
    with torch.no_grad():
        label_embeddings = model.encode_text(labels, device)

    # Chargement des métadonnées selon le dataset
    if dataset_name == 'esc50':
        csv_path = os.path.join(data_home, "meta", "esc50.csv")
        audio_dir = os.path.join(data_home, "audio")
        df = pd.read_csv(csv_path)
    else: # urbansound8k
        csv_path = os.path.join(data_home, "metadata", "UrbanSound8K.csv")
        audio_dir = os.path.join(data_home, "audio")
        df = pd.read_csv(csv_path)

    correct, total = 0, 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if dataset_name == 'esc50':
            file_path = os.path.join(audio_dir, row['filename'])
            target_label = row['category']
        else:
            file_path = os.path.join(audio_dir, f"fold{row['fold']}", row['slice_file_name'])
            target_label = row['class']

        try:
            waveform, sr = torchaudio.load(file_path)
            if waveform.shape[0] > 1: waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resampling 32kHz & Pad/Truncate 10s
            if sr != 32000:
                waveform = torchaudio.transforms.Resample(sr, 32000)(waveform)
            
            max_samples = 32000 * 10
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
            else:
                waveform = F.pad(waveform, (0, max_samples - waveform.shape[1]))

            with torch.no_grad():
                audio_embed = model.encode_audio(waveform.to(device))
            
            similarities = torch.matmul(audio_embed, label_embeddings.T)
            pred_idx = torch.argmax(similarities, dim=-1).item()

            if labels[pred_idx].replace("_", " ") == target_label.replace("_", " "):
                correct += 1
            total += 1
        except Exception:
            continue

    accuracy = correct / total if total > 0 else 0
    print(f"\nPrécision pour {dataset_name} : {accuracy:.4f}")
    return accuracy

import os
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
import numpy as np
from models import AudioTextCounterfactualModel

def run_zero_shot_eval_us8k_rigorous(checkpoint_path, data_home, device):
    print(f"\n--- Évaluation Rigoureuse UrbanSound8K (10-Fold) ---")
    
    # 1. Charger le modèle et les étiquettes
    model = AudioTextCounterfactualModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.audio_encoder.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    labels = ["air conditioner", "car horn", "children playing", "dog bark", 
              "drilling", "engine idling", "gunshot", "jackhammer", "siren", "street music"]
    
    with torch.no_grad():
        label_embeddings = model.encode_text(labels, device)

    # 2. Charger les métadonnées
    df = pd.read_csv(os.path.join(data_home, "metadata", "UrbanSound8K.csv"))
    audio_dir = os.path.join(data_home, "audio")

    fold_accuracies = []

    # 3. Évaluation par Fold
    for fold in range(1, 11):
        fold_df = df[df['fold'] == fold]
        correct, total = 0, 0
        
        print(f"Évaluation du Fold {fold}...")
        for _, row in tqdm(fold_df.iterrows(), total=len(fold_df), leave=False):
            file_path = os.path.join(audio_dir, f"fold{fold}", row['slice_file_name'])
            
            try:
                waveform, sr = torchaudio.load(file_path)
                if waveform.shape[0] > 1: waveform = waveform.mean(dim=0, keepdim=True)
                if sr != 32000:
                    waveform = torchaudio.transforms.Resample(sr, 32000)(waveform)
                
                max_samples = 32000 * 10
                waveform = waveform[:, :max_samples] if waveform.shape[1] > max_samples else F.pad(waveform, (0, max_samples - waveform.shape[1]))

                with torch.no_grad():
                    audio_embed = model.encode_audio(waveform.to(device))
                
                sims = torch.matmul(audio_embed, label_embeddings.T)
                # ... à l'intérieur de la boucle de prédiction ...
                pred_idx = torch.argmax(sims).item()
                predicted_label = labels[pred_idx].replace(" ", "_").lower()
                actual_label = row['class'].lower()

                if predicted_label == actual_label:
                    correct += 1
                total += 1
            except: continue

        acc = correct / total if total > 0 else 0
        fold_accuracies.append(acc)
        print(f"Fold {fold} Accuracy: {acc:.4f}")

    # 4. Résultats finaux
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"\n=====================================")
    print(f"Moyenne finale (10-fold): {mean_acc:.4f} (+/- {std_acc:.4f})")
    print(f"=====================================")
    return mean_acc