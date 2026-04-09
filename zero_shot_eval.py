import os
import torch
import torch.nn.functional as F
import soundata
import numpy as np
from tqdm import tqdm

# Importation des modèles définis par votre collègue
from models import AudioTextCounterfactualModel

def run_zero_shot_eval(checkpoint_path, data_home, device):
    """
    Exécute l'évaluation Zero-Shot sur UrbanSound8K.
    """
    # 1. Chargement du modèle et des poids
    print(f"Chargement du modèle depuis {checkpoint_path}...")
    model = AudioTextCounterfactualModel().to(device)
    
    # Chargement du dictionnaire d'état (state_dict)
    # PyTorch gère le dossier extrait comme un fichier unique
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.audio_encoder.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Configuration du dataset UrbanSound8K via soundata
    dataset = soundata.initialize('urbansound8k', data_home=data_home)
    # Les 10 classes officielles citées dans l'article [cite: 152]
    us8k_labels = [
        "air conditioner", "car horn", "children playing", "dog bark", 
        "drilling", "engine idling", "gunshot", "jackhammer", 
        "siren", "street music"
    ]
    
    # Encodage des étiquettes de texte (réalisé une seule fois) [cite: 89, 140]
    print("Encodage des étiquettes de texte...")
    with torch.no_grad():
        label_embeddings = model.encode_text(us8k_labels, device) # (10, 512)

    # 3. Boucle d'évaluation
    clips = dataset.load_clips()
    correct_predictions = 0
    total_clips = 0

    print("Début de l'évaluation Zero-Shot...")
    for clip_id, clip in tqdm(clips.items()):
        # Chargement de l'audio
        audio, sr = clip.audio
        
        # Prétraitement : Mono et Resampling à 32kHz (comme dans le dataset.py)
        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.mean(dim=0)
        
        # Resampling si nécessaire (PANNs attend 32kHz) [cite: 143]
        if sr != 32000:
            import torchaudio.transforms as T
            resampler = T.Resample(sr, 32000)
            audio_tensor = resampler(audio_tensor)

        # Troncature/Padding à 10 secondes [cite: 145]
        max_samples = 32000 * 10
        if audio_tensor.shape[0] > max_samples:
            audio_tensor = audio_tensor[:max_samples]
        else:
            audio_tensor = F.pad(audio_tensor, (0, max_samples - audio_tensor.shape[0]))

        # Inférence audio
        with torch.no_grad():
            audio_embedding = model.encode_audio(audio_tensor.unsqueeze(0).to(device)) # (1, 512)

        # Calcul de la similarité cosinus [cite: 160]
        # Similarity = (1, 512) @ (512, 10) -> (1, 10)
        similarities = torch.matmul(audio_embedding, label_embeddings.T)
        prediction_idx = torch.argmax(similarities, dim=-1).item()

        # Vérification
        if us8k_labels[prediction_idx] == clip.class_label:
            correct_predictions += 1
        total_clips += 1

    accuracy = correct_predictions / total_clips
    print(f"\nPrécision Top-1 sur UrbanSound8K : {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    # Configuration des chemins
    CHECKPOINT = "/content/models/checkpoint_epoch_30"
    DATA_HOME = "/content/data/urbansound8k"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(CHECKPOINT):
        print(f"Erreur : Checkpoint introuvable à {CHECKPOINT}")
    else:
        run_zero_shot_eval(CHECKPOINT, DATA_HOME, DEVICE)