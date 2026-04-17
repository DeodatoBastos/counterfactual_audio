"""Plot the figures for the qualitative results of the report."""

import os
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from models import AudioTextCounterfactualModel
from zero_shot_eval import load_model_weights, preprocess_audio

def get_random_us8k_sample(data_home):
    """Select randomly a  file and its label from the metadata of US8K."""
    metadata_path = os.path.join(data_home, "metadata", "UrbanSound8K.csv")
    df = pd.read_csv(metadata_path)
    sample = df.sample(n=1).iloc[0]
    
    file_path = os.path.join(
        data_home, "audio", f"fold{sample['fold']}", sample['slice_file_name']
    )
    return file_path, sample['class']

def plot_us8k(model, audio_path, true_label, device, save_path=None):
    """Plot probabilities predicted by the model."""
    # Labels
    labels = ["air_conditioner", "car_horn", "children_playing", "dog_bark",
              "drilling", "engine_idling", "gunshot", "jackhammer", "siren", "street_music"]

    waveform = preprocess_audio(audio_path).to(device)
    with torch.no_grad():
        audio_embed = model.encode_audio(waveform)
        text_embs = model.encode_text(labels, device)
        sims = torch.matmul(audio_embed, text_embs.T).squeeze().cpu().numpy()
        
    # Softmax (T=10) for contrast
    probs = np.exp(sims * 10) / np.sum(np.exp(sims * 10))

    # True class in green, the rest in grey
    colors = []
    for l in labels:
        if l.lower().replace(" ", "_") == true_label.lower().replace(" ", "_"):
            colors.append('#2ca02c') # Vert
        else:
            colors.append('#d3d3d3') # Gris clair

    plt.figure(figsize=(10, 6))
    plt.bar(labels, probs, color=colors)
    
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Probability")
    plt.title(f"Qualitative Analysis\nFile: {os.path.basename(audio_path)} | True Label: {true_label}")
    # plt.ylim(0, 1.0) # Fixer l'axe Y pour mieux comparer
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # CONFIGURATION
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT = "checkpoints/best_model.pth"
    DATA_HOME = "data/urbansound8k"

    # Initialization
    net = AudioTextCounterfactualModel().to(DEVICE)
    load_model_weights(CHECKPOINT, DEVICE, net.audio_encoder)

    # Select a sample and plot the  associated figure
    sample_path, label = get_random_us8k_sample(DATA_HOME)
    plot_us8k(net, sample_path, label, DEVICE)
