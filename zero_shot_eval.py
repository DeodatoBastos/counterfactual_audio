import torch
import torch.nn.functional as F
from models import AudioTextCounterfactualModel

def zero_shot_classification(model, audio_waveform, class_labels, device):
    model.eval()
    
    # 1. Encode the audio (using ResNet38 + your Adapter)
    with torch.no_grad():
        # Ensure waveform is (Batch, Samples)
        audio_embed = model.encode_audio(audio_waveform.to(device)) # 512-dim
        
    # 2. Encode all class labels (e.g., "Dog", "Rain", "Siren")
    # The paper uses frozen CLIP text encoders for this [cite: 140, 141]
    label_embeds = model.encode_text(class_labels, device) # (Num_Classes, 512)
    
    # 3. Calculate Cosine Similarity 
    # Similarity = (Audio @ Labels.T)
    similarities = torch.matmul(audio_embed, label_embeds.T)
    
    # 4. Get the predicted class index
    prediction = torch.argmax(similarities, dim=-1)
    return prediction