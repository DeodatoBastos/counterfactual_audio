import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import CLIPTokenizer, CLIPTextModel


class AudioEncoder(nn.Module):
    """
    Audio Encoder based on ResNet architecture.
    The paper uses PANNs ResNet-38; here we use PyTorch's standard ResNet-34/50 
    adapted for audio (1-channel input) as the closest native equivalent.
    """
    def __init__(self, embedding_dim=512):
        super().__init__()
        # Load a standard pretrained ResNet
        self.base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Modify the first convolutional layer to accept 1-channel Log-Mel Spectrograms
        # instead of standard 3-channel RGB images
        self.base.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Replace the final classification head with an Adapter Layer
        in_features = self.base.fc.in_features
        self.base.fc = nn.Identity() 

        # Adapter to map the ResNet features into the CLIP latent space
        self.adapter = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x):
        # x shape: (Batch, 1, Freq, Time)
        features = self.base(x)
        embeddings = self.adapter(features)

        # L2 Normalize the embeddings to match CLIP's spatial distribution
        return F.normalize(embeddings, p=2, dim=-1)


class AudioTextCounterfactualModel(nn.Module):
    """
    Combines the trainable Audio Encoder and the Frozen CLIP Text Encoder.
    """
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.audio_encoder = AudioEncoder(embedding_dim=512)

        # Load HuggingFace CLIP Text Encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name)

        # The paper specifies the text encoder is frozen at all stages
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def encode_audio(self, audio):
        return self.audio_encoder(audio)

    def encode_text(self, text_list, device):
        # Tokenize the batch of raw string captions
        inputs = self.tokenizer(
            text_list, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)

        # Forward pass through frozen CLIP model
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)

        # Extract the pooled text embeddings and L2 Normalize
        text_embeds = outputs.pooler_output
        return F.normalize(text_embeds, p=2, dim=-1)