import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel

from panns_models import ResNet38


class AudioEncoder(nn.Module):
    """
    Audio Encoder based on ResNet architecture.
    The paper uses PANNs ResNet-38; here we use PyTorch's standard ResNet-34/50 
    adapted for audio (1-channel input) as the closest native equivalent.
    """
    def __init__(self, checkpoint_path="models/ResNet38_mAP=0.434.pth", embedding_dim=512):
        super().__init__()

        self.base = ResNet38(
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527 # AudioSet has 527 classes
        )

        print(f"Loading PANNs ResNet-38 weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.base.load_state_dict(checkpoint['model'])

        # The PANNs ResNet38 outputs a 2048-dimensional feature vector.
        # We project this down to the 512-dimensional CLIP text space.
        self.adapter = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        # The spectrogram extractors already have freeze_parameters=True in your __init__,
        # but we also explicitly freeze the early convolutional operations to save massive VRAM.
        self.base.bn0.requires_grad_(False)
        self.base.conv_block1.requires_grad_(False)

        # The core 'resnet' has 4 macro-layers (defined by layers=[3, 4, 6, 3]).
        # Freezing the first two macro-layers (7 residual blocks) will cut VRAM usage in half
        # without hurting performance, as these layers only detect low-level acoustic edges.
        self.base.resnet.layer1.requires_grad_(False)
        self.base.resnet.layer2.requires_grad_(False)
        
        # We don't use the final AudioSet classifier, so it shouldn't track gradients
        self.base.fc_audioset.requires_grad_(False)
    def forward(self, waveform):
        # Input shape: (Batch_size, Audio_length)

        # The PANNs forward pass returns a dictionary.
        # 'embedding' is the 2048-d vector extracted right before the final classification layer.
        output_dict = self.base(waveform, mixup_lambda=None)
        deep_audio_features = output_dict['embedding']

        # Map to CLIP space and normalize
        embeddings = self.adapter(deep_audio_features)
        return F.normalize(embeddings, )


class AudioTextCounterfactualModel(nn.Module):
    """
    Combines the trainable Audio Encoder and the Frozen CLIP Text Encoder.
    """
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.audio_encoder = AudioEncoder(embedding_dim=512)

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

        with torch.no_grad():
            outputs = self.text_encoder(**inputs)

        text_embeds = outputs.pooler_output
        return F.normalize(text_embeds, p=2, dim=-1)