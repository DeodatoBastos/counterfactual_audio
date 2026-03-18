import torch
import torchaudio
import pandas as pd
import random
from torch.utils.data import Dataset
import torch.nn.functional as F


class CounterfactualAudioDataset(Dataset):
    """
    Dataset for loading audio and text pairs for Counterfactual Audio Learning
    """
    def __init__(self, metadata_path: str, target_sr: int = 32000, duration: int = 10):
        """
        Args:
            metadata_path (str): Path to a CSV file containing the dataset metadata.
                                 Expected columns: ['audio_path', 'caption', 'counterfactual']
            target_sr (int): Target sampling rate for the audio (default 32kHz).
            duration (int): Fixed length of the audio segment in seconds (default 10s).
        """
        self.metadata = pd.read_csv(metadata_path)

        self.target_sr = target_sr
        self.duration = duration
        self.max_samples = self.target_sr * self.duration

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=1024,
            win_length=1024,
            hop_length=320,
            f_min=50.0,
            f_max=14000.0,
            n_mels=64,
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80.0)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = row['audio_path']
        caption = row['caption']
        counterfactual = row['counterfactual']

        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono if the audio is stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if the native sample rate doesn't match the target (32kHz)
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        num_frames = waveform.shape[1]

        if num_frames > self.max_samples:
            # Randomly truncate to a contiguous 10-second segment for training
            start_idx = random.randint(0, num_frames - self.max_samples)
            waveform = waveform[:, start_idx : start_idx + self.max_samples]
        elif num_frames < self.max_samples:
            # Zero padding applied for shorter clips
            padding = self.max_samples - num_frames
            waveform = F.pad(waveform, (0, padding))

        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)

        return log_mel_spec, caption, counterfactual

# ==========================================
# Usage Example
# ==========================================
if __name__ == "__main__":
    # Expected CSV structure:
    # audio_path,caption,counterfactual
    # data/audio1.wav,"A dog is barking.","A cat is meowing."
    dataset = CounterfactualAudioDataset("path/to/metadata.csv")

    log_mel_spec, caption, counterfactual = dataset[0]

    print(f"Log Mel Spectrogram Shape: {log_mel_spec.shape}")
    print(f"Original Caption: {caption}")
    print(f"Counterfactual Caption: {counterfactual}")