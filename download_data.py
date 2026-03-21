import torchaudio

# aac-datasets attempts to import AudioMetaData for type hinting, but it was 
# removed in newer versions of PyTorch. This dummy class safely bypasses the crash.
if not hasattr(torchaudio, 'AudioMetaData'):
    torchaudio.AudioMetaData = type('AudioMetaData', (), {})

from aac_datasets import Clotho, AudioCaps, MACS

# 1. Download Clotho (Fastest, hosted on Zenodo)
print("Downloading Clotho...")
clotho_train = Clotho(root="./data", subset="dev", download=True)
clotho_val = Clotho(root="./data", subset="val", download=True)
clotho_test = Clotho(root="./data", subset="eval", download=True)

# 2. Download MACS (Hosted on Zenodo)
print("Downloading MACS...")
macs_train = MACS(root="./data", subset="full", download=True)

# 3. Download AudioCaps (Downloads from YouTube via yt-dlp)
# Note: Some YouTube videos may have been removed over time. 
# yt-dlp might output warnings for missing videos.
print("Downloading AudioCaps...")
audiocaps_train = AudioCaps(
    root="./data", 
    subset="train", 
    download=True,
    ytdlp_opts=[
        "--cookies", "cookies.txt",
        "--extractor-args", "youtube:player_client=tv",
        "--sleep-requests", "1",
        "--ignore-errors"
    ]
)
