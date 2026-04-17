# Causal-Audio Framework: Counterfactual Audio-Language Alignment

This repository contains the implementation and ablation study for an audio-text retrieval model trained using causal counterfactuals. It builds upon the standard Contrastive Language-Audio Pretraining (CLAP) paradigm by introducing "hard negative" text captions generated via a Large Language Model (LLM) to enforce fine-grained acoustic learning.

## Overview
Standard contrastive models often rely on "easy" in-batch negatives, leading to shortcut learning (e.g., matching broad environmental noise rather than specific acoustic events). This project implements a dual-encoder architecture that trains on triplets `(Audio, Factual Text, Counterfactual Text)` to force the model to identify specific causal acoustic features.

### Architecture
* **Audio Encoder:** PANNs ResNet-38 (Pretrained on AudioSet) + Projection Adapter.
* **Text Encoder:** CLIP (Frozen) to generate 512-dimensional semantic embeddings.
* **Loss Functions:**
  * `baseline`: Standard InfoNCE (Symmetric Cross-Entropy).
  * `counterfactual`: Triplet Margin Loss (Angular) + InfoNCE Factual Consistency.

## Repository Structure
* `train.py`: The main training script handling both standard CLAP and Counterfactual setups.
* `eval.py`: Automated evaluation suite that scans checkpoints and calculates Top-1 and Top-10 text-to-audio retrieval scores on Clotho.
* `zero_shot_eval.py`: Script to run rigorous K-fold zero-shot classification on environmental datasets (ESC-50 and UrbanSound8K).
* `data/`: Contains the pre-processed `.csv` metadata and audio files.
* `models/`: Directory where `.pth` model checkpoints are saved.

## Requirements
To install the necessary dependencies, run:
```bash
pip install torch torchaudio transformers pandas tqdm numpy
```

## Data Preparation

### 1. Downloading the Datasets
This implementation utilizes several open-source audio datasets for training and evaluation (AudioCaps was excluded due to current availability constraints):
* **Clotho v2:** Used for Text-to-Audio retrieval evaluation. Contains audio clips of 15 to 30 seconds, each with 5 human-annotated captions.
* **MACS:** Used for training. Contains audio clips with multiple human annotations.
* **ESC-50 & UrbanSound8K:** Used for zero-shot classification evaluation.

Download and extract the audio `.wav` files into a designated `data/audio/` directory.

### 2. Audio Preprocessing
All audio files must be standardized before training and evaluation. The data loader handles the following transformations:
* **Resampling:** All audio is resampled to **32 kHz**.
* **Truncation/Padding:** Audio clips are randomly truncated into contiguous **10-second segments** for training. Clips shorter than 10 seconds are zero-padded.
* **Spectrogram Extraction:** The raw waveform is converted into Logarithmic Mel Spectrograms using a window size of 1024 frames, hop size of 320 frames, 64 Mel bins, spanning 50 – 14,000 Hz.

### 3. Metadata Structure
The training script expects `.csv` files mapping the audio files to their respective factual and counterfactual captions. The LLM-generated counterfactuals should be organized as follows:

| audio_path | factual_caption | counterfactual_caption |
| :--- | :--- | :--- |
| `data/audio/file1.wav` | A dog barks in the distance. | A dog howls in the distance. |
| `data/audio/file2.wav` | Water flows rapidly down a stream. | Water trickles slowly down a stream. |

*Note: For the standard CLAP baseline runs, the `counterfactual_caption` column is ignored.*

## Usage

### 1. Training
The training script uses `argparse` to switch between the baseline CLAP InfoNCE loss and the custom Counterfactual loss. It also utilizes the **OneCycleLR** scheduler to safely fine-tune the ResNet-38 backbone alongside randomly initialized adapters.

**Train the Counterfactual Model:**
```bash
python train.py --mode counterfactual --batch_size 32 --epochs 30 --lr 1e-4
```

**Train the Baseline CLAP Model:**
```bash
python train.py --mode baseline --batch_size 32 --epochs 30 --lr 1e-4
```

### 2. Text-to-Audio Retrieval Evaluation
The evaluation script scans the `models/` directory for all saved checkpoints, deduplicates the 1-to-5 audio-caption mappings in Clotho, and outputs a formatted summary table of Top-1 and Top-10 accuracies.

```bash
python eval.py --models_dir models --eval_csv data/clotho_eval_metadata.csv --output_csv results.csv
```

### 3. Zero-Shot Classification
To test the generalized acoustic representations, use the zero-shot evaluation script. It uses the frozen CLIP text encoder to generate prompt embeddings and directly maps them to the audio embeddings using cosine similarity.


```bash
python zero_shot_eval.py
```

## Ablation Study Insights
We conducted an ablation study focusing on two main variables:
1. **Backbone Fine-tuning:** Comparing a fully frozen ResNet-38 backbone against partial fine-tuning (updating the adapter and final two macro-layers).
2. **Learning Rate Scheduling:** Comparing a static learning rate against a OneCycle scheduler. 

Our findings indicate that partially fine-tuning the backbone combined with the OneCycle scheduler (to prevent catastrophic forgetting during the adapter's warmup phase) yields the highest Top-10 retrieval accuracy, outperforming both fully frozen configurations and static learning rates.

---

## Acknowledgements & Citation
This implementation and ablation study is based on the methodology proposed in the following paper:

> **Vosoughi, A., Bondi, L., Wu, H.-H., & Xu, C.** (2024). *Learning Audio Concepts from Counterfactual Natural Language*. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2024), 366-370. 

```bibtex
@inproceedings{Vosoughi2024LearningAC,
  title={Learning Audio Concepts from Counterfactual Natural Language},
  author={Vosoughi, Ali and Bondi, Luca and Wu, Ho-Hsiang and Xu, Chenliang},
  booktitle={2024 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
  pages={366--370},
  year={2024},
  organization={IEEE}
}
```