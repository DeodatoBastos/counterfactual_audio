import os
import argparse

from dotenv import load_dotenv

import transformers
import torch
from torch.utils.data import DataLoader

from dataset import CounterfactualAudioDataset
from models import AudioTextCounterfactualModel
from utils import evaluate_retrieval, set_seed


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Train Counterfactual Audio-Text Model")
    arg_parser.add_argument("--checkpoint", type=str, default="models/checkpoint_epoch_15.pth", help="Path to checkpoint to resume training")
    arg_parser.add_argument("--model", type=str, default="models/counterfactual_audio_encoder.pth", help="Path to checkpoint to resume training")
    arg_parser.add_argument("--bs", type=int, default=32, help="Batch size for training")
    arg_parser.add_argument("--num_workers", type=int, default=20, help="Number of workers for DataLoader")
    arg_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    arg_parser.add_argument("--deterministic", type=bool, default=False, help="Use deterministic operations for reproducibility")
    args = arg_parser.parse_args()

    print("Args:")
    for key, value in vars(args).items():
        print(f"    > {key}: {value}")
    print()

    load_dotenv()
    transformers.logging.set_verbosity_error()
    set_seed(args.seed, args.deterministic)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    print(f"Using device: {device}")

    bs = args.bs
    resume_checkpoint = args.checkpoint
    model_path = args.model

    model = AudioTextCounterfactualModel().to(device)

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Loading checkpoint '{resume_checkpoint}'.")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.audio_encoder.load_state_dict(checkpoint['model_state_dict'])
    elif model_path and os.path.exists(model_path):
        print(f"Loading model '{model_path}'.")
        weights = torch.load(model_path, map_location=device)
        model.audio_encoder.load_state_dict(weights)
    else:
        raise ValueError("No pretrained model found.")

    test_dataset = CounterfactualAudioDataset("data/clotho_eval_metadata.csv") 
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=args.num_workers)
    top1_acc, top10_acc = evaluate_retrieval(model, test_loader, device)

    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-10 Accuracy: {top10_acc:.4f}")
