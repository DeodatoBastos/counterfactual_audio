import os
import csv
import glob
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
    arg_parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume training")
    arg_parser.add_argument("--bs", type=int, default=32, help="Batch size for training")
    arg_parser.add_argument("--num_workers", type=int, default=20, help="Number of workers for DataLoader")
    arg_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    arg_parser.add_argument("--deterministic", type=bool, default=False, help="Use deterministic operations for reproducibility")
    arg_parser.add_argument("--out_file", type=str, default="evaluation_results.csv", help="Path to save the results CSV")
    args = arg_parser.parse_args()

    print("Args:")
    for key, value in vars(args).items():
        print(f"    > {key}: {value}")
    print()

    load_dotenv()
    transformers.logging.set_verbosity_error()
    set_seed(args.seed, args.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dataset = CounterfactualAudioDataset("data/clotho_eval_metadata.csv")
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
    model = AudioTextCounterfactualModel().to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint_files = [args.checkpoint]
    else:
        search_pattern = os.path.join("models/", "checkpoint*.pth")
        checkpoint_files = sorted(glob.glob(search_pattern))

    results_summary = {}
    for ckpt_path in checkpoint_files:
        filename = os.path.basename(ckpt_path)
        print(f"{'='*50}")
        print(f"Evaluating: {filename}")
        print(f"{'='*50}")

        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.audio_encoder.load_state_dict(checkpoint['model_state_dict'])

            top1, top10 = evaluate_retrieval(model, test_loader, device)

            print(f"Result -> Top-1: {top1:.4f} | Top-10: {top10:.4f}\n")
            results_summary[filename] = {"Top-1": top1, "Top-10": top10}

        except Exception as e:
            print(f"Failed to evaluate {filename}. Error: {e}\n")

    print("\n" + "="*60)
    print(f"{'Summary':^60}")
    print("="*60)
    print(f"{'Checkpoint Name':<35} | {'Top-1':<8} | {'Top-10':<8}")
    print("-" * 60)

    for filename, scores in results_summary.items():
        name = filename.removeprefix("checkpoint_").removesuffix("_epoch_30.pth")
        print(f"{name:<35} | {scores['Top-1']:.4f}   | {scores['Top-10']:.4f}")
    print("="*60)

    with open(args.out_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Checkpoint Name", "Top-1 Accuracy", "Top-10 Accuracy"])
        for filename, scores in results_summary.items():
            writer.writerow([filename, f"{scores['Top-1']:.4f}", f"{scores['Top-10']:.4f}"])

