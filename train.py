import os
import argparse

from dotenv import load_dotenv

import transformers
import torch
from torch.utils.data import DataLoader

from dataset import CounterfactualAudioDataset
from models import AudioTextCounterfactualModel
from utils import CounterfactualLoss, train, evaluate_retrieval


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Train Counterfactual Audio-Text Model")
    arg_parser.add_argument("--checkpoint", type=str, default="models/checkpoint_epoch_15.pth", help="Path to checkpoint to resume training")
    arg_parser.add_argument("--bs", type=int, default=32, help="Batch size for training")
    arg_parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    arg_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    arg_parser.add_argument("--w1", type=float, default=1.0, help="Weight for Angle Loss")
    arg_parser.add_argument("--w2", type=float, default=100.0, help="Weight for Factual Consistency Loss")
    arg_parser.add_argument("--num_workers", type=int, default=20, help="Number of workers for DataLoader")
    args = arg_parser.parse_args()

    for key, value in vars(args).items():
        print(f"{key}: {value}")

    load_dotenv()
    transformers.logging.set_verbosity_error()

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    print(f"Using device: {device}")

    bs = args.bs
    epochs = args.epochs
    lr = args.lr
    w1 = args.w1
    w2 = args.w2
    resume_checkpoint = args.checkpoint

    train_dataset = CounterfactualAudioDataset("data/metadata.csv")
    test_dataset = CounterfactualAudioDataset("data/clotho_eval_metadata.csv") 

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=args.num_workers)

    model = AudioTextCounterfactualModel().to(device)
    criterion = CounterfactualLoss(margin=0.1, w1=w1, w2=w2)

    optimizer = torch.optim.AdamW(model.audio_encoder.parameters(), lr=lr)
    scaler = torch.GradScaler(device_type)

    start_epoch = 0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Loading checkpoint '{resume_checkpoint}'...")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.audio_encoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed training from epoch {start_epoch + 1}")
    elif resume_checkpoint:
        print(f"Checkpoint '{resume_checkpoint}' not found. Starting from scratch.")

    train(model, optimizer, train_loader, criterion, start_epoch, epochs, device)
    top1_acc, top10_acc = evaluate_retrieval(model, test_loader, device)

    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-10 Accuracy: {top10_acc:.4f}")
