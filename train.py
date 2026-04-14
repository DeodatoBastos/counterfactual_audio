import os
import argparse

from dotenv import load_dotenv

import transformers
import torch
from torch.utils.data import DataLoader

from dataset import CounterfactualAudioDataset
from models import AudioTextCounterfactualModel
from utils import CLAPLoss, CounterfactualLoss, train, evaluate_retrieval, set_seed


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Train Counterfactual Audio-Text Model")
    arg_parser.add_argument("--checkpoint", type=str, default="models/checkpoint_epoch_15.pth", help="Path to checkpoint to resume training")
    arg_parser.add_argument("--bs", type=int, default=32, help="Batch size for training")
    arg_parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    arg_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    arg_parser.add_argument("--w1", type=float, default=1.0, help="Weight for Angle Loss")
    arg_parser.add_argument("--w2", type=float, default=100.0, help="Weight for Factual Consistency Loss")
    arg_parser.add_argument("--num_workers", type=int, default=20, help="Number of workers for DataLoader")
    arg_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    arg_parser.add_argument("--deterministic", type=bool, default=False, help="Use deterministic operations for reproducibility")
    arg_parser.add_argument("--freeze", type=bool, default=False, help="Freeze all the audio backbone")
    arg_parser.add_argument("--mode", type=str, default="counterfactual", help="The loss used to train the model")
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
    epochs = args.epochs
    lr = args.lr
    w1 = args.w1
    w2 = args.w2
    resume_checkpoint = args.checkpoint

    train_dataset = CounterfactualAudioDataset("data/metadata.csv")
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    model = AudioTextCounterfactualModel(freeze=args.freeze).to(device)
    if args.mode == "baseline":
        criterion = CLAPLoss().to(device)
    else:
        criterion = CounterfactualLoss(margin=0.1, w1=w1, w2=w2).to(device)
    optimizer = torch.optim.AdamW(
        list(model.audio_encoder.parameters()) + \
        list(criterion.parameters()),
        lr=lr
    )

    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=.1, anneal_strategy="cos",
        div_factor=10.0, final_div_factor=100.0,
    )

    start_epoch = 0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Loading checkpoint '{resume_checkpoint}'.")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.audio_encoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed training from epoch {start_epoch + 1}")
    elif resume_checkpoint:
        print(f"Checkpoint '{resume_checkpoint}' not found. Starting from scratch.")

    train(model, optimizer, scheduler, train_loader, criterion, start_epoch, epochs, device, args.mode)

    test_dataset = CounterfactualAudioDataset("data/clotho_eval_metadata.csv") 
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=args.num_workers)
    top1_acc, top10_acc = evaluate_retrieval(model, test_loader, device)

    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-10 Accuracy: {top10_acc:.4f}")
