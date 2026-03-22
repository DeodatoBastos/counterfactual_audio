from tqdm import tqdm
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from models import AudioTextCounterfactualModel


class CounterfactualLoss(nn.Module):
    """
    Composite Loss function combining Factual Consistency and Angle Loss
    """
    def __init__(self, margin=0.1, w1=1.0, w2=100.0):
        super().__init__()
        self.margin = margin
        self.w1 = w1  # Angle Loss Weight
        self.w2 = w2  # Factual Consistency Weight
        self.mse = nn.MSELoss()

    def forward(self, audio_embeds: Tensor, factual_embeds: Tensor, counterfactual_embeds: Tensor):
        # Equation 5: Factual Consistency Loss (Mean Squared Error)
        # Drives the audio embedding towards the factual caption
        l_factual: Tensor = self.mse(audio_embeds, factual_embeds)

        # Equation 3 & 4: Angle Loss (Triplet Margin Cosine)
        # Cosine similarity is the dot product of L2 normalized vectors
        cos_factual: Tensor = torch.sum(audio_embeds * factual_embeds, dim=-1)
        cos_counterfactual: Tensor = torch.sum(audio_embeds * counterfactual_embeds, dim=-1)

        # Punishes the model if the counterfactual similarity is greater than the factual similarity
        # L_angle = max(0, cos(a, cf) - cos(a, f) + margin)
        l_angle: Tensor = torch.mean(torch.clamp(cos_counterfactual - cos_factual + self.margin, min=0.0))

        # Equation 6: Total Composite Loss
        total_loss: Tensor = (self.w1 * l_angle) + (self.w2 * l_factual)

        return total_loss, l_angle, l_factual


def evaluate_retrieval(model: AudioTextCounterfactualModel, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    """
    Evaluates the model on the Language-Based Audio Retrieval task.
    For each text query, computes cosine similarity with all audio embeddings and ranks them.
    Returns Top-1 and Top-10 Recall metrics.
    """
    model.eval()
    all_audio_embeds = []
    all_text_embeds = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            audio, captions, _ = batch
            audio = audio.to(device)

            audio_embeds = model.encode_audio(audio)
            text_embeds = model.encode_text(captions, device)

            all_audio_embeds.append(audio_embeds.cpu())
            all_text_embeds.append(text_embeds.cpu())

    all_audio_embeds = torch.cat(all_audio_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)

    # Compute Cosine Similarity Matrix (N_text_queries, N_audio_database)
    # Because vectors are normalized, matrix multiplication yields cosine similarity
    sim_matrix = torch.matmul(all_text_embeds, all_audio_embeds.T)

    N = sim_matrix.shape[0]
    top1_correct = 0
    top10_correct = 0

    # Calculate Top-k Recall
    for i in range(N):
        ranked_indices = torch.argsort(sim_matrix[i], descending=True)

        # The correct audio for text query `i` is at index `i`
        if ranked_indices[0] == i:
            top1_correct += 1

        if i in ranked_indices[:10]:
            top10_correct += 1

    top1_acc = top1_correct / N
    top10_acc = top10_correct / N

    return top1_acc, top10_acc


def train(model: AudioTextCounterfactualModel, optimizer: torch.optim.Optimizer,
          train_loader: DataLoader, criterion: CounterfactualLoss, start_epoch: int,
          epochs: int, device: torch.device):
    """Trains the model using the provided DataLoader and loss function.
    Saves checkpoints every 5 epochs and the final model weights at the end.
    """
    model.train()
    for epoch in range(start_epoch, epochs):
        total_loss_epoch = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, batch in progress_bar:
            audio, factual_captions, counterfactual_captions = batch
            audio = audio.to(device)

            optimizer.zero_grad()

            audio_embeds = model.encode_audio(audio)
            factual_embeds = model.encode_text(factual_captions, device)
            counterfactual_embeds = model.encode_text(counterfactual_captions, device)

            loss, l_angle, l_factual = criterion(audio_embeds, factual_embeds, counterfactual_embeds)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.audio_encoder.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss_epoch += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

            if batch_idx == len(train_loader) - 1:
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Avg Loss": f"{total_loss_epoch/len(train_loader):.4f}"})

        if (epoch + 1) % 5 == 0:  # Save checkpoint every epoch
            checkpoint_path = f"models/checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.audio_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved at '{checkpoint_path}'")

    torch.save(model.audio_encoder.state_dict(), "models/counterfactual_audio_encoder.pth")
    print("Training Finished. Model weights saved.")
