import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.stats import pearsonr, spearmanr
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from ImageDataset_score import ScoreDataset
from ImageDataset import ImageDataset
from Model import Drew, Drew2, Drew3
from utils import load_image_and_prompt, get_optimizer_and_scheduler, load_image_for_score, rank_loss, \
    get_optimizer_and_scheduler_cosine


def train(current_step):
    model.train()
    aesthetics_total_loss, misalignment_total_loss = 0.0, 0.0
    batch_counter = 0

    for batch in tqdm(train_loader, desc="Training"):
        aesthetics_score = batch['scores'].squeeze().to(device)  # [5]
        filenames = [filename[0] for filename in batch['filenames']]  # [5]

        images = load_image_for_score(filenames)  # [5, C=3, H=224, W=224]
        images = torch.stack(images, dim=0).to(device)  # [5, C, H, W]

        with autocast('cuda'):
            outputs = model(images)  # [5]
            mse_loss = loss_fn(outputs, aesthetics_score)
            rk_loss = rank_loss(outputs)
            loss = mse_loss + lambda_loss * rk_loss
            # loss = mse_loss * lambda_loss + rk_loss
            # loss = mse_loss
            # loss = rk_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()  # Use Rank and MSE for back propagation
        scaler.step(optimizer)
        scaler.update()

        aesthetics_total_loss += mse_loss.item()  # record MSE Loss

        if batch_counter < 20:
            print(f"Batch {batch_counter + 1} - MSE Loss: {mse_loss.item():.4f}")
            batch_counter += 1

    scheduler.step()

    num_batches = len(train_loader)
    print(f"Aesthetics Score Train Loss: {aesthetics_total_loss / num_batches:.4f}")

    current_step += 1
    return current_step


def evaluate():
    model.eval()
    aesthetics_preds, aesthetics_gt = [], []
    aesthetics_total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            filenames = batch['filename']  # [bs]
            aesthetics_score = batch['scores']["aesthetics_score"].to(device)
            artifact_maps = batch['artifact_map']
            misalignment_maps = batch['misalignment_map']

            images, prompts, artifact_maps, misalignment_maps = load_image_and_prompt(
                filenames, artifact_maps, misalignment_maps)

            images = torch.stack(images, dim=0).to(device)  # (batch_size, C, H, W)

            with autocast('cuda'):
                outputs = model(images)
                loss = loss_fn(outputs, aesthetics_score)
                aesthetics_preds.append(outputs.detach().cpu().numpy())
                aesthetics_gt.append(aesthetics_score.detach().cpu().numpy())

            aesthetics_total_loss += loss

        aesthetics_plcc, _ = pearsonr(np.concatenate(aesthetics_preds).flatten(),
                                      np.concatenate(aesthetics_gt).flatten())
        aesthetics_srcc, _ = spearmanr(np.concatenate(aesthetics_preds).flatten(),
                                       np.concatenate(aesthetics_gt).flatten())

        num_batches = len(test_loader)
        print(f"Aesthetics Score Test Loss: {aesthetics_total_loss / num_batches:.4f}")
        print(f"Aesthetics PLCC: {aesthetics_plcc:.4f}, Aesthetics SRCC: {aesthetics_srcc:.4f}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler('cuda')
model = Drew().to(device)

# checkpoint = torch.load("./checkpoints/Drew_step_0.pth")
# model.load_state_dict(checkpoint)

batch_size = 8

train_dir = "./Data_richhf18k/torch/train"
dev_dir = "./Data_richhf18k/torch/dev"
test_dir = "./Data_richhf18k/torch/test"
train_dataset = ScoreDataset()
dev_dataset = ImageDataset(dev_dir)
test_dataset = ImageDataset(test_dir)
# train batch must be 5
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=10, pin_memory=True)
# dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

for name, param in model.named_parameters():
    if (
            "shared" in name
            or re.search(r"layer\.0(?!\d)", name)
            or re.search(r"layer\.1(?!\d)", name)
            or re.search(r"layer\.2(?!\d)", name)
            or re.search(r"layer\.3(?!\d)", name)
            or re.search(r"layer\.4(?!\d)", name)
            or re.search(r"layer\.5(?!\d)", name)
            or re.search(r"layer\.6(?!\d)", name)
            or re.search(r"layer\.7(?!\d)", name)
            or re.search(r"layer\.8(?!\d)", name)
            or re.search(r"layer\.9(?!\d)", name)
    ):
        param.requires_grad = False
    if "embeddings" in name:
        param.requires_grad = False

# for name, param in model.named_parameters():
#     if "embeddings" in name:
#         param.requires_grad = False
#     if (name.startswith('swin.encoder.layers.0')
#             or name.startswith('swin.encoder.layers.1')):
#         param.requires_grad = False
# for i in range(14):
#     block = model.swin.encoder.layers[2].blocks[i]
#     for name, param in block.named_parameters():
#         param.requires_grad = False

loss_fn = nn.MSELoss()
lambda_loss = 0.75

warmup_iterations = 4
num_iterations = 12
base_learning_rate = 0.005
max_lr = 2e-5
optimizer, scheduler = get_optimizer_and_scheduler_cosine(model, max_lr)
# optimizer, scheduler = get_optimizer_and_scheduler(model, base_learning_rate)


def main():
    save_dir = "./checkpoints"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    current_step = 1
    while current_step < num_iterations:
        print(f"Training Step {current_step}/{num_iterations}")
        print(f"Learning rate {optimizer.param_groups[0]['lr']:.6f}")

        current_step = train(current_step)
        evaluate()

        save_path = f"{save_dir}/Drew_step_{current_step - 1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    print("Training completed!")


if __name__ == "__main__":
    main()

