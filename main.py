import torch
import torch.optim as optim
import os
import argparse
import config
from dataset import get_dataloader
from model import CVAE
from utils import WarmupCosineLR
from trainer import train
from download_data import download_celeba

def main(args):
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    
    print(f"Using device: {config.DEVICE}")
    print(f"Running for {config.NUM_EPOCHS} epochs with batch size {config.BATCH_SIZE}.")

    try:
        data_root = download_celeba()
    except Exception as e:
        print(f"Error downloading data: {e}")
        return

    config.DATA_ROOT = data_root
    config.IMG_DIR = os.path.join(data_root, "img_align_celeba/img_align_celeba")
    config.CSV_PATH = os.path.join(data_root, "list_attr_celeba.csv")
    
    if not os.path.exists(config.IMG_DIR) or not os.path.exists(config.CSV_PATH):
        print("Error: Could not find 'img_align_celeba' or 'list_attr_celeba.csv' in dataset directory.")
        return

    print("Loading dataset...")
    dataloader = get_dataloader(config)
    
    model = CVAE(
        latent_dim=config.LATENT_DIM,
        num_attrs=config.NUM_ATTRS,
        base_channels=config.BASE_CHANNELS
    ).to(config.DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    total_steps = len(dataloader) * config.NUM_EPOCHS
    warmup_steps = len(dataloader) * config.WARMUP_EPOCHS
    
    scheduler = WarmupCosineLR(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        max_lr=config.LEARNING_RATE
    )

    train(model, dataloader, optimizer, scheduler, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CVAE on CelebA.")
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=config.NUM_EPOCHS, 
        help=f"Number of training epochs (default: {config.NUM_EPOCHS})"
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=config.BATCH_SIZE, 
        help=f"Batch size for training (default: {config.BATCH_SIZE})"
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=config.LEARNING_RATE, 
        help=f"Peak learning rate (default: {config.LEARNING_RATE})"
    )
    
    args = parser.parse_args()
    main(args)