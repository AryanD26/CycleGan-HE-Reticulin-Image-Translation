# ==============================================================================
# train.py
#
# This script trains a CycleGAN model for image-to-image translation.
# It is designed to be run from the command line, with all hyperparameters
# and paths configurable via arguments.
#
# Usage:
#   python train.py --run-id my_first_run --num-epochs 50 --batch-size 16
#
# To resume training:
#   python train.py --run-id my_first_run --load-model --epoch-to-load-from 20
# ==============================================================================

import os
import argparse
import random
import time
import logging
import itertools
import glob
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

# --- Import custom modules from the 'src' directory ---
# This assumes your models.py and dataset.py are in a 'src' folder.
from src.models import UNetGenerator, Discriminator # Using the U-Net from Training (1).ipynb
from src.dataset import PairedImageDataset

# ==============================================================================
# 1. UTILITY CLASSES AND FUNCTIONS
# ==============================================================================

class ReplayBuffer:
    """
    A buffer to store a history of previously generated images. This helps
    stabilize GAN training by training the discriminator on a mix of recent
    and older fake images.
    """
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data_batch):
        images_to_return = []
        for element in data_batch.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                images_to_return.append(element)
            else:
                if random.random() > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    images_to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    images_to_return.append(element)
        return torch.cat(images_to_return)

def save_checkpoint(model, optimizer, scaler, filename):
    """Saves model, optimizer, and scaler states to a file."""
    logging.info(f" => Saving checkpoint: {filename}")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, scaler, lr, checkpoint_file):
    """Loads model, optimizer, and scaler states from a file."""
    logging.info(f" => Loading checkpoint: {checkpoint_file}")
    try:
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(checkpoint["state_dict"])
        if optimizer and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        if scaler and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        logging.info("Checkpoint loaded successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to load checkpoint {checkpoint_file}: {e}")
        return False

# ==============================================================================
# 2. MAIN TRAINING FUNCTION
# ==============================================================================

def main(args):
    """
    Main function to set up and run the training pipeline.
    """
    # --- Setup Device and Directories ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    os.makedirs(os.path.join(args.samples_dir, args.run_id), exist_ok=True)

    # --- Setup Logging ---
    log_filename = f"train_log_{args.run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(args.logs_dir, log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_filepath), logging.StreamHandler()]
    )

    logging.info("--- Starting New Training Session ---")
    logging.info(f"Run ID: {args.run_id}")
    logging.info(f"Device: {DEVICE}")
    logging.info(f"Hyperparameters: {vars(args)}")

    # --- Reproducibility ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if DEVICE == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # --- Initialize Models, Optimizers, and Losses ---
    gen_H = UNetGenerator(img_channels=3).to(DEVICE) # H&E -> Reticulin
    gen_R = UNetGenerator(img_channels=3).to(DEVICE) # Reticulin -> H&E
    disc_H = Discriminator(in_channels=3).to(DEVICE) # Discriminator for H&E
    disc_R = Discriminator(in_channels=3).to(DEVICE) # Discriminator for Reticulin

    opt_gen = optim.Adam(itertools.chain(gen_H.parameters(), gen_R.parameters()), lr=args.lr_gen, betas=(0.5, 0.999))
    opt_disc_H = optim.Adam(disc_H.parameters(), lr=args.lr_disc, betas=(0.5, 0.999))
    opt_disc_R = optim.Adam(disc_R.parameters(), lr=args.lr_disc, betas=(0.5, 0.999))

    adv_loss_fn = nn.MSELoss()
    cycle_loss_fn = nn.L1Loss()
    identity_loss_fn = nn.L1Loss()

    scaler_gen = torch.cuda.amp.GradScaler()
    scaler_disc_H = torch.cuda.amp.GradScaler()
    scaler_disc_R = torch.cuda.amp.GradScaler()

    # --- Handle Resuming from Checkpoint ---
    start_epoch = 1
    run_checkpoints_dir = os.path.join(args.checkpoints_dir, args.run_id)
    os.makedirs(run_checkpoints_dir, exist_ok=True)

    if args.load_model:
        epoch_to_load = args.epoch_to_load_from
        logging.info(f"Attempting to resume from epoch {epoch_to_load}...")
        
        success_gh = load_checkpoint(gen_H, opt_gen, scaler_gen, args.lr_gen, os.path.join(run_checkpoints_dir, f"genh_epoch{epoch_to_load}.pth.tar"))
        success_gr = load_checkpoint(gen_R, opt_gen, scaler_gen, args.lr_gen, os.path.join(run_checkpoints_dir, f"genr_epoch{epoch_to_load}.pth.tar"))
        success_dh = load_checkpoint(disc_H, opt_disc_H, scaler_disc_H, args.lr_disc, os.path.join(run_checkpoints_dir, f"disch_epoch{epoch_to_load}.pth.tar"))
        success_dr = load_checkpoint(disc_R, opt_disc_R, scaler_disc_R, args.lr_disc, os.path.join(run_checkpoints_dir, f"discr_epoch{epoch_to_load}.pth.tar"))
        
        if all([success_gh, success_gr, success_dh, success_dr]):
            start_epoch = epoch_to_load + 1
            logging.info(f"Resume successful. Starting training from epoch {start_epoch}.")
        else:
            logging.error("Resume failed. Starting from scratch.")
            start_epoch = 1

    # --- Data Loading ---
    train_dataset = PairedImageDataset(
        root_H_folder=args.train_h_dir,
        root_R_folder=args.train_r_dir,
        domain_name="Train",
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    vis_dataset = PairedImageDataset(root_H_folder=args.test_h_dir, root_R_folder=args.test_r_dir, domain_name="Visualization")
    vis_dataloader = DataLoader(vis_dataset, batch_size=1, shuffle=False)
    vis_sample_H, vis_sample_R = next(iter(vis_dataloader))

    # --- Replay Buffers ---
    buffer_fake_H = ReplayBuffer()
    buffer_fake_R = ReplayBuffer()

    # --- GPU-side Transforms ---
    gpu_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    # ==========================================================================
    # 3. TRAINING LOOP
    # ==========================================================================
    for epoch in range(start_epoch, args.num_epochs + 1):
        loop = tqdm(train_dataloader, desc=f"Epoch [{epoch}/{args.num_epochs}]", leave=True)
        
        for batch_idx, (real_H_raw, real_R_raw) in enumerate(loop):
            real_H = gpu_transform(real_H_raw.to(DEVICE))
            real_R = gpu_transform(real_R_raw.to(DEVICE))

            # --- Train Discriminators ---
            with torch.cuda.amp.autocast():
                # Discriminator H
                fake_H = gen_R(real_R)
                D_H_real = disc_H(real_H)
                D_H_fake = disc_H(buffer_fake_H.push_and_pop(fake_H.detach()))
                D_H_loss = (adv_loss_fn(D_H_real, torch.ones_like(D_H_real)) + adv_loss_fn(D_H_fake, torch.zeros_like(D_H_fake))) / 2

                # Discriminator R
                fake_R = gen_H(real_H)
                D_R_real = disc_R(real_R)
                D_R_fake = disc_R(buffer_fake_R.push_and_pop(fake_R.detach()))
                D_R_loss = (adv_loss_fn(D_R_real, torch.ones_like(D_R_real)) + adv_loss_fn(D_R_fake, torch.zeros_like(D_R_fake))) / 2

            # Backward pass for discriminators
            opt_disc_H.zero_grad()
            scaler_disc_H.scale(D_H_loss).backward()
            scaler_disc_H.step(opt_disc_H)
            scaler_disc_H.update()

            opt_disc_R.zero_grad()
            scaler_disc_R.scale(D_R_loss).backward()
            scaler_disc_R.step(opt_disc_R)
            scaler_disc_R.update()

            # --- Train Generators ---
            with torch.cuda.amp.autocast():
                # Adversarial loss
                D_H_fake = disc_H(fake_H)
                D_R_fake = disc_R(fake_R)
                loss_G_H_adv = adv_loss_fn(D_H_fake, torch.ones_like(D_H_fake))
                loss_G_R_adv = adv_loss_fn(D_R_fake, torch.ones_like(D_R_fake))

                # Cycle-consistency loss
                cycled_H = gen_R(fake_R)
                loss_cycle_H = cycle_loss_fn(real_H, cycled_H)
                cycled_R = gen_H(fake_H)
                loss_cycle_R = cycle_loss_fn(real_R, cycled_R)
                
                # Identity loss
                identity_H = gen_R(real_H)
                loss_identity_H = identity_loss_fn(real_H, identity_H)
                identity_R = gen_H(real_R)
                loss_identity_R = identity_loss_fn(real_R, identity_R)

                # Total generator loss
                total_G_loss = (
                    loss_G_H_adv + loss_G_R_adv +
                    (loss_cycle_H * args.lambda_cycle) +
                    (loss_cycle_R * args.lambda_cycle) +
                    (loss_identity_H * args.lambda_identity) +
                    (loss_identity_R * args.lambda_identity)
                )

            # Backward pass for generators
            opt_gen.zero_grad()
            scaler_gen.scale(total_G_loss).backward()
            scaler_gen.step(opt_gen)
            scaler_gen.update()

            # Update progress bar
            loop.set_postfix(G_loss=total_G_loss.item(), D_H=D_H_loss.item(), D_R=D_R_loss.item())

        # --- End of Epoch Actions ---
        if epoch % args.save_samples_every == 0:
            gen_H.eval()
            gen_R.eval()
            with torch.no_grad():
                fake_R_sample = gen_H(vis_sample_H.to(DEVICE))
                fake_H_sample = gen_R(vis_sample_R.to(DEVICE))
                save_image(fake_R_sample * 0.5 + 0.5, os.path.join(args.samples_dir, args.run_id, f"sample_R_epoch{epoch}.png"))
                save_image(fake_H_sample * 0.5 + 0.5, os.path.join(args.samples_dir, args.run_id, f"sample_H_epoch{epoch}.png"))
            gen_H.train()
            gen_R.train()

        if epoch % args.save_model_every == 0:
            save_checkpoint(gen_H, opt_gen, scaler_gen, os.path.join(run_checkpoints_dir, f"genh_epoch{epoch}.pth.tar"))
            save_checkpoint(gen_R, opt_gen, scaler_gen, os.path.join(run_checkpoints_dir, f"genr_epoch{epoch}.pth.tar"))
            save_checkpoint(disc_H, opt_disc_H, scaler_disc_H, os.path.join(run_checkpoints_dir, f"disch_epoch{epoch}.pth.tar"))
            save_checkpoint(disc_R, opt_disc_R, scaler_disc_R, os.path.join(run_checkpoints_dir, f"discr_epoch{epoch}.pth.tar"))
            
    logging.info("--- Training Complete ---")

# ==============================================================================
# 4. SCRIPT ENTRYPOINT
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a CycleGAN model for stain translation.")

    # --- Path Arguments ---
    parser.add_argument('--train-h-dir', type=str, required=True, help='Path to the training directory for H&E images.')
    parser.add_argument('--train-r-dir', type=str, required=True, help='Path to the training directory for Reticulin images.')
    parser.add_argument('--test-h-dir', type=str, required=True, help='Path to the test directory for H&E images.')
    parser.add_argument('--test-r-dir', type=str, required=True, help='Path to the test directory for Reticulin images.')
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints', help='Directory to save model checkpoints.')
    parser.add_argument('--logs-dir', type=str, default='logs', help='Directory to save training logs.')
    parser.add_argument('--samples-dir', type=str, default='saved_images', help='Directory to save sample generated images.')

    # --- Training Hyperparameters ---
    parser.add_argument('--run-id', type=str, required=True, help='A unique name for this training run.')
    parser.add_argument('--num-epochs', type=int, default=50, help='Total number of epochs to train.')
    parser.add_argument('--batch-size', type=int, default=16, help='Number of images per batch.')
    parser.add_argument('--img-size', type=int, default=512, help='Image size to which all images will be resized.')
    parser.add_argument('--lr-gen', type=float, default=2e-4, help='Learning rate for the generators.')
    parser.add_argument('--lr-disc', type=float, default=2e-4, help='Learning rate for the discriminators.')
    parser.add_argument('--lambda-cycle', type=float, default=10.0, help='Weight for the cycle-consistency loss.')
    parser.add_argument('--lambda-identity', type=float, default=5.0, help='Weight for the identity loss.')
    parser.add_argument('--steps-per-epoch', type=int, default=2000, help='Number of batches to process per epoch for faster feedback.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of worker threads for the DataLoader.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    
    # --- Frequency Arguments ---
    parser.add_argument('--save-model-every', type=int, default=2, help='Frequency (in epochs) to save model checkpoints.')
    parser.add_argument('--save-samples-every', type=int, default=5, help='Frequency (in epochs) to save sample images.')
    
    # --- Resume Training Arguments ---
    parser.add_argument('--load-model', action='store_true', help='Flag to resume training from a checkpoint.')
    parser.add_argument('--epoch-to-load-from', type=int, default=0, help='The epoch number of the checkpoint to load.')
    
    args = parser.parse_args()
    main(args)