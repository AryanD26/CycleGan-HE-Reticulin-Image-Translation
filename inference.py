# ==============================================================================
# inference.py
#
# This script evaluates a trained CycleGAN model. It performs two main tasks:
# 1. Generates a visual gallery of sample translations.
# 2. Calculates the Fréchet Inception Distance (FID) for both translation directions.
#
# Usage:
#   python inference.py --run-id my_first_run --epoch 50
# ==============================================================================

import os
import argparse
import glob
from PIL import Image

import torch
from torchvision import transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm
import tifffile
from torch_fidelity import calculate_metrics

# --- Import custom modules from the 'src' directory ---
from src.models import UNetGenerator # Ensure this matches the generator used in training
from src.dataset import PairedImageDataset

# ==============================================================================
# 1. UTILITY FUNCTIONS
# ==============================================================================

def load_generator(checkpoint_path, device):
    """Loads a generator model from a checkpoint file."""
    model = UNetGenerator(img_channels=3).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        print(f"Successfully loaded generator from: {os.path.basename(checkpoint_path)}")
        return model
    except FileNotFoundError:
        print(f"!!! ERROR: Checkpoint file not found at '{checkpoint_path}'.")
        return None
    except Exception as e:
        print(f"!!! ERROR: Failed to load checkpoint. Reason: {e}")
        return None

def denormalize(t):
    """Converts a tensor from the [-1, 1] range back to [0, 1] for saving."""
    return torch.clamp((t + 1.0) / 2.0, 0.0, 1.0)

def generate_outputs(gen, dataloader, device, output_dir, num_gallery_images, img_size):
    """
    Generates all fake images for a given generator and saves a small gallery.
    """
    os.makedirs(output_dir, exist_ok=True)
    gallery_dir = os.path.join(os.path.dirname(output_dir), "visual_gallery")
    os.makedirs(gallery_dir, exist_ok=True)
    
    gen.eval()
    count = 0
    with torch.no_grad():
        for real_raw, _ in tqdm(dataloader, desc=f"Generating images in {os.path.basename(output_dir)}"):
            real_img = transforms.Resize((img_size, img_size), antialias=True)(real_raw.to(device))
            fake_img_batch = gen(real_img)
            
            for i in range(fake_img_batch.size(0)):
                # Save every generated image as PNG for FID calculation
                save_image(denormalize(fake_img_batch[i]), os.path.join(output_dir, f"fake_{count:05d}.png"))
                
                # Save a few examples to the gallery
                if count < num_gallery_images:
                    # Create a side-by-side comparison image
                    comparison = torch.cat([denormalize(real_img[i].cpu()), denormalize(fake_img_batch[i].cpu())], dim=2)
                    save_image(comparison, os.path.join(gallery_dir, f"gallery_{os.path.basename(output_dir)}_{count:02d}.png"))
                
                count += 1
    print(f"Generated {count} images and saved them to {output_dir}")
    print(f"Visual gallery samples saved to {gallery_dir}")


def convert_tif_folder_to_png(tif_folder, png_folder):
    """
    Converts a folder of .tif files to .png files for FID calculation.
    Skips if the PNG folder already seems complete.
    """
    print(f"\nPreparing real images for FID: Converting .tif to .png")
    print(f"Source: {tif_folder}\nDestination: {png_folder}")
    os.makedirs(png_folder, exist_ok=True)
    
    tif_files = glob.glob(os.path.join(tif_folder, "*.tif"))
    num_pngs = len(glob.glob(os.path.join(png_folder, "*.png")))

    if len(tif_files) == num_pngs and num_pngs > 0:
        print("PNG directory already seems complete. Skipping conversion.")
        return

    for tif_path in tqdm(tif_files, desc="Converting TIF to PNG"):
        try:
            img_np = tifffile.imread(tif_path)
            if img_np.dtype != 'uint8':
                img_np = (img_np / 256).astype('uint8') if img_np.dtype == 'uint16' else img_np.astype('uint8')
            img_pil = Image.fromarray(img_np)
            filename = os.path.basename(tif_path).replace('.tif', '.png')
            img_pil.save(os.path.join(png_folder, filename))
        except Exception as e:
            print(f"Could not convert {tif_path}: {e}")

# ==============================================================================
# 2. MAIN INFERENCE FUNCTION
# ==============================================================================

def main(args):
    """
    Main function to set up and run the evaluation pipeline.
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Starting Evaluation on Device: {DEVICE} ---")
    print(f"Run ID: {args.run_id}, Epoch: {args.epoch}")

    # --- Setup Paths ---
    run_checkpoints_dir = os.path.join(args.checkpoints_dir, args.run_id)
    eval_output_dir = os.path.join("evaluation_results", args.run_id, f"epoch_{args.epoch}")
    os.makedirs(eval_output_dir, exist_ok=True)

    # --- Load Models ---
    checkpoint_path_H = os.path.join(run_checkpoints_dir, f"genh_epoch{args.epoch}.pth.tar") # H&E -> Reticulin
    checkpoint_path_R = os.path.join(run_checkpoints_dir, f"genr_epoch{args.epoch}.pth.tar") # Reticulin -> H&E
    
    gen_H = load_generator(checkpoint_path_H, DEVICE)
    gen_R = load_generator(checkpoint_path_R, DEVICE)

    if not gen_H or not gen_R:
        print("!!! Halting due to model loading failure.")
        return

    # --- Load Data ---
    # We only need the test sets for evaluation
    test_dataset = PairedImageDataset(root_H_folder=args.test_h_dir, root_R_folder=args.test_r_dir, domain_name="Evaluation")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- Generate Fake Images ---
    # These will be saved as PNGs for FID calculation
    fake_R_dir = os.path.join(eval_output_dir, "fake_Reticulin_from_H")
    fake_H_dir = os.path.join(eval_output_dir, "fake_H&E_from_R")
    
    generate_outputs(gen_H, test_dataloader, DEVICE, fake_R_dir, args.num_gallery, args.img_size)
    generate_outputs(gen_R, test_dataloader, DEVICE, fake_H_dir, args.num_gallery, args.img_size)
    
    # --- Convert Real TIFFs to PNGs for a fair comparison ---
    real_H_png_dir = os.path.join(eval_output_dir, "real_H&E_png")
    real_R_png_dir = os.path.join(eval_output_dir, "real_Reticulin_png")
    convert_tif_folder_to_png(args.test_h_dir, real_H_png_dir)
    convert_tif_folder_to_png(args.test_r_dir, real_R_png_dir)

    # --- Calculate FID Scores ---
    print("\n--- Calculating FID Scores ---")
    try:
        # FID for Reticulin -> H&E translation
        metrics_H = calculate_metrics(input1=real_H_png_dir, input2=fake_H_dir, cuda=(DEVICE=="cuda"), fid=True)
        fid_H = metrics_H['frechet_inception_distance']
        print(f"======================================================")
        print(f"  FID (Real H&E vs. Fake H&E): {fid_H:.4f}")
        print(f"======================================================")

        # FID for H&E -> Reticulin translation
        metrics_R = calculate_metrics(input1=real_R_png_dir, input2=fake_R_dir, cuda=(DEVICE=="cuda"), fid=True)
        fid_R = metrics_R['frechet_inception_distance']
        print(f"======================================================")
        print(f"  FID (Real Reticulin vs. Fake Reticulin): {fid_R:.4f}")
        print(f"======================================================")
    except Exception as e:
        print(f"!!! FID calculation failed: {e}")

    print("\n--- Evaluation Complete ---")

# ==============================================================================
# 3. SCRIPT ENTRYPOINT
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained CycleGAN model.")

    # --- Path Arguments ---
    parser.add_argument('--test-h-dir', type=str, required=True, help='Path to the test directory for H&E images.')
    parser.add_argument('--test-r-dir', type=str, required=True, help='Path to the test directory for Reticulin images.')
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints', help='Directory where model checkpoints are saved.')
    
    # --- Evaluation Parameters ---
    parser.add_argument('--run-id', type=str, required=True, help='The unique name of the training run to evaluate.')
    parser.add_argument('--epoch', type=int, required=True, help='The epoch number of the checkpoint to evaluate.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for faster inference.')
    parser.add_argument('--img-size', type=int, default=512, help='Image size used during training.')
    parser.add_argument('--num-gallery', type=int, default=20, help='Number of images to save in the visual gallery.')

    args = parser.parse_args()
    main(args)