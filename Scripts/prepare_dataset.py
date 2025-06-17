# ==============================================================================
# src/prepare_dataset.py
# ==============================================================================
# This script prepares the final training-ready dataset from a folder of
# pre-generated patches. It performs a three-step process:
# 1. Verification: Scans all .tif patches to find and remove corrupt or empty files.
# 2. Splitting: Randomly splits the clean file list into training and testing sets.
# 3. Copying: Copies the files into a clean train/test directory structure.
#
# This script should be run twice: once for the H&E patches and once for the
# Reticulin patches.
# ==============================================================================

import os
import glob
import shutil
import random
import logging
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tifffile

# ==============================================================================
# 1. UTILITY FUNCTIONS
# ==============================================================================

def verify_and_clean_patches(patch_folder):
    """
    Scans a directory of TIFF patches, identifies corrupt or empty files,
    and deletes them.
    """
    print(f"\n--- Verifying and Cleaning Patches in: {patch_folder} ---")
    
    # Use recursive glob to find all .tif files, even in subdirectories
    search_pattern = os.path.join(patch_folder, "**/*.tif")
    all_patches = glob.glob(search_pattern, recursive=True)
    
    if not all_patches:
        print("No .tif patches found in this directory. Skipping.")
        return []

    print(f"Found {len(all_patches)} patches to verify...")
    
    corrupt_files = set()
    for file_path in tqdm(all_patches, desc="Verifying patches"):
        try:
            # The most robust check: try to open the file.
            # This catches both empty files and structurally corrupt ones.
            with tifffile.TiffFile(file_path) as tif:
                # Attempting to access the array data will trigger errors on corrupt files
                _ = tif.pages[0].asarray()
        except Exception as e:
            corrupt_files.add(file_path)
            logging.warning(f"Flagged corrupt file: {os.path.basename(file_path)} | Reason: {e}")

    if corrupt_files:
        print(f"\nFound {len(corrupt_files)} corrupt files. Deleting them...")
        for f_path in tqdm(corrupt_files, desc="Deleting corrupt files"):
            try:
                os.remove(f_path)
            except OSError as e:
                print(f"Error deleting {f_path}: {e}")
        print("Deletion of corrupt files complete.")
    else:
        print("No corrupt files found.")
        
    # Return the list of clean, valid files
    clean_files = [p for p in all_patches if p not in corrupt_files]
    return clean_files


def split_and_copy_files(file_list, output_dir, test_size, random_state):
    """
    Splits a list of files into train/test sets and copies them to new folders.
    """
    print(f"\n--- Splitting {len(file_list)} clean files into train/test sets ---")
    
    # Create the base output directory and the train/test subdirectories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Use scikit-learn for a reproducible, stratified split
    train_files, test_files = train_test_split(
        file_list,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    print(f"Training set size: {len(train_files)}")
    print(f"Test set size: {len(test_files)}")
    
    # Copy files to their new homes
    print(f"\nCopying files to {output_dir}...")
    for f in tqdm(train_files, desc="Copying train files"):
        shutil.copy(f, train_dir)
        
    for f in tqdm(test_files, desc="Copying test files"):
        shutil.copy(f, test_dir)
        
    print("File copying complete.")


# ==============================================================================
# 2. MAIN EXECUTION SCRIPT
# ==============================================================================

def main(args):
    """Main function to orchestrate the entire preparation pipeline."""
    
    print(f"{'='*80}")
    print(f"Starting Data Preparation Pipeline for: {args.source_dir}")
    print(f"{'='*80}")

    # Step 1: Verify and clean the source directory of patches
    clean_patch_files = verify_and_clean_patches(args.source_dir)
    
    if not clean_patch_files:
        print("No valid patch files remaining after cleaning. Halting.")
        return
        
    # Step 2: Split the clean file list and copy to the destination
    split_and_copy_files(
        file_list=clean_patch_files,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    print(f"\n--- Pipeline finished for {args.source_dir} ---")
    print(f"Final training-ready dataset is located at: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean, split, and organize a patch dataset for training.")
    
    parser.add_argument('--source_dir', type=str, required=True, help='The source directory containing the generated patches (e.g., H&E_Patches).')
    parser.add_argument('--output_dir', type=str, required=True, help='The destination directory for the final train/test split (e.g., H&E_split_dataset).')
    parser.add_argument('--test_size', type=float, default=0.2, help='The proportion of the dataset to allocate to the test set (e.g., 0.2 for 20%).')
    parser.add_argument('--random_state', type=int, default=42, help='A seed for the random number generator to ensure reproducible splits.')
    
    args = parser.parse_args()
    main(args)