# ==============================================================================
# Retic_Patch.py
# ==============================================================================
# This script processes Whole Slide Images (WSIs) of Reticulin stains. It performs
# tissue segmentation, extracts patches of a specified size, and saves both
# the patches and their coordinates in an XML file.
# ==============================================================================

import os
import time
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import xml.etree.ElementTree as ET
from xml.dom import minidom
import tifffile
import glob

# Allow PIL to open very large images without raising an error.
Image.MAX_IMAGE_PIXELS = None

# ==============================================================================
# 1. CONFIGURATION SETTINGS
# ==============================================================================
# --- Input Settings ---
# Directory containing the raw Reticulin Whole Slide Images.
INPUT_DIR = 'Images'
# Glob pattern to find the specific Reticulin files to process.
INPUT_PATTERN = '*Retic*.ome.tif' 

# --- Output Settings ---
# Main directory where all processed outputs will be saved.
OUTPUT_DIR = 'Output'

# --- Patch & Masking Settings ---
PATCH_SIZE = 512
STRIDE = 512  # Set to PATCH_SIZE for non-overlapping patches.
TISSUE_THRESHOLD_PERCENT = 0.5 # A patch must contain at least this much tissue to be saved.
OTSU_THRESHOLD_METHOD = cv2.THRESH_BINARY_INV # Inverts the mask so tissue is white (255).

# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================

def create_tissue_mask(image_np):
    """Creates a binary tissue mask using Otsu's thresholding."""
    print("  Creating tissue mask from full-resolution image...")
    
    if image_np.ndim == 3 and image_np.shape[2] >= 3:
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image_np

    _, mask = cv2.threshold(gray_image, 0, 255, OTSU_THRESHOLD_METHOD + cv2.THRESH_OTSU)
    print("  ...Tissue mask created.")
    return mask

def check_tissue_percentage(mask_patch, threshold):
    """Checks if the percentage of tissue in a patch exceeds a threshold."""
    if mask_patch.size == 0: return False
    tissue_pixels = np.count_nonzero(mask_patch)
    return (tissue_pixels / mask_patch.size) >= threshold

def prettify_xml(elem):
    """Returns a pretty-printed XML string for an ElementTree element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

# ==============================================================================
# 3. CORE IMAGE PROCESSING FUNCTION
# ==============================================================================

def process_image(image_path):
    """Loads, masks, and extracts patches from a single WSI."""
    print(f"\n{'='*80}\nProcessing WSI: {os.path.basename(image_path)}\n{'='*80}")
    
    image_filename = os.path.basename(image_path)
    base_name = f"{image_filename.split('_')[0]}_{image_filename.split('_')[1]}"
    
    # --- Setup Output Directories and XML ---
    image_output_dir = os.path.join(OUTPUT_DIR, base_name)
    patches_output_dir = os.path.join(image_output_dir, 'patches')
    os.makedirs(patches_output_dir, exist_ok=True)
    
    xml_output_path = os.path.join(image_output_dir, f'{base_name}_patch_coordinates.xml')
    
    if os.path.exists(xml_output_path):
        tree = ET.parse(xml_output_path)
        xml_root = tree.getroot()
        print(f"Appending to existing XML file: {xml_output_path}")
    else:
        xml_root = ET.Element("Patches")
        xml_root.set("PatchSize", str(PATCH_SIZE))
        print(f"Creating new XML file: {xml_output_path}")

    # --- Load Image ---
    try:
        print("Loading image with tifffile...")
        retic_image_np = tifffile.imread(image_path)
        if retic_image_np.dtype != np.uint8:
            retic_image_np = (retic_image_np / 256).astype(np.uint8) if retic_image_np.dtype == np.uint16 else retic_image_np.astype(np.uint8)
        print(f"Image loaded. Shape: {retic_image_np.shape}")
    except Exception as e:
        print(f"!!! FATAL ERROR loading {image_filename}: {e}")
        return 0

    # --- Create and Save Tissue Mask ---
    tissue_mask = create_tissue_mask(retic_image_np)
    mask_output_path = os.path.join(image_output_dir, f'tissue_mask_{image_filename}.png')
    cv2.imwrite(mask_output_path, tissue_mask)
    print(f"Tissue mask saved to: {mask_output_path}")

    # --- Generate and Save Patches ---
    print("Extracting and saving patches...")
    height, width, _ = retic_image_np.shape
    saved_count = 0
    start_index = len(xml_root.findall("Patch"))
    
    coords_to_process = []
    for y in range(0, height - PATCH_SIZE + 1, STRIDE):
        for x in range(0, width - PATCH_SIZE + 1, STRIDE):
            coords_to_process.append((x, y))

    for x, y in tqdm(coords_to_process, desc="Saving Patches"):
        mask_patch = tissue_mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
        if check_tissue_percentage(mask_patch, TISSUE_THRESHOLD_PERCENT):
            patch_np = retic_image_np[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            
            patch_filename = f"{base_name}_patch_{start_index + saved_count:06d}.tif"
            patch_save_path = os.path.join(patches_output_dir, patch_filename)
            tifffile.imsave(patch_save_path, patch_np, compress=6)

            patch_element = ET.SubElement(xml_root, "Patch")
            patch_element.set("FileName", os.path.join('patches', patch_filename))
            patch_element.set("X", str(x))
            patch_element.set("Y", str(y))
            patch_element.set("SourceImage", image_filename)
            saved_count += 1
            
    print(f"Saved {saved_count} new patches.")

    # --- Write Final XML File ---
    if saved_count > 0:
        with open(xml_output_path, "w", encoding="utf-8") as f:
            f.write(prettify_xml(xml_root))
        print(f"Updated XML coordinate file saved.")
        
    return saved_count

# ==============================================================================
# 4. MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    print("--- Starting Reticulin WSI Patch Extraction ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    search_path = os.path.join(INPUT_DIR, INPUT_PATTERN)
    input_files = glob.glob(search_path)
    
    if not input_files:
        print(f"!!! No files found matching '{search_path}'. Exiting.")
        exit()
    
    print(f"Found {len(input_files)} Reticulin image(s) to process.")
    
    total_patches = 0
    for image_file in input_files:
        total_patches += process_image(image_file)
    
    print(f"\n{'='*80}\nAll Reticulin processing complete. Total patches extracted in this run: {total_patches}\n{'='*80}")