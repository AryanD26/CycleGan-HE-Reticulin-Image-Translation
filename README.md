# CycleGAN for H&E to Reticulin Stain Translation

This repository contains a complete, end-to-end pipeline for training and evaluating a CycleGAN model to perform image-to-image translation between H&E and Reticulin histopathology stains. The model architecture is based on a U-Net generator, which is effective at preserving high-resolution details.

---

## Project Workflow

This project is structured as a complete, multi-step pipeline. The scripts should be run in the following order:
1.  **Data Preprocessing:**
    *   `scripts/H&E_Patching.py` & `scripts/Reticulin_Patching.py`: Extract small, manageable patches from large Whole Slide Images (WSIs).
    *   `scripts/prepare_dataset.py`: Cleans the extracted patches and splits them into `train` and `test` sets.
2.  **Model Training:**
    *   `train.py`: Trains the CycleGAN model using the prepared datasets.
3.  **Model Evaluation:**
    *   `inference.py`: Evaluates a trained model's performance using a visual gallery and FID scores.

## Project Structure

```
.
├── scripts/                # Data preprocessing scripts
│   ├── H&E_Patching.py
│   ├── Reticulin_Patching.py
│   └── prepare_dataset.py
├── src/                    # Core source code for models and datasets
│   ├── dataset.py
│   └── models.py
├── train.py                # Main script to train the model
├── inference.py            # Main script to evaluate a trained model
├── requirements.txt        # Python dependencies for reproducibility
├── .gitignore              # Specifies files for Git to ignore
└── README.md               # This documentation file
```

---

## Dataset Description

The dataset used to train and evaluate the models in this project was generated from a private collection of **94 whole-slide images (WSIs)**.

The preprocessing pipeline, detailed in the `scripts/` directory, was used to create the final set of image patches. This process involved:
1.  **Tissue Segmentation:** Identifying tissue regions in each WSI to avoid sampling empty background areas.
2.  **Patch Extraction:** Tiling the identified tissue regions into non-overlapping 512x512 pixel patches.
3.  **Data Cleaning & Splitting:** Verifying all patches were valid (not corrupt) and randomly splitting them into training and testing sets.

### Final Patch Distribution

The final number of patches used for the experiment is detailed below:

| Dataset Split | H&E Patches | Reticulin Patches |
| :-------------- | :---------- | :---------------- |
| **Training Set**  | 211,664     | 147,052           |
| **Testing Set**   | 52,916      | 36,765            |

---

## Getting Started

### 1. Prerequisites
- Python 3.8+
- An NVIDIA GPU with CUDA support is highly recommended for training.
- Your own raw Whole Slide Images in `.ome.tif` format (if you wish to replicate the preprocessing).

### 2. Installation
First, clone the repository and install the required Python packages.

```bash
# Clone this repository
git clone https://github.com/YOUR_USERNAME/CycleGAN-HE-Reticulin-Image-Translation.git
cd CycleGAN-HE-Reticulin-Image-Translation

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install all required packages
pip install -r requirements.txt
```

### 3. Data Setup
To run the preprocessing scripts, you must first set up your raw data.

**Step A: Place Raw Images**
- Create a folder named `Images/` in the root directory of this project.
- Place all your raw H&E and Reticulin `.ome.tif` WSIs inside this `Images/` folder.

**Step B: Run Preprocessing Scripts**
- The preprocessing scripts (detailed in the next section) will automatically use the images from the `Images/` folder.
- After running, your project folder will contain the final `H&E_split_dataset/` and `Retic_split_dataset/` directories needed for training.

*(Note: The `Images/`, `H&E_split_dataset/`, and `Retic_split_dataset/` folders are all listed in the `.gitignore` file and will not be uploaded to GitHub.)*

---

## Step-by-Step Pipeline Execution

### Step 1: Data Preprocessing

These scripts will process your raw WSIs from the `Images/` folder and create the final datasets needed for training.

**1A. Extract Patches from WSIs**
This process will create an `Output/` directory containing the extracted patches.

```bash
# Process H&E images
python scripts/H&E_Patching.py

# Process Reticulin images
python scripts/Reticulin_Patching.py
```

**1B. Clean and Split Patches into Train/Test Sets**
This step takes the raw patches from `Output/`, cleans them of any corrupt files, and creates the final `H&E_split_dataset/` and `Retic_split_dataset/` folders.

```bash
# Create H&E train/test split (assuming 80/20 split)
python scripts/prepare_dataset.py --source_dir Output/ --output_dir H&E_split_dataset --test_size 0.2

# Create Reticulin train/test split (assuming 80/20 split)
python scripts/prepare_dataset.py --source_dir Output/ --output_dir Retic_split_dataset --test_size 0.2
```

### Step 2: Model Training

Use the `train.py` script to train the model. All hyperparameters can be configured via command-line arguments. You must provide a unique `--run-id` to organize your checkpoints and logs.

```bash
# Example training command
python train.py \
    --run-id unet_run_01 \
    --train-h-dir H&E_split_dataset/train \
    --train-r-dir Retic_split_dataset/train \
    --test-h-dir H&E_split_dataset/test \
    --test-r-dir Retic_split_dataset/test \
    --num-epochs 50 \
    --batch-size 16
```
To resume training, add the `--load-model` and `--epoch-to-load-from <epoch_number>` flags.

### Step 3: Inference and Evaluation

After training is complete, use `inference.py` to evaluate a specific checkpoint.

```bash
# Example inference command
python inference.py \
    --run-id unet_run_01 \
    --epoch 50 \
    --test-h-dir H&E_split_dataset/test \
    --test-r-dir Retic_split_dataset/test
```
This script will create an `evaluation_results/` directory containing a visual gallery and the final FID scores.

---

## Results

Using the dataset described above, our model achieved the following Fréchet Inception Distance (FID) scores after 50 epochs:

-   **FID (Real H&E vs. Fake H&E):** 33.36
-   **FID (Real Reticulin vs. Fake Reticulin):** 42.12

*(Lower FID is better, indicating that the generated images are statistically more similar to real images.)*
