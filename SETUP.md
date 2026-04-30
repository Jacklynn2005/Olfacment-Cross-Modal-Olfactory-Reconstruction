# Setup and Installation Guide: Olfacment

This guide provides the necessary steps to configure the environment, prepare the dataset, and execute the training pipeline for the cross-modal olfactory reconstruction project.

## 1. Prerequisites

### Hardware Requirements
* **GPU:** NVIDIA GPU with at least 16GB VRAM is strongly recommended for training the Diffusion U-Net and COIP models.
* **Storage:** Approximately 50GB of disk space for the NYC Smells dataset and model checkpoints.

### Software Requirements
* **Operating System:** Linux (Ubuntu 20.04+ recommended) or macOS (training on Apple Silicon supported but CUDA is preferred).
* **Python:** Version 3.8 or higher.
* **CUDA:** Version 11.0 or higher (for GPU acceleration).

## 2. Installation

Clone the repository and install the required Python libraries. It is recommended to use a virtual environment or a Conda environment.

```bash
# Create and activate a conda environment
conda create -n ny-smells python=3.9
conda activate ny-smells

# Install core deep learning libraries
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install Hugging Face and Diffusion libraries
pip install diffusers transformers

# Install data processing and utility libraries
pip install pandas scikit-learn pillow tqdm matplotlib
```

## 3. Data Preparation
### Dataset Acquisition

Ensure the NYC Smells dataset (Voxel51) is structured correctly in your data directory:

```Plaintext
data/
└── nyc_smells/
    ├── images/
    └── signals/ (32-channel .npy or .h5 files)
```
    
### Partitioning & Stratification

Run the stratification script to ensure that samples from the same Object ID are grouped together to prevent data leakage.

```bash
python scripts/partition_data.py --metadata metadata.csv --output_dir ./indices/
This will generate train_ids.txt, val_ids.txt, and test_ids.txt.
```

### Calibration

Before training the VAE, calculate the per-channel normalization constants to handle the varying physical resistances of the 32 sensors.

```bash
python scripts/calculate_constants.py --train_ids ./indices/train_ids.txt
Output: sensor_metadata.pt
```

## 4. Execution Pipeline
The project must be trained in a specific modular sequence. Each stage saves checkpoints that are required by the subsequent stage.

### Step 1: Signal VAE Training

Train the VAE to learn the "grammar" of the olfactory signals.

```bash
python train_vae.py --config configs/vae_config.yaml
Target Output: checkpoints/best_vae.pt
```

### Step 2: COIP Alignment

Train the Vision and Olfactory encoders to align in a shared latent space.

```bash
python train_coip.py --vae_path checkpoints/best_vae.pt
Target Output: checkpoints/best_coip.pt
```

### Step 3: Latent Diffusion Training

Train the U-Net sculptor to generate scent latents guided by image features.

```bash
python train_diffusion.py --vae_path checkpoints/best_vae.pt --coip_path checkpoints/best_coip.pt
Target Output: checkpoints/best_diffusion_unet.pt
```

## 5. Inference & Evaluation
To generate a fragrance recommendation from a single image, use the provided ScentInferenceSystem wrapper.

```Python

from inference import ScentInferenceSystem

# Configuration
weight_paths = {
    'vae': 'checkpoints/best_vae.pt',
    'vision': 'checkpoints/best_coip.pt',
    'unet': 'checkpoints/best_diffusion_unet.pt'
}

# Run System
system = ScentInferenceSystem(weight_paths=weight_paths, meta_path='sensor_metadata.pt')
result = system.predict_scent("path/to/image.jpg")
```
