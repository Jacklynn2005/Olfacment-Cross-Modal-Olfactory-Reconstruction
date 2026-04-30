setup_content = """# Setup and Installation Guide: Olfacment

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
