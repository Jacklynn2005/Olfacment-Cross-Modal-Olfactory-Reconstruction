# Olfacment: Cross-Modal-Olfactory-Reconstruction
**Olfacment** is a generative framework that reconstructs high-fidelity 32-channel chemical signals from environmental images. By bridging the gap between computer vision and olfactory chemistry, the system generates physically plausible sensor responses to recommend real-world fragrances corresponding to a visual scene.

---

### What it Does
The project implements a three-stage pipeline to transform pixels into chemistry. First, a Signal VAE learns the "grammar" of 32-channel electronic nose data to ensure reconstructions respect sensor physics. Second, a COIP (Contrastive Olfactory-Image Pre-training) model aligns visual features from a Vision Transformer with chemical features from a 1D-CNN into a shared embedding space. Finally, a Latent Diffusion Model guided by Classifier-Free Guidance (CFG) sculpts a clean olfactory latent vector from Gaussian noise. The system then decodes this latent into a time-series signal and performs a vector search against a fragrance database to identify the closest product match based on cosine similarity.

---

### Quick Start

1.  **Install Dependencies:**
    ```bash
    pip install torch torchvision diffusers transformers pandas pillow scikit-learn
    ```

2.  **Initialize the Inference Engine:**
    Ensure your pre-trained weights (`best_vae.pt`, `best_coip.pt`, `best_diff.pt`) and `sensor_metadata.pt` are in the root directory.

3.  **Run a Prediction:**
    ```python
    from inference import ScentInferenceSystem
    
    # Initialize with Duke cluster paths
    system = ScentInferenceSystem(weight_paths=my_paths, fragrance_db=my_gallery)
    
    # Generate recommendation and printed report
    fragrance, score, signal = system.predict_scent("nyc_street_view.jpg")
    ```

---

### Video Links
* **Project Demo:** [[Link to Demo Video]](https://prodduke-my.sharepoint.com/:f:/r/personal/jr528_duke_edu/Documents/CS%20372%20PF?csf=1&web=1&e=MwdsmF)
* **Technical Walkthrough:** [[Link to Technical Walkthrough]](https://prodduke-my.sharepoint.com/:f:/r/personal/jr528_duke_edu/Documents/CS%20372%20PF?csf=1&web=1&e=MwdsmF)

---

### Evaluation

<img width="1953" height="864" alt="0a60003df1f57663e8092032a43d5014" src="https://github.com/user-attachments/assets/3cb41497-7e4c-423f-9803-ce921ba0c50c" />

<img width="506" height="400" alt="image" src="https://github.com/user-attachments/assets/1a124b5f-5e53-47a4-8163-a7edbfc02d79" />

<img width="635" height="289" alt="image" src="https://github.com/user-attachments/assets/96f91b09-c2c3-4dbe-8c4d-3732937d6992" />

<img width="401" height="184" alt="image" src="https://github.com/user-attachments/assets/54c66ebe-383c-4253-825d-1f98887ffedb" />

---


