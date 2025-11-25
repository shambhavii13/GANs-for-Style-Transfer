**Generative Adversarial Networks for Image Reconstruction and Style Transfer with Perceptual Loss Enhancement**

This repository contains PyTorch implementations for training and testing Style Transfer on various image-to-image translation datasets (e.g., apple2orange, vangogh2photo). It includes preprocessing, model training, image generation, and evaluation scripts.\

---

1. Environment Setup
   Create a virtual environment (optional but recommended)
   ```bash
   python3 -m venv cycle_env
   source cycle_env/bin/activate

Install the required dependencies:
```bash
pip install -r requirements.txt

2. Download Dataset
```bash
   bash download_dataset.sh apple2orange
 
