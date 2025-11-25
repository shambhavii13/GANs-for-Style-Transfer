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
```
2. Download Dataset
   ```bash
   bash download_dataset.sh apple2orange
   bash download_dataset.sh vangogh2photo

The dataset will be placed under:
```bash
/datasets/
```

3. Training
   Use train.py to train the model.\
   --root: Put Dataset Path\
   --name: Specify apple2orange or vangog2photo

   Example-
   ```bash
   python train.py --root /path/to/datasets/apple2orange --name apple2orange
This will create checkpoint files under:
```bash
checkpoints/apple2orange/
```

4. Evaluation
   Use test.py to generate trnslated image from the train model\
   --epoch: specify checkpoint of the epoch to be loaded\
   --name: specify name of dataset (apple2orange or vangogh2photo)\
   --direction: A2B or B2A\
   --model_name: specify name of checkpoint folder ( keep same as --name)

   Example-
   ```bash
   python test.py \
    --epoch 20 \
    --name apple2orange \
    --model_name apple2orange \
    --direction A2B
Generated images will be saved to:
```bash
results/<model_name>/<direction>/
```



`

