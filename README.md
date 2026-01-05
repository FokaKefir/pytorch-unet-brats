# Multimodal Brain Tumor Segmentation (BraTS 2020)

This repository contains a PyTorch-based deep learning pipeline for automated brain tumor segmentation using the **BraTS 2020** dataset. The project explores the evolution of U-Net architectures, from the vanilla implementation to **Attention-guided Residual U-Nets**.

## 🚀 Key Features

* **Flexible Architectures:** Choice between three models:
* `UNet1`: Standard Encoder-Decoder.
* `UNet2`: Residual U-Net for better gradient flow.
* `UNet3`: Attention-Residual U-Net for localized focus.


* **Hybrid Optimization:** Loss function combining **Categorical Cross-Entropy/Focal Loss** and **Dice Loss** ().
* **Production-Ready Logging:** Full integration with **Weights & Biases (W&B)** for:
* Real-time metric tracking.
* Gradient and weight distribution histograms.
* Visual segmentation comparison artifacts.


* **Robust Evaluation:** Comprehensive statistical analysis including Multi-class Confusion Matrices, Sensitivity, Specificity, and Dice Scores.



## 📂 Project Structure

```bash
.
├── notebook.ipynb        # Main research and training pipeline
├── models/               # Locally saved trained model weights (.pth)
├── wandb/                # Local experiment logs (ignored by git)
└── .gitignore            # Clean repository management

```



## 🛠️ Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/brats-segmentation.git
cd brats-segmentation
```


2. **Dataset:**
Ensure the `brats20-dataset-training-validation` folder is in the root directory.
*(Note: The dataset is excluded from the repository via .gitignore due to size).*
3. **Weights & Biases:**
Login to your W&B account to track the experiments:
```bash
wandb login
```



## 🧠 Training Configuration

The project uses a centralized configuration object for reproducibility:

| Parameter | Value | Description |
| --- | --- | --- |
| `init_lr` | 1e-3 | Initial learning rate |
| `weight_decay` | 0.00008 | AdamW regularization |
| `percent_dice_loss` | 0.5 | Weight of Dice Loss in Hybrid Loss |
| `model_depth` | 3 - 4 | Number of downsampling steps |
| `es_patience` | 10 | Early stopping patience |


## 📊 Evaluation & Results

The pipeline produces a detailed evaluation matrix at the end of the test phase. It automatically calculates:

* **Voxel-wise Accuracy**
* **Class-wise Dice Scores**
* **Sensitivity & Specificity**

### Visual Results

The model generates side-by-side comparisons of **Input MRI Modalities**, **Ground Truth**, and **Model Prediction** for qualitative assessment, which are automatically synced to your W&B dashboard.


## 📖 How To Use

### 1. Data Preparation

Place the BraTS 2020 dataset in the root directory. The structure should look like this:

```text
.
└── brats20-dataset-training-validation/
    ├── BraTS20_Training_001/
    │   ├── BraTS20_Training_001_flair.nii.gz
    │   ├── BraTS20_Training_001_t1ce.nii.gz
    │   └── ...
    └── ...

```

### 2. Configure Experiment

Inside the `notebook.ipynb`, locate the `config` dictionary. You can easily switch models and adjust hyperparameters:

```python
config = dict(
    model_name='unet3', # Options: 'unet1', 'unet2', 'unet3'
    model_depth=4,
    model_filters=256,
    init_lr=1e-3,
    # ... other params
)
```

### 3. Run Training

1. Open `notebook.ipynb` in VS Code or Jupyter.
2. Run all cells.
3. The script will automatically:
* Initialize a new **Weights & Biases** run.
* Build the selected U-Net architecture.
* Start the training loop with Early Stopping.
* Save the best model to `models/model.pth`.



### 4. Inference & Visualization

After training, the final cells will pick a random slice from the test set and generate a side-by-side comparison:

* **Input Channels** (FLAIR, T1ce, etc.)
* **Ground Truth** (Expert annotation)
* **Model Prediction** (AI output)



## 📝 License

This project is for educational and research purposes as part of the BraTS 2020 challenge.


