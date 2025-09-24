### 🛠️ Setup & Training

#### 1. Install Dependencies  
First, install the required packages:

```bash
pip install torchao liger_kernel pyarrow tensorboard
```

> 💡 **Note**: `torchao` and `liger_kernel` may require a recent version of PyTorch (≥2.3) and a CUDA-enabled environment for optimal performance.

#### 2. Prepare Data  
1. Download all files from this repository.  
2. Place them in a single working directory.  
3. Inside this directory, create a subfolder named `128`.  
4. Download the training data (Parquet files) into the `128/` folder:  
   🔗 **[TinyCorpus-v2](https://huggingface.co/datasets/xTimeCrystal/TinyCorpus-v2)**

#### 3. File Structure  
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

#### 4. Start Training  
Run the training script from inside your-training-folder:

```bash
python trainGPT-token.py
```

> By default, the script logs training loss and other metrics to a directory called `runs/` using PyTorch’s `SummaryWriter`.

#### 5. Monitor Training with TensorBoard  
While training is running (or after it finishes), launch TensorBoard to visualize the loss curve:

```bash
tensorboard --logdir=runs
```

Then open your browser and go to:  
👉 **http://localhost:6006**

You’ll see a real-time plots of the training loss (refreshes every 30s).

#### 6. Troubleshooting Out-of-Memory (OOM) Errors  
If you encounter memory issues, open `trainGPT-token.py` and adjust one or both of the following:

- Reduce model size:
  ```python
  'input_dims': 512   # default 768
  ```
- Reduce batch size:
  ```python
  batch_size = 32     # default 64
  ```

Smaller values will lower VRAM usage at the cost of training speed or stability.
