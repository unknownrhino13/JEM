# ⚡ Joint Energy-Based Models (JEM)

Training Joint Energy-Based Models for image generation and classification, with two variants:

- **Pixel-space JEM** (`train.py`) — operates directly on RGB images; supports CIFAR-10 and Gloss datasets 
- **Latent-space JEM** (`train_imagenet.py`) — encodes images into the latent space of a frozen Stable Diffusion VAE before running SGLD; enables ImageNet-scale (1000-class, 224×224) training

Both variants use a Wide ResNet energy function trained with Stochastic Gradient Langevin Dynamics (SGLD) and support joint objectives.

---

## 🛠️ Environment

Python 3.9+ and CUDA 12+ are recommended.

```bash
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate
pip install clean-fid
pip install tqdm matplotlib numpy
```

The latent-space variant uses `stabilityai/sd-vae-ft-mse` from the Hugging Face Hub, downloaded automatically on first run. 


### FrozenVAEWrapper

The VAE wrapper (`vae_kl.py`) loads the SD VAE, freezes all parameters, and optionally compiles encode/decode with `torch.compile`. It exposes two methods used during training:

```python
from vae_kl import FrozenVAEWrapper

vae = FrozenVAEWrapper(
    model_name="stabilityai/sd-vae-ft-mse",
    device="cuda",
    dtype=t.float16,
    use_tiling=False,
    use_compile=True,
)

z = vae.encode(x)      # (B, 4, H/8, W/8), scaled by VAE scaling_factor
x_rec = vae.decode(z)  # (B, 3, H, W), values in [-1, 1]
```

For 224×224 inputs, latents are `(B, 4, 28, 28)`. The encoder mean is used by default (`sample_posterior=False`); pass `sample_posterior=True` to sample from the posterior instead.

---

## 📦 Data

**CIFAR-10** is downloaded automatically via torchvision.

**Gloss** — [dataset page](https://www.nature.com/articles/s41562-021-01097-6#data-availability).

**ImageNet (ILSVRC 2012)** — requires registration at https://image-net.org. Once downloaded, extract into the standard synset folder structure expected by torchvision.

**CIFAR-10H** — used for uncertainty evaluation only. Download from https://github.com/jcpeterson/cifar-10h (`data/cifar10h-probs.npy` — soft human labels for the CIFAR-10 test set).

## 🚀 Training

```bash
python train.py \        # pixel-space (CIFAR-10, Gloss)
python train_imagenet.py \  # latent-space (ImageNet)
  --p $alpha \
  --n_epochs $n_epochs \
  --n_steps 20 \
  --print_every 200 \
  --lr .0001 \
  --dataset $dataset \
  --optimizer adam \
  --p_x_weight $p_x_weight \
  --p_y_given_x_weight $p_y_given_x_weight \
  --p_x_y_weight 0.0 \
  --sigma .05 \
  --width $width \
  --depth $depth \
  --save_dir ./${alpha}/${seed}/ \
  --plot_uncond \
  --warmup_iters 1000 \
  --batch_size $batch_size \
  --seed $seed \
  --latent_dim $latent_dim \
  --data_root $data_root \
```

The weights are set as:

```bash
p_x_weight=$alpha
p_y_given_x_weight= 1.0 - $alpha
```

So the two terms always sum to 1. The generative and discriminative loss terms are additionally rebalanced by their gradient norm ratio during training.

## 🧪 Evaluation

Run `eval_cifar10h.py` to evaluate a trained checkpoint on CIFAR-10 and CIFAR-10H:

```bash
python eval_cifar10h.py \
  --load_path best_valid_ckpt.pt \
  --cifar10h_probs_path cifar10h-probs.npy \
  --output_dir ./eval_results
```

Reports three metrics against the CIFAR-10 test set:

- **Top-1 accuracy** — standard hard-label accuracy
- **Soft cross-entropy** — cross-entropy between model predictions and human soft labels
- **KL divergence** — KL(human || model), measuring how well the model's predictive distribution matches human uncertainty

For purely generative checkpoints (`alpha=1.0`), the energy function has no classifier head, so the script automatically extracts penultimate features and trains a **linear probe** on the CIFAR-10 training set before evaluating.

Results are saved to `--output_dir` as `.txt` summaries and per-image `.npy` arrays.

