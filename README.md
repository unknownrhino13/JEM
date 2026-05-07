# ⚡ Joint Energy-Based Models (JEM)

Training Joint Energy-Based Models for image generation and classification, with two variants:

- **Pixel-space JEM** (`train.py`) — operates directly on RGB images; supports CIFAR-10 and Gloss datasets with built-in FID evaluation via Clean-FID
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

**Gloss** — contact the authors via the [dataset page](https://www.nature.com/articles/s41562-021-01097-6#data-availability).

**ImageNet (ILSVRC 2012)** — requires registration at https://image-net.org. Once downloaded, extract into the standard synset folder structure expected by torchvision.

**CIFAR-10H** — used for uncertainty evaluation only. Download from https://github.com/jcpeterson/cifar-10h (`data/cifar10h-probs.npy` — soft human labels for the CIFAR-10 test set).
