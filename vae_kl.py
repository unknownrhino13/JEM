from diffusers import AutoencoderKL, AutoencoderTiny
import torch as t
import torch.nn as nn


class FrozenVAEWrapper(nn.Module):
    def __init__(
        self,
        model_name="stabilityai/sd-vae-ft-mse",
        device="cuda",
        dtype=t.float32,
        use_tiling=False,
        use_compile=True,
        compile_mode="max-autotune",
        compile_fullgraph=True,
    ):
        super().__init__()

        self.vae = AutoencoderKL.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )

        self.vae.eval()

        for p in self.vae.parameters():
            p.requires_grad = False

        self.vae.to(device)

        if use_tiling and hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling()

        if use_compile:
            self.vae.decode = t.compile(
                self.vae.decode,
                mode=compile_mode,
                fullgraph=False,
            )

            self.vae.encode = t.compile(
                self.vae.encode,
                mode=compile_mode,
                fullgraph=False,
            )

    @property
    def device(self):
        return next(self.vae.parameters()).device

    @property
    def dtype(self):
        return next(self.vae.parameters()).dtype

    @t.no_grad()
    def encode(self, x, sample_posterior=False):
        x = x.to(device=self.device, dtype=self.dtype)

        enc = self.vae.encode(x)
        posterior = enc.latent_dist
        z = posterior.sample() if sample_posterior else posterior.mean
        z = z * self.vae.config.scaling_factor

        return z

    @t.no_grad()
    def decode(self, z):
        z = z.to(device=self.device, dtype=self.dtype)

        dec = self.vae.decode(z)
        return dec.sample

    @property
    def latent_channels(self):
        if hasattr(self.vae.config, "latent_channels"):
            return self.vae.config.latent_channels
        return 4
