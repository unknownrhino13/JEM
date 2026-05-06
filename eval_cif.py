import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

import wideresnet


# ============================
# Model
# ============================
class F(nn.Module):
    def __init__(
        self,
        depth=28,
        width=2,
        norm=None,
        dropout_rate=0.0,
        n_classes=10,
        latent_dim=512,
        in_proj=640,
    ):
        super().__init__()
        self.f = wideresnet.Wide_ResNet(
            depth,
            width,
            norm=norm,
            dropout_rate=dropout_rate,
            latent_dim=latent_dim,
            in_proj=in_proj,
        )
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, n_classes)

    def forward(self, x, y=None):
        z = self.f(x)
        return self.energy_output(z).squeeze(-1)

    def classify(self, x):
        z = self.f(x)
        return self.class_output(z)


class CCF(F):
    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1)
        return torch.gather(logits, 1, y[:, None]).squeeze(1)


class LinearProbe(nn.Module):
    def __init__(self, feat_dim, n_classes=10):
        super().__init__()
        self.linear = nn.Linear(feat_dim, n_classes)

    def forward(self, z):
        return self.linear(z)

def get_alpha_from_load_path(load_path):
    try:
        alpha_str = os.path.basename(os.path.dirname(os.path.dirname(load_path)))
        return float(alpha_str)
    except Exception:
        return None


def get_penult_z(model, x):
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.f(x)
    return model.f(x)


def extract_all_latents(model, loader, device):
    model.eval()
    latents, labels = [], []

    with torch.inference_mode():
        for images, y in loader:
            images = images.to(device, non_blocking=True)
            z = get_penult_z(model, images)
            latents.append(z.cpu())
            labels.append(y.cpu())

    return torch.cat(latents, dim=0), torch.cat(labels, dim=0)


def train_linear_probe(probe, loader, device, epochs=50, lr=1e-3):
    probe.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(probe.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss, total_correct, total_n = 0.0, 0, 0
        for z, y in loader:
            z = z.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = probe(z)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * z.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_n += z.size(0)

        print(
            f"[probe] epoch {epoch+1}/{epochs} | "
            f"loss={total_loss / total_n:.6f} | "
            f"acc={total_correct / total_n:.6f}"
        )

def soft_cross_entropy_per_sample(logits, target_probs):
    log_probs = torch.log_softmax(logits, dim=1)
    return -(target_probs * log_probs).sum(dim=1)


def kl_human_to_model_per_sample(logits, target_probs, eps=1e-12):
    p = torch.clamp(target_probs, min=eps)
    p = p / p.sum(dim=1, keepdim=True)

    q = torch.softmax(logits, dim=1)
    q = torch.clamp(q, min=eps)
    q = q / q.sum(dim=1, keepdim=True)

    return (p * (torch.log(p) - torch.log(q))).sum(dim=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate CIFAR-10/CIFAR-10H: top-1 accuracy, soft-label cross-entropy, KL divergence."
    )
    parser.add_argument("--load_path", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--output_dir", type=str, default="cifar10h_eval")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_ccf", action="store_true")
    parser.add_argument(
        "--cifar10h_probs_path",
        type=str,
        default="../cifar-10h/data/cifar10h-probs.npy",
    )
    parser.add_argument("--probe_epochs", type=int, default=50)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    params_path = os.path.join(os.path.dirname(args.load_path), "params.txt")
    with open(params_path, "r") as f:
        params = json.load(f)

    latent_dim = params.get("latent_dim", 512)
    depth = params.get("depth", 28)
    width = params.get("width", 2)
    norm = params.get("norm", None)
    in_proj = params.get("in_proj", 640)

    model_cls = CCF if args.use_ccf else F
    model = model_cls(
        depth=depth,
        width=width,
        norm=norm,
        n_classes=10,
        latent_dim=latent_dim,
        in_proj=in_proj,
    )

    ckpt = torch.load(args.load_path, map_location="cpu")
    model = torch.compile(model)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    alpha_val = get_alpha_from_load_path(args.load_path)
    use_probe = (alpha_val is not None and abs(alpha_val - 1.0) < 1e-8)

    probe = None
    test_latents = None

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    cifar10_test = torchvision.datasets.CIFAR10(
        root="../cifar10",
        train=False,
        download=True,
        transform=transform_test,
    )

    test_loader = DataLoader(
        cifar10_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


    # alpha = 1.0: extract latents + train linear probe
    if use_probe:
        cifar10_train = torchvision.datasets.CIFAR10(
            root="../cifar10",
            train=True,
            download=True,
            transform=transform_test,
        )
        train_loader = DataLoader(
            cifar10_train,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        feat_dim = (
            model._orig_mod.f.last_dim
            if hasattr(model, "_orig_mod")
            else model.f.last_dim
        )
        probe = LinearProbe(feat_dim=feat_dim, n_classes=10).to(device)

        print("Extracting train latents...")
        train_z, train_y = extract_all_latents(model, train_loader, device)

        print("Extracting test latents...")
        test_latents, _ = extract_all_latents(model, test_loader, device)

        train_latent_loader = DataLoader(
            TensorDataset(train_z, train_y),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        train_linear_probe(
            probe, train_latent_loader, device,
            epochs=args.probe_epochs, lr=args.probe_lr,
        )


    cifar10h_probs = np.load(args.cifar10h_probs_path)
    assert cifar10h_probs.shape == (10000, 10), f"Got {cifar10h_probs.shape}"
    human_probs = torch.tensor(cifar10h_probs, dtype=torch.float32)

    total_ce, total_kl, total_correct, total_n = 0.0, 0.0, 0, 0
    start = 0
    all_ce, all_kl = [], []

    if probe is not None:
        probe.eval()

    with torch.inference_mode():
        for images, labels in test_loader:
            bsz = images.size(0)
            end = start + bsz

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            target_probs = human_probs[start:end].to(device, non_blocking=True)

            if probe is not None:
                z = test_latents[start:end].to(device, non_blocking=True)
                logits = probe(z)
            else:
                logits = model.classify(images)

            ce = soft_cross_entropy_per_sample(logits, target_probs)
            kl = kl_human_to_model_per_sample(logits, target_probs)

            total_ce += ce.sum().item()
            total_kl += kl.sum().item()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_n += bsz

            all_ce.append(ce.cpu())
            all_kl.append(kl.cpu())

            start = end

    mean_ce = total_ce / total_n
    mean_kl = total_kl / total_n
    mean_acc = total_correct / total_n

    all_ce = torch.cat(all_ce).numpy()
    all_kl = torch.cat(all_kl).numpy()

    with open(os.path.join(args.output_dir, "cifar10_accuracy.txt"), "w") as f:
        f.write(f"{mean_acc:.8f}\n")
    with open(os.path.join(args.output_dir, "cifar10h_cross_entropy.txt"), "w") as f:
        f.write(f"{mean_ce:.8f}\n")
    with open(os.path.join(args.output_dir, "cifar10h_kl_divergence.txt"), "w") as f:
        f.write(f"{mean_kl:.8f}\n")

    np.save(os.path.join(args.output_dir, "cifar10h_cross_entropy_per_image.npy"), all_ce)
    np.save(os.path.join(args.output_dir, "cifar10h_kl_divergence_per_image.npy"), all_kl)

    print(f"CIFAR-10 top-1 accuracy:           {mean_acc:.8f}")
    print(f"CIFAR-10H soft cross-entropy:      {mean_ce:.8f}")
    print(f"CIFAR-10H KL(human || model):      {mean_kl:.8f}")
    print(f"Saved to {args.output_dir}/")
