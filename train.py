# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import utils
import torch as t, torch.nn as nn, torch.nn.functional as tnnF, torch.distributions as tdist
from torch.utils.data import DataLoader, Dataset
import torchvision as tv, torchvision.transforms as tr
import os
import sys
import argparse
import numpy as np
import wideresnet
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from plot_utils import plot_losses
from cleanfid.fid import frechet_distance, get_batch_features
from cleanfid.features import build_feature_extractor
import random
import itertools
from gloss_dataset import GlossDataset
from torch.utils.data import Subset
from torchvision import datasets

if not t.cuda.is_available():
    print("ERROR: CUDA is not available.")

t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False
t.backends.cudnn.enabled = True
im_sz = 128
n_ch = 3
conditionals = []


class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


class F(nn.Module):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10, latent_dim=512, in_proj=8192):
        super(F, self).__init__()
        self.f = wideresnet.Wide_ResNet(
            depth,
            width,
            norm=norm,
            dropout_rate=dropout_rate,
            latent_dim=latent_dim,
            in_proj=in_proj
        )
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, n_classes)
        self.class_dropout = nn.Dropout(0.1)

    def forward(self, x, y=None):
        penult_z = self.f(x)
        return self.energy_output(penult_z).squeeze()

    def classify(self, x):
        penult_z = self.f(x)
        return self.class_output(penult_z).squeeze()

class CCF(F):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10, latent_dim=512, in_proj=8192):
        super(CCF, self).__init__(depth, width, norm=norm, dropout_rate=dropout_rate, n_classes=n_classes, latent_dim=latent_dim, in_proj=in_proj)

    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1)
        else:
            return t.gather(logits, 1, y[:, None])


def cycle(loader):
    while True:
        for data in loader:
            yield data


def grad_norm_from_loss(loss, params):
    """Average L2 grad norm over a list of params, without touching .grad."""
    grads = t.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    norms = [g.norm(2) for g in grads if g is not None]
    if not norms:
        return t.tensor(0., device=loss.device)
    
    return t.stack(norms).mean()

def init_random(args, bs, device):
    return t.empty(bs, args.n_ch, args.im_sz, args.im_sz, device=device).uniform_(-1, 1)

def get_model_and_buffer(args, device, sample_q):
    model_cls = F if args.uncond else CCF
    if args.dataset == "cifar10":
        in_proj = 640
    elif args.dataset == "imagenet":
        in_proj = 2048
    else:
        in_proj = 10240 if args.width == 10 else 8192
    f = model_cls(args.depth, args.width, args.norm, dropout_rate=args.dropout_rate, n_classes=args.n_classes, latent_dim=args.latent_dim, in_proj=in_proj)
    if not args.uncond:
        assert args.buffer_size % args.n_classes == 0, "Buffer size must be divisible by args.n_classes"
    if args.load_path is None:
        # make replay buffer
        replay_buffer = init_random(args, args.buffer_size, device)
    else:
        print(f"loading model from {args.load_path}")
        ckpt_dict = t.load(args.load_path)
        f.load_state_dict(ckpt_dict["model_state_dict"])
        replay_buffer = ckpt_dict["replay_buffer"]
    f = f.to(device)
    return f, replay_buffer

def get_data(args):
    # ---------------------------
    # Transforms
    # ---------------------------
    if args.dataset == "cifar10":
        transform_train = tr.Compose([
            tr.Pad(4, padding_mode="reflect"),
            tr.RandomCrop(args.im_sz),
            tr.RandomHorizontalFlip(),
            tr.ToTensor(),
            tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            lambda x: x + args.sigma * t.randn_like(x)
        ])

        transform_test = tr.Compose([
            tr.ToTensor(),
            tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    else:
        transform_train = tr.Compose([
            tr.ToTensor(),
            tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            lambda x: x + args.sigma * t.randn_like(x)
        ])

        transform_test = tr.Compose([
            tr.ToTensor(),
            tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    if args.dataset == "gloss":
        all_inds = np.arange(args.n_train + args.n_valid + args.n_test)
        train_inds = all_inds[:args.n_train]
        valid_inds = all_inds[args.n_train:args.n_train + args.n_valid]
        train_labeled_inds = train_inds

        dset_train = GlossDataset(args.data_root, indices=train_inds, transform=transform_train)
        dset_train_labeled = GlossDataset(args.data_root, indices=train_labeled_inds, transform=transform_train)
        dset_valid = GlossDataset(args.data_root, indices=valid_inds, transform=transform_test)
        dset_test = None

    elif args.dataset == "cifar10":
        cifar_train_full = datasets.CIFAR10(
            root=args.data_root,
            train=True,
            download=False,
            transform=transform_train
        )

        cifar_train_full_eval = datasets.CIFAR10(
            root=args.data_root,
            train=True,
            download=False,
            transform=transform_test
        )

        cifar_test_full = datasets.CIFAR10(
            root=args.data_root,
            train=False,
            download=False,
            transform=transform_test
        )

        n_total = len(cifar_train_full)
        n_valid = int(0.05 * n_total)
        n_train = n_total - n_valid

        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(n_total)

        train_inds = perm[:n_train]
        valid_inds = perm[n_train:]
        train_labeled_inds = train_inds

        dset_train = Subset(cifar_train_full, train_inds)
        dset_train_labeled = Subset(cifar_train_full, train_labeled_inds)
        dset_valid = Subset(cifar_train_full_eval, valid_inds)
        dset_test = cifar_test_full

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Size of dset_train (general training set): {len(dset_train)} samples")
    print(f"Size of dset_train_labeled (balanced labeled subset): {len(dset_train_labeled)} samples")
    print(f"Size of dset_valid (validation set): {len(dset_valid)} samples")
    if dset_test is not None:
        print(f"Size of dset_test (test set): {len(dset_test)} samples")

    # ---------------------------
    # DataLoaders
    # ---------------------------

    dload_train = DataLoader(
        dset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
    )

    dload_train_labeled = DataLoader(
        dset_train_labeled,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
    )
    dload_train_labeled = cycle(dload_train_labeled)

    dload_valid = DataLoader(
        dset_valid,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=6,
        drop_last=False
    )

    dload_test = None if dset_test is None else DataLoader(
        dset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=6,
        drop_last=False
    )

    return dload_train, dload_train_labeled, dload_valid, dload_test

def sample_p_0(replay_buffer, bs, y=None):
    if len(replay_buffer) == 0:
        return init_random(args, bs, device), []
    
    if y is not None:
        buffer_size = len(replay_buffer) // args.n_classes
        inds = (y * buffer_size + t.randint(0, buffer_size, (bs,), device=device))
    else:
        inds = t.randint(0, len(replay_buffer), (bs,), device=device)
    
    samples = replay_buffer[inds]
    
    if args.reinit_freq > 0:
        choose_random = t.rand(bs, device=device) < args.reinit_freq
        if choose_random.any():
            samples[choose_random] = t.empty_like(samples[choose_random]).uniform_(-1, 1)
    
    return samples, inds

def get_sample_q(args, device):
    def sample_p_0(replay_buffer, bs, y=None):
        if len(replay_buffer) == 0:
            return init_random(args, bs, device), []
        buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // args.n_classes
        inds = t.randint(0, buffer_size, (bs,))
        # if cond, convert inds to class conditional inds
        if y is not None:
            inds = y.cpu() * buffer_size + inds
            assert not args.uncond, "Can't drawn conditional samples without giving me y"
        buffer_samples = replay_buffer[inds]
        random_samples = init_random(args, bs, device)
        choose_random = t.rand(bs, device=device) < args.reinit_freq
 
        samples = buffer_samples.clone()
        samples[choose_random] = random_samples[choose_random]
        return samples, inds
    
    def sample_q(f, replay_buffer, y=None, n_steps=args.n_steps, retain = True):
        """this func takes in replay_buffer now so we have the option to sample from
        scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
        """
        f.eval()
        # get batch size
        bs = args.batch_size if y is None else y.size(0)
        # generate initial samples and buffer inds of those samples (if buffer is used)
        init_sample, buffer_inds = sample_p_0(replay_buffer, bs=bs, y=y)
        x_k = init_sample.clone().detach().requires_grad_(retain)
        # sgld
        for k in range(n_steps):
            current_lr = args.sgld_lr * (1 - k / n_steps)
            f_prime = t.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=retain)[0]

            f_prime.clamp_(-args.eps, args.eps)
            x_k.data.add_(current_lr * f_prime)
            x_k.data.add_(args.sgld_std * t.randn_like(x_k))
            x_k.data.clamp_(-1.0, 1.0)

        f.train()
        final_samples = x_k.detach()
        # update replay buffer
        if len(replay_buffer) > 0:
            replay_buffer[buffer_inds] = final_samples
        return final_samples
    
    return sample_q


def eval_classification(f, dload, device):
    f.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_n = 0

    with t.no_grad():
        for x_p_d, y_p_d in dload:
            x_p_d = x_p_d.to(device, non_blocking=True)
            y_p_d = y_p_d.to(device, non_blocking=True)

            logits = f.classify(x_p_d)
            loss = tnnF.cross_entropy(logits, y_p_d, reduction='sum')
            correct = (logits.argmax(dim=1) == y_p_d).sum()

            bs = y_p_d.size(0)
            total_loss += loss.item()
            total_correct += correct.item()
            total_n += bs

            del x_p_d, y_p_d, logits, loss, correct

    return total_correct / total_n, total_loss / total_n



def to_cleanfid_range(x):
    return (x.clamp(-1, 1) + 1.0) * 127.5


def eval_generation_FID(
    f, dload_valid, replay_buffer, device, args, sample_q, feat_model
):
    n_batches = min(args.fid_batches, len(dload_valid))

    real_feats = []
    fake_feats = []

    for batch_idx, (x_p_d, _) in enumerate(dload_valid):
        if batch_idx >= n_batches:
            break

        x_p_d = x_p_d.to(device, non_blocking=True)

        with t.cuda.amp.autocast(dtype=t.bfloat16):
            if args.class_cond_p_x_sample:
                y_q = t.randint(
                    0, args.n_classes, (x_p_d.size(0),), device=device
                )
                x_q = sample_q(f, replay_buffer, y=y_q)
            else:
                x_q = sample_q(f, replay_buffer)

            if args.dataset == "cifar10":
                x_p_d_fid = to_cleanfid_range(t.nn.functional.interpolate(
                    x_p_d, size=(128, 128), mode="bilinear", align_corners=False
                ))
                x_q_fid = to_cleanfid_range(t.nn.functional.interpolate(
                    x_q, size=(128, 128), mode="bilinear", align_corners=False
                ))
            else:
                x_p_d_fid = x_p_d
                x_q_fid = x_q
        real_in = x_p_d_fid
        fake_in = x_q_fid

        real_feats.append(get_batch_features(real_in, feat_model, device))
        fake_feats.append(get_batch_features(fake_in, feat_model, device))

        del x_p_d, x_q, x_p_d_fid, x_q_fid

    real_feats = np.concatenate(real_feats, axis=0)
    fake_feats = np.concatenate(fake_feats, axis=0)

    mu1 = np.mean(real_feats, axis=0)
    sigma1 = np.cov(real_feats, rowvar=False)
    mu2 = np.mean(fake_feats, axis=0)
    sigma2 = np.cov(fake_feats, rowvar=False)

    return frechet_distance(mu1, sigma1, mu2, sigma2)


def checkpoint(f, buffer, tag, args, device, epoch):
    f.cpu()
    ckpt_dict = {
        "model_state_dict": f.state_dict(),
        "replay_buffer": buffer,
        "epoch": epoch,
        "width": args.width,
        "depth": args.depth,
        "latent_dim": args.latent_dim
    }
    t.save(ckpt_dict, os.path.join(args.save_dir, tag))
    f.to(device)


loss_avgs = {k: 1.0 for k in ['p_x_loss', 'p_y_given_x_loss', 'p_x_y_loss']}

def main(args):
    utils.makedirs(args.save_dir)
    with open(f'{args.save_dir}/params.txt', 'w') as f:
        json.dump(args.__dict__, f)
    if args.print_to_log:
        sys.stdout = open(f'{args.save_dir}/log.txt', 'w')

    np.random.seed(args.seed) 
    t.manual_seed(args.seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(args.seed)
    
    scaler = t.cuda.amp.GradScaler() if t.cuda.is_available() else None

    # datasets
    dload_train, dload_train_labeled, dload_valid, dload_test = get_data(args)

    warmup_iters = max(0, args.warmup_iters)
    iters_per_epoch = len(dload_train)
    warmup_epochs = int(np.ceil(warmup_iters / float(iters_per_epoch))) if warmup_iters > 0 else 0

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    sample_q = get_sample_q(args, device)
    f, replay_buffer = get_model_and_buffer(args, device, sample_q)
    f = t.compile(f)
    sqrt = lambda x: int(t.sqrt(t.Tensor([x])))
    plot = lambda p, x: tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

    # optimizer
    params = f.class_output.parameters() if args.clf_only else f.parameters()
    if args.optimizer == "adam":
        optim = t.optim.Adam(params, lr=args.lr, betas=[.9, .999], weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optim = t.optim.SGD(params, lr=args.lr, momentum=.9, weight_decay=args.weight_decay)
    else:
        optim = t.optim.AdamW(params, lr=args.lr, betas=[.9, .999], weight_decay=5e-5)
    
    stop_training = False
    best_fp = float('inf')
    best_fp_fq_diff = float('inf')
    best_fid = float('inf')
    best_valid_acc_fid = float('inf')
    best_reg_mse = float('inf')
    best_reg_r2 = -float('inf')
    best_valid_acc = 0.0
    cur_iter = 0
    current_fid = None
    losses_arrays = {
        'p_x_loss': [],
        'p_y_given_x_loss': [],
        'p_x_y_loss': [],
        'acc': [],
        'fid': [],
    }

    # ---- gradient balancing state ----
    grad_params = [p for p in f.parameters() if p.requires_grad]
    grad_eps = 1e-8
    # ----------------------------------

    scheduler = None
    if args.scheduler == "cosine":
        cosine_epochs = max(1, args.n_epochs - warmup_epochs)
        scheduler = t.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=cosine_epochs,
            eta_min=1e-5,
        )
                    

    feat_model = build_feature_extractor(device=device, mode="legacy_pytorch", use_dataparallel=False)
    for epoch in range(args.n_epochs):
        if epoch in args.decay_epochs:
            if args.scheduler != "cosine":
                for param_group in optim.param_groups:
                    param_group["lr"] *= args.decay_rate
                print(f"Decaying optimizer lr to {optim.param_groups[0]['lr']:.8f}")

            args.sgld_lr *= args.decay_rate
            print(f"Decaying sgld_lr to {args.sgld_lr:.8f}")

        for i, (x_p_d, _) in tqdm(enumerate(dload_train)):
            cur_iter += 1

            if warmup_iters > 0 and cur_iter <= warmup_iters:
                lr = args.lr * cur_iter / float(warmup_iters)
                for param_group in optim.param_groups:
                    param_group['lr'] = lr

            x_p_d = x_p_d.to(device, non_blocking=True)
            x_lab, y_lab = dload_train_labeled.__next__()
            x_lab = x_lab.to(device, non_blocking=True)
            y_lab = y_lab.to(device, non_blocking=True)

            L = 0.
            l_p_x = 0
            l_p_y_given_x = 0
            l_p_x_y = 0
            l_reg = 0

            # ----- P(x) term -----
            if args.p_x_weight > 0:
                with t.cuda.amp.autocast(dtype=t.bfloat16):
                    if args.class_cond_p_x_sample:
                        assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                        y_q = t.randint(0, args.n_classes, (args.batch_size,)).to(device)
                        x_q = sample_q(f, replay_buffer, y=y_q)
                    else:
                        x_q = sample_q(f, replay_buffer)

                    fp_all = f(x_p_d)
                    fq_all = f(x_q)

                fp = fp_all.mean()
                fq = fq_all.mean()
                l_p_x = -(fp - fq)

            # ----- P(y|x) term -----
            if args.p_y_given_x_weight > 0:
                with t.cuda.amp.autocast(dtype=t.bfloat16):
                    logits = f.classify(x_lab)
                    l_p_y_given_x = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)(logits, y_lab)

            # ----- P(x,y) term (kept simple) -----
            if args.p_x_y_weight > 0:
                assert not args.uncond, "this objective can only be trained for class-conditional EBM DUUUUUUUUHHHH!!!"
                x_q_lab = sample_q(f, replay_buffer, y=y_lab)
                fp_xy, fq_xy = f(x_lab, y_lab).mean(), f(x_q_lab, y_lab).mean()
                l_p_x_y = -(fp_xy - fq_xy)
                if cur_iter % args.print_every == 0:
                    losses_arrays["p_x_y_loss"].append(l_p_x_y.item())
                    print('P(x, y) | {}:{:>d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f} d={:>14.9f}'.format(
                        epoch, i, fp_xy, fq_xy, fp_xy - fq_xy))
            
            if (args.p_x_weight > 0) and (args.p_y_given_x_weight > 0):

                # Gradient norms for each term
                g_p_x  = grad_norm_from_loss(l_p_x, grad_params).detach()
                g_p_yx = grad_norm_from_loss(l_p_y_given_x, grad_params).detach()
                

                correction = (g_p_yx2 / (g_p_x2 + grad_eps))
        
                L = args.p_x_weight * correction * l_p_x + args.p_y_given_x_weight * l_p_y_given_x
            
            elif args.p_x_weight > 0:
                # Only generative objective
                L = args.p_x_weight * l_p_x

            elif args.p_y_given_x_weight > 0:
                # Only discriminative objective
                L = args.p_y_given_x_weight * l_p_y_given_x

            if (not t.isfinite(L)) or (L.abs().item() > 1e7):
                print(f"Skipping bad batch at epoch={epoch}, iter={i}, L={L.item()}")
                optim.zero_grad(set_to_none=True)
                continue
            

            optim.zero_grad()
            if scaler is not None:
                scaler.scale(L).backward()
                scaler.step(optim)
                scaler.update()
            else:
                L.backward()
                optim.step()
            #cur_iter += 1
               
            if cur_iter % args.print_every == 0:
               pass

        if epoch % args.ckpt_every == 0:
            checkpoint(f, replay_buffer, f'ckpt_{epoch}.pt', args, device, epoch)

        if epoch % args.eval_every == 0:
            f.eval()
            output_parts = [f"Iteration {cur_iter}: Total Loss = {L.item():.6f}"]
            if l_p_x is not None:
                output_parts.append(f"l_p_x = {l_p_x:.6f}")
            if l_p_y_given_x is not None:
                output_parts.append(f"l_p_y_given_x = {l_p_y_given_x:.6f}")
            if l_p_x_y is not None:
                output_parts.append(f"l_p_x_y = {l_p_x_y:.6f}")
            print(", ".join(output_parts))

            if args.p_x_weight > 0:

                if args.plot_uncond:
                    if args.class_cond_p_x_sample:
                        assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                        y_q = t.randint(0, args.n_classes, (args.batch_size,)).to(device)
                        x_q = sample_q(f, replay_buffer, y=y_q)
                    else:
                        x_q = sample_q(f, replay_buffer)
                    plot('{}/x_q_{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q)
                if args.plot_cond:  # generate class-conditional samples
                    y = t.arange(0, args.n_classes)[None].repeat(args.n_classes, 1).transpose(1, 0).contiguous().view(-1).to(device)
                    x_q_y = sample_q(f, replay_buffer, y=y)
                
                fp_all = f(x_p_d)
                fq_all = f(x_q)
                

                fp = fp_all.mean().detach()
                fq = fq_all.mean().detach()     


                losses_arrays["p_x_loss"].append(l_p_x.item())
                print('P(x) | {}:{:>d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f} d={:>14.9f}'.format(
                    epoch, i, fp.item(), fq.item(), fp.item() - fq.item()))
                

                current_fid= eval_generation_FID(f, dload_valid, replay_buffer, device, args, sample_q, feat_model)
                losses_arrays["fid"].append(current_fid)
                print(f"fid: {current_fid}")
                if best_fid > current_fid:
                    best_fid = current_fid
                    checkpoint(f, replay_buffer, f"best_fid_model_ckpt.pt", args, device, epoch)
                    plot('{}/x_q_{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q)
                    print(f"New best fid: {best_fid}, model saved for epoch {epoch}")


            if args.p_y_given_x_weight > 0:
                correct, loss = eval_classification(f, dload_valid, device)
                losses_arrays["p_y_given_x_loss"].append(loss)
                losses_arrays["acc"].append(correct)
                print("Epoch {}: Valid Loss {}, Valid Acc {}".format(epoch, loss, correct))
                if correct > best_valid_acc:
                    best_valid_acc = correct
                    print("Best Valid Acc!: {}".format(correct))
                    checkpoint(f, replay_buffer, f"best_valid_ckpt.pt", args, device, epoch)
                    if correct >= 1.0:
                        print(f"Validation accuracy reached 1.0 at epoch={epoch}. Stopping training.")
                        stop_training = True

                print("Epoch {}".format(epoch))

            f.train()
            plot_losses(losses_arrays, args, args.save_dir)
        
        checkpoint(f, replay_buffer, f"last_ckpt.pt", args, device, epoch)
        if stop_training:
            break
        if scheduler is not None and cur_iter > warmup_iters:
            scheduler.step()
            print(f"Cosine LR stepped -> lr = {optim.param_groups[0]['lr']:.8f}")
        t.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Energy Based Models")
    parser.add_argument("--dataset", type=str, default="gloss", choices=["cifar10", "svhn", "cifar100", "gloss", "imagenet"])
    parser.add_argument("--data_root", type=str, default="../data")
    # optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 300, 325],
                        help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.9,
                        help="learning rate decay multiplier")
    parser.add_argument("--clf_only", action="store_true", help="If set, then only train the classifier")
    parser.add_argument("--labels_per_class", type=int, default=-1,
                        help="number of labeled examples per class, if zero then use all labels")
    parser.add_argument("--optimizer", choices=["adam", "sgd","adamw"], default="adamw")
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=150)
    parser.add_argument("--warmup_iters", type=int, default=-1,
                        help="number of iters to linearly increase learning rate, if -1 then no warmmup")
    # loss weighting
    parser.add_argument("--p_x_weight", type=float, default=1.)
    parser.add_argument("--p_y_given_x_weight", type=float, default=1.)
    parser.add_argument("--p_x_y_weight", type=float, default=0.)
    # regularization
    parser.add_argument("--dropout_rate", type=float, default=0.04)
    parser.add_argument("--sigma", type=float, default=5e-2,
                        help="stddev of gaussian noise to add to input, .03 works but .1 is more stable")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # network
    parser.add_argument("--norm", type=str, default=None, choices=[None, "norm", "batch", "instance", "layer", "act"],
                        help="norm to add to weights, none works fine")
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=30,
                        help="number of steps of SGLD per iteration, 100 works for short-run, 20 works for PCD")
    parser.add_argument("--width", type=int, default=10, help="WRN width parameter")
    parser.add_argument("--depth", type=int, default=28, help="WRN depth parameter")
    parser.add_argument("--uncond", action="store_true", help="If set, then the EBM is unconditional")
    parser.add_argument("--class_cond_p_x_sample", action="store_true",
                        help="If set we sample from p(y)p(x|y), othewise sample from p(x),"
                             "Sample quality higher if set, but classification accuracy better if not.")
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--reinit_freq", type=float, default=.05)
    parser.add_argument("--sgld_lr", type=float, default=1.0)
    parser.add_argument("--sgld_std", type=float, default=9e-3)
    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='./experiment')
    parser.add_argument("--ckpt_every", type=int, default=1000, help="Epochs between checkpoint save")
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=100, help="Iterations between print")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--print_to_log", action="store_true", help="If true, directs std-out to log file")
    parser.add_argument("--plot_cond", action="store_true", help="If set, save class-conditional samples")
    parser.add_argument("--plot_uncond", action="store_true", help="If set, save unconditional samples")
    parser.add_argument("--n_train", type=int, default=250)
    parser.add_argument("--n_valid", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--p", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eps", type=float, default=1.0)
    parser.add_argument("--clamp", action="store_true", default=None)
    parser.add_argument("--lights", type=int, default=None)
    parser.add_argument("--latent_dim", type=int, default=1000)
    parser.add_argument("--fid_batches", type=int, default=20)
    parser.add_argument("--label_smoothing", type=float, default=0.05)

    args = parser.parse_args()
    
    if args.dataset == "cifar10":
        args.n_classes = 10
        args.im_sz = 32
        args.n_ch = 3

    elif args.dataset == "gloss":
        args.n_classes = 2
        args.im_sz = 128
        args.n_ch = 3


    main(args)
