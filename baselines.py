import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from data import ImageDataset, make_probe_transform
from logging_utils import save_baseline_row, save_supervised_row
from models import Encoder
from probes import run_all_probes


def random_baseline(device, probe_train_loader, probe_test_loader, num_classes, args):
    print("\n--- Random (untrained) encoder ---")
    enc = Encoder(args.emb_dim, spiking=True).to(device)
    acc_lin, acc_knn, acc_lin_mem, acc_knn_mem = run_all_probes(
        enc, probe_train_loader, probe_test_loader, device, num_classes)
    print(f"[random] linear {acc_lin:.2f}%  knn {acc_knn:.2f}%  "
          f"| mem linear {acc_lin_mem:.2f}%  mem knn {acc_knn_mem:.2f}%")
    save_baseline_row("./probe_baselines.csv", "random",
                      acc_lin=acc_lin, acc_knn=acc_knn,
                      acc_lin_mem=acc_lin_mem, acc_knn_mem=acc_knn_mem)
    return acc_lin, acc_knn, acc_lin_mem, acc_knn_mem


def supervised_baseline(args, device, train_root, val_root, num_classes,
                        probe_train_loader, probe_test_loader):
    print("\n--- Supervised ResNet-18 ---")

    curve_csv = "./supervised_curve.csv"
    if os.path.exists(curve_csv):
        os.remove(curve_csv)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.35, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = make_probe_transform(args.img_size)

    train_ds = ImageDataset(train_root, train_tf)
    test_ds = ImageDataset(val_root, test_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device == "cuda"),
                              drop_last=True, persistent_workers=(args.workers > 0))
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=args.workers, pin_memory=(device == "cuda"))

    enc = Encoder(args.emb_dim, spiking=False).to(device)
    head = nn.Linear(args.emb_dim, num_classes).to(device)
    params = list(enc.parameters()) + list(head.parameters())

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    use_amp = (device == "cuda")
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    steps_per_epoch = max(1, len(train_loader))
    total = args.epochs * steps_per_epoch
    warmup = max(1, min(100, total // 20))

    def lr_at(s):
        if s < warmup:
            return args.lr * s / warmup
        pr = (s - warmup) / max(1, total - warmup)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * pr))

    step = 0
    for epoch in range(args.epochs):
        enc.train()
        head.train()
        t0 = time.time()
        rl = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits = head(enc(x))
                loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            rl += loss.item()
            n += 1

        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            enc.eval()
            head.eval()
            correct = 0
            seen = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    pred = head(enc(x)).argmax(1)
                    correct += (pred == y).sum().item()
                    seen += y.numel()
            acc = correct / seen * 100.0
            save_supervised_row(curve_csv, epoch, rl / max(n, 1), acc)
            print(f"[supervised] epoch {epoch} loss {rl / max(n, 1):.4f} "
                  f"val {acc:.2f}%  ({time.time() - t0:.1f}s)")

    enc.eval()
    head.eval()
    correct = 0
    seen = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = head(enc(x)).argmax(1)
            correct += (pred == y).sum().item()
            seen += y.numel()
    acc = correct / seen * 100.0
    print(f"[supervised] final val {acc:.2f}%")
    save_baseline_row("./probe_baselines.csv", "supervised", top1=acc)

    print("[supervised] probing frozen ResNet-18 features...")
    acc_lin_feat, acc_knn_feat, _, _ = run_all_probes(
        enc, probe_train_loader, probe_test_loader, device, num_classes)
    print(f"[supervised-features] linear {acc_lin_feat:.2f}%  "
          f"knn {acc_knn_feat:.2f}%")
    save_baseline_row("./probe_baselines.csv", "supervised_features",
                      acc_lin=acc_lin_feat, acc_knn=acc_knn_feat)

    return acc