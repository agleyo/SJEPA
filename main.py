import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from baselines import random_baseline, supervised_baseline
from data import (
    ImageDataset,
    MultiViewTransform,
    download_data,
    make_probe_transform,
    make_train_transforms,
)
from train import train_arm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.03)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--n-global", type=int, default=3)
    p.add_argument("--n-local", type=int, default=0)
    p.add_argument("--emb-dim", type=int, default=1024)
    p.add_argument("--proj-dim", type=int, default=4096)
    p.add_argument("--proj-hidden", type=int, default=1024)
    p.add_argument("--spike-rate", type=float, default=0.3,
                   help="Target firing rate for spiking projectors. "
                        "Variance hinge target is sqrt(p*(1-p)).")
    p.add_argument("--vicreg-sim-coeff", type=float, default=25.0)
    p.add_argument("--vicreg-std-coeff", type=float, default=25.0)
    p.add_argument("--vicreg-cov-coeff", type=float, default=1.0)
    p.add_argument("--rate-reg-weight", type=float, default=1.00)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--probe-every", type=int, default=5)
    p.add_argument("--supervised-epochs", type=int, default=100)
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--arms", nargs="+",
                   default=["vicreg_hyb", "vicreg_bin", "vicreg"])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        torch.cuda.set_device(0)
        torch.cuda.init()

    train_root = download_data(args.data_dir)
    val_root = Path(train_root).parent / "val"

    gt, lt = make_train_transforms(args.img_size)
    train_ds = ImageDataset(train_root, MultiViewTransform(gt, lt, args.n_global, args.n_local))
    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=(device == "cuda"),
                    drop_last=True, persistent_workers=(args.workers > 0))
    print(f"Steps/epoch: {len(dl)} "
          f"| micro-batch {args.batch_size} x accum {args.accum_steps} "
          f"= effective {args.batch_size * args.accum_steps}")

    probe_tf = make_probe_transform(args.img_size)
    probe_train = ImageDataset(train_root, probe_tf)
    probe_test = ImageDataset(val_root, probe_tf)
    probe_train_loader = DataLoader(probe_train, batch_size=256, shuffle=False,
                                    num_workers=args.workers, pin_memory=(device == "cuda"))
    probe_test_loader = DataLoader(probe_test, batch_size=256, shuffle=False,
                                   num_workers=args.workers, pin_memory=(device == "cuda"))

    if not args.skip_baselines:
        print("\nBaselines")
        random_baseline(device, probe_train_loader, probe_test_loader,
                        probe_train.num_classes, args)

        sup_epochs = args.supervised_epochs if args.supervised_epochs is not None else args.epochs
        sup_args = argparse.Namespace(**vars(args))
        sup_args.epochs = sup_epochs
        supervised_baseline(sup_args, device, train_root, val_root,
                            probe_train.num_classes,
                            probe_train_loader, probe_test_loader)

    for arm in args.arms:
        train_arm(arm, args, device, dl, probe_train_loader, probe_test_loader,
                  probe_train.num_classes)

    print("All arms finished.")


if __name__ == "__main__":
    main()