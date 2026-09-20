import math
import os
import time

import torch

from losses import vicreg_loss
from logging_utils import save_probe_row
from models import Encoder, Projector
from probes import run_all_probes


def train_arm(arm, args, device, train_loader, probe_train_loader,
              probe_test_loader, num_classes):
    print(f"\n===== Training arm: {arm} =====")
    spiking_enc = (arm == "vicreg_bin")
    spiking_proj = (arm != "vicreg")
    emb_dim = args.emb_dim

    probe_csv = f"./probe_{arm}.csv"
    if os.path.exists(probe_csv):
        os.remove(probe_csv)

    enc = Encoder(emb_dim, spiking=spiking_enc).to(device)
    proj = Projector(emb_dim, args.proj_hidden, args.proj_dim, spiking=spiking_proj).to(device)

    if spiking_proj:
        target_std = math.sqrt(args.spike_rate * (1.0 - args.spike_rate))
    else:
        target_std = 1.0

    print(f"Target std for variance hinge: {target_std:.4f} "
          f"(spike_rate={args.spike_rate if spiking_proj else 'n/a'})")

    params = list(enc.parameters()) + list(proj.parameters())

    print(f"Params: {sum(p.numel() for p in params) / 1e6:.2f}M "
          f"| accum={args.accum_steps}")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    use_amp = (device == "cuda")
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    accum_steps = max(1, args.accum_steps)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accum_steps))
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
        proj.train()
        t0 = time.time()
        rp = rr = 0.0
        rs = 0.0
        n = 0
        opt.zero_grad(set_to_none=True)

        for i, (views, _) in enumerate(train_loader):
            views = views.to(device, non_blocking=True)
            B, V, C, H, W = views.shape
            flat = views.reshape(B * V, C, H, W)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                h = enc(flat)
                z = proj(h).reshape(B, V, -1)

                if spiking_proj:
                    target_std = 0.5
                    target_rate = args.spike_rate
                    rate_coeff = args.rate_reg_weight
                else:
                    target_std = 1.0
                    target_rate = None
                    rate_coeff = 0.0

                loss, sim_l, var_l, cov_l, rate_l = vicreg_loss(
                    z.float(),
                    sim_coeff=args.vicreg_sim_coeff,
                    std_coeff=args.vicreg_std_coeff,
                    cov_coeff=args.vicreg_cov_coeff,
                    target_std=target_std,
                    target_rate=target_rate,
                    rate_coeff=rate_coeff,
                )
                pred_val = sim_l.item()
                reg_val = loss.item()

            (loss / accum_steps).backward()

            rp += pred_val
            rr += reg_val
            if spiking_proj:
                rs += z.float().mean().item()
            n += 1

            if (i + 1) % accum_steps == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % 20 == 0:
                    rate_str = f"rate {rs / max(n, 1):.4f} " if spiking_proj else ""
                    print(f"ep {epoch:3d} step {step:6d}/{total} "
                          f"pred {rp / max(n, 1):.4f} reg {rr / max(n, 1):.4f} "
                          f"{rate_str}"
                          f"loss {loss.item():.4f} lr {lr_at(step):.2e}")

        if len(train_loader) % accum_steps != 0:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

        dt = time.time() - t0
        rate_str = f"rate {rs / max(n, 1):.4f} " if spiking_proj else ""
        print(f"epoch {epoch} done in {dt:.1f}s | "
              f"pred {rp / max(n, 1):.4f} reg {rr / max(n, 1):.4f} {rate_str}")

        if (epoch + 1) % args.probe_every == 0 or epoch == args.epochs - 1:
            acc_lin, acc_knn, acc_lin_mem, acc_knn_mem = run_all_probes(
                enc, probe_train_loader, probe_test_loader, device, num_classes)
            save_probe_row(probe_csv, epoch, arm,
                           acc_lin, acc_knn, acc_lin_mem, acc_knn_mem)
            if acc_lin_mem is not None:
                print(f"[probe] epoch {epoch} linear {acc_lin:.2f}%  knn {acc_knn:.2f}%  "
                      f"| mem linear {acc_lin_mem:.2f}%  mem knn {acc_knn_mem:.2f}%")
            else:
                print(f"[probe] epoch {epoch} linear {acc_lin:.2f}%  knn {acc_knn:.2f}%")

    ckpt_path = f"./{arm}_ckpt.pt"
    save = {"enc": enc.state_dict(), "proj": proj.state_dict(),
            "args": vars(args), "epoch": args.epochs - 1, "arm": arm}
    torch.save(save, ckpt_path)
    print(f"Saved {arm} to {ckpt_path}")
    return enc, proj