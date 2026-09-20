import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def extract_features(encoder, loader, device):
    encoder.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            h = encoder(x).float()
        feats.append(h.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def linear_probe(encoder, train_loader, test_loader, device, num_classes,
                 epochs=200, lr=1e-2, wd=1e-4):
    ftr, ytr = extract_features(encoder, train_loader, device)
    fte, yte = extract_features(encoder, test_loader, device)
    mu = ftr.mean(0, keepdim=True)
    sd = ftr.std(0, keepdim=True).clamp_min(1e-6)
    ftr, fte = (ftr - mu) / sd, (fte - mu) / sd
    clf = nn.Linear(ftr.shape[1], num_classes).to(device)

    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ftr, ytr = ftr.to(device), ytr.to(device)
    fte, yte = fte.to(device), yte.to(device)
    n = ftr.shape[0]
    bs = 512
    for _ in range(epochs):
        clf.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(clf(ftr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sch.step()
    clf.eval()
    with torch.no_grad():
        pred = clf(fte).argmax(1)
        acc = (pred == yte).float().mean().item()
    return acc * 100.0


@torch.no_grad()
def knn_probe(encoder, train_loader, test_loader, device, k=20, T=0.07):
    ftr, ytr = extract_features(encoder, train_loader, device)
    fte, yte = extract_features(encoder, test_loader, device)
    ftr = F.normalize(ftr, dim=-1).to(device)
    fte = F.normalize(fte, dim=-1).to(device)
    ytr = ytr.to(device)
    num_classes = int(ytr.max().item()) + 1
    correct = 0
    bs = 1024
    for i in range(0, fte.shape[0], bs):
        q = fte[i:i + bs]
        sim = q @ ftr.T
        topv, topi = sim.topk(k, dim=1)
        w = (topv / T).softmax(dim=1)
        votes = torch.zeros(q.shape[0], num_classes, device=device)
        votes.scatter_add_(1, ytr[topi], w)
        correct += (votes.argmax(1) == yte[i:i + bs].to(device)).sum().item()
    return correct / fte.shape[0] * 100.0


def run_all_probes(enc, probe_train_loader, probe_test_loader, device, num_classes):
    acc_lin = linear_probe(enc, probe_train_loader, probe_test_loader, device,
                           num_classes=num_classes)
    acc_knn = knn_probe(enc, probe_train_loader, probe_test_loader, device)
    if enc.spiking:
        enc.probe_mem = True
        acc_lin_mem = linear_probe(enc, probe_train_loader, probe_test_loader, device,
                                   num_classes=num_classes)
        acc_knn_mem = knn_probe(enc, probe_train_loader, probe_test_loader, device)
        enc.probe_mem = False
        return acc_lin, acc_knn, acc_lin_mem, acc_knn_mem
    return acc_lin, acc_knn, None, None