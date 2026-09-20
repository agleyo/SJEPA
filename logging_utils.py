import csv
import os


def save_probe_row(path, epoch, arm, acc_lin, acc_knn, acc_lin_mem, acc_knn_mem):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["epoch", "arm", "acc_lin", "acc_knn", "acc_lin_mem", "acc_knn_mem"])
        w.writerow([
            epoch, arm,
            f"{acc_lin:.4f}", f"{acc_knn:.4f}",
            "" if acc_lin_mem is None else f"{acc_lin_mem:.4f}",
            "" if acc_knn_mem is None else f"{acc_knn_mem:.4f}",
        ])


def save_baseline_row(path, name, acc_lin=None, acc_knn=None,
                      acc_lin_mem=None, acc_knn_mem=None, top1=None):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["name", "acc_lin", "acc_knn", "acc_lin_mem", "acc_knn_mem", "top1"])
        w.writerow([
            name,
            "" if acc_lin is None else f"{acc_lin:.4f}",
            "" if acc_knn is None else f"{acc_knn:.4f}",
            "" if acc_lin_mem is None else f"{acc_lin_mem:.4f}",
            "" if acc_knn_mem is None else f"{acc_knn_mem:.4f}",
            "" if top1 is None else f"{top1:.4f}",
        ])


def save_supervised_row(path, epoch, loss, acc):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["epoch", "loss", "val_acc"])
        w.writerow([epoch, f"{loss:.6f}", f"{acc:.4f}"])