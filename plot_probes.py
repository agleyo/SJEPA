import os
import csv
import argparse
import matplotlib.pyplot as plt


def load_probe_csv(path):
    data = {"epoch": [], "acc_lin": [], "acc_knn": [],
            "acc_lin_mem": [], "acc_knn_mem": []}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            data["epoch"].append(int(row["epoch"]))
            data["acc_lin"].append(
                float(row["acc_lin"]) if row.get("acc_lin") else None)
            data["acc_knn"].append(
                float(row["acc_knn"]) if row.get("acc_knn") else None)
            data["acc_lin_mem"].append(
                float(row["acc_lin_mem"]) if row.get("acc_lin_mem") else None)
            data["acc_knn_mem"].append(
                float(row["acc_knn_mem"]) if row.get("acc_knn_mem") else None)
    return data


def load_baselines(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["name"]] = {
                k: (float(row[k]) if row.get(k) else None)
                for k in ("acc_lin", "acc_knn", "acc_lin_mem",
                          "acc_knn_mem", "top1")
            }
    return out


def load_supervised_curve(path):
    epochs, losses, accs = [], [], []
    if not os.path.exists(path):
        return epochs, losses, accs
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            losses.append(float(row["loss"]))
            accs.append(float(row["val_acc"]))
    return epochs, losses, accs


def plot_series(ax, xs, ys, **kwargs):
    """Plot (x, y) pairs, skipping any y that is None."""
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not pairs:
        return False
    xs_, ys_ = zip(*pairs)
    ax.plot(xs_, ys_, **kwargs)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+",
                   default=["vicreg_bin", "vicreg_hyb", "vicreg"])
    p.add_argument("--baselines-csv", default="./probe_baselines.csv")
    p.add_argument("--supervised-csv", default="./supervised_curve.csv")
    p.add_argument("--out", default="./probe_curves.png")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    baselines = load_baselines(args.baselines_csv)
    rand = baselines.get("random", {})
    sup = baselines.get("supervised", {})
    sup_epochs, _, sup_accs = load_supervised_curve(args.supervised_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_lin, ax_knn = axes[0], axes[1]

    for arm in args.arms:
        path = f"./probe_{arm}.csv"
        if not os.path.exists(path):
            print(f"Missing {path}, skipping")
            continue
        d = load_probe_csv(path)

        plot_series(ax_lin, d["epoch"], d["acc_lin"],
                    marker="o", ms=3, lw=1.2, label=arm)
        plot_series(ax_knn, d["epoch"], d["acc_knn"],
                    marker="o", ms=3, lw=1.2, label=arm)

        if any(v is not None for v in d["acc_lin_mem"]):
            plot_series(ax_lin, d["epoch"], d["acc_lin_mem"],
                        marker="o", ms=3, lw=1.2, ls=":",
                        label=f"{arm} (mem)")
        if any(v is not None for v in d["acc_knn_mem"]):
            plot_series(ax_knn, d["epoch"], d["acc_knn_mem"],
                        marker="o", ms=3, lw=1.2, ls=":",
                        label=f"{arm} (mem)")

    if rand.get("acc_lin") is not None:
        ax_lin.axhline(rand["acc_lin"], ls="--", color="purple", lw=2.0,
                       label="random init (lin)")
    if rand.get("acc_lin_mem") is not None:
        ax_lin.axhline(rand["acc_lin_mem"], ls=":", color="pink", lw=2.0,
                       label="random init (lin, mem)")
    if rand.get("acc_knn") is not None:
        ax_knn.axhline(rand["acc_knn"], ls="--", color="purple", lw=2.0,
                       label="random init (knn)")
    if rand.get("acc_knn_mem") is not None:
        ax_knn.axhline(rand["acc_knn_mem"], ls=":", color="pink", lw=2.0,
                       label="random init (knn, mem)")

    if sup.get("top1") is not None:
        ax_lin.axhline(sup["top1"], ls="--", color="tab:red", alpha=0.7,
                       lw=2.0, label="supervised")
        ax_knn.axhline(sup["top1"], ls="--", color="tab:red", alpha=0.7,
                       lw=2.0, label="supervised")


    ax_lin.set_title("Linear probe accuracy")
    ax_lin.set_xlabel("epoch")
    ax_lin.set_ylabel("accuracy (%)")

    ax_knn.set_title("kNN probe accuracy")
    ax_knn.set_xlabel("epoch")
    ax_knn.set_ylabel("accuracy (%)")

    for ax in (ax_lin, ax_knn):
        ax.grid(True)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()