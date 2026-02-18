import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from lunar_fno.data.spectrogram_dataset import SpectrogramDataset
from lunar_fno.models.fno2d import SimpleBlock2d
from lunar_fno.utils.io_contracts import load_npz, get_required_arrays
from lunar_fno.utils.metrics import compute_binary_metrics, seed_metrics_to_frame
from lunar_fno.utils.reproducibility import set_all_seeds


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        if y_batch.dim() > 1 and y_batch.size(-1) == 1:
            y_batch = y_batch.squeeze(-1)

        optimizer.zero_grad()
        outputs = model(x_batch)
        if outputs.dim() > 1 and outputs.size(-1) == 1:
            outputs = outputs.squeeze(-1)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def eval_model(model, loader, criterion, device, threshold=0.5):
    model.eval()
    losses, preds, trues, probs = [], [], [], []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        if y_batch.dim() > 1 and y_batch.size(-1) == 1:
            y_batch = y_batch.squeeze(-1)

        outputs = model(x_batch)
        if outputs.dim() > 1 and outputs.size(-1) == 1:
            outputs = outputs.squeeze(-1)

        loss = criterion(outputs, y_batch)
        losses.append(loss.item())

        p = outputs.detach().cpu().numpy()
        t = y_batch.detach().cpu().numpy()
        preds.append((p > threshold).astype(int))
        trues.append(t.astype(int))
        probs.append(p)

    return (
        float(np.mean(losses)),
        np.concatenate(preds),
        np.concatenate(trues),
        np.concatenate(probs),
    )


def save_roc_plot(y_true, y_prob, out_path: Path, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig = plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_cm_plot(y_true, y_pred, out_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(4.8, 4.2))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.colorbar()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(config: dict):
    seed_list = [int(s) for s in config["seed_list"]]
    train_cfg = config["training"]
    model_cfg = config["model"]
    paths_cfg = config["paths"]
    runtime_cfg = config.get("runtime", {})

    device_name = runtime_cfg.get("device", "cuda")
    device = torch.device(device_name if (device_name == "cpu" or torch.cuda.is_available()) else "cpu")

    out_dir = Path(paths_cfg["output_dir"])
    eq_dir = out_dir / "EQ_results"
    mq_dir = out_dir / "MQ_results"
    models_dir = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    eq_dir.mkdir(parents=True, exist_ok=True)
    mq_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    eq_npz = load_npz(paths_cfg["eq_npz"])
    mq_npz = load_npz(paths_cfg["mq_npz"])

    x_eq, y_eq = get_required_arrays(eq_npz, paths_cfg["x_key"], paths_cfg["y_key"])
    x_mq, y_mq = get_required_arrays(mq_npz, paths_cfg["x_key"], paths_cfg["y_key"])

    split_seed = int(train_cfg.get("split_seed", 1234))
    rng = np.random.default_rng(split_seed)
    idx = rng.permutation(len(x_eq))
    n_train = int(0.8 * len(x_eq))
    n_val = int(0.1 * len(x_eq))
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    x_train, y_train = x_eq[train_idx], y_eq[train_idx]
    x_val, y_val = x_eq[val_idx], y_eq[val_idx]
    x_test, y_test = x_eq[test_idx], y_eq[test_idx]

    eq_metrics, mq_metrics = {}, {}
    early_stop_rows = []

    for seed in seed_list:
        set_all_seeds(seed)
        print(f"\n=== Seed {seed} ===")

        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            SpectrogramDataset(x_train, y_train),
            batch_size=int(train_cfg["batch_size"]),
            shuffle=True,
            generator=generator,
            num_workers=int(runtime_cfg.get("num_workers", 0)),
        )
        train_eval_loader = DataLoader(
            SpectrogramDataset(x_train, y_train),
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(runtime_cfg.get("num_workers", 0)),
        )
        val_loader = DataLoader(
            SpectrogramDataset(x_val, y_val),
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(runtime_cfg.get("num_workers", 0)),
        )
        test_loader = DataLoader(
            SpectrogramDataset(x_test, y_test),
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(runtime_cfg.get("num_workers", 0)),
        )
        mq_loader = DataLoader(
            SpectrogramDataset(x_mq, y_mq),
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(runtime_cfg.get("num_workers", 0)),
        )

        model = SimpleBlock2d(
            modes1=int(model_cfg["modes1"]),
            modes2=int(model_cfg["modes2"]),
            width=int(model_cfg["width"]),
        ).to(device)

        criterion = nn.BCELoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(train_cfg["scheduler_factor"]),
            patience=int(train_cfg["scheduler_patience"]),
        )

        history = []
        best_val_loss = float("inf")
        wait = 0
        stop_epoch = int(train_cfg["num_epochs"])

        for ep in range(int(train_cfg["num_epochs"])):
            tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            _, tr_pred, tr_true, _ = eval_model(model, train_eval_loader, criterion, device)
            val_loss, val_pred, val_true, _ = eval_model(model, val_loader, criterion, device)

            tr_acc = float((tr_pred == tr_true).mean())
            val_acc = float((val_pred == val_true).mean())
            history.append({"epoch": ep + 1, "train_loss": tr_loss, "val_loss": val_loss, "train_acc": tr_acc, "val_acc": val_acc})
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                torch.save(model.state_dict(), models_dir / f"best_model_seed{seed}.pth")
            else:
                wait += 1
                if wait >= int(train_cfg["patience"]):
                    stop_epoch = ep + 1
                    print(f"Early stopping at epoch {stop_epoch}")
                    break

        torch.save(model.state_dict(), models_dir / f"final_model_seed{seed}.pth")
        pd.DataFrame(history).to_csv(out_dir / f"training_history_seed{seed}.csv", index=False)

        _, eq_pred, eq_true, eq_prob = eval_model(model, test_loader, criterion, device)
        _, mq_pred, mq_true, mq_prob = eval_model(model, mq_loader, criterion, device)

        eq_metrics[seed] = compute_binary_metrics(eq_true, eq_prob, threshold=0.5)
        mq_metrics[seed] = compute_binary_metrics(mq_true, mq_prob, threshold=0.5)

        save_roc_plot(eq_true, eq_prob, eq_dir / f"roc_seed{seed}.png", title=f"EQ ROC (seed {seed})")
        save_roc_plot(mq_true, mq_prob, mq_dir / f"roc_seed{seed}.png", title=f"MQ ROC (seed {seed})")
        save_cm_plot(eq_true, eq_pred, eq_dir / f"cm_seed{seed}.png", title=f"EQ Confusion (seed {seed})")
        save_cm_plot(mq_true, mq_pred, mq_dir / f"cm_seed{seed}.png", title=f"MQ Confusion (seed {seed})")

        early_stop_rows.append({"modality": "spectrogram", "seed": seed, "early_stop_epoch": stop_epoch})

    seed_metrics_to_frame(eq_metrics).to_csv(out_dir / "metrics_eq.csv")
    seed_metrics_to_frame(mq_metrics).to_csv(out_dir / "metrics_mq.csv")
    pd.DataFrame(early_stop_rows).to_csv(out_dir / "early_stopping.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to spectrogram config YAML")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    run(config)
