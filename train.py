import argparse
import pathlib

import torch
import wandb
import yaml
import time

from dataloader import SemanticKITTI, collate
from model import PointNetSeg, loss_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset",
        required=True,
        help="path to the dataset",
    )

    parser.add_argument(
        "--checkpoints",
        type=str,
        default="checkpoints",
        help="path to save model checkpoints",
    )

    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_dec", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument(
        "--sem-kitty-conf", default="semantic-kitti.yaml"
    )

    args = parser.parse_args()

    with open(args.sem_kitty_conf, "r") as f:
        kitty_conf = yaml.load(f, yaml.Loader)

    cfg = dict(
        model="PointNetSeg",
        dataset="SemanticKITTI",
        batch_size=args.batch_size,
        lr=args.lr,
        weight_dec=args.weight_dec,
        epochs=args.epochs,
    )

    wandb.login()

    ckpt_dir = pathlib.Path(args.checkpoints)
    ckpt_dir.mkdir(exist_ok=True)
    best_acc = 0.0
    global_step = 0

    ds_train = SemanticKITTI(args.dataset, kitty_conf, "train")
    ds_val = SemanticKITTI(args.dataset, kitty_conf, "valid")
    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val, batch_size=cfg["batch_size"], collate_fn=collate
    )

    wandb_run = wandb.init(
        project="semkitti-pointnet",
        config=cfg,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = max(kitty_conf["learning_map"].values()) + 1
    class_names = [
        kitty_conf["labels"][kitty_conf["learning_map_inv"][l]]
        for l in range(num_classes)
    ]
    net = PointNetSeg(num_classes).to(device)
    wandb.watch(net, log="gradients", log_freq=100)

    num_params = sum(p.numel() for p in net.parameters()) / 1e6  # millions
    wandb.config.update({"params_million": num_params})

    opt = torch.optim.AdamW(
        net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_dec"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    for epoch in range(cfg["epochs"]):
        epoch_start_wall = time.time()

        net.train()
        running_loss = 0
        for step, (pts, lbl) in enumerate(dl_train):
            global_step += 1
            pts, lbl = pts.to(device), lbl.to(device)
            logits = net(pts)
            loss = loss_fn(logits, lbl)
            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            if step != 0 and step % 20 == 0:
                wandb.log(
                    {
                        "train/loss": running_loss / 20,
                        "train/lr": scheduler.get_last_lr()[0],
                    },
                    step=epoch * len(dl_train) + step,
                )
                running_loss = 0

        # quick val IoU
        net.eval()
        correct = total = 0
        conf_mat = torch.zeros(
            (num_classes, num_classes), dtype=torch.long, device=device
        )
        all_preds, all_labels = [], []
        num_pc = 0
        infer_time = 0.0

        with torch.no_grad():
            for pts, lbl in dl_val:
                pts, lbl = pts.to(device), lbl.to(device)
                torch.cuda.synchronize() if device.type == "cuda" else None
                t0 = time.perf_counter()
                pred = net(pts).argmax(-1)
                torch.cuda.synchronize() if device.type == "cuda" else None
                infer_time += time.perf_counter() - t0
                num_pc = pts.shape[0]

                # ignore invalid labels
                mask = lbl != -1
                pred = pred[mask]
                lbl = lbl[mask]

                all_preds.append(pred.cpu())
                all_labels.append(lbl.cpu())

                idx = lbl * num_classes + pred
                conf_mat += torch.bincount(
                    idx, minlength=num_classes * num_classes
                ).reshape(num_classes, num_classes)

        elapsed_wall = time.time() - epoch_start_wall
        gpu_hours = elapsed_wall * torch.cuda.device_count() / 3600.0
        secs_per_pc = infer_time / max(1, num_pc)

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        conf_rows = conf_mat[1:, :]  # ignore class 0 (unlabeled)
        TP = conf_mat[1:, 1:].diag().float()
        col_sums = conf_rows[:, 1:].sum(0).float()
        FP = col_sums - TP
        row_sums = conf_rows.sum(1).float()
        FN = row_sums - TP

        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        iou = TP / (TP + FP + FN + 1e-6)

        # Macro averages
        mPrecision = precision.mean().item()
        mRecall = recall.mean().item()
        mF1 = f1.mean().item()
        mIoU = iou.mean().item()
        mAcc = TP.sum().item() / conf_mat.sum().item()

        # micro averages
        micro_precision = TP.sum() / (TP.sum() + FP.sum() + 1e-6)
        micro_recall = TP.sum() / (TP.sum() + FN.sum() + 1e-6)
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-6)
        )
        micro_iou = TP.sum() / (TP.sum() + FP.sum() + FN.sum() + 1e-6)
        micro_acc = TP.sum() / conf_mat.sum()
        micro_acc = micro_acc.item()
        micro_recall = micro_recall.item()
        micro_precision = micro_precision.item()
        micro_f1 = micro_f1.item()
        micro_iou = micro_iou.item()

        metrics_dict = {
            "iou": iou.cpu().numpy(),
            "precision": precision.cpu().numpy(),
            "recall": recall.cpu().numpy(),
            "f1": f1.cpu().numpy(),
            "support": row_sums.cpu().numpy(),
        }
        table_columns = ["metric"] + list(class_names[1:])
        pc_table = wandb.Table(columns=table_columns)
        for metric_name, arr in metrics_dict.items():
            vals = [float(x) for x in arr]
            if metric_name != "support":
                vals = [round(x * 100, 2) for x in vals]
            pc_table.add_data(metric_name, *vals)

        ckpt_name = ckpt_dir / f"epoch{epoch:03d}_acc{mAcc:.3f}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model": net.state_dict(),
                "optimizer": opt.state_dict(),
                "mAcc": mAcc,
                "mPrecision": mPrecision,
                "mRecall": mRecall,
                "mIoU": mIoU,
                "mF1": mF1,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1,
                "micro_iou": micro_iou,
                "micro_acc": micro_acc,
            },
            ckpt_name,
        )

        wandb.log(
            {
                "val/mAcc": mAcc,
                "val/mPrecision": mPrecision,
                "val/mRecall": mRecall,
                "val/mIou": mIoU,
                "val/mF1": mF1,
                "val/micro_acc": micro_acc,
                "val/micro_precision": micro_precision,
                "val/micro_recall": micro_recall,
                "val/micro_iou": micro_iou,
                "val/micro_f1": micro_f1,
                "val/per_class_metrics": pc_table,
                "val/conf_mat": wandb.plot.confusion_matrix(
                    y_true=all_labels,
                    preds=all_preds,
                    class_names=class_names,
                ),
                "perf/seconds_per_pc": secs_per_pc,
                "perf/gpu_hours_epoch": gpu_hours,
            }, step=global_step)

        artifact = wandb.Artifact(
            name=f"pointnet_light_epoch{epoch:03d}",
            type="model",
            metadata=dict(
                mAcc=mAcc,
                precision=mPrecision,
                recall=mRecall,
                iou=mIoU,
                f1=mF1,
                epoch=epoch,
            ),
        )
        artifact.add_file(str(ckpt_name))
        wandb_run.log_artifact(artifact)
        if mAcc > best_acc:
            best_acc = mAcc
            wandb_run.link_artifact(artifact, aliases=["best"], target_path="model")

        print(
            f"epoch {epoch:02d} | mAcc {mAcc:.3%} | mIoU {mIoU:.3%} | "
            f"mF1 {mF1:.3%} | mPrec {mPrecision:.3%} | mRec {mRecall:.3%}"
            f" | micro_acc {micro_acc:.3%} | micro_iou {micro_iou:.3%} | "
            f"micro_f1 {micro_f1:.3%} | micro_precision {micro_precision:.3%} | "
            f"micro_recall {micro_recall:.3%}"
        )
        scheduler.step()

    wandb_run.finish()
