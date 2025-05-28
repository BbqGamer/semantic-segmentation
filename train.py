import argparse
import pathlib

import torch
import wandb
import yaml

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
    parser.add_argument("--sem-kitty-conf", default="semantic-kitti-api/config/semantic-kitti.yaml")

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
    
    num_classes = len(set(kitty_conf['learning_map'].values()))
    net = PointNetSeg(num_classes).to(device)
    wandb.watch(net, log="gradients", log_freq=100)

    opt = torch.optim.AdamW(
        net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_dec"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    for epoch in range(cfg["epochs"]):
        net.train()
        running_loss = 0
        for step, (pts, lbl) in enumerate(dl_train):
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
        conf_mat = np.zeros((num_classes, num_classes), dtype=torch.long, device=device)
        all_preds, all_labels = [], []
        with torch.no_grad():
            for pts, lbl in dl_val:
                pts, lbl = pts.to(device), lbl.to(device)
                pred = net(pts).argmax(-1)

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

		all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        TP = conf_mat.diag().float()
        FP = conf_mat.sum(0).float() - TP
        FN = conf_mat.sum(1).float() - TP
        
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        iou = TP / (TP + FP + FN + 1e-6)

        mPrecision = precision.mean().item()
        mRecall = recall.mean().item()
        mF1 = f1.mean().item()
        mIoU = iou.mean().item()
        mAcc = TP.sum().item() / conf_mat.sum().item()
        
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
            },
            ckpt_name,
        )

		wandb.log(
            {
                "val/acc": mAcc,
                "val/precision": mPrecision,
                "val/recall": mRecall,
                "val/iou": mIoU,
                "val/f1": mF1,
				"val/conf_mat": wanb.plot.confusion_matrix(
					y_true=all_labels,
					preds=all_preds,
					class_names=class_names,
				)
            }
        )

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
        )
        scheduler.step()

    wandb_run.finish()
