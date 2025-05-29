from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import List

import numpy as np
import torch
import tqdm
import wandb

from model import PointNetSeg  # local model definition

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_scan(path: pathlib.Path) -> np.ndarray:
    """Load a raw Velodyne scan (.bin) → (N, 3) float32 array."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def save_label(arr: np.ndarray, path: pathlib.Path):
    """Save uint32 semantic IDs to `.label` (little‑endian)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.astype(np.uint32).tofile(path)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_single(net: torch.nn.Module, pts: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(pts).unsqueeze(0).to(device)  # (1, N, 3)
    pred = net(tensor).argmax(-1).squeeze(0).cpu().numpy().astype(np.uint32)
    return pred


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("PointNetSeg inference (no batching)")
    parser.add_argument("--artifact", required=True, help="W&B artifact, e.g. user/project/model:best")
    parser.add_argument("--dataset", type=pathlib.Path, required=True, help="SemanticKITTI root directory")
    parser.add_argument("--sequence", default="00", help="Sequence id (00–21)")
    parser.add_argument("--output", type=pathlib.Path, default="predictions", help="Output root directory")
    args = parser.parse_args()

    # -------------------- W&B download --------------------
    wandb.login()
    with wandb.init(job_type="inference", anonymous="must") as run:
        art = run.use_artifact(args.artifact, type="model")
        ckpt_dir = pathlib.Path(art.download())
        ckpt_path = next(ckpt_dir.glob("*.pth"))
        print("Downloaded checkpoint →", ckpt_path)

    # -------------------- Model --------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PointNetSeg().to(device)
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state.get("model", state))
    net.eval()

    # -------------------- Scan list --------------------
    seq_velo = args.dataset / "sequences" / args.sequence / "velodyne"
    scans: List[pathlib.Path] = sorted(seq_velo.glob("*.bin"))
    if not scans:
        print("No scans found in", seq_velo)
        sys.exit(1)

    print(f"Running inference on {len(scans)} scans (sequence {args.sequence}) …")
    t_total = 0.0
    out_pred_dir = args.output / "sequences" / args.sequence / "predictions"

    for bin_path in tqdm.tqdm(scans, desc="inference"):
        pts = load_scan(bin_path)
        t0 = time.perf_counter()
        pred = predict_single(net, pts, device)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t_total += time.perf_counter() - t0

        out_path = out_pred_dir / bin_path.name.replace(".bin", ".label")
        save_label(pred, out_path)

    avg_time = t_total / len(scans)
    print("Saved predictions under", out_pred_dir)
    print(f"Average inference time: {avg_time:.4f} s per scan")


if __name__ == "__main__":
    main()

