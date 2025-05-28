from pathlib import Path

import numpy as np
import torch


class SemanticKITTI(torch.utils.data.Dataset):
    def __init__(self, root, split="train", kitty_config):
        self.files = []
        self.config = kitty_config
        root = Path(root)
        if split == "train":
            sequences = self.config['split']['train'] 
        else:
            sequences = self.config['split']['valid']

        self.files = []
        for seq in sequences:
            seq_dir = root / "sequences" / f"{seq:02}"
            if not seq_dir.is_dir():
                continue

            bins = (seq_dir / "velodyne").glob("*.bin")
            for b in bins:
                lbl = seq_dir / "labels" / (b.stem + ".label")
                if lbl.exists():
                    self.files.append((b, lbl))
        print(f"[SemanticKITTI] Scanned: {len(self.files)} files")

    def __len__(self):
        return len(self.files)

    def _read_scan(self, bin_path, lbl_path):
        pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        lbls = np.fromfile(lbl_path, dtype=np.uint32) & 0xFFFF  # strip instance id
        lbls = np.array([self.config['learning_map'][l] for l in lbls])
        return pts, lbls

    def __getitem__(self, idx):
        bin_path, lbl_path = self.files[idx]
        pts, lbl = self._read_scan(bin_path, lbl_path)  # (N,4) & (N,)
        return torch.from_numpy(pts), torch.from_numpy(lbl)


def collate(batch):
    pts, lbl = list(zip(*batch))
    return torch.nn.utils.rnn.pad_sequence(
        pts, batch_first=True
    ), torch.nn.utils.rnn.pad_sequence(lbl, batch_first=True, padding_value=-1)


if __name__ == "__main__":
    import open3d as o3d

    ds_train = SemanticKITTI("dataset", "train")
    pts, lbl = ds_train[0]
    print("points:", pts.shape)  # e.g. torch.Size([38742, 4])
    print("labels:", lbl.shape, "  unique:", torch.unique(lbl)[:10])

    pts_np = pts[:, :3].numpy()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_np))
    o3d.visualization.draw_geometries([pcd])
