import torch

TRAIN = ["00", "01", "02", "03", "04", "05", "06", "07", "10"]
VAL = ["08"]

learning_map = {
  0 : 0, 1 : 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5, 30: 6, 31: 7,
  32: 8, 40: 9, 44: 1, 48: 1, 49: 1, 50: 1, 51: 1, 52: 0, 60: 9, 70: 1, 71: 1,
  72: 1, 80: 1, 81: 1, 99: 0, 252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5,
  258: 4, 259: 5,
}

class SemKITTI_SLIM(torch.utils.data.Dataset):
    def __init__(self, root, split="train"):
        self.files = []
        root = Path(root)
        sequences = TRAIN if split=="train" else VAL

		self.files = []
        for seq in sequences:
			seq_dir = root/"sequences"/seq
            if not seq_dir.is_dir(): continue

            bins = (seq_dir/"velodyne").glob("*.bin")
            for b in bins:
                lbl = seq_dir/"labels"/(b.stem + ".label")
                if lbl.exists():
                    self.files.append((b,lbl))
        print(f"[SemanticKITTIRaw] {len(self.files)} scans from {len(seq_dirs)} sequences")

    def __len__(self):  return self.n_scans

    def _read_scan(self, bin_path, lbl_path):
        pts  = np.fromfile(bin_path, dtype=np.float32).reshape(-1,4)
        lbls = np.fromfile(lbl_path, dtype=np.uint32) & 0xFFFF  # strip instance id
        return pts, lbls

	def __getitem__(self, idx):
        bin_path, lbl_path = self.files[idx]
        pts,lbl = self._read_scan(bin_path, lbl_path)           # (N,4) & (N,)
        return torch.from_numpy(pts), torch.from_numpy(lbl)

def collate(batch):
    pts, lbl = list(zip(*batch))
    return torch.nn.utils.rnn.pad_sequence(pts, batch_first=True), \
           torch.nn.utils.rnn.pad_sequence(lbl, batch_first=True, padding_value=-1)

