import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.eye_(self.fc[-1].bias.view(k, k))

    def forward(self, x):
        # x: B×k×N
        B = x.size(0)
        feat = self.mlp(x).max(-1)[0]
        mat = self.fc(feat).view(B, self.k, self.k)
        return torch.bmm(mat, x)


class PointNetSeg(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.input_tnet = TNet(3)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.feature_tnet = TNet(64)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.seg_head = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(256, num_classes, 1),
        )

    def forward(self, x):
        # x: B×N×4  (xyzI)
        x = x[..., :3].transpose(1, 2)  # B×3×N
        x = self.input_tnet(x)
        x = self.mlp1(x)
        x = self.feature_tnet(x)
        local = x  # 64×N
        global_feat = self.mlp2(x).max(-1, keepdim=True)[0].repeat(1, 1, x.size(-1))
        feat = torch.cat([local, global_feat], 1)  # 1088×N
        logits = self.seg_head(feat).transpose(1, 2)  # B×N×C
        return logits


def loss_fn(pred, target):
    # target: -1 are padding points
    mask = target != -1
    return F.cross_entropy(pred[mask], target[mask], ignore_index=-1)
