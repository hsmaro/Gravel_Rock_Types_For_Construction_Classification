# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


# -------- Loss 구현들 --------
## 샘플 수가 적을수록 높은 가중치 부여
def get_weighted_cross_entropy_loss(df: pd.DataFrame, label_column: str):
    # 1. 각 클래스별 샘플 수 계산 (정수 라벨 기준, 오름차순 정렬)
    value_counts = df[label_column].value_counts().sort_index()
    class_counts = value_counts.tolist()
    # 2. 클래스별 가중치 계산 (샘플 수가 적을수록 가중치 높게)
    total = sum(class_counts)
    weights = [total / c for c in class_counts]
    norm_weights = [w / sum(weights) for w in weights]
    # 3. Tensor로 변환 후 Loss 반환
    class_weights = torch.tensor(norm_weights, dtype=torch.float32)
    return nn.CrossEntropyLoss(weight=class_weights)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_probs = F.log_softmax(input, dim=-1)
        n_classes = input.size(-1)
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (n_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))