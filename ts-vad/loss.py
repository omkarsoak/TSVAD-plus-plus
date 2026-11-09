import torch
import torch.nn as nn

class Loss(nn.Module):
    """Output head + BCE loss.

    Default head: x is (B, T, 384) per-frame features; a Linear(384, max_speaker)
    produces one logit per speaker per frame -> (B, max_speaker, T).

    Legacy head (legacy_head=True): x is (B, max_speaker, T, 384 // max_speaker) from the
    old reshape; a Linear(384 // max_speaker, 1) is applied. Only for old checkpoints.
    """
    def __init__(self, max_speaker, legacy_head=False, d_model=384):
        super(Loss, self).__init__()
        if d_model % max_speaker != 0 and legacy_head:
            raise ValueError("legacy_head needs 384 divisible by max_speaker (got %d)" % max_speaker)
        self.max_speaker = max_speaker
        self.legacy_head = legacy_head
        if legacy_head:
            self.fc = nn.Linear(d_model // max_speaker, 1)
        else:
            self.fc = nn.Linear(d_model, max_speaker)
        self.loss = nn.BCEWithLogitsLoss(reduction = 'mean')
        self.m = nn.Sigmoid()

    def forward(self, x, labels=None): # labels: B, max_speaker, T
        if self.legacy_head:
            x = self.fc(x).squeeze(-1)          # B, max_speaker, T
        else:
            x = self.fc(x).transpose(1, 2)      # B, T, max_speaker -> B, max_speaker, T

        total_loss = 0
        for i in range(self.max_speaker):
            total_loss += self.loss(x[:, i, :], labels[:, i, :])

        x = self.m(x)
        x = x.data.cpu().numpy()

        return total_loss / self.max_speaker, x
