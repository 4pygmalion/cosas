import torch


class DiceLoss(torch.nn.Module):

    def __init__(self, smooth=0.01):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        inputs = torch.sigmoid(logits)
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )
        return 1 - dice


class DiceXentropy(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceXentropy, self).__init__()
        self.dice = DiceLoss(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        dice_loss = self.dice(logits, targets)
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

        return dice_loss + bce_loss
