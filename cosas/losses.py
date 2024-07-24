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
    
class MCC_Loss(torch.nn.Module):
    """
    Calculates the proposed Matthews Correlation Coefficient-based loss.

    Args:
        inputs (torch.Tensor): 1-hot encoded predictions
        targets (torch.Tensor): 1-hot encoded ground truth
    
    Reference:
        https://github.com/kakumarabhishek/MCC-Loss/blob/main/loss.py
    """

    def __init__(self):
        super(MCC_Loss, self).__init__()

    def forward(self, inputs, targets):
        """
        MCC = (TP.TN - FP.FN) / sqrt((TP+FP) . (TP+FN) . (TN+FP) . (TN+FN))
        where TP, TN, FP, and FN are elements in the confusion matrix.
        """
        tp = torch.sum(torch.mul(inputs, targets))
        tn = torch.sum(torch.mul((1 - inputs), (1 - targets)))
        fp = torch.sum(torch.mul(inputs, (1 - targets)))
        fn = torch.sum(torch.mul((1 - inputs), targets))

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(
            torch.add(tp, 1, fp)
            * torch.add(tp, 1, fn)
            * torch.add(tn, 1, fp)
            * torch.add(tn, 1, fn)
        )

        # Adding 1 to the denominator to avoid divide-by-zero errors.
        mcc = torch.div(numerator.sum(), denominator.sum() + 1.0)
        return 1 - mcc

LOSS_REGISTRY = {
    "dicebce": DiceXentropy,
    "dice": DiceLoss,
    "mcc": MCC_Loss
}