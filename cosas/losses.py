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

class IoULoss(torch.nn.Module):
    def __init__(self, smooth=1e-4):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        logits = torch.sigmoid(logits)
        intersection = (logits * targets).sum()
        total = (logits + targets).sum()
        union = total - intersection
        iou_score = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou_score
    
class DiceIoU(torch.nn.Module):
    def __init__(self, smooth=1e-4):
        super(DiceIoU, self).__init__()
        self.dice = DiceLoss(smooth)
        self.iou = IoULoss(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        dice_loss = self.dice(logits, targets)
        iou_loss = self.iou(logits, targets)

        return dice_loss + iou_loss 
    
class MCCLosswithLogits(torch.nn.Module):
    """
    Calculates the proposed Matthews Correlation Coefficient-based loss.

    Args:
        inputs (torch.Tensor): 1-hot encoded predictions
        targets (torch.Tensor): 1-hot encoded ground truth
    
    Reference:
        https://github.com/kakumarabhishek/MCC-Loss/blob/main/loss.py
    """

    def __init__(self):
        super(MCCLosswithLogits, self).__init__()

    def forward(self, logits, targets):
        """
        
        Note:
            위의 모든 코드가 logits값을 입력값으로 받고 있어서, logtis->confidence [0,1]으로 변경
            MCC = (TP.TN - FP.FN) / sqrt((TP+FP) . (TP+FN) . (TN+FP) . (TN+FN))
            where TP, TN, FP, and FN are elements in the confusion matrix.
        
        
        """
        pred = torch.sigmoid(logits)
        tp = torch.sum(torch.mul(pred, targets))
        tn = torch.sum(torch.mul((1 - pred), (1 - targets)))
        fp = torch.sum(torch.mul(pred, (1 - targets)))
        fn = torch.sum(torch.mul((1 - pred), targets))

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
    "mcc": MCCLosswithLogits,
    "diceiou": DiceIoU,
}