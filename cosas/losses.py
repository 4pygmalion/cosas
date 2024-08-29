import torch
import torch.nn as nn


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
        confidences = torch.sigmoid(logits)
        intersection = (confidences * targets).sum()
        total = (confidences + targets).sum()
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


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR

    Reference
    https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    """

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]

        # (2N, D): View 1(1N, D) concat View 2(1N, D)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # 분모(z_i, z_a)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class SparsityLoss(nn.Module):
    def __init__(self):
        super(SparsityLoss, self).__init__()

    def forward(self, vector, density):
        """Sparsity loss

        Args:
            vector (torch.Tensor): (N, 2, 3, W, H)
            density (torch.Tensor): (N, 2, W, H)

        Returns:
            _type_: _description_
        """
        stain1_vector, stain2_vector = torch.unbind(vector, dim=1)  # (N, 3, W, H)
        stain1_density, stain2_density = torch.unbind(
            density.unsqueeze(2), dim=1
        )  # (N, 1, W, H)

        stain1_sparsity = torch.norm(stain1_vector * stain1_density, p=2, dim=1) ** 2
        stain2_sparsity = torch.norm(stain2_vector * stain2_density, p=2, dim=1) ** 2

        penalty = stain1_sparsity + stain2_sparsity
        return penalty.mean()


class AELoss(torch.nn.Module):
    def __init__(self, use_sparisty_loss: bool, alpha: float = 1):
        super(AELoss, self).__init__()
        self.mcc = MCCLosswithLogits()
        self.sparsity_loss = SparsityLoss()
        self.use_sparisty_loss = use_sparisty_loss
        self.alpha = alpha

    def forward(self, recon_x, x, logits, targets, vector, desnity):
        mask_error = self.mcc(logits, targets)
        recon_error = torch.nn.functional.mse_loss(recon_x, x)
        loss = mask_error + self.alpha * recon_error

        if self.use_sparisty_loss:
            sparisty_penalty = self.sparsity_loss(vector, desnity)
            loss += sparisty_penalty
            return loss

        return loss


LOSS_REGISTRY = {
    "bce": torch.nn.BCEWithLogitsLoss,
    "iou": IoULoss,
    "dicebce": DiceXentropy,
    "dice": DiceLoss,
    "mcc": MCCLosswithLogits,
    "diceiou": DiceIoU,
    "multi-task": AELoss,
}
