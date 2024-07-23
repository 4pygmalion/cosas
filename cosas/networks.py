import torch
import segmentation_models_pytorch as smp
from torchvision.transforms import Resize
from efficientnet_pytorch.utils import Conv2dStaticSamePadding
from cosas.transforms import tesellation, reverse_tesellation


class PyramidSeg(torch.nn.Module):
    def __init__(self):
        super(PyramidSeg, self).__init__()  # 부모 클래스 초기화
        self.level1_size = (224 * 3, 224 * 3)
        self.level2_size = (224, 224)
        self.level0 = smp.FPN(
            encoder_name="efficientnet-b7",
            encoder_weights="imagenet",
            classes=2,
        )
        self.level0.encoder._conv_stem = Conv2dStaticSamePadding(
            3, 64, kernel_size=(3, 3), stride=(2, 2), bias=False, image_size=224
        )
        self.level1 = smp.FPN(
            encoder_name="efficientnet-b7",
            encoder_weights="imagenet",
            classes=2,
        )
        self.level1.encoder._conv_stem = Conv2dStaticSamePadding(
            5, 64, kernel_size=(3, 3), stride=(2, 2), bias=False, image_size=224
        )
        self.level2 = smp.FPN(
            encoder_name="efficientnet-b7",
            encoder_weights="imagenet",
            classes=1,
        )
        self.level2.encoder._conv_stem = Conv2dStaticSamePadding(
            5, 64, kernel_size=(3, 3), stride=(2, 2), bias=False, image_size=224
        )

    def forward(self, xs: torch.Tensor):
        if not hasattr(self, "device"):
            self.device = self.level2.encoder._conv_stem.weight.device

        assert xs.ndim == 4, "Input must be a 4D tensor"
        original_shape = xs.shape[2:]

        res = list()
        for x in xs:
            patched_x = tesellation(x.unsqueeze(0)).squeeze(0)
            level0_output = self.level0(patched_x)
            level0_output = reverse_tesellation(
                level0_output, original_shape, device=self.device
            )
            level0_fusion = torch.concat(
                [x.unsqueeze(0), level0_output.unsqueeze(0)], axis=1
            )

            downsampled_o = Resize(self.level1_size)(level0_fusion)
            level1_inputs = tesellation(downsampled_o)  # (1, N, C, 224*3, 224*3)
            level1_output = self.level1(level1_inputs.squeeze(0))
            level1_output = reverse_tesellation(
                level1_output, self.level1_size, device=self.device
            )
            downsampled_x = Resize(self.level2_size)(x)
            resized_level1_output = Resize(self.level2_size)(level1_output)
            level1_fusion = torch.concat(
                [downsampled_x, resized_level1_output], axis=0
            ).unsqueeze(0)

            level2_output = self.level2(level1_fusion)
            res.append(level2_output)

        res = torch.concat(res, dim=0)
        return Resize(original_shape)(res)


MODEL_REGISTRY = {
    "pyramid": PyramidSeg,
}
