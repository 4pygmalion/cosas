from typing import Optional, List, Union

from copy import deepcopy
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision.transforms import Resize
from einops import rearrange, repeat
from efficientnet_pytorch.utils import Conv2dStaticSamePadding
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base.heads import SegmentationHead, ClassificationHead
from segmentation_models_pytorch.base.model import SegmentationModel
from segmentation_models_pytorch.encoders import get_encoder
from transformers import (
    SegformerForSemanticSegmentation,
)
from segmentation_models_pytorch.base import modules as md

from cosas.transforms import tesellation, reverse_tesellation


class PyramidSeg(torch.nn.Module):
    def __init__(self):
        super(PyramidSeg, self).__init__()  # 부모 클래스 초기화
        self.level1_size = (256 * 3, 256 * 3)
        self.level2_size = (256, 256)
        self.level0 = smp.FPN(
            encoder_name="efficientnet-b1",
            encoder_weights="imagenet",
            classes=2,
        )
        self.level0.encoder._conv_stem = Conv2dStaticSamePadding(
            3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False, image_size=256
        )
        self.level1 = smp.FPN(
            encoder_name="efficientnet-b1",
            encoder_weights="imagenet",
            classes=2,
        )
        self.level1.encoder._conv_stem = Conv2dStaticSamePadding(
            5, 32, kernel_size=(3, 3), stride=(2, 2), bias=False, image_size=256
        )
        self.level2 = smp.FPN(
            encoder_name="efficientnet-b1",
            encoder_weights="imagenet",
            classes=1,
        )
        self.level2.encoder._conv_stem = Conv2dStaticSamePadding(
            5, 32, kernel_size=(3, 3), stride=(2, 2), bias=False, image_size=256
        )

    def forward(self, xs: torch.Tensor):
        if not hasattr(self, "device"):
            self.device = self.level2.encoder._conv_stem.weight.device

        assert xs.ndim == 4, "Input must be a 4D tensor"
        original_shape = xs.shape[2:]

        res = list()
        for x in xs:
            patched_x = tesellation(x.unsqueeze(0), size=self.level2_size).squeeze(0)
            level0_output = self.level0(patched_x)
            level0_output = reverse_tesellation(
                level0_output, original_shape, device=self.device
            )
            level0_fusion = torch.concat(
                [x.unsqueeze(0), level0_output.unsqueeze(0)], axis=1
            )

            downsampled_o = Resize(self.level1_size)(level0_fusion)
            level1_inputs = tesellation(downsampled_o, size=self.level2_size)
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


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = (embedding_dim // head_num) ** (1 / 2)

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, mask=None):
        qkv = self.qkv_layer(x)

        query, key, value = tuple(
            rearrange(qkv, "b t (d k h ) -> k b h t d ", k=3, h=self.head_num)
        )
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk

        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)

        attention = torch.softmax(energy, dim=-1)

        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = self.out_attention(x)

        return x


class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = self.mlp_layers(x)

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        _x = self.multi_head_attention(x)
        _x = self.dropout(_x)
        x = x + _x
        x = self.layer_norm1(x)

        _x = self.mlp(x)
        x = x + _x
        x = self.layer_norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()

        self.layer_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embedding_dim, head_num, mlp_dim)
                for _ in range(block_num)
            ]
        )

    def forward(self, x):
        for layer_block in self.layer_blocks:
            x = layer_block(x)

        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_dim,
        in_channels,
        embedding_dim,
        head_num,
        mlp_dim,
        block_num,
        patch_dim,
        classification=True,
        num_classes=1,
    ):
        super().__init__()

        self.patch_dim = patch_dim
        self.classification = classification
        self.num_tokens = (img_dim // patch_dim) ** 2
        self.token_dim = in_channels * (patch_dim**2)

        self.projection = nn.Linear(self.token_dim, embedding_dim)
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.dropout = nn.Dropout(0.1)

        self.transformer = TransformerEncoder(
            embedding_dim, head_num, mlp_dim, block_num
        )

        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        img_patches = rearrange(
            x,
            "b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)",
            patch_x=self.patch_dim,
            patch_y=self.patch_dim,
        )

        batch_size, tokens, _ = img_patches.shape

        project = self.projection(img_patches)
        token = repeat(
            self.cls_token, "b ... -> (b batch_size) ...", batch_size=batch_size
        )

        patches = torch.cat([token, project], dim=1)
        patches += self.embedding[: tokens + 1, :]

        x = self.dropout(patches)
        x = self.transformer(x)
        x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]

        return x


class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )

        width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=2,
            groups=1,
            padding=1,
            dilation=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = x + x_down
        x = self.relu(x)

        return x


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=True
        )
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)

        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)

        x = self.layer(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        img_dim,
        in_channels,
        out_channels,
        head_num,
        mlp_dim,
        block_num,
        patch_dim,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)

        self.vit_img_dim = img_dim // patch_dim
        self.vit = ViT(
            self.vit_img_dim,
            out_channels * 8,
            out_channels * 8,
            head_num,
            mlp_dim,
            block_num,
            patch_dim=1,
            classification=False,
        )

        self.conv2 = nn.Conv2d(
            out_channels * 8, 512, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu(x)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x = self.encoder3(x3)

        x = self.vit(x)
        x = rearrange(x, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        return x, x1, x2, x3


class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(
            int(out_channels * 1 / 2), int(out_channels * 1 / 8)
        )

        self.conv1 = nn.Conv2d(int(out_channels * 1 / 8), class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.conv1(x)

        return x


class TransUNet(nn.Module):
    """

    Source:
        https://github.com/mkara44/transunet_pytorch/blob/main/utils/transunet.py

    """

    def __init__(
        self,
        img_dim=1024,
        in_channels=3,
        out_channels=128,
        head_num=4,
        mlp_dim=512,
        block_num=8,
        patch_dim=16,
        class_num=1,
    ):
        super().__init__()

        self.encoder = Encoder(
            img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim
        )

        self.decoder = Decoder(out_channels, class_num)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(*z)

        return x


class MultiTaskAE(torch.nn.Module):
    def __init__(self, architecture: str, encoder_name, input_size=(224, 224)):
        super(MultiTaskAE, self).__init__()

        self.encoder_name = encoder_name
        self.input_size = input_size
        self.architecture = getattr(smp, architecture)(
            encoder_name=self.encoder_name, classes=6
        )

        self.stain_vec_head = self.architecture.segmentation_head

        self.encoder = self.architecture.encoder
        self.stain_den = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=(256, 128, 64, 32, 2),
        )
        self.stain_den_head = SegmentationHead(
            in_channels=2, out_channels=2, activation=None
        )
        self.mask_head = SegmentationHead(
            in_channels=8, out_channels=1, activation=None
        )

    def reconstruction(self, x):
        z = self.architecture.encoder(x)

        # Stain vectors (B, 2, 3, W, H)
        x = self.architecture.decoder(*z)  # (6, W, H)
        x = self.stain_vec_head(x)  # (B, 6, W, H)
        stain_vectors = x.view(-1, 2, 3, *self.input_size)  # (B, 2, 3, W, H)

        # Stain Density (B, 2, W, H)
        x_d = self.stain_den(*z)  # (B, 2, W, H)
        stain_density = self.stain_den_head(x_d)

        recon = torch.einsum("bscwh,bswh->bcwh", stain_vectors, stain_density)
        recon = torch.clip(recon, -1, 1)

        return {"recon": recon, "vector": stain_vectors, "density": stain_density}

    def forward(self, x):
        w, h = x.shape[-2:]

        output = self.reconstruction(x)
        recon = output["recon"]
        vector = output["vector"]
        density = output["density"]

        batch_size = x.shape[0]
        stain_info = torch.concat(
            [vector.view(batch_size, -1, w, h), density.view(batch_size, -1, w, h)],
            axis=1,
        )

        return {
            "recon": recon,
            "mask": self.mask_head(stain_info),
            "vector": output["vector"],
            "density": output["density"],
        }


class MultiTaskTransAE(torch.nn.Module):
    def __init__(
        self,
        architecture: str,
        encoder_name,
        input_size=(224, 224),
        in_channels=3,
        out_channels=128,
        head_num=4,
        mlp_dim=512,
        block_num=8,
        patch_dim=16,
    ):
        super(MultiTaskTransAE, self).__init__()

        img_dim = input_size[0]
        self.encoder = Encoder(
            img_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            head_num=head_num,
            mlp_dim=mlp_dim,
            block_num=block_num,
            patch_dim=patch_dim,
        )
        self.decoder = Decoder(out_channels=out_channels, class_num=6)
        self.stain_app = Decoder(out_channels=out_channels, class_num=2)
        self.input_size = input_size
        self.segmentation_head = SegmentationHead(
            in_channels=6, out_channels=6, activation=None
        )
        self.mask_head = SegmentationHead(
            in_channels=8, out_channels=1, activation=None
        )

    def reconstruction(self, x):
        z = self.encoder(x)

        # Stain vectors (B, 2, 3, W, H)
        x = self.decoder(*z)  # (6, W, H)
        x = self.segmentation_head(x)  # (B, 6, W, H)
        stain_vectors = x.view(-1, 2, 3, *self.input_size)  # (B, 2, 3, W, H)

        # Stain Density (B, 2, W, H)
        stain_density = self.stain_app(*z)  # (B, 2, W, H)

        recon = torch.einsum("bscwh,bswh->bcwh", stain_vectors, stain_density)
        recon = torch.clip(recon, -1, 1)

        return {"recon": recon, "vector": stain_vectors, "density": stain_density}

    def forward(self, x):
        w, h = x.shape[-2:]

        output = self.reconstruction(x)
        recon = output["recon"]
        vector = output["vector"]
        density = output["density"]

        batch_size = x.shape[0]
        stain_info = torch.concat(
            [vector.view(batch_size, -1, w, h), density.view(batch_size, -1, w, h)],
            axis=1,
        )

        return {
            "recon": recon,
            "mask": self.mask_head(stain_info),
            "vector": output["vector"],
            "density": output["density"],
        }


class Segformer(torch.nn.Module):
    def __init__(self):
        super(Segformer, self).__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b5", ignore_mismatched_sizes=True
        )
        self.model.decode_head.classifier = torch.nn.Conv2d(
            768,
            1,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x: torch.Tensor):
        b, c, w, h = x.shape
        segmenter_output = self.model.forward(x)

        return nn.functional.interpolate(
            segmenter_output.logits, size=(w, h), mode="bilinear"
        )


class TransposeReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(4, 4),
        padding=1,
        stride=2,
        output_padding=0,
        use_batchnorm=True,
    ):

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(TransposeReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.conv1_in_features = in_channels + skip_channels
        self.conv1 = TransposeReLU(
            in_channels,
            in_channels,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = self.conv1(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        x = self.conv2(x)

        return x


class TransposeUnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class TransposeUnet(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = TransposeUnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()


class Siamformer(torch.nn.Module):
    def __init__(self):
        super(Siamformer, self).__init__()
        self.size = (512, 512)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b5", ignore_mismatched_sizes=True
        )
        self.model.decode_head.classifier = torch.nn.Conv2d(
            768,
            1,
            kernel_size=1,
            stride=1,
        )
        self.zoom_pooling_conv = torch.nn.Conv2d(
            2, 1, kernel_size=(1, 1), stride=(1, 1)
        )

    def _forward_low_mag(self, x: torch.Tensor):
        x = nn.functional.interpolate(x, size=self.size, mode="bilinear")
        segmenter_output = self.model.forward(x)

        return nn.functional.interpolate(
            segmenter_output.logits, size=self.size, mode="bilinear"
        )

    def _forward_high_mag_one_image(self, x: torch.Tensor):
        """
        Params
            x (torch.Tensor): x가 (1024, 1024)인 경우
        """

        patches = torch.concat(
            [
                x[..., :512, :512],  # 좌상단 패치
                x[..., :512, 512:],  # 우상단 패치
                x[..., 512:, :512],  # 좌하단 패치
                x[..., 512:, 512:],  # 우하단 패치
            ],
            dim=0,
        )

        segmenter_output = self.model.forward(patches)  # (N, c, 128, 128)
        logits = segmenter_output.logits

        top = torch.concat([logits[0], logits[1]], dim=-1)
        bottom = torch.concat([logits[2], logits[3]], dim=-1)
        res = torch.concat([top, bottom], dim=-2).unsqueeze(0)

        return torch.nn.functional.interpolate(res, size=self.size, mode="bilinear")

    def forward(self, x: torch.Tensor):
        original_size = x.shape[-2:]

        logit_low_mag = self._forward_low_mag(x)

        logit_high_mags = list()
        for image_tensor in x:
            logit_high_mag = self._forward_high_mag_one_image(image_tensor.unsqueeze(0))
            logit_high_mags.append(logit_high_mag)

        logit_high_mags = torch.concat(logit_high_mags, dim=0)

        z = torch.concat([logit_low_mag, logit_high_mags], dim=1)  # (N, 2, W, H)

        return torch.nn.functional.interpolate(
            self.zoom_pooling_conv(z),
            size=original_size,
            mode="bilinear",
        )


class StainReconSegformer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b5", ignore_mismatched_sizes=True
        )
        self.encoder = model.segformer.encoder
        self.stain_matrix_decoder = deepcopy(model.decode_head)
        self.stain_matrix_decoder.classifier = torch.nn.Conv2d(
            768,
            6,
            kernel_size=1,
            stride=1,
        )

        self.stain_density_decoder = deepcopy(model.decode_head)
        self.stain_density_decoder.classifier = torch.nn.Conv2d(
            768,
            2,
            kernel_size=1,
            stride=1,
        )

        self.mask_head = torch.nn.Sequential(
            torch.nn.Conv2d(8, 8, kernel_size=1, stride=1),
            torch.nn.Conv2d(8, 1, kernel_size=1, stride=1),
        )

    def reconstruction(self, x):
        w, h = x.shape[-2:]
        z = self.encoder(
            x,
            output_hidden_states=True,
        ).hidden_states

        # Stain vectors (B, 2, 3, W, H)
        x = self.stain_matrix_decoder(z)  # (1, 6, W, H)
        stain_matrix = x.view(-1, 2, 3, int(w / 4), int(h / 4))  # (B, 2, 3, W, H)

        # Stain Density (B, 2, W, H)
        stain_density = self.stain_density_decoder(z)  # (B, 2, W, H)
        recon = torch.einsum("bscwh,bswh->bcwh", stain_matrix, stain_density)

        return {"recon": recon, "vector": stain_matrix, "density": stain_density}

    def forward(self, x):
        w, h = x.shape[-2:]

        z_w, z_h = int(w / 4), int(h / 4)

        output = self.reconstruction(x)
        recon = output["recon"]
        vector = output["vector"]
        density = output["density"]

        batch_size = x.shape[0]
        stain_info = torch.concat(
            [
                vector.view(batch_size, -1, z_w, z_h),
                density.view(batch_size, -1, z_w, z_h),
            ],
            axis=1,
        )

        return {
            "recon": torch.nn.functional.interpolate(
                recon, size=(w, h), mode="bilinear"
            ),
            "mask": torch.nn.functional.interpolate(
                self.mask_head(stain_info), size=(w, h), mode="bilinear"
            ),
            "vector": output["vector"],
            "density": output["density"],
        }


class StainPredictSegformer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b5", ignore_mismatched_sizes=True
        )
        self.encoder = model.segformer.encoder
        self.mask_head = deepcopy(model.decode_head)
        self.mask_head.classifier = torch.nn.Conv2d(
            768,
            1,
            kernel_size=1,
            stride=1,
        )

        self.stain_head = deepcopy(model.decode_head)
        self.stain_head.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(
                768,
                2,
                kernel_size=1,
                stride=1,
            ),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        w, h = x.shape[-2:]
        z = self.encoder(
            x,
            output_hidden_states=True,
        ).hidden_states

        mask = self.mask_head(z)
        stain = self.stain_head(z)

        return {
            "density": torch.nn.functional.interpolate(
                stain, size=(w, h), mode="bilinear"
            ),
            "mask": torch.nn.functional.interpolate(mask, size=(w, h), mode="bilinear"),
        }


class ImagelevelMultiTaskAE(torch.nn.Module):
    def __init__(self, architecture: str, encoder_name, input_size=(224, 224)):
        super(ImagelevelMultiTaskAE, self).__init__()

        self.encoder_name = encoder_name
        self.input_size = input_size
        self.architecture = getattr(smp, architecture)(
            encoder_name=self.encoder_name, classes=6
        )

        self.stain_vec_head = self.architecture.segmentation_head

        self.encoder = self.architecture.encoder
        self.stain_den = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=(256, 128, 64, 32, 2),
        )
        self.stain_den_head = SegmentationHead(
            in_channels=2, out_channels=2, activation=None
        )
        self.mask_head = SegmentationHead(
            in_channels=8, out_channels=1, activation=None
        )
        self.classifier = ClassificationHead(
            in_channels=8,
            classes=1,
        )

    def reconstruction(self, x):
        z = self.architecture.encoder(x)

        # Stain vectors (B, 2, 3, W, H)
        x = self.architecture.decoder(*z)  # (6, W, H)
        x = self.stain_vec_head(x)  # (B, 6, W, H)
        stain_vectors = x.view(-1, 2, 3, *self.input_size)  # (B, 2, 3, W, H)

        # Stain Density (B, 2, W, H)
        x_d = self.stain_den(*z)  # (B, 2, W, H)
        stain_density = self.stain_den_head(x_d)

        recon = torch.einsum("bscwh,bswh->bcwh", stain_vectors, stain_density)

        return {"recon": recon, "vector": stain_vectors, "density": stain_density}

    def forward(self, x):
        w, h = x.shape[-2:]

        output = self.reconstruction(x)
        recon = output["recon"]
        vector = output["vector"]
        density = output["density"]

        batch_size = x.shape[0]
        stain_info = torch.concat(
            [vector.view(batch_size, -1, w, h), density.view(batch_size, -1, w, h)],
            axis=1,
        )

        return {
            "recon": recon,
            "mask": self.mask_head(stain_info),
            "logit": self.classifier(stain_info),
            "vector": output["vector"],
            "density": output["density"],
        }


class EnsembleModel_Segform_MTaskAE(torch.nn.Module):
    """
    # continual learning
    >>> model = EnsembleModel()
    >>> model1 = mlflow.pytorch.load_model()
    >>> model.model1.params = model1

    # init
    >>> model = EnsembleModel()

    """

    def __init__(self, aggregation_method="majority_voting"):
        super(EnsembleModel_Segform_MTaskAE, self).__init__()
        self.model1 = MultiTaskAE(
            architecture="Unet", encoder_name="efficientnet-b7", input_size=(640, 640)
        )
        self.model2 = Segformer()
        self.aggregation_method = aggregation_method

    def majority_voting(self, outputs):
        stacked_outputs = torch.stack(outputs, dim=0)
        voted_output = torch.mode(stacked_outputs, dim=0).values
        return voted_output

    def max_confidence(self, outputs):
        stacked_outputs = torch.stack(outputs, dim=0)
        max_conf_output, _ = torch.max(stacked_outputs, dim=0).values
        return max_conf_output

    def forward(self, x):
        """
        AE input_size = 640
        Segformer input_size = 512
        """
        x1 = torch.nn.functional.interpolate(x, size=640, mode="bilinear")
        x2 = torch.nn.functional.interpolate(x, size=512, mode="bilinear")

        output1 = torch.nn.functional.interpolate(
            self.model1(x1)["mask"], size=512, mode="bilinear"
        )
        output2 = self.model2(x2)

        if self.aggregation_method == "majority_voting":
            output = self.majority_voting([output1, output2])

        elif self.aggregation_method == "max_confidence":
            output = self.max_confidence([output1, output2])

        else:
            raise ValueError(
                "Unsupported aggregation method. "
                "Choose 'majority_voting', or 'max_confidence'"
            )

        return output


MODEL_REGISTRY = {
    "pyramid": PyramidSeg,
    "transunet": TransUNet,
    "autoencoder": MultiTaskAE,
    "segformer": Segformer,
    "transpose_unet": TransposeUnet,
    "siamformer": Siamformer,
    "recon_segformer": StainReconSegformer,
    "stainsegformer": StainPredictSegformer,
    "imagelevel_multitask": ImagelevelMultiTaskAE,
    "ensemble_segform_multitaskae": EnsembleModel_Segform_MTaskAE,
}
