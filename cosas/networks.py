import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision.transforms import Resize
from einops import rearrange, repeat
from efficientnet_pytorch.utils import Conv2dStaticSamePadding
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base.heads import SegmentationHead

from .transforms import tesellation, reverse_tesellation, od_to_rgb, rgb_to_od


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
        self.architecture = getattr(smp, architecture)(
            encoder_name=encoder_name, classes=6, activation=torch.nn.ReLU
        )
        self.encoder_name = encoder_name
        self.input_size = input_size

        self.stain_vec_head = self.architecture.segmentation_head

        self.encoder = self.architecture.encoder
        self.stain_den = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=(256, 128, 64, 32, 2),
        )
        self.stain_den_head = SegmentationHead(
            in_channels=2, out_channels=2, activation=torch.nn.ReLU
        )
        self.mask_head = SegmentationHead(
            in_channels=8, out_channels=1, activation=None
        )

    def reconstruction(self, x):
        """
        Args:
            x (torch.Tensor): torch tensor image (float32)

        return
            dict:
             - recon (torch.Tensor): reconstructed image (float32)
             - vector (torch.Tensor): stain vectors (float32)
             - density (torch.Tensor): stain density (float32)
        """

        x = rgb_to_od(x)

        z = self.architecture.encoder(x)

        # Stain vectors (B, 2, 3, W, H)
        x = self.architecture.decoder(*z)  # (6, W, H)
        x = self.stain_vec_head(x)  # (B, 6, W, H)
        stain_vectors = x.view(-1, 2, 3, *self.input_size)  # (B, 2, 3, W, H)

        # Stain Density (B, 2, W, H)
        x_d = self.stain_den(*z)  # (B, 2, W, H)
        stain_density = self.stain_den_head(x_d)

        recon = torch.einsum("bscwh,bswh->bcwh", stain_vectors, stain_density)
        recon = od_to_rgb(recon)

        return {"recon": recon, "vector": stain_vectors, "denisty": stain_density}

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): torch tensor image (float32)

        """

        w, h = x.shape[-2:]

        output = self.reconstruction(x)
        recon = output["recon"]
        vector = output["vector"]
        density = output["denisty"]

        batch_size = x.shape[0]
        stain_info = torch.concat(
            [vector.view(batch_size, -1, w, h), density.view(batch_size, -1, w, h)],
            axis=1,
        )

        return {
            "recon": recon,
            "mask": self.mask_head(stain_info),
            "vector": output["vector"],
            "density": output["denisty"],
        }


MODEL_REGISTRY = {
    "pyramid": PyramidSeg,
    "transunet": TransUNet,
    "autoencoder": MultiTaskAE,
}
