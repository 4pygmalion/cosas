import os
import argparse

import cv2
import numpy as np
import torch
import SimpleITK
import albumentations as A
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2

from cosas.normalization import SPCNNormalizer
from cosas.transforms import discard_minor_prediction

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")

    return parser.parse_args()


def read_image(path) -> np.ndarray:
    image = SimpleITK.ReadImage(path)
    image_array = SimpleITK.GetArrayFromImage(image)  # BGR

    return cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)


def preprocess_image(image_array: np.ndarray, device: str):
    """[summary] preprocessing input image into model format"""

    transform = A.Compose(
        [
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return transform(image=image_array)["image"].to(device).unsqueeze(0)


def postprocess_image(confidences: torch.Tensor, original_size):
    """[summary] postprocessing model result to original image format"""

    if confidences.ndim != 4:
        raise ValueError(f"confidences should be 4D tensor, got {confidences.ndim}D")

    confidences_array: np.ndarray = confidences.cpu().numpy()[0]
    confidences_array = np.squeeze(confidences_array, axis=0)
    upsampled_confidences = A.Resize(*original_size)(image=confidences_array)["image"]

    pred_mask = (upsampled_confidences >= 0.5).astype(np.uint8)

    return discard_minor_prediction(pred_mask)


def write_image(path, result):
    pred_mask = SimpleITK.GetImageFromArray(result)
    SimpleITK.WriteImage(pred_mask, path, useCompression=False)
    return


def main():

    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dry_run:
        input_dir = "task2/input/domain1/images/adenocarcinoma-image"
        output_dir = "task2/output/images/adenocarcinoma-mask"
    else:
        input_dir = "/input/images/adenocarcinoma-image"
        output_dir = "/output/images/adenocarcinoma-mask"

    os.makedirs(output_dir, exist_ok=True)

    # 모델 load
    model_path = os.path.join(CURRENT_DIR, "model.pth")
    model = torch.load(model_path).eval().to(device)

    normalizer = SPCNNormalizer()
    target_image = np.array(Image.open("target_image.png"))
    normalizer.fit(target_image)
    for filename in os.listdir(input_dir):
        if filename.endswith(".mha"):
            output_path = os.path.join(output_dir, filename)
            input_path = os.path.join(input_dir, filename)
            try:
                raw_image = read_image(input_path)
            except Exception as e:
                print(e)

            # norm_image = normalizer.transform(raw_image)
            x: torch.Tensor = preprocess_image(raw_image, device)
            with torch.no_grad():
                logit = model(x)
                confidence: torch.Tensor = torch.sigmoid(logit)

            original_size = raw_image.shape[:2]
            result = postprocess_image(confidence, original_size=original_size)
            write_image(output_path, result)


if __name__ == "__main__":
    main()
