import os
from typing import List

import numpy as np
import torch
import SimpleITK
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def read_image(path) -> np.ndarray:
    image = SimpleITK.ReadImage(path)
    return SimpleITK.GetArrayFromImage(image)


def preprocess_image(image_array: np.ndarray, device: str):
    """[summary] preprocessing input image into model format"""

    # TODO: 하드코딩이 아니라, A.Compose을 직렬화해서 불러올수 있도록
    transform = A.Compose(
        [
            A.Resize(384, 384),
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

    return (upsampled_confidences >= 0.5).astype(np.uint8)


def write_image(path, result):
    pred_mask = SimpleITK.GetImageFromArray(result)
    SimpleITK.WriteImage(pred_mask, path, useCompression=False)
    return


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dir = "/input/images/adenocarcinoma-image"
    output_dir = "/output/images/adenocarcinoma-mask"

    os.makedirs(output_dir, exist_ok=True)

    # 모델 load
    model_path = os.path.join(CURRENT_DIR, "model.pth")
    model = torch.load(model_path).eval().to(device)

    with torch.no_grad():
        for filename in os.listdir(input_dir):
            if filename.endswith(".mha"):
                output_path = os.path.join(output_dir, filename)
                try:
                    input_path = os.path.join(input_dir, filename)
                    raw_image = read_image(input_path)
                    original_size = raw_image.shape[:2]

                    x: torch.Tensor = preprocess_image(raw_image, device)
                    confidences: torch.Tensor = model(x)["mask"]
                    result = postprocess_image(confidences, original_size=original_size)
                    write_image(output_path, result)

                except Exception as error:
                    print(error)


if __name__ == "__main__":
    main()
