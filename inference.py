import os
from typing import List

import numpy as np
import torch
import SimpleITK 
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def read_image(path) -> List[np.ndarray]:
    image = SimpleITK.ReadImage(path)
    return SimpleITK.GetArrayFromImage(image)

def preprocess_image(image_array:np.ndarray):
    """[summary] preprocessing input image into model format"""
    
    # TODO: 하드코딩이 아니라, A.Compose을 직렬화해서 불러올수 있도록
    transform = A.Compose(
        [
            A.Resize(384, 384),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    
    return transform(image=image_array)['image'].to("cuda").unsqueeze(0)

def postprocess_image(image_array:np.ndarray, original_size):
    """[summary] postprocessing model result to original image format"""
    
    transform = A.Compose(
        [
            A.Resize(*original_size),
        ]
    )
    return transform(image=image_array)['image']

def write_image(path, result):
    image = SimpleITK.GetImageFromArray(result)
    SimpleITK.WriteImage(image, path, useCompression=False)
    return 


def main():
    
    input_dir = '/input/images/adenocarcinoma-image'
    output_dir = '/output/images/adenocarcinoma-mask'
    os.makedirs(output_dir, exist_ok=True)

    # 모델 load 
    model_path = os.path.join(CURRENT_DIR, "model.pth")
    model = torch.load(model_path).eval()
        
    with torch.no_grad():
        for filename in os.listdir(input_dir):
            if filename.endswith('.mha'):
                output_path = os.path.join(output_dir, filename)
                try:
                    input_path = os.path.join(input_dir, filename)
                    raw_image = read_image(input_path)
                    original_size = raw_image.shape[-2:]
                    
                    x:torch.Tensor = preprocess_image(raw_image)
                    pred = model(x)
                    pred_mask = pred["mask"].cpu().numpy()[0]
                    
                    result = postprocess_image(pred_mask, original_size=original_size)
                    write_image(output_path, result)
                    
                except Exception as error:
                    print(error)

if __name__=="__main__":
    main()