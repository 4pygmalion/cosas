import os
import np
import torch
import mlflow
import SimpleITK #대회 주최 측에서 .mha 파일 쓰는데 문제 없다고 함
from typing import List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def read_image(path) -> List[np.ndarray]:
    image = SimpleITK.ReadImage(path)
    return SimpleITK.GetArrayFromImage(image)

# TODO
def preprocess_image(image_array):
    """[summary] preprocessing input image into model format"""
    return

# TODO
def postprocess_image(image_array):
    """[summary] postprocessing model result to original image format"""
    return

# TODO
def write_image(path, result):
    # 주최 제공 템플릿
    image = SimpleITK.GetImageFromArray(result)
    SimpleITK.WriteImage(image, path, useCompression=False)
    return 


def main():
    # 대회 주최측에서 제공한 input/output 경로 (output 있는지 확인해달라고 함)
    input_dir = '/input/images/adenocarcinoma-image'
    output_dir = '/output/images/adenocarcinoma-mask'

    os.makedirs(output_dir, exist_ok=True)

    # 모델 load 
    model_path = os.path.join(CURRENT_DIR, "checkpoint.pth")
    model = torch.load(model_path)

    with torch.no_grad():
        for filename in os.listdir(input_dir):
            if filename.endswith('.mha'):
                output_path = os.path.join(output_dir, filename)
                try:
                    input_path = os.path.join(input_dir, filename)
                    raw_image = read_image(input_path)
                    image = preprocess_image(raw_image)

                    raw_result = model.eval(image)
                    #TODO
                    result = postprocess_image(raw_result)
                    write_image(output_path, result)
                except Exception as error:
                    print(error)

if __name__=="__main__":
    main()