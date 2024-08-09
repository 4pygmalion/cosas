import os
import argparse

from typing import List

import numpy as np
from PIL import Image
from sklearn.model_selection import KFold, train_test_split

from cosas.data_model import COSASData
from cosas.paths import DATA_DIR


EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXP_DIR)
MMSEG_DIR = os.path.join(ROOT_DIR, "mmsegmentation")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument

def save_image(images:List[np.ndarray], output_dir:str):
    for i, image in enumerate(images):
        Image.fromarray(image).save(os.path.join(output_dir, f"{i}.png"), format="PNG")
        
    return 
    

if __name__ == "__main__":

    print("Load COSAS data")
    cosas_data = COSASData(os.path.join(DATA_DIR, "task2"))
    cosas_data.load()
    
    cosas_dir = os.path.join(MMSEG_DIR, "data", "cosas")
    os.makedirs(cosas_dir, exist_ok=True)
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_val_indices, test_indices) in enumerate(
            folds.split(cosas_data.images, cosas_data.masks), start=1
        ):
            train_val_images = [cosas_data.images[i] for i in train_val_indices]
            train_val_masks = [cosas_data.masks[i] for i in train_val_indices]
            test_images = [cosas_data.images[i] for i in test_indices]
            test_masks = [cosas_data.masks[i] for i in test_indices]

            train_images, val_images, train_masks, val_masks = train_test_split(
                train_val_images, train_val_masks, test_size=0.2, random_state=42
            )
            data_partition = {
                "train": {"images": train_images, "masks": train_masks},
                "val": {"images": val_images, "masks": val_masks},
                "test": {"images": test_images, "masks": test_masks},
            }
            for phase, xy in data_partition.items():
                for inout_put, array in xy.items():
                    output_dir = os.path.join(cosas_dir, f"fold{fold}", phase, inout_put)
                    os.makedirs(output_dir, exist_ok=True)
                    save_image(array, output_dir)
                    
            print(f"Fold ({fold}) saved")

            
            
            
            
            
            
            
            