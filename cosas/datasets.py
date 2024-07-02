import os 
from dataclasses import dataclass

@dataclass
class COSASData:
    data_dir:str
    
    def __post_init__(self):
        for scanner in os.path