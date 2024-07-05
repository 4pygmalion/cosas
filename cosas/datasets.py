from torch.utils.data import Dataset

from .data_model import ScannerData, COSASData


class COSASdataset(Dataset):
    def __init__(self, cosas_data: COSASData):
        cosas_data.load()
        self.cosas_data = cosas_data
