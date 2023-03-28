import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ImageDatasetAll(Dataset):
    """ Image Dataset """

    def __init__(self, df, root_dir, transform=None):
        """

        :param csv_file: Path to csv file with image labels
        :param root_dir: Directory with all the images
        :param transform: Optional transform to be applied on a sample
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # NB: Column names are imgname, exp0, exp1, exp2, exp3
        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        # NB: Reshaping for modelling
        image = image.resize((224, 224), Image.ANTIALIAS)

        label1 = self.df.iloc[idx, 1]
        label1 = np.array(label1)
        label2 = self.df.iloc[idx, 2]
        label2 = np.array(label2)
        label3 = self.df.iloc[idx, 3]
        label3 = np.array(label3)
        label4 = self.df.iloc[idx, 4]
        label4 = np.array(label4)
    
        if self.transform:
            image = self.transform(image)

        return image, label1, label2, label3, label4, self.df.iloc[idx, 0]

