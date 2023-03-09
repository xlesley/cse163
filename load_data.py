import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class LoadData(Dataset):
    def __init__(self, data_dir):
        self.__data_dir = data_dir
        self.__categories = [os.path.join(self.__data_dir, name) for name in os.listdir(self.__data_dir) if os.path.isdir(os.path.join(self.__data_dir, name))]
        self.__label_list = []
        self.__image_list = []
        
        for category in self.__categories:
            image_paths = [os.path.join(category, name) for name in os.listdir(category) if name.endswith('.jpg') or name.endswith('.jpeg') or name.endswith('.png')]
            label = os.path.basename(category)

            for file_path in image_paths:
                img = Image.open(file_path)
                img_array = np.array(img)
                img_array = img_array / 255.0

                self.__label_list.append(label)
                self.__image_list.append(img_array)
        
        self.__labels = sorted(list(set(self.__label_list)))
        self.__label_to_idx = {label: i for i, label in enumerate(self.__labels)}
        self.__idx_to_label = {i: label for i, label in enumerate(self.__labels)}

    def __getitem__(self, index):
        label = self.__label_list[index]
        image = self.__image_list[index]
        label_idx = self.__label_to_idx[label]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        image_tensor = torch.tensor(image, dtype=torch.float)
        return image_tensor, label_tensor

    def __len__(self):
        return len(self.__image_list)

    def get_labels(self):
        return self.__labels

    def get_label(self, index):
        return self.__idx_to_label[index]
