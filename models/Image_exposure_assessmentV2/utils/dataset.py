import os
from torchvision import transforms
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision.datasets.folder import default_loader
# import albumentations as albu
from PIL import Image
import cv2 as cv
import tqdm

from torch.autograd import Variable


class IELDataset(Dataset):
    def __init__(self, path_to_csv, images_path,label_path, if_train):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.label_path = label_path
        self.if_train = if_train
        if if_train:
            self.transform = transforms.Compose([
                transforms.PILToTensor()
                ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
                ])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        image_id = row['img_id']

        # print('id:',image_id)
        # print('ref_id:', ref_id)
        #print(f'{image_id}'.split('.png')[-2] + '.npy')
        image_path = os.path.join(self.images_path, f'{image_id}')
        label_path = os.path.join(self.label_path, f'{image_id}'.split('.png')[-2] + '.npy') #注意后面这个ref和原始图像要放到一起

        image = Image.open(image_path)
        numpy_file = np.load(label_path)

        img_tensor = self.transform(image)/255.0
        label_tensor = torch.from_numpy(numpy_file)/255.0

        return img_tensor,label_tensor

class IELTestDataset(Dataset):
    def __init__(self, path_to_csv, images_path,label_path, if_train):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.label_path = label_path
        self.if_train = if_train
        if if_train:
            self.transform = transforms.Compose([
                transforms.PILToTensor()
                ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
                ])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        image_id = row['img_id']

        # print('id:',image_id)
        # print('ref_id:', ref_id)
        #print(f'{image_id}'.split('.png')[-2] + '.npy')
        image_path = os.path.join(self.images_path, f'{image_id}')
        label_path = os.path.join(self.label_path, f'{image_id}'.split('.png')[-2] + '.npy') #注意后面这个ref和原始图像要放到一起

        image = Image.open(image_path)
        numpy_file = np.load(label_path)

        img_tensor = self.transform(image)/255.0
        label_tensor = torch.from_numpy(numpy_file)/255.0

        return img_tensor,label_tensor

#if __name__ == '__main__':


    # image = Image.open("T:/dataset_exp/256/img/a0001-jmac_DSC1459_P4.png")
    # numpy_file = np.load("T:/dataset_exp/256/label-h/a0001-jmac_DSC1459_P4.npy")
    #
    # img_tensor = transforms.functional.pil_to_tensor(image)
    # label_tensor = torch.from_numpy(numpy_file)/255.0
    # print(label_tensor)

