import torch
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os
from torch.utils.data import Dataset



class CancerDataset(Dataset):
    def __init__(self, data, target, datapath, modetype, transform):
        super(CancerDataset, self).__init__()
        self._data = data
        self._target = target
        self._modetype = modetype 
        self._transform = transform
        self._datapath = datapath
        #self.image_file_list = [s for s in os.listdir(datapath+'test/')]
        #if self.modetype == "train":
        #    self.fileName = pd.read_csv(os.path.join(dataFolder, "train.csv"))
        #elif self.modetype == "val":
        #    self.fileName = pd.read_csv(os.path.join(dataFolder, "val.csv"))
        if self._modetype == "test":
            self.fileName = pd.read_csv(os.path.join(datapath, 'sample_submission.csv'))
    
    def __len__(self):
        if self._modetype == "test":
            return len(self.fileName.set_index('id').index.values)
        else: 
            return len(self._data)

    def __getitem__(self, idx):
        #labelList = np.asarray(self.fileName.set_index('id')['label'].values)
        #if self.modetype == "test":
        #    imgName = os.path.join(self.folder, 'test', imageFileList[idx]+'.tif')
        #else:
        #    imgName = os.path.join(self.folder, 'train', imageFileList[idx]+'.tif')
        #label = labelList[idx]
        #img = Image.open(imgName)
        #img = cv2.cvtColor(np.asarray(img).astype(np.float32), cv2.COLOR_RGB2BGR)
        #img = cv2.resize(img, (224, )*2, interpolation = cv2.INTER_LINEAR)
        if self._modetype == "test":
            imageFileList = self.fileName.set_index('id').index.values
            imageName = os.path.join(self._datapath, 'test', imageFileList[idx] + '.tif')
            label = 0
        else:
            imageName = os.path.join(self._datapath, 'train', self._data[idx] + '.tif')
            #print(imageName)
            label = self._target[idx]
        if os.path.exists(imageName)==False:
            print(imageName)
        img = Image.open(imageName)
        if self._transform is not None:
            img = self._transform(img)
        #img = img.transpose(2, 0, 1)
        return img, label

        

