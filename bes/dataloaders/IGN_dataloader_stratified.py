from __future__ import print_function, division
import os
import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import seaborn as sns
from dataloaders.dataset_helper import findallimagesosm, findallimagesosm_nopartition, load_mask_coverage, get_all_images_in_folder
from dataloaders.img_helper import  cutimage
import matplotlib.pyplot as plt
import numpy as np
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import csv

def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i :
            return i

class OSMDataset_stratified(Dataset):
    '''Characterizes an OSM dataset for PyTorch
    https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel '''
    def __init__(self, list_IDs, labels, label_mask = 'house', transforms=None,  mask_classes =None):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.transforms = transforms
        self.label_mask = label_mask
        self.mask_class = mask_classes


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)


    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        img_name = self.list_IDs[index]
        label_name = self.labels[index]
        # Load data and get label
        try:
            X = io.imread(img_name)
            y = io.imread(label_name)
        except:
            X = None
            y = None
            print(img_name)
            print(label_name)
        if self.label_mask =='house':
            y =np.expand_dims(1.0-(y[:, :, 2]/255.0), axis=3) #take only the red channel of the image


        # if self.transform:
        #     X = self.transform(X) # optional transform

        return (X, y)

    def load_all_files(self):
        self.imgs_df = list()
        self.lbls_df = list()
        self.coverage = list()
        len_imgs = len(self.list_IDs)
        for i in range(len_imgs):
            X,y = self.__getitem__(i)
            #self.imgs_df.append(X)
            #self.lbls_df.append(y)
            self.coverage.append(self.get_coverage(y))
    def get_coverage(self, mask):
        '''return the coverage of the mask per image'''
        H, W, C = mask.shape
        mask = np.squeeze(mask)
        return np.sum(mask) / (H*W)



if __name__ == '__main__':

    train_df = {}

    # Datasets
    folder_imgs = "D:/programming/datasets/inria/aerialimagelabeling/AerialImageDataset/train/images/"
    folder_lbls = "D:/programming/datasets/inria/aerialimagelabeling/AerialImageDataset/train/gt/"

    imgs = get_all_images_in_folder(folder_imgs, ".tif")
    lbls = get_all_images_in_folder(folder_lbls, ".tif")

    folder_saved_imgs = 'D:/programming/datasets/inria/aerialimagelabeling/inria_processed_augmented/'
    all_images_abs_path =[]
    all_labels_abs_path = []
    for i in range(len(imgs)):
        img_name = imgs[i]
        lbl_name = lbls[i]
        X = io.imread(img_name)
        y = io.imread(lbl_name)

        #img_name = img_name[-36:-4] # for IGN
        img_name = (os.path.basename(img_name))[:-4] # for INRIA print(os.path.basename(img_name))

        all_images_abs_path += cutimage(X, size=(512, 512), stride=(12, 12), path= folder_saved_imgs, dir_name = img_name, name_indx='img.png')
        all_labels_abs_path += cutimage(y, size=(512, 512), stride=(12, 12), path= folder_saved_imgs, dir_name = img_name,  name_indx='lbl.png')

    dictionary = dict(zip( all_images_abs_path, all_labels_abs_path))
    with open(folder_saved_imgs + '/img_pairs.csv', 'w') as f:
        for key, value in dictionary.items():
            f.write("%s,%s\n" % (key,  value))
