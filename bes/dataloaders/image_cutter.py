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



# Datasets
folder_imgs = "D:/programming/datasets/inria/aerialimagelabeling/AerialImageDataset/train/im_val/"
folder_lbls = "D:/programming/datasets/inria/aerialimagelabeling/AerialImageDataset/train/gt_val/"

imgs = get_all_images_in_folder(folder_imgs, ".tif")
lbls = get_all_images_in_folder(folder_lbls, ".tif")

folder_saved_imgs = 'D:/programming/datasets/inria/aerialimagelabeling/512/'
all_images_abs_path = []
all_labels_abs_path = []
for i in range(len(imgs)):
    img_name = imgs[i]
    lbl_name = lbls[i]
    X = io.imread(img_name)
    y = io.imread(lbl_name)

    # img_name = img_name[-36:-4] # for IGN
    img_name = (os.path.basename(img_name))[:-4]  # for INRIA print(os.path.basename(img_name))


    all_images_abs_path += cutimage(X, size=(512, 512), stride=(12, 12), path=folder_saved_imgs+'val_frames/', dir_name=img_name,
                                    name_indx='img.png')
    all_labels_abs_path += cutimage(y, size=(512, 512), stride=(12, 12), path=folder_saved_imgs+'val_masks/', dir_name=img_name,
                                   name_indx='lbl.png')


# dictionary = dict(zip(all_images_abs_path, all_labels_abs_path))
# with open(folder_saved_imgs + '/img_pairs.csv', 'w') as f:
#     for key, value in dictionary.items():
#         f.write("%s,%s\n" % (key, value))
