from __future__ import print_function, division
from iteround import saferound

import random
from PIL import Image

from dataloaders.dataset_helper import load_scv_file
from sklearn.model_selection import train_test_split
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from matplotlib.image import imread
from segmentation_models.backbones import get_preprocessing as process_image
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from scipy.ndimage.morphology import binary_erosion


from dataloaders.img_helper import show_sample
import numpy as np
import keras





def get_stratified_sampling(list_IDs, mask_coverage,batch_size):
    '''
    :param list_IDs: list of image indexes
    :param mask_coverage: corresponding mask coverage as a dictionary imageId-mask coverage values
    :param batch_size:
    :return:
    indexes of the batch size stratified
    '''
    n_bins = [0,  0.1, 0.2, 0.3, 0.4, 1] # 4 bins

    hist, val =  np.histogram(np.fromiter(mask_coverage.values(), dtype=float), bins = n_bins, normed = 1)

    bin1 = [key for key, val in mask_coverage.items() if float(val) <= 0.1]
    bin2 = [key for key, val in mask_coverage.items() if float(val) > 0.1 and float(val) <= 0.2]
    bin3 = [key for key, val in mask_coverage.items() if float(val) > 0.2 and float(val) <= 0.3]
    bin4 = [key for key, val in mask_coverage.items() if float(val) > 0.3 and float(val) <= 0.4]
    bin5 = [key for key, val in mask_coverage.items() if float(val) > 0.4]

    random.shuffle(bin1)
    random.shuffle(bin2)  # shuffle all mini dict
    random.shuffle(bin3)
    random.shuffle(bin4)
    random.shuffle(bin5)

    samples_per_bin = saferound(hist/sum(hist) * batch_size, places=0)

    indexes =[]
    for i in bin1[:int(samples_per_bin[0])]:
        indexes.append(list_IDs.index(i))
    for i in bin2[:int(samples_per_bin[1])]:
        indexes.append(list_IDs.index(i))
    for i in  bin3[:int(samples_per_bin[2])]:
        indexes.append(list_IDs.index(i))
    for i in bin4[:int(samples_per_bin[3])]:
        indexes.append(list_IDs.index(i))
    for i in bin5[:int(samples_per_bin[4])]:
        indexes.append(list_IDs.index(i))

    return indexes

class InriaDatagenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(512, 512), n_channels_img=3, n_channel_mask=1, shuffle=True, coverage = None, stratified_sampling = False, Transform = False, Process_function = None, Flip = False, Dilation = None, aux_loss = False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels_img = n_channels_img
        self.n_channels_lbl = n_channel_mask
        self.shuffle = shuffle
        self.on_epoch_end()
        self.mask_coverage = coverage
        self.stratified_sampling = stratified_sampling
        self.transform = Transform
        self.preprocess_input = Process_function
        self.Flip = Flip
        self.Dilation = Dilation
        self.Two_GT = aux_loss

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index, processing = True):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        if self.stratified_sampling:
            # TO DO get indexes given a distribution
            list_IDs_temp = get_stratified_sampling(self.labels, self.mask_coverage, self.batch_size)
        else:
            # Find list of IDs
            list_IDs_temp = [k for k in indexes]
        if processing!= True:
            temp_transform = self.transform  #temp variable, to switch back
            self.transform = False
        X, y = self.__data_generation(list_IDs_temp)
        if processing != True:
            self.transform = temp_transform # switch back


        return X, y




    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels_img), dtype = np.float32)
        y = np.empty((self.batch_size, *self.dim, self.n_channels_lbl),dtype = np.float32)
        if self.Two_GT:
            y_class_presence = np.empty((self.batch_size, self.n_channels_lbl),dtype = np.float32)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img_name = self.list_IDs[ID]
            label_name = self.labels[ID]
            # Load data and get label

            try:
                # Relative Path
                yp = Image.open(label_name)
                Xp = Image.open(img_name)
            except IOError:
                print("unexpected error with  the file :", img_name, label_name)
                img_name = self.list_IDs[0]
                label_name = self.labels[0]  # if wrong - always take the first one
                Xp = Image.open(img_name)
                yp = Image.open(label_name)

            Xp = Xp.resize((self.dim[0], self.dim[0]), Image.ANTIALIAS)
            yp = yp.resize((self.dim[0], self.dim[0]), Image.ANTIALIAS)
            Xs = np.array(Xp)
            ys = np.array(yp) # convert to numpy
            yp.close()
            Xp.close()

            if self.transform:
                try:
                    Xs = self.preprocess_input(Xs) # VGG processing of an image
                except:
                    print('Something is wrong with the transofmraiton function')
            if self.n_channels_lbl == 1:
                ys = np.expand_dims(ys/ 255.0, axis=3)  # divide image on 255 and add a dummy dimension
                if self.Dilation is not None:
                    ys = binary_erosion(ys, structure=np.ones((self.Dilation, self.Dilation, 1)), iterations=1, mask=None, border_value=0,
                                        origin=0, brute_force=False).astype(ys.dtype)
            if self.n_channels_lbl > 1:
                foreground = (ys/ 255.0)
                background = np.ones_like(ys)
                background = background -  foreground
                ys = np.dstack((background, foreground))



            if self.Flip:
                # flip in 50% of cases
                flip_yes = np.random.choice(2, 1)
                if flip_yes:
                    Xs = np.fliplr(Xs)
                    ys = np.fliplr(ys)
                flip_yes = np.random.choice(2, 1)
                if flip_yes:
                    Xs = np.flipud(Xs)
                    ys = np.flipud(ys)

            X[i,] = Xs
            # Store class
            y[i,] = ys
            if self.Two_GT:
                y_class_presence[i,] = (np.sum(ys)) > 0

        if self.Two_GT:
            y = [y, y_class_presence]
        return X, y


    def load_whole_dataset(self):
        batch_size_temp = self.batch_size
        self.batch_size = len(self.list_IDs)
        list_IDs_temp = np.arange(len(self.list_IDs))
        X, y = self.__data_generation(list_IDs_temp)
        self.batch_size = batch_size_temp # just get back to te pre-defined value
        return X, y

if __name__ == '__main__':
# main file test

    # define backbone since I will do image processing accordingly
    BACKBONE = 'resnet34'
    preprocess_input = process_image(BACKBONE)

    # train and validation splits
    import glob
    X_val = sorted(glob.glob("D:/programming/datasets/inria/aerialimagelabeling/inria_processed_from_folder256/val_frames/*.png"))
    y_val = sorted(glob.glob("D:/programming/datasets/inria/aerialimagelabeling/inria_processed_from_folder256/val_masks/*.png"))
    print('The num'
          'ber of validation samples is = %d' % (len(X_val)))


    coverage = {}
    gr_truth = np.zeros((len(y_val),1))
    for i in range(len(y_val)):
        mask = imread(y_val[i])
        H, W = mask.shape
        gr_truth[i] =  np.sum(mask) / (H*W)
        coverage[y_val[i]] = gr_truth[i]

    params = {'dim': (256, 256),
              'batch_size': 5,
              'n_channels_img': 3,
              'n_channel_mask': 1,
              'shuffle':False,
              'Dilation': 1}

    validation_generator = InriaDatagenerator(X_val, y_val, Transform=True, stratified_sampling=False, coverage=coverage,
                                             Process_function=preprocess_input, **params)
    (X, y) = validation_generator.__getitem__(0)
    for i in range(X.shape[0]):
        show_sample(X[i, :], (np.squeeze(y[i, :])))



