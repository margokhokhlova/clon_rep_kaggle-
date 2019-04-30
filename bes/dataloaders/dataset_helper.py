import os
import glob
import numpy as np
import csv
from PIL import Image

def findallimagesosm(folder):
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    subfolders_names = [f.name for f in os.scandir(folder) if f.is_dir()] #find all files in folders
    images_ID = {}
    labels_ID = {}

    for counter, folder in enumerate(subfolders):
        images = sorted(glob.glob(folder + "/*img.png"))
        labels = sorted(glob.glob(folder + "/*lbl.png"))
        images_ID[subfolders_names[counter]] = images
        labels_ID[subfolders_names[counter]] = labels


    return images_ID, labels_ID

def get_all_images_in_folder(folder, img_type):
    images = sorted(glob.glob(folder + "*" + img_type))
    return images


def findallimagesosm_nopartition(folder, with_label = True):
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    subfolders_names = [f.name for f in os.scandir(folder) if f.is_dir()] #find all files in folders
    images_ID = {}
    labels_ID = {}
    all_images_id = []
    all_labels_id = []

    for counter, folder in enumerate(subfolders):
        if with_label:
            images = sorted(glob.glob(folder + "/*img.png"))
            labels = sorted(glob.glob(folder + "/*lbl.png"))
            images_ID[subfolders_names[counter]] = images
            labels_ID[subfolders_names[counter]] = labels
            all_images_id = np.concatenate((all_images_id, images), axis = 0)
            all_labels_id= np.concatenate((all_labels_id, labels), axis = 0)

        else:
            images = sorted(glob.glob(folder + "/*img.png"))
            images_ID[subfolders_names[counter]] = images
            all_images_id = np.concatenate((all_images_id, images), axis=0)
    if with_label:
        return all_images_id, all_labels_id
    else:
        return all_images_id




def load_mask_coverage(file_name_coverage):
    masks_coverage = {}
    with open(file_name_coverage) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # print(row['ID'], row['val'])
            masks_coverage[row['ID']] = row['val']

    return masks_coverage


def load_scv_file(file_name_coverage):
    Xname_Yname = {}
    with open(file_name_coverage) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # print(row['ID'], row['val'])
            Xname_Yname[row['img']] = row['lbl']

    return Xname_Yname



def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i :
            return i

def load_batch(img_names, img_size = (512, 512), colors =3,  process_function = None):
    batch_size = len(img_names)
    imgs = np.empty((batch_size, *img_size, colors), dtype = np.float32) # empty storage array
    for i in range(batch_size):
        Xp = Image.open(img_names[i])
        Xs = Xp.resize(img_size, Image.ANTIALIAS)
        Xs = np.array(Xs)
        if process_function is not None:
            Xs = process_function(Xs)  # model-specific processing of an image
        imgs[i,:] = Xs

    return imgs

def save_batch(imgs, IDs,  folder_path, img_size = (512, 512), img_format ='.tif'):
    dir_name = os.path.basename(IDs[0][:-12])
    try:  # create a directory to store all the images
        os.mkdir((folder_path + dir_name))
    except OSError:
        print("Creation of the directory %s failed" % (folder_path + dir_name))
    for i in range(len(imgs)):
        name = IDs[i][IDs[i].size-12:]
        new_path = folder_path+dir_name +'/' + name
        img = img_frombytes(np.squeeze(imgs[i, :]))
        img = img.resize(img_size, Image.ANTIALIAS)
        img.save(new_path)
    return imgs


def load_img(img_name,  process_function = None):
    Xp = Image.open(img_name)
    Xp = np.array(Xp)

    if process_function is not None:
        try:
            Xs = process_function(Xp)  # model-specific processing of an image
            Xp = Xs
        except:
            print('Something is wrong with the transofmraiton function')
    return Xp
def save_img(img, ID,  folder_path, img_size = (512, 512), img_format ='.tif'):
    dir_name = os.path.basename(ID[:-12])
    try:  # create a directory to store all the images
        os.mkdir(folder_path + dir_name)
    except OSError:
        print("Creation of the directory %s failed" % folder_path + dir_name)
    name = ID[len(ID)-12:]
    new_path = folder_path + dir_name + '/' + name +img_format
    img = img_frombytes(np.squeeze(img))
    img = img.resize(img_size, Image.ANTIALIAS)
    img.save(new_path)

# The standard work-around: first convert to greyscale
def img_grey(data):
    return Image.fromarray(data * 255, mode='L').convert('1')

def img_frombytes(data):
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)

if __name__ == '__main__':

    test_Ids = findallimagesosm_nopartition('D:/programming/datasets/inria/aerialimagelabeling/inria_processed_test', False)
    path_results = 'D:/programming/datasets/inria/aerialimagelabeling/results_test/'
    # load batches of 100 images, then predict the label on them, and save it
    step = 100 # 100 subimages for the big image
    for batch in range(0, len(test_Ids), 100):
        img_batch = load_batch(test_Ids[batch:batch+step], img_size=(256, 256))
        masks_pred = np.zeros((step, 256, 256))
        save_batch(masks_pred, test_Ids[batch:batch+step], path_results)

