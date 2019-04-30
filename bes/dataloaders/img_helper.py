import numpy as np
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize, rescale
from dataloaders.dataset_helper import get_all_images_in_folder
import glob

import os
#from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral

def cutimage(image, size = (512,512), stride = (12,12),  path = None, dir_name = None, name_indx = '.png'):
    ''' function cuts high res images to a smaller resolution
    and returns the array of num_images.
    In this implementation, the frames overlap by stride pixels '''
    try:
        H, W, C = image.shape
    except:
        H, W = image.shape


    # try: # create a directory to store all the images
    #     os.mkdir(path + dir_name)
    # except OSError:
    #     print("Creation of the directory %s failed" % path + dir_name)

    num_images = np.floor(image.shape[0]/(size[0]-stride[0])) * np.floor(image.shape[1]/(size[1]-stride[1])) #check how many images are there
    list_files = []
    #cropped_images = np.zeros((int(num_images), size[0], size[1], C)) # array N_images, H, W, ColorChannels
    index = 0
    for height in range(0, H-stride[0], size[0]-stride[0]):
        for width in range(0, W-stride[1], size[1]-stride[1]):
            if height == 0 and width == 0:
                cropped_image = image[height:height + size[0], width:width + size[1]]
            if height == 0 and width != 0:
                cropped_image = image[height:height+size[0], width-stride[1]: width-stride[1] + size[1]]
            if height !=0 and width == 0:
                cropped_image = image[height-stride[0]:height -stride[0] + size[0], width:width + size[1]]
            if height!=0 and width !=0:
                cropped_image = image[height-stride[0]:height -stride[0] + size[0],width-stride[1]:width-stride[1] + size[1]]
            if path is not None:
                new_path = path + dir_name + '_'+ str(index).zfill(4) + name_indx
            io.imsave(new_path, cropped_image)
            list_files.append(new_path)
            #cropped_images[0,:,:,:] = cropped_image
            index +=1
    return list_files


def show_img_gtprederror(img, lbl, gt):
    pred_gt = lbl+ gt
    fig = plt.figure(figsize=(12, 8))
    fig.add_subplot(1,2,1)
    plt.imshow(img)
    plt.title("Image")
    fig.add_subplot(1,2,2)
    plt.imshow(pred_gt, cmap = 'summer')
    plt.title("Generated Label + GT")
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()


def show_sample(img, lbl):
    #img = img.astype(np.uint8)
    C = None
    try:
        H, W, C = lbl.shape
    except:
        H, W =lbl.shape
    if C is None:
        fig = plt.figure()
        fig.add_subplot(1,2,1)
        plt.suptitle("Image")
        plt.imshow(img)
        fig.add_subplot(1,2,2)
        plt.imshow(lbl, cmap = 'bone')
        plt.suptitle("Mask")
        plt.pause(0.001)  # pause a bit so that plots are updated
        plt.show()
    else:
        fig = plt.figure()
        fig.add_subplot(1, 3, 1)
        plt.suptitle("Image")
        plt.imshow(img)
        fig.add_subplot(1, 3, 2)
        plt.imshow(lbl[:,:,0], cmap='bone')
        plt.suptitle("Mask class 1")
        fig.add_subplot(1, 3, 3)
        plt.imshow(lbl[:, :, 1], cmap='bone')
        plt.suptitle("Mask class 2")
        plt.pause(0.001)  # pause a bit so that plots are updated
        plt.show()


def show_sample_gt(img, lbl, gt, undoprocessing = False):
    """Show image with labels"""
    mean = [0.485, 0.456, 0.406]  # from KERAS Processing
    std = [0.229, 0.224, 0.225]

    if undoprocessing:
        img = img.astype(np.float64)
        img[:, :,0] /= 1/std[0]
        img[:, :,1] /= 1/std[1]
        img[:, :,2] /= 1/std[2]
        img[:, :,0] += mean[0]
        img[:, :,1] += mean[1]
        img[:, :,2] += mean[2]
        img = img[:,:,::-1]
        img = (255*img).astype(np.int16)

    fig = plt.figure(figsize=(10, 8))
    fig.add_subplot(1,3,1)
    plt.imshow(img)
    plt.title("Image")
    fig.add_subplot(1,3,2)
    plt.imshow(lbl, cmap = 'bone')
    plt.title("Generated Label")
    fig.add_subplot(1, 3, 3)
    plt.imshow(gt, cmap = 'bone')
    plt.title("Ground Truth")
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()



def make_multi_class_image(y, ignore_channel = None):
    ''' the function takes the original image as H, W, C
    and transforms it into H,W, C + 1 binary images,
    where C = number of classes + the background class.
    The images are expected to have classes coded as different color channels by pure color
    i.e the value is 255'''

    if ignore_channel is not None:
        H,W, or_ch = y.shape
        y_red = np.zeros((H,W, or_ch-1))
        counter = 0
        for i in range(or_ch):
            if i!=ignore_channel:
                y_red[:,:,counter] = y[:,:,i]
                counter += 1
        y = y_red

    H,W,C = y.shape


    ys = np.zeros((H,W, C+1))


    for i in range(C): # each channel
        ys[:, :, i] = (1.0 - (y[:, :, i] / 255.0))  # take only the one channel of the image
    ys[:, :, C] = 1.0 - (np.sum(ys,keepdims = 1)>0) # background
    return ys





def assemble_images_from_tiles(folder, save_path, stride, im_size =(5000,5000), tile_size =(512, 512)):
    '''
    :param folder: folder with images
    :param save_path: the path to folder where image will be saved
    :param stride: the overlapping stride between the images
    :return:
    The function merges images from folder into a one single image.
    The image is then saved in the save_path folder
    '''
    im_name = os.path.basename(folder)
    img_list = get_all_images_in_folder(folder +'/', '.png')
    img = np.zeros(im_size)
    # direction to assemble: H -> W
    img_per_width = im_size[1] / (tile_size[1]- stride)
    tile_no_overlap = (tile_size[1]- stride) # images which I finally take
    h = 0
    w = 0
    for i in range(len(img_list)):
         im = Image.open(img_list[i]).convert('L')
         im = np.asarray(im)
         img[h*tile_no_overlap:h*tile_no_overlap+tile_no_overlap, w*tile_no_overlap:w*tile_no_overlap+tile_no_overlap] = im[0:tile_no_overlap, 0:tile_no_overlap]
         w+=1
         if w == img_per_width: #when we reached the end of the row
            h+=1
            w=0
    img[img.astype(int)>0] = 255
    # save the resulting image in a folder
    save_img_path = save_path + im_name + '.tif'
    io.imsave(save_img_path, img)




#def save_images_batch(batch_img, )

#
# # Original_image = Image which has to labelled
# # Mask image = Which has been labelled by some technique..
# def crf(original_image, mask_img):
#     """
#     Function which returns the labelled image after applying CRF
#     taken from here https://www.kaggle.com/nafisur/crf-nr
#     """
#     # Converting annotated image to RGB if it is Gray scale
#     if (len(mask_img.shape) < 3):
#         mask_img = gray2rgb(mask_img)
#
#     #     #Converting the annotations RGB color to single 32 bit integer
#     annotated_label = mask_img[:, :, 0] + (mask_img[:, :, 1] << 8) + (mask_img[:, :, 2] << 16)
#
#     #     # Convert the 32bit integer color to 0,1, 2, ... labels.
#     colors, labels = np.unique(annotated_label, return_inverse=True)
#
#     n_labels = 2
#
#     # Setting up the CRF model
#     d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)
#
#     # get unary potentials (neg log probability)
#     U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
#     d.setUnaryEnergy(U)
#
#     # This adds the color-independent term, features are the locations only.
#     d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
#                           normalization=dcrf.NORMALIZE_SYMMETRIC)
#
#     # Run Inference for 10 steps
#     Q = d.inference(10)
#
#     # Find out the most probable class for each pixel.
#     MAP = np.argmax(Q, axis=0)
#
#     return MAP.reshape((original_image.shape[0], original_image.shape[1]))
#

def get_coverage(mask):
        '''return the coverage of the mask per image'''

        try:
            H, W, C = mask.shape
        except:
            H, W = mask.shape
        mask = np.squeeze(mask)
        return np.sum(mask) / (H*W)


def smooth_mask(mask, ups_rate = 10):
    """ the function upsamples the mask to a big size and the downsamples it back,
    using the bilinear interpolation. I am getting smooth borders when doing it """
    mask = np.squeeze(mask)
    H, W = mask.shape
    upsampled = rescale(mask, ups_rate, anti_aliasing=True)
    downsampled =rescale(upsampled,  1.0 / ups_rate, anti_aliasing=True)
    downsampled = np.expand_dims(downsampled,  axis=2)
    return downsampled


def smooth_batch(pr):
    ''' function to calculate the mean IOU for a batch, calculating only for
    the positive class
    '''
    batch_size = pr.shape[0]
    for i in range(batch_size):
        pr[i,:] = smooth_mask(pr[i,:])
    return pr


def assemble_images_from_tiles_draft(folder, save_path, stride, im_name, im_correlation =(9,6), tile_size =(512, 512)):
    img_list = sorted(glob.glob(folder + '*.png'))
    i=0
    H = im_correlation[0]*(tile_size[0]-stride) +stride
    W = im_correlation[1] * (tile_size[1] - stride) +stride
    image = np.zeros((H,W))
    for height in range(0,H-  stride, tile_size[0] - stride):
        for width in range(0,W-stride, tile_size[1] - stride):
            # if height == 0 and width == 0:
            tile =Image.open(img_list[i])
            #plt.imshow(tile)
            #plt.show()
            tile = np.array(tile)
            image[height:height + tile_size[0], width:width + tile_size[1]] += tile
            plt.imshow(image)
            plt.show()
            i+=1
    image[image.astype(int) > 0] = 255
    # save the resulting image in a folder
    save_img_path = save_path + im_name + '.png'
    io.imsave(save_img_path,image.astype(int))


if __name__ == '__main__':
    # save_patch = 'D:/programming/datasets/inria/aerialimagelabeling/results_test/final_images/'
    # root_folder ='D:/programming/datasets/inria/aerialimagelabeling/inria_processed_test/'
    # subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    # for folder in subfolders:
    #     assemble_images_from_tiles(folder, save_patch,stride=262, im_size =(5000,5000))
    save_patch = 'D:/programming/datasets/inria/aerialimagelabeling/results_saved/'
    img_folder ='D:/programming/datasets/inria/aerialimagelabeling/results_saved/best_algo_input_smoothed/best_algo_input/'
    assemble_images_from_tiles_draft(img_folder, save_patch, stride = 256, im_correlation = (6,9), im_name='best_algo_output')
        #
        #
        # ### draft
        # for height in range(0, H, size[0] - stride[0]):
        #     for width in range(0, W, size[1] - stride[1]):
        #         if height == 0 and width == 0:
        #             cropped_image = image[height:height + size[0], width:width + size[1]]
        #         if height == 0 and width != 0:
        #             cropped_image = image[height:height + size[0], width - stride[1]:width - stride[1] + size[1]]
        #         if height != 0 and width == 0:
        #             cropped_image = image[height - stride[0]:height - stride[0] + size[0], width:width + size[1]]
        #         if height != 0 and width != 0:
        #             cropped_image = image[height - stride[0]:height - stride[0] + size[0],
        #                             width - stride[1]:width - stride[1] + size[1]]
        #         if path is not None:
        #             new_path = path + dir_name + '_' + str(index).zfill(4) + name_indx
        #             cropped_image = np.fliplr(cropped_image)
        #             io.imsave(new_path, cropped_image)
        #         list_files.append(new_path)
        #         # cropped_images[0,:,:,:] = cropped_image
        #         index += 1
        # return list_files
