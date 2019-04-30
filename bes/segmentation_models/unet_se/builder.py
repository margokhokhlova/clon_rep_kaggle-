from keras.layers import Conv2D, Lambda, Dense, Multiply, Add, Flatten, AveragePooling2D, Reshape
from keras.models import Model
import keras.backend as K

from keras.losses import binary_crossentropy

from .blocks import Transpose2D_block
from .blocks import Upsample2D_block
from ..utils import get_layer_number, to_tuple
import tensorflow as tf

from keras.layers import Layer


def cse_block(prevlayer, prefix):
    mean = Lambda(lambda xin: K.mean(xin, axis=[1, 2]))(prevlayer)
    lin1 = Dense(K.int_shape(prevlayer)[3] // 2, name=prefix + 'cse_lin1', activation='relu')(mean)
    lin2 = Dense(K.int_shape(prevlayer)[3], name=prefix + 'cse_lin2', activation='sigmoid')(lin1)
    x = Multiply()([prevlayer, lin2])
    return x


def sse_block(prevlayer, prefix):
    # # Bug? Should be 1 here?
    # conv = Conv2D(K.int_shape(prevlayer)[3], (1, 1), padding="same", kernel_initializer="he_normal",
    #               activation='sigmoid', strides=(1, 1),
    #               name=prefix + "_conv")(prevlayer)
    # conv = Multiply(name=prefix + "_mul")([prevlayer, conv])
    # return conv Conv2D(1,(1,1), padding='valid')

    conv = Conv2D(1, (1, 1), padding="same", kernel_initializer="he_normal", activation='sigmoid', strides=(1, 1),
                  name=prefix + "_conv")(prevlayer)
    conv = Multiply(name=prefix + "_mul")([prevlayer, conv])
    return conv


def csse_block(x, prefix):
    '''
    Implementation of Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks
    https://arxiv.org/abs/1803.02579
    '''
    cse = cse_block(x, prefix)
    sse = sse_block(x, prefix)
    x = Add(name=prefix + "_csse_mul")([cse, sse])

    return x


# def encoding_block(backbone_features,classes,  prefix):
#
#     X = backbone_features
#
#     X = AveragePooling2D((2, 2), name="avg_pool")(X)
#     # BxDxHxW => Bx(HW)xD
#
#     B, W, H, D = X.shape
#
#     X = tf.reshape(X,  [tf.shape(X)[0], H*W, D])
#
#     X = Flatten()(X)
#     # Dense layer with Sigmoid activation to predict the presence of the object on the picture by evaluating the encoded features
#     X = Dense(classes, input_shape=(tf.shape(X)[1],),  name=prefix, activation='sigmoid')(X)
#
#     return X # x should be the probability whether the image has the house on it or not!




def build_unet(backbone, classes, skip_connection_layers,
               decoder_filters=(512, 256, 128, 64, 32),
               upsample_rates=(2, 2, 2, 2, 2),
               n_upsample_blocks=5,
               block_type='upsampling',
               activation='sigmoid',
               use_batchnorm=False,
               seloss=True):
    input = backbone.input
    x = backbone.output
    classes = classes

    hyper_list = []

    if seloss:
        ''' Keras implementation of the Context Encoding for Semantic Segmentation module, but here without the feature upweighting
          input - the backbone output from the network, math:`\mathcal{R}^{B\times 16\times 16\times D}` (where :math:`B` is batch)
          output - BxNbClasses Tensor  - predicts whether there is an object or not, for my binary case this is B x one single number 
          TO DO: code his as a layer wrapper! '''
        X = AveragePooling2D((2, 2), name="avg_pool")(x)
        # BxDxHxW => Bx(HW)xD  MAYBE reshape?
        #X = Reshape((tf.shape(X)[1]*tf.shape(X)[2], tf.shape(X)[3]))(X)
        X = Flatten()(X)


        # Dense layer with Sigmoid activation to predict the presence of the object on the picture by evaluating the encoded features
        auxiliary_output = Dense(classes, input_shape=(tf.shape(X)[1],), name='se_auxillary', activation='sigmoid')(X)

        #auxiliary_output = encoding_block(x, classes, prefix='margo_se_aux_output')
        #hyper_list.append(auxiliary_output)


    if block_type == 'transpose':
        up_block = Transpose2D_block
    else:
        up_block = Upsample2D_block

    # convert layer names to indices
    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                            for l in skip_connection_layers])


    for i in range(n_upsample_blocks):

        # check if there is a skip connection
        skip_connection = None
        if i < len(skip_connection_idx):
            skip_connection = backbone.layers[skip_connection_idx[i]].output

        upsample_rate = to_tuple(upsample_rates[i])

        x = up_block(decoder_filters[i], i, upsample_rate=upsample_rate,
                     skip=skip_connection, use_batchnorm=use_batchnorm)(x)

        if i==0:
            x = sse_block(x, prefix='sse_block_{}'.format(i)) # only one squeeze module in the layer 1

        hyper_list.append(x)

    output = x
    if seloss:
        output = [x, auxiliary_output]
    model = Model(inputs=input, outputs=output)

    return model, hyper_list


