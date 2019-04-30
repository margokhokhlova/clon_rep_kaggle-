from keras.engine.training import Model
from keras.layers import SpatialDropout2D, Conv2D
from segmentation_models import Unet, Unet_SE


def unet_resnext_50(input_shape, freeze_encoder):
    resnet_base, hyper_list = Unet(backbone_name='resnext50',
                                   input_shape=input_shape,
                                   input_tensor=None,
                                   encoder_weights='imagenet',
                                   freeze_encoder=freeze_encoder,
                                   skip_connections='default',
                                   decoder_block_type='transpose',
                                   decoder_filters=(512, 256, 128, 64, 32),
                                   decoder_use_batchnorm=True,
                                   n_upsample_blocks=5,
                                   upsample_rates=(2, 2, 2, 2, 2),
                                   classes=1,
                                   activation='sigmoid')

    x = resnet_base.output #SpatialDropout2D(0.2)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(x)

    model = Model(resnet_base.input, x)

    return model


def unet_resnext_50_lovasz(input_shape, freeze_encoder):
    resnet_base, hyper_list = Unet(backbone_name='resnext50',
                                   input_shape=input_shape,
                                   input_tensor=None,
                                   encoder_weights='imagenet',
                                   freeze_encoder=freeze_encoder,
                                   skip_connections='default',
                                   decoder_block_type='transpose',
                                   decoder_filters=(512, 256, 128, 64, 32),
                                   decoder_use_batchnorm=True,
                                   n_upsample_blocks=5,
                                   upsample_rates=(2, 2, 2, 2, 2),
                                   classes=1,
                                   activation='linear')

    x = resnet_base.output#SpatialDropout2D(0.2)()
    x = Conv2D(1, (1, 1), name="prediction")(x)

    model = Model(resnet_base.input, x)

    return model


def unet_resnext_50_margo(input_shape, freeze_encoder, seloss = True):
    resnet_base, hyper_list = Unet_SE(backbone_name='resnext50',
                                   input_shape=input_shape,
                                   input_tensor=None,
                                   encoder_weights='imagenet',
                                   freeze_encoder=freeze_encoder,
                                   skip_connections='default',
                                   decoder_block_type='transpose',
                                   decoder_filters=(512, 256, 128, 64, 32),
                                   decoder_use_batchnorm=True,
                                   n_upsample_blocks=5,
                                   upsample_rates=(2, 2, 2, 2, 2),
                                   classes=1,
                                   activation='sigmoid',
                                   seloss = seloss)

    x,  aux_x = resnet_base.output #SpatialDropout2D(0.2)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(x)
    output = x
    if seloss:
        output = [x,  aux_x ]
    model = Model(inputs=resnet_base.input, outputs=output)
    print(hyper_list)

    return model