from models.models_zoo import unet_resnext_50, unet_resnext_50_lovasz, unet_resnext_50_margo
from segmentation_models.backbones.preprocessing import get_preprocessing


def get_model(network, input_shape, freeze_encoder):
    if network == 'unet_resnext_50':
        model = unet_resnext_50(input_shape, freeze_encoder)
        return model
    elif network == 'unet_resnext_50_lovasz':
        model = unet_resnext_50_lovasz(input_shape, freeze_encoder)
        return model
    elif  network == 'unet_resnext_50_margo':
        model = unet_resnext_50_margo(input_shape, freeze_encoder, seloss = True)
        return model
    else:
        raise ValueError('Unknown network ' + network)

    return model
