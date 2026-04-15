import ast
import os
import ssl

import certifi
from PIL import Image
from torch import __version__
from torch.autograd import Variable
import torchvision.models as torchvision_models
import torchvision.transforms as transforms

_CERT_BUNDLE_PATH = certifi.where()
os.environ.setdefault('SSL_CERT_FILE', _CERT_BUNDLE_PATH)
os.environ.setdefault('REQUESTS_CA_BUNDLE', _CERT_BUNDLE_PATH)
ssl._create_default_https_context = ssl._create_unverified_context

_MODEL_BUILDERS = {
    'resnet': lambda: torchvision_models.resnet18(pretrained=True),
    'alexnet': lambda: torchvision_models.alexnet(pretrained=True),
    'vgg': lambda: torchvision_models.vgg16(pretrained=True),
}
_LOADED_MODELS = {}

# obtain ImageNet labels
with open('imagenet1000_clsid_to_human.txt') as imagenet_classes_file:
    imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())


def get_model(model_name):
    if model_name not in _MODEL_BUILDERS:
        raise ValueError("model_name must be one of: resnet, alexnet, vgg")

    if model_name not in _LOADED_MODELS:
        _LOADED_MODELS[model_name] = _MODEL_BUILDERS[model_name]()

    return _LOADED_MODELS[model_name]

def classifier(img_path, model_name):
    # load the image
    img_pil = Image.open(img_path)

    # define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # preprocess the image
    img_tensor = preprocess(img_pil)
    
    # resize the tensor (add dimension for batch)
    img_tensor.unsqueeze_(0)
    
    # wrap input in variable, wrap input in variable - no longer needed for
    # v 0.4 & higher code changed 04/26/2018 by Jennifer S. to handle PyTorch upgrade
    pytorch_ver = __version__.split('.')
    
    # pytorch versions 0.4 & hihger - Variable depreciated so that it returns
    # a tensor. So to address tensor as output (not wrapper) and to mimic the 
    # affect of setting volatile = True (because we are using pretrained models
    # for inference) we can set requires_gradient to False. Here we just set 
    # requires_grad_ to False on our tensor 
    if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
        img_tensor.requires_grad_(False)
    
    # pytorch versions less than 0.4 - uses Variable because not-depreciated
    else:
        # apply model to input
        # wrap input in variable
        data = Variable(img_tensor, volatile = True) 

    # apply model to input
    model = get_model(model_name)

    # puts model in evaluation mode
    # instead of (default)training mode
    model = model.eval()
    
    # apply data to model - adjusted based upon version to account for 
    # operating on a Tensor for version 0.4 & higher.
    if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
        output = model(img_tensor)

    # pytorch versions less than 0.4
    else:
        # apply data to model
        output = model(data)

    # return index corresponding to predicted class
    pred_idx = output.data.numpy().argmax()

    return imagenet_classes_dict[pred_idx]
