import torch
from .resgcnet import *
from .ml_decoder import MLDecoder
from torchvision import models
import timm
from models.unet import Encoder
# 當創建新環境時，需要將ultralytics/nn/modules內的Classify class做更動
from ultralytics.nn.tasks import ClassificationModel


def get_optim(optim_name:str, model, lr:float):
    """optim_name = [adam | adamw | sgd | rmsprop | adagrad"""
    optim_name = optim_name.lower()
    if optim_name == "adam":
        return torch.optim.Adam(model.parameters(),lr = lr)
    elif optim_name == "adamw":
        return torch.optim.AdamW(model.parameters(),lr = lr)
    elif optim_name == "sgd":
        return torch.optim.SGD(model.parameters(),lr = lr)
    elif optim_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(),lr = lr)
    elif optim_name == "adagrad":
        return torch.optim.Adagrad(model.parameters(),lr = lr)
    else:
        print(f'Don\'t find the model: {optim_name} . default optimizer is adam')
        return torch.optim.Adam(model.parameters(),lr = lr)
    
def initialize_model(model_name, num_classes,use_custom_clf=True, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet50
        """
        model_ft = timm.create_model('resnet50',pretrained=True)
        num_ftrs = model_ft.fc.in_features
        if use_custom_clf:
            # model_ft.fc = Classifier(num_ftrs, num_classes,reduction_rate=4)
            model_ft.global_pool = nn.Identity()
            model_ft.fc = MLDecoder(num_classes)
        else:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet152":
        """ Resnet152
        """
        model_ft = timm.create_model('resnet152',pretrained=True)
        num_ftrs = model_ft.fc.in_features
        if use_custom_clf:
            model_ft.global_pool = nn.Identity()
            model_ft.fc = MLDecoder(num_classes)
        else:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)

    # elif model_name == "alexnet":
    #     """ Alexnet
    #     """
    #     model_ft = models.alexnet(pretrained=use_pretrained)
    #     num_ftrs = model_ft.classifier[6].in_features
    #     if use_custom_clf:
    #         model_ft.classifier[6] = Classifier(num_ftrs, num_classes,reduction_rate=4)
    #     else:
    #         model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    # elif model_name == "vgg":
    #     """ VGG11_bn
    #     """
    #     model_ft = models.vgg11_bn(pretrained=use_pretrained)
    #     num_ftrs = model_ft.classifier[6].in_features
    #     if use_custom_clf:
    #         model_ft.classifier[6] = Classifier(num_ftrs, num_classes,reduction_rate=4)
    #     else:
    #         model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    # elif model_name == "squeezenet":
    #     """ Squeezenet
    #     """
    #     model_ft = models.squeezenet1_0(pretrained=use_pretrained)
    #     model_ft.classifier[1] = Classifier(num_ftrs, num_classes,reduction_rate=4)
    #     model_ft.num_classes = num_classes

    # elif model_name == "densenet":
    #     """ Densenet
    #     """
    #     model_ft = models.densenet121(pretrained=use_pretrained)
    #     num_ftrs = model_ft.classifier.in_features
    #     if use_custom_clf:
    #         model_ft.classifier = Classifier(num_ftrs, num_classes,reduction_rate=4)
    #     else:
    #         model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    # elif model_name == "inception":
    #     """ Inception v3
    #     Be careful, expects (299,299) sized images and has auxiliary output
    #     """
    #     model_ft = models.inception_v3(pretrained=use_pretrained)
    #     # Handle the auxilary net
    #     num_ftrs = model_ft.AuxLogits.fc.in_features
    #     model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    #     # Handle the primary net
    #     num_ftrs = model_ft.fc.in_features
    #     if use_custom_clf:
    #         model_ft.fc = Classifier(num_ftrs, num_classes,reduction_rate=4)
    #     else:
    #         model_ft.fc = nn.Linear(num_ftrs,num_classes)
    
    elif model_name == "yolov8":
        """Yolov8 by ultralytic, please install the mudule ultralytic first"""
        if use_pretrained:
            model_obj = ClassificationModel(cfg='yolov8n-cls.yaml', # build a new model from YAML
                                        model=None,ch=3,
                                        nc= num_classes,
                                        cutoff=10,verbose=True)
            model_ft = model_obj.model.cpu()
            model_ft.load_state_dict(torch.load("yolov8n-cls.pt",map_location='cpu'),strict=False)
        else:
            model_obj = ClassificationModel(cfg='yolov8n-cls.yaml', # build a new model from YAML
                                        model=None,ch=3,
                                        nc= num_classes,
                                        cutoff=10,verbose=True)  
            model_ft = model_obj.model

    elif model_name == "mldecoder":
        encoder = Encoder(3,n_layers=4, is_normalize=True)
        decoder = MLDecoder(num_classes,initial_num_features=64*(2**4))
        model_ft = nn.Sequential(encoder,decoder)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft