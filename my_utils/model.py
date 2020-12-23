import torchvision.models as models
import torch.nn as nn
from my_utils.my_models import resnet_for_tiny as rst


def model(network_name, num_classes, pretrained=False):
    try:
        return eval(network_name + '({}, {})'.format(num_classes, pretrained))
        # return locals()[network_name](pretrained)
    except Exception as e:
        print('[Error]', e)
        exit(1)


# Insert your model function here!
# Ref.1: https://pytorch.org/docs/stable/torchvision/models.html
# Ref.2: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# Ref.3: https://stackoverflow.com/questions/63015883/pytorch-based-resnet18-achieves-low-accuracy-on-cifar100

def resnet18_for_tiny(num_classes, pretrained):
    return rst.resnet18_for_tiny(num_classes=num_classes)


def resnet34_for_tiny(num_classes, pretrained):
    return rst.resnet34_for_tiny(num_classes=num_classes)


def resnet50_for_tiny(num_classes, pretrained):
    return rst.resnet50_for_tiny(num_classes=num_classes)


def resnet101_for_tiny(num_classes, pretrained):
    return rst.resnet101_for_tiny(num_classes=num_classes)


def resnet152_for_tiny(num_classes, pretrained):
    return rst.resnet152_for_tiny(num_classes=num_classes)


def resnet18(num_classes, pretrained):
    net = models.resnet18(pretrained)
    net.fc = nn.Linear(512, num_classes)
    return net


def resnet34(num_classes, pretrained):
    net = models.resnet34(pretrained)
    net.fc = nn.Linear(512, num_classes)
    return net


def resnet50(num_classes, pretrained):
    net = models.resnet50(pretrained)
    net.fc = nn.Linear(512, num_classes)
    return net


def resnet101(num_classes, pretrained):
    net = models.resnet101(pretrained)
    net.fc = nn.Linear(512, num_classes)
    return net


def resnet152(num_classes, pretrained):
    net = models.resnet152(pretrained)
    net.fc = nn.Linear(512, num_classes)
    return net

