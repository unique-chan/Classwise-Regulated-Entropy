import torchvision.models as models
import torch.nn as nn


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

