import torchvision.models as models


def model(network_name='resnet18', pretrained=False):
    try:
        return eval(network_name + '({})'.format(pretrained))
        # return locals()[network_name](pretrained)
    except Exception as e:
        print('[Error]', e)
        exit(1)


# Insert your model function here!
# https://pytorch.org/docs/stable/torchvision/models.html


def resnet18(pretrained):
    return models.resnet18(pretrained)


def resnet34(pretrained):
    return models.resnet34(pretrained)


def resnet50(pretrained):
    return models.resnet50(pretrained)


def resnet101(pretrained):
    return models.resnet101(pretrained)


def resnet152(pretrained):
    return models.resnet152(pretrained)


