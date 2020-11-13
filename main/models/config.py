from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, resnext101_32x4d, resnext101_64x4d, wide_resnet50_2, wide_resnet101_2
from .densent import densenet121, densenet161, densenet169, densenet201
from .inception import inception_v3

models = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_32x4d', 'resnext101_64x4d', 'wide_resnet50_2', 'wide_resnet101_2', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'inception_v3']

def get_model(name,use_softpool, **kwargs):
    net = None
    if 'res' in name.lower():
        if '18' in name.lower():
            net = resnet18(use_softpool=use_softpool, **kwargs)
        elif '34' in name.lower():
            net = resnet34(use_softpool=use_softpool, **kwargs)
        elif '50' in name.lower():
            if 'xt' in name.lower():
                net = resnext50_32x4d(use_softpool=use_softpool, **kwargs)
            elif 'wide' in name.lower():
                net = wide_resnet50_2(use_softpool=use_softpool, **kwargs)
            else:
                net = resnet50(use_softpool=use_softpool, **kwargs)
        elif '101' in name.lower():
            if 'xt' in name.lower():
                if '32x4d' in name.lower():
                    net = resnext101_32x4d(use_softpool=use_softpool, **kwargs)
                elif '64x4d' in name.lower():
                    net = resnext101_64x4d(use_softpool=use_softpool, **kwargs)
                elif '32x8d' in name.lower():
                    net = resnext101_32x8d(use_softpool=use_softpool, **kwargs)
            elif 'wide' in name.lower():
                net = wide_resnet101_2(use_softpool=use_softpool, **kwargs)
            else:
                net = resnet101(use_softpool=use_softpool, **kwargs)
        elif '152' in name.lower():
            net = resnet152(use_softpool=use_softpool, **kwargs)

    elif 'densenet' in name.lower():
        if '121' in name.lower():
            net = densenet121(use_softpool=use_softpool, **kwargs)
        elif '161' in name.lower():
            net = densenet161(use_softpool=use_softpool, **kwargs)
        elif '169' in name.lower():
            net = densenet169(use_softpool=use_softpool, **kwargs)
        elif '201' in name.lower():
            net = densenet201(use_softpool=use_softpool, **kwargs)

    elif 'inception' in name.lower():
        net = inception_v3(use_softpool=use_softpool, **kwargs)

    if net is None:
        print('Selected architecture not implemented !')
        raise NotImplementedError

    return net
