


def define_Model(opt):
    opt_net = opt['net']

    net_type = opt_net["net_type"]

    if net_type == 'nextvit':
        from models.network_nextvit import NextViT as net
        model = net(stem_chs=opt_net["stem_chs"], depths=opt_net["depths"], path_dropout=opt_net["path_dropout"], num_classes=opt_net["num_classes"])
    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(net_type))

    print('Training model [{:s}] is created.'.format(model.__class__.__name__))
    return model
