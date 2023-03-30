
def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['type'].lower()

    if dataset_type in ['sport']:
        from data.sport_dataset import SportDataset as Dataset
    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = Dataset(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
