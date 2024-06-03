import torch
import dataloaders.UAV as UAV


def get_dataset_name(mode):
    if mode == "ade20k":
        return "Ade20kDataset"
    if mode == "cityscapes":
        return "CityscapesDataset"
    else:
        ValueError("There is no such dataset regime as %s" % mode)


def get_dataloaders(opt):
    dataset_name, competition, mode = opt.dataset_mode.split('_')

    if mode == 'train':
        dataset_train = UAV.UAVDataset(opt, competition, 'train')
        dataset_val = UAV.UAVDataset(opt, competition, 'valid')
    elif mode == 'test':
        dataset_train = UAV.UAVDataset(opt, competition, 'test')
        dataset_val = UAV.UAVDataset(opt, competition, 'test')
    elif mode == 'full':
        dataset_train = UAV.UAVDataset(opt, competition, 'full-train')
        dataset_val = UAV.UAVDataset(opt, competition, 'full-valid')
    elif mode in ['public', 'private']:
        dataset_train = UAV.UAVDataset(opt, competition, mode)
        dataset_val = UAV.UAVDataset(opt, competition, mode)
    print("Created %s, size train: %d, size val: %d" % (dataset_name, len(dataset_train), len(dataset_val)))

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = opt.batch_size, shuffle = True, drop_last=True, num_workers=20, pin_memory=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size = opt.batch_size, shuffle = False, drop_last=False, num_workers=20, pin_memory=True)

    return dataloader_train, dataloader_val
