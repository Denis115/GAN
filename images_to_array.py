import torch

import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from common import *

def get_dataloader(path, batch_size, workers):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = dset.ImageFolder(root=path,
                           transform=transforms.Compose([
                               transforms.Resize(IMAGE_SIZE),
                               transforms.CenterCrop(IMAGE_SIZE),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
    