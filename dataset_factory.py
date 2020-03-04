from __future__ import print_function
import torch
from torchvision import datasets, transforms


def get_dataset_by_id(args,kwargs):

    if args.dataset_name == 'MNIST':

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        # to know the dataset better
        dataiter = iter(train_loader)
        images, labels = dataiter.next()
        input_dim = images.shape[2]*images.shape[3]
        ouptut_dim = 10 # TODO: how do I extract this from the dataloader?


    return train_loader, test_loader, input_dim, ouptut_dim