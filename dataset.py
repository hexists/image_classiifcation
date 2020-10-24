#!/usr/bin/env python3


import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch

dataset = dset.ImageFolder(root="images/",
                           transform=transforms.Compose([
                               transforms.Resize(128),      # 한 축을 128로 조절하고
                               transforms.CenterCrop(128),  # square를 한 후,
                               transforms.ToTensor(),       # Tensor로 바꾸고 (0~1로 자동으로 normalize)
                               transforms.Normalize((0.5, 0.5, 0.5),  # -1 ~ 1 사이로 normalize
                                                    (0.5, 0.5, 0.5)), # (c - m)/s 니까...
                           ]))
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=2,
                                         shuffle=True,
                                         num_workers=8)
for i, data in enumerate(dataloader):
    print(data[0].size())  # input image
    print(data[1])         # class label
