#!/usr/bin/env python
# coding: utf-8

# ref: https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html

import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import argparse
import os
import sys
import glob
import shutil
from collections import defaultdict


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.image_paths = glob.glob(path + '/*.jpg')
        self.transform = transform

    def __getitem__(self, index):
        x = Image.open(self.image_paths[index])
        if self.transform is not None:
            x = self.transform(x)

        return (self.image_paths[index], x)

    def __len__(self):
        return len(self.image_paths)

 
'''
1) 이미지 사이즈를 256 X 256로 변경
2) Tensor로 변환
3) [-1, 1] 범위로 정규화
'''

'''
transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
'''

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def preprocess_train_dataset(src_path, dst_path_prefix='./images'):

    file_list = [fname for fname in os.listdir(src_path) if fname.endswith('.jpg')]
    
    print("preprocess_train_dataset")
    print("file_list: {}".format(file_list[:10]))
    
    img_cls = defaultdict(list)
    for fname in file_list:
        cls = '_'.join(fname.split('_')[:2])
        img_cls[cls].append(fname)
    
    for cls, files in img_cls.items():
        dst_path = '{}/{}'.format(dst_path_prefix, cls)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
    
        for fname in files:
            src_file = '{}/{}'.format(src_path, fname)
            dst_file = '{}/{}'.format(dst_path, fname)
            shutil.copy(src_file, dst_file) 

    print("End preprocess('./images') ...")


def inference(model, classes, device, loader):
    print('inference')
    fp = open('result.tsv', 'w')
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            name, data = data
            data = data.to(device)
            outputs = model(data)
            pred = outputs.argmax(dim=1, keepdim=True)
            fname = name[0].split('/')[-1]
            pclass, dclass = classes[pred.item()].split('_')
            fp.writelines('{}\t{}\t{}\n'.format(fname, pclass, dclass))
            if i % 100 == 0:
                print('{}\t{}\t{}'.format(fname, pclass, dclass))
    fp.close()
    print('End inference: result.tsv')


def set_train_loader(path, batch_size):
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    
    data = datasets.ImageFolder(root=path, transform=transform)
    classes = data.classes
    
    valid_len = int(len(data) * 0.2)
    train_data, valid_data = data_utils.random_split(data, (len(data) - valid_len, valid_len))
    
    train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = data_utils.DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print('===== train_loader sample')
    for i, data in enumerate(train_loader):
        print('input image: {}'.format(data[0].size()))  # input image
        print('class label: {}'.format(data[1]))         # class label
        break
    
    print('===== valid_loader sample')
    for i, data in enumerate(valid_loader):
        print('input image: {}'.format(data[0].size()))  # input image
        print('class label: {}'.format(data[1]))         # class label
        break
        
    print('===== Class: {}'.format(classes))

    '''
    # 학습용 이미지를 무작위로 가져오기
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    
    # 이미지 보여주기
    imshow(torchvision.utils.make_grid(images))
    # 정답(label) 출력
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    '''
    return train_loader, valid_loader, classes


def set_test_loader(path):
    test_data = TestDataset(path, transform=transform)
    test_loader = data_utils.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2)
    
    print('===== test_loader sample')
    for i, data in enumerate(test_loader):
        names, data = data
        print(names, data.size())
        break

    print('Success ...')
    return test_loader


def print_progress(msg, progress):
    max_progress = int((progress*100)/2)
    remain=50-max_progress
    buff="{}\t[".format( msg )
    for i in range( max_progress ): buff+="⬛"
    buff+="⬜"*remain
    buff+="]:{:.2f}%\r".format( progress*100 )
    sys.stderr.write(buff)


def train(model, criterion, device, train_loader, optimizer):
    model.train()
    train_losses = []
    correct = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        progress = (i + 1) / float(len(train_loader))
        print_progress('train', progress)
    print(file=sys.stderr)

    return np.mean(train_losses), 100. * correct / len(train_loader.dataset)


def valid(model, criterion, device, valid_loader):
    model.eval()
    valid_losses = []
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(valid_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            valid_loss = criterion(outputs, target)
            valid_losses.append(valid_loss.item())
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            progress = (i + 1) / float(len(valid_loader))
            print_progress('valid', progress)
        print(file=sys.stderr)

    return np.mean(valid_losses), 100. * correct / len(valid_loader.dataset)


def set_model():
    # 미리 학습된 모델은 그대로 고정된 Feature로 두고, 맨 위에 Fully Connected Layer만 추가
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False
    
    # 새로 생성된 모듈의 매개변수는 기본값이 requires_grad=True 임
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 20)
    model_conv = model_conv.to(device)
    return model_conv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Image Classification')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-b', '--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('-n', '--num_iters', default=30, type=int, help='number of iterations')
    parser.add_argument('--mode', default='train', choices=['train', 'inference'], type=str, help='run mode')
    parser.add_argument('--model', default='./checkpoint/ckpt.pth', type=str, help='model path')
    parser.add_argument('--train_path', default='./nipa_dataset/train', type=str, help='train dataset path')
    parser.add_argument('--test_path', default='./nipa_dataset/test', type=str, help='test dataset path')
    args = parser.parse_args()

    no_cuda = True
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = set_model(device)

    if args.mode == 'inference':
        test_loader = set_test_loader(args.test_path)
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint['net'])
        classes = checkpoint['classes']
        inference(model, classes, device, test_loader)
    else:
        # preprocess_train_dataset(args.train_path, './images')
        train_loader, valid_loader, classes = set_train_loader('./images', args.batch_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

        date_time = datetime.now().strftime('%Y%m%d%H%M')
        writer = SummaryWriter('./runs/{}'.format(date_time))

        best_acc = 0
        for epoch in range(1, args.num_iters + 1):
            tr_loss, tr_acc = train(model, criterion, device, train_loader, optimizer)
            va_loss, va_acc = valid(model, criterion, device, valid_loader)
        
            if va_acc > best_acc:
                print('===== Saving Model, acc = {} => {}'.format(best_acc, va_acc)) 
                state = {
                    'net': model.state_dict(),
                    'acc': va_acc,
                    'classes': classes,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/ckpt.pth')
                best_acc = va_acc
        
            print("{} / {}\ttrain loss, acc : {:.4f}, {:.4f}, valid loss, acc: {:.4f}, {:.4f}\n".format(epoch, args.num_iters, tr_loss, tr_acc, va_loss, va_acc), file=sys.stderr)
        
            writer.add_scalar('{}/{}'.format('loss', 'train'), tr_loss, epoch)
            writer.add_scalar('{}/{}'.format('loss', 'valid'), va_loss, epoch)
            writer.add_scalar('{}/{}'.format('acc', 'train'), tr_acc, epoch)
            writer.add_scalar('{}/{}'.format('acc', 'valid'), va_acc, epoch)
        
        writer.close()
        print('Finished Training, Best acc: {}'.format(best_acc))

        test_loader = set_test_loader(args.test_path)
        inference(model, classes, device, test_loader)
