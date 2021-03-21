#!/usr/bin/env python3
import argparse
import sys
import os
__thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert (0, os.path.join (__thisdir))

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import torchvision
from torchvision import transforms
import torchvision.utils as vutils
import logging
import os
import time
import datetime
import random
import attack_model
import numpy as np
import torchvision.models as models
from target_models import *
from utils import *



if __name__ == '__main__':


    # if not os.path.exists('log'):
    #     os.mkdir('log')

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG)
        # level=logging.INFO,
        # filename='log/Fashion-MNIST_GUAP_'+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+'.log')
    logging.getLogger('matplotlib.font_manager').disabled = True


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Fashion-MNIST',
                        choices = ('Fashion-MNIST', 'CIFAR10', 'IMAGENET'))
    parser.add_argument('--lr', type=float, required=False, default=0.01, help='Learning rate')
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--l2reg', type=float, default=0.0001, help='weight factor for l2 regularization')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--tau',  type=float, default=0.1, help='max flow magnitude, default=0.1')
    parser.add_argument('--eps', type=float, default=0.1, help='allow for linf noise. default=0.1')
    parser.add_argument('--model', type=str, default='modelA',
                        choices = ('modelA', # Fashion-MNIST
                                   'VGG19', 'ResNet101', 'DenseNet121', # CIFAR10
                                   'VGG16', 'VGG19', 'ResNet152', 'GoogLeNet', # IMAGENET
                                   ),
                        help='modelA for Fashion-MNIST, VGG19/ResNet101/DenseNet121 for CIFAR10, VGG16/VGG19/ResNet152/GoogLeNet for IMAGENET')
    parser.add_argument('--manualSeed', type=int, default=5198, help='manual seed')
    parser.add_argument('--gpuid', type=str, default='0', help='multi gpuid')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--limited', action='store_true', help='used limited training data (1/10) for Universal Perturabtion Generation')
    parser.add_argument('--resume', action='store_true', help='load pretrained model')
    parser.add_argument('--outdir', type=str, default='GUAP_output', help='output dir')

    args = parser.parse_args()
    logger.info(args)
    tau = args.tau
    lr = args.lr
    dataSet = args.dataset
    batch_size = args.batch_size
    eps = args.eps
    model_name = args.model
    epochs = args.epochs
    gpuid = args.gpuid
    outdir = args.outdir

    model_file = lambda f: os.path.join ('saved_models', f)

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
        device_ids = [ i for i in range (torch.cuda.device_count())]
        print('number of gpu:', len(device_ids))
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(args.manualSeed)
        torch.cuda.manual_seed_all(args.manualSeed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device('cuda')
    else:
        device_ids = []
        device = torch.device('cpu')


    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)


    print('Generalizing Universal Adversarial Examples:')
    print('==> Preparing data..')

    if dataSet == 'Fashion-MNIST':
        dataset_mean = [0.5,]
        dataset_std = [0.5,]
        transform_data = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean,
                                 std=dataset_std)
        ])

    elif dataSet == 'CIFAR10':
        dataset_mean = [0.4914, 0.4822, 0.4465]
        dataset_std = [0.2023, 0.1994, 0.2010]

        transform_data = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean,
                                 std=dataset_std)
        ])


    elif dataSet == 'IMAGENET':
        dataset_mean = [0.485, 0.456, 0.406]
        dataset_std = [0.229, 0.224, 0.225]
        transform_data = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean,
                                 std=dataset_std)
        ])

    else:
        raise NotImplementedError

    if dataSet == 'Fashion-MNIST':
        trainset = torchvision.datasets.FashionMNIST(root='fashion-data', train=True, download=True, transform=transform_data)
        testset = torchvision.datasets.FashionMNIST(root='fashion-data', train=False, download=True, transform=transform_data)
        if model_name == 'modelA':
            model = modelA()
            model.load_state_dict(torch.load(model_file ('fashion_mnist_modela.pth'),map_location=device))
        else:
            raise NotImplementedError
    elif dataSet == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='cifar10-data', train=True, download=True, transform=transform_data)
        testset = torchvision.datasets.CIFAR10(root='cifar10-data', train=False, download=True, transform=transform_data)
        if model_name == 'VGG19':
            model = VGG('VGG19')
            model.load_state_dict(torch.load(model_file ('cifar10_vgg19.pth'),map_location=device)['net'])
        elif model_name == 'ResNet101':
            model = ResNet101()
            model.load_state_dict(torch.load(model_file ('cifar10_resnet101.pth'),map_location=device)['net'])
        elif model_name == 'DenseNet121':
            model = DenseNet121()
            model.load_state_dict(torch.load(model_file ('cifar10_dense121.pth'),map_location=device)['net'])
        else:
            raise NotImplementedError

    elif dataSet == 'IMAGENET':
        imagenet_dir = '/mnt/storage0_8/torch_datasets/ILSVRC'
        if not os.path.isdir (imagenet_dir):
          raise NotADirectoryError (imagenet_dir)
        trainset = torchvision.datasets.ImageFolder(os.path.join (imagenet_dir, 'train'),
                                                    transform = transform_data)
        testset = torchvision.datasets.ImageFolder(os.path.join (imagenet_dir, 'val'),
                                                   transform=transform_data)
        if model_name == 'VGG19':
            model = models.vgg19(pretrained=True)
        elif model_name == 'ResNet152':
            model = models.resnet152(pretrained=True)
        elif model_name == 'VGG16':
            model = models.vgg16(pretrained=True)
        elif model_name == 'GoogLeNet':
            model = models.googlenet(pretrained=True)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    nc,H,W = trainset.__getitem__(0)[0].shape

    mu = torch.Tensor((dataset_mean)).unsqueeze(-1).unsqueeze(-1).to(device)
    std = torch.Tensor((dataset_std)).unsqueeze(-1).unsqueeze(-1).to(device)
    unnormalize = lambda x: x*std + mu
    normalize = lambda x: (x-mu)/std


    for params in model.parameters():
        params.requires_grad = False
    model.eval()


    netAttacker = attack_model.Generator(1,nc,H)
    netAttacker.apply(weights_init)


    if len(device_ids)>1:
        model= nn.DataParallel(model,device_ids=device_ids)
        netAttacker = nn.DataParallel(netAttacker,device_ids=device_ids)


    model = model.to(device)
    netAttacker = netAttacker.to(device)

    if args.resume:
        netAttacker.load_state_dict(torch.load(model_file ('Fashion-MNIST_GUAP_pretrained_model.pth'),map_location=device))






    noise = torch.FloatTensor(1, 1, H, W)
    noise = noise.to(device)
    noise = Variable(noise)
    torch.nn.init.normal_(noise, mean=0, std=1.)

    loss_flow = Loss_flow()

    optimizer = torch.optim.Adam(netAttacker.parameters(), lr=lr, betas=(args.beta1, 0.999), weight_decay=args.l2reg)

    bestatt = 0.
    bestloss = 10000


    logger.info('Epoch \t Time \t Tr_loss \t Tr_acc \t Tr_stASR \t Tr_noiseASR \t Tr_GUAP_ASR ')

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0
        train_n = 0
        train_attack_rate = 0
        train_st_rate = 0
        train_noise_rate = 0
        train_ori_acc = 0

        netAttacker.train()
        model.eval()

        for i, (X, y) in enumerate(train_loader):

            if args.limited and i == (len(train_loader)//10):
                break

            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            train_ori_logits = model(X)
            flow_field,perb_noise = netAttacker(noise)

            L_flow = loss_flow(flow_field)
            flow_field = flow_field *tau/L_flow
            perb_noise = perb_noise* eps

            X_st = flow_st(unnormalize(X),flow_field)
            X_noise = unnormalize(X)+ perb_noise
            X_noise = normalize(torch.clamp(X_noise, 0, 1))
            X_adv = X_st +perb_noise
            X_adv = normalize(torch.clamp(X_adv, 0, 1))

            logits_st = model(normalize(X_st))
            logits_noise = model(X_noise)
            logits_adv = model(X_adv)
            adv_lossall = F.cross_entropy(logits_adv, train_ori_logits.max(1)[1], reduction = 'none')+1
            adv_loss = -torch.mean(torch.log(adv_lossall))
            adv_loss.backward()
            optimizer.step()

            train_ori_acc += (train_ori_logits.max(1)[1] == y).sum().item()
            train_loss += adv_loss.item() * y.size(0)
            train_attack_rate += ((logits_adv.max(1)[1] != train_ori_logits.max(1)[1])).sum().item()
            train_st_rate += ((logits_st.max(1)[1] != train_ori_logits.max(1)[1])).sum().item()
            train_noise_rate += ((logits_noise.max(1)[1] != train_ori_logits.max(1)[1])).sum().item()
            train_n += y.size(0)

        train_time = time.time()
        logger.info('%d \t %.2f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
                    epoch, train_time - start_time, train_loss / train_n, train_ori_acc/train_n, train_st_rate/train_n,train_noise_rate/train_n, train_attack_rate/train_n)

        if bestatt<train_attack_rate/train_n and bestloss>train_loss / train_n:
            bestloss = train_loss / train_n
            bestatt = train_attack_rate/train_n
            bestflow = flow_field
            bestnoise = perb_noise

    print('Best train ASR:',end = '\t')
    print(bestatt)
    flow_field = bestflow
    perb_noise = bestnoise

    if dataSet == 'IMAGENET':
        num_showfig =5
    else:
        num_showfig =10

    print('==> start testing ..')
    test_ori_acc = 0
    test_n = 0
    test_adv_loss = 0
    test_adv_acc = 0
    test_attack_rate = 0
    test_st_rate = 0
    test_noise_rate = 0

    start_time = time.time()

    clean_np = np.empty((0,nc, H, W))
    st_np = np.empty((0,nc, H, W))
    perb_np = np.empty((0, nc, H, W))
    clean_preds_np = np.empty(0)
    perb_preds_np = np.empty(0)
    model.eval()
    netAttacker.eval()

    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)

            test_ori_logits = model(X)
            X_st = flow_st(unnormalize(X),flow_field)
            X_noise = unnormalize(X)+ perb_noise
            X_noise = normalize(torch.clamp(X_noise, 0, 1))
            X_perb = X_st+ perb_noise
            X_perb = normalize(torch.clamp(X_perb, 0, 1))

            X_st = normalize(X_st)

            test_logits_st = model(X_st)
            test_logits_noise = model(X_noise)
            test_logits_adv = model(X_perb)
            test_ori_acc += (test_ori_logits.max(1)[1] == y).sum().item()
            test_adv_acc += (test_logits_adv.max(1)[1] == y).sum().item()
            adv_lossall = F.cross_entropy(test_logits_adv, test_ori_logits.max(1)[1], reduction = 'none')+1
            adv_loss = -torch.mean(torch.log(adv_lossall))
            test_adv_loss += adv_loss.item() * y.size(0)
            success_bool = (test_logits_adv.max(1)[1] != test_ori_logits.max(1)[1])
            test_attack_rate += success_bool.sum().item()
            test_st_rate += ((test_logits_st.max(1)[1] != test_ori_logits.max(1)[1])).sum().item()
            test_noise_rate += ((test_logits_noise.max(1)[1] != test_ori_logits.max(1)[1])).sum().item()

            if len(clean_preds_np)<num_showfig:
                clean_np = np.append(clean_np, X[success_bool].data.cpu(),axis=0)
                st_np = np.append(st_np, X_st[success_bool].data.cpu(),axis=0)
                perb_np = np.append(perb_np, X_perb[success_bool].data.cpu(),axis=0)
                clean_preds_np = np.append(clean_preds_np, test_ori_logits.max(1)[1][success_bool].data.cpu())
                perb_preds_np = np.append(perb_preds_np,test_logits_adv.max(1)[1][success_bool].data.cpu())

            test_n += y.size(0)

    test_time = time.time()

    test_asr = np.round(test_attack_rate/test_n,4)

    logger.info('Time \t  Test_loss \t Ori_Acc \t Perb_Acc \t Te_stASR \t Te_noiseASR\t Te_GUAP_ASR ')
    logger.info('%.2f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f', test_time - start_time, test_adv_loss/test_n, test_ori_acc/test_n, test_adv_acc/test_n, test_st_rate/test_n, test_noise_rate/test_n, test_attack_rate/test_n)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    file_id = f'{dataSet}_{model_name}_tau{str(tau)}_eps{str(eps)}_ASR{str(test_asr)}'
    flow_file = os.path.join (outdir, f'flow_{file_id}.npy')
    noise_file = os.path.join (outdir, f'noise_{file_id}.npy')
    GUAP_model_file = os.path.join (outdir, f'GUAP_model_{file_id}.pth')

    np.save(flow_file, bestflow.data.cpu().numpy().astype(np.float32))
    np.save(noise_file, bestnoise.data.cpu().numpy().astype(np.float32))
    torch.save(netAttacker.state_dict(), GUAP_model_file)

    clean = unnormalize(torch.from_numpy(clean_np[:num_showfig]).to(device)).cpu().clamp(0,1)
    st = unnormalize(torch.from_numpy(st_np[:num_showfig]).to(device)).cpu().clamp(0,1)
    adv = unnormalize(torch.from_numpy(perb_np[:num_showfig]).to(device)).cpu().clamp(0,1)

    middlenoise1 = st - clean
    middlenoise2 = adv - st
    for i in range(num_showfig):
      middlenoise1[i] = norm_ip(middlenoise1[i])
      middlenoise2[i] = norm_ip(perb_noise.detach().unsqueeze(0).cpu())

    fig = plt.figure(figsize=(10, 5))
    grid = vutils.make_grid(torch.cat((clean,middlenoise1,st,middlenoise2,adv)).float(),nrow=num_showfig)

    savefig = os.path.join (outdir, 'savefig')
    if not os.path.exists (savefig):
        os.mkdir (savefig)

    imgrid_file = os.path.join (savefig, f'{file_id}.png')
    plt.imsave(imgrid_file, grid.numpy().transpose((1, 2, 0)))
