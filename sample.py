import os
from os.path import basename
from argparse import ArgumentParser
from collections import OrderedDict
import itertools

import numpy as np
import random

import PIL
from PIL import Image
from pdb import set_trace as st

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.cuda

import util.other as other
import util.folder as folder
from util.visualizer import Visualizer
from util.pool import ImagePool
import util.html as html


class DivCycleGAN():

    def __init__(self, args):
        self.savepath = args.savepath
        self.loadpath = args.loadpath
        self.initiallr = args.lr
        self.lr = args.lr
        self.epoch = args.epoch
        self.decayepoch = args.decayepoch
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.batchsize = args.batchsize
        self.numberclasses = args.numberclasses
        self.cropsize = args.cropsize

        self.inputA = torch.cuda.FloatTensor(self.batchsize, 3, self.cropsize, self.cropsize)
        self.inputB = torch.cuda.FloatTensor(self.batchsize, 3, self.cropsize, self.cropsize)
        self.mask = torch.cuda.LongTensor(self.batchsize, 1, self.cropsize, self.cropsize)
        self.maskEach = torch.cuda.ByteTensor(self.batchsize, self.numberclasses, self.cropsize, self.cropsize)
        self.random1 = torch.zeros(self.batchsize, 1, self.cropsize, self.cropsize).cuda()
        self.random2 = torch.zeros(self.batchsize, 1, self.cropsize, self.cropsize).cuda()

        self.zerocomparer = Variable(torch.zeros(self.batchsize, 3, self.cropsize, self.cropsize).cuda())

        self.construct_generator(args)
        self.construct_discriminator(args)

        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()), lr=args.lr, betas=(0.5, 0.999))
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()), lr=args.lr, betas=(0.5, 0.999))

        self.lossGAN = GANLoss()
        self.lossL1 = torch.nn.L1Loss()
        self.lossDiv = self.diversity_loss


        self.imagePool = ImagePool()
        self.imagePool2 = ImagePool()
        self.imagePool3 = ImagePool()

    def initialize(self):
        self.D_A.apply(self.weight_initialization)
        self.G_A.apply(self.weight_initialization)
        self.D_B.apply(self.weight_initialization)
        self.G_B.apply(self.weight_initialization)

    def load(self):
        pathG_A = os.path.join(self.savepath, "G_A_latest.pth")
        pathG_B = os.path.join(self.savepath, "G_B_latest.pth")
        pathD_A = os.path.join(self.savepath, "D_A_latest.pth")
        pathD_B = os.path.join(self.savepath, "D_B_latest.pth")
        self.G_A.load_state_dict(torch.load(pathG_A))
        self.G_B.load_state_dict(torch.load(pathG_B))
        self.D_A.load_state_dict(torch.load(pathD_A))
        self.D_B.load_state_dict(torch.load(pathD_B))

    def weight_initialization(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def construct_discriminator(self, args):
        self.D_A = MultilayerCNN(args.channel, layers=args.layerD, norm=args.normlayer)
        self.D_B = MultilayerCNN(args.channel, layers=args.layerD, norm=args.normlayer)

        self.D_A.cuda()
        self.D_B.cuda()

    def construct_generator(self, args):
        if args.generator == "unet256":
            self.G_A = UNet(8, offset=0, nc=args.channel, norm=args.normlayer)
            self.G_B = UNet(8, offset=1, nc=args.channel, norm=args.normlayer)
        elif args.generator == 'resnet':
            self.G_A = Resnet(3, 3, args.channel, norm_layer=args.normlayer, use_dropout=True, n_blocks=9)
            self.G_B = Resnet(4, 3, args.channel, norm_layer=args.normlayer, use_dropout=True, n_blocks=9)
        else:
            raise Exception("[Error]: No this type of generator.")

        self.G_A.cuda()
        self.G_B.cuda()

    def forward(self, inputA, inputB, mask, randomVector):
        self.randomVector = Variable(torch.from_numpy(np.abs(randomVector)).float().cuda())

        self.inputA.copy_(inputA)
        self.inputB.copy_(inputB)
        self.mask.copy_(mask)

        self.random2.zero_()
        for label in range(self.numberclasses):
            torch.eq(self.mask.float(), 1.0*label, out=self.maskEach[:, label, :, :])
            self.random2 += randomVector[label] * self.maskEach[:, label, :, :].float()

        self.realA = Variable(self.inputA.cuda())
        self.realB = Variable(self.inputB.cuda())
        self.label = Variable(self.mask.cuda())
        self.noise1 = Variable(self.random1.cuda())
        self.noise2 = Variable(self.random2.cuda())

        self.fakeA = self.G_A.forward(self.realB)
        self.recB = self.G_B.forward(torch.cat([self.fakeA, self.noise1], 1))

        self.fakeB = self.G_B.forward(torch.cat([self.realA, self.noise1], 1))
        self.recA = self.G_A.forward(self.fakeB)

        self.fakeB2 = self.G_B.forward(torch.cat([self.realA, self.noise2], 1))


    def backprop_D(self):
        self.optimizerD.zero_grad()

        fakeB_pool = self.imagePool.query(self.fakeB)
        PredictionFake1 = self.D_B.forward(fakeB_pool.detach())
        self.lossD_B_Fake1 = self.lossGAN(PredictionFake1, False)

        fakeB2_pool = self.imagePool2.query(self.fakeB2)
        PredictionFake2 = self.D_B.forward(fakeB2_pool.detach())
        self.lossD_B_Fake2 = self.lossGAN(PredictionFake2, False)

        self.lossD_BFake = (self.lossD_B_Fake1 + self.lossD_B_Fake2) * 0.5

        predictionRealB = self.D_B.forward(self.realB)
        self.lossD_BReal = self.lossGAN(predictionRealB, True)

        self.lossD_B = (self.lossD_BFake + self.lossD_BReal) * 0.5

        fakeA_pool = self.imagePool3.query(self.fakeA)
        PredictionFakeA = self.D_A.forward(fakeA_pool.detach())
        self.lossD_AFake = self.lossGAN(PredictionFakeA, False)

        predictionRealA = self.D_A.forward(self.realA)
        self.lossD_AReal = self.lossGAN(predictionRealA, True)

        self.lossD_A = (self.lossD_AFake + self.lossD_AReal) * 0.5

        self.lossD = self.lossD_A + self.lossD_B

        self.lossD.backward()
        self.optimizerD.step()

    def backprop_G(self):
        self.optimizerG.zero_grad()

        prediction = self.D_B.forward(self.fakeB)
        prediction2 = self.D_B.forward(self.fakeB2)
        self.lossG_BGAN1 = self.lossGAN(prediction, True)
        self.lossG_BGAN2 = self.lossGAN(prediction2, True)
        self.lossG_BGAN = (self.lossG_BGAN1 + self.lossG_BGAN2) * 0.5

        predictionA = self.D_A.forward(self.realA)
        self.lossG_AGAN = self.lossGAN(predictionA, True)

        self.lossGGAN = self.lossG_AGAN + self.lossG_BGAN


        self.lossGCycleA = self.lossL1(self.recA, self.realA)
        self.lossGCycleB = self.lossL1(self.recB, self.realB)

        self.lossGCycle = self.lossGCycleA + self.lossGCycleB

        # loss of diversity
        self.lossGDiv = 0
        for label in range(self.numberclasses):
            labelMask = self.maskEach[:, label, :, :]
            if self.current_epoch > 100:
                self.lossGDiv += torch.abs(self.randomVector[label]) * self.lossDiv(self.fakeB, self.fakeB2, labelMask)
            else:
                self.lossGDiv += self.current_epoch / 100.0 * torch.abs(self.randomVector[label]) *  self.lossDiv(self.fakeB, self.fakeB2, labelMask)

        self.lossG = 10.0 * self.lossGGAN + self.alpha * self.lossGCycle + self.gamma * self.lossGDiv

        self.lossG.backward()
        self.optimizerG.step()

    def set_epoch(self, i):
        self.current_epoch = i

    def decay(self):
        factor = self.initiallr / (self.epoch - self.decayepoch)
        lr = self.lr - factor
        for paramGroup in self.optimizerD.param_groups:
            paramGroup['lr'] = lr
        for paramGroup in self.optimizerG.param_groups:
            paramGroup['lr'] = lr
        print('update learning rate: %f -> %f' % (self.lr, lr))
        self.lr = lr

    def save(self, epoch=0):
        pathG_A = os.path.join(self.savepath, "G_A_epoch%d.pth" % epoch)
        latestG_A = os.path.join(self.savepath, "G_A_latest.pth")
        pathG_B = os.path.join(self.savepath, "G_B_epoch%d.pth" % epoch)
        latestG_B = os.path.join(self.savepath, "G_B_latest.pth")
        torch.save(self.G_A.cpu().state_dict(), pathG_A)
        torch.save(self.G_A.cpu().state_dict(), latestG_A)
        torch.save(self.G_B.cpu().state_dict(), pathG_B)
        torch.save(self.G_B.cpu().state_dict(), latestG_B)
        self.G_A.cuda()
        self.G_B.cuda()

        pathD_A = os.path.join(self.savepath, "D_A_epoch%d.pth" % epoch)
        latestD_A = os.path.join(self.savepath, "D_A_latest.pth")
        pathD_B = os.path.join(self.savepath, "D_B_epoch%d.pth" % epoch)
        latestD_B = os.path.join(self.savepath, "D_B_latest.pth")
        torch.save(self.D_A.cpu().state_dict(), pathD_A)
        torch.save(self.D_A.cpu().state_dict(), latestD_A)
        torch.save(self.D_B.cpu().state_dict(), pathD_B)
        torch.save(self.D_B.cpu().state_dict(), latestD_B)
        self.D_A.cuda()
        self.D_B.cuda()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.lossGGAN.data[0]),
                            ('G_Cycle', self.lossGCycle.data[0]),
                            ('D_A', self.lossD_A.data[0]),
                            ('D_B', self.lossD_B.data[0]),
                            ('G_Div', self.lossGDiv.data[0])])

    def get_current_visuals(self):
        realA = other.tensor2im(self.realA.data)
        fakeB = other.tensor2im(self.fakeB.data)
        fakeB2 = other.tensor2im(self.fakeB2.data)
        realB = other.tensor2im(self.realB.data)
        return OrderedDict([('real_A', realA), ('fake_B', fakeB), ('fake_B2', fakeB2), ('real_B', realB)])

    def cross_entropy2d(self, input, target, weight=None, sizeAverage=True):
        n, c, h, w = input.size()
        logP = F.log_softmax(input)
        logP = logP.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        logP = logP[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        logP = logP.view(-1, c)

        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(logP, target, weight=weight, size_average=False)
        if sizeAverage:
            loss /= mask.data.sum()
        return loss

    def diversity_loss(self, inputA, inputB, mask):
        maskpixel = torch.sum(mask)

        if maskpixel > 0:
            totalpixel = torch.numel(mask)
            mask3 = Variable(torch.cat([mask, mask, mask], 1).float())
            regularized = torch.mul(inputA - inputB, mask3)
            return F.relu(0.3 - self.lossL1(regularized, self.zerocomparer) * totalpixel / maskpixel)
            #return -self.lossL1(regularized, self.zerocomparer) * totalpixel / maskpixel
        else:
            return 0


class GANLoss(nn.Module):
    def __init__(self, targetRealLabel=1.0, targetFakeLabel=0.0):
        super(GANLoss, self).__init__()
        self.realLabel = targetRealLabel
        self.fakeLabel = targetFakeLabel
        self.realLabelVar = None
        self.fakeLabelVar = None
        self.Tensor = torch.FloatTensor
        self.loss = nn.BCELoss()

    def get_target_tensor(self, input, targetIsReal):
        targetTensor = None
        if targetIsReal:
            createLabel = ((self.realLabelVar is None) or (self.realLabelVar.numel() != input.numel()))
            if createLabel:
                realTensor = self.Tensor(input.size()).fill_(self.realLabel)
                self.realLabelVar = Variable(realTensor, requires_grad=False)
            targetTensor = self.realLabelVar
        else:
            createLabel = ((self.fakeLabelVar is None) or (self.fakeLabelVar.numel() != input.numel()))
            if createLabel:
                fakeTensor = self.Tensor(input.size()).fill_(self.fakeLabel)
                self.fakeLabelVar = Variable(fakeTensor, requires_grad=False)
            targetTensor = self.fakeLabelVar
        return targetTensor

    def __call__(self, input, targetIsReal):
        targetTensor = self.get_target_tensor(input, targetIsReal)
        return self.loss(input, targetTensor.cuda())


class Resnet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Resnet, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class UNet(nn.Module):
    def __init__(self, numDownsample, offset=0, nc=64, norm=nn.BatchNorm2d):
        super(UNet, self).__init__()

        unet = UNetBlock(nc*8, nc*8, norm=norm, innermost=True)
        for i in range(numDownsample - 5):
            unet = UNetBlock(nc*8, nc*8, submodule=unet, norm=norm, dropout=True)
        unet = UNetBlock(nc*4, nc*8, submodule=unet, norm=norm)
        unet = UNetBlock(nc*2, nc*4, submodule=unet, norm=norm)
        unet = UNetBlock(nc, nc*2, submodule=unet, norm=norm)
        unet = UNetBlock(3, nc, offset, submodule=unet, norm=norm, outermost=True)

        self.network = unet

    def forward(self, input):
        return self.network(input)


class UNetBlock(nn.Module):
    def __init__(self, outChannel, inChannel, offset=0, submodule=None, norm=nn.BatchNorm2d, outermost=False, innermost=False, dropout=False):
        super(UNetBlock, self).__init__()
        self.outermost = outermost

        downconv = nn.Conv2d(outChannel, inChannel, kernel_size=4, stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2)
        downnorm = norm(inChannel)
        uprelu = nn.ReLU()
        upnorm = norm(outChannel)

        if outermost:
            upconv = nn.ConvTranspose2d(inChannel * 2, outChannel, kernel_size=4, stride=2, padding=1)
            down = [nn.Conv2d(outChannel+offset, inChannel, kernel_size=4, stride=2, padding=1)]
            up = [uprelu, upconv, nn.Tanh()]
            network = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inChannel, outChannel, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            network = down + up
        else:
            upconv = nn.ConvTranspose2d(inChannel * 2, outChannel, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if dropout:
                network = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                network = down + [submodule] + up

        self.network = nn.Sequential(*network)

    def forward(self, x):
        if self.outermost:
            return self.network(x)
        else:
            return torch.cat([self.network(x), x], 1)


class MultilayerCNN(nn.Module):
    def __init__(self, nc=64, layers=3, norm=nn.BatchNorm2d):
        super(MultilayerCNN, self).__init__()

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(3, nc, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2)
        ]

        nfMult = 1
        nfMultPrev = 1
        for n in range(1, layers):
            nfMultPrev = nfMult
            nfMult = min(2**n, 8)
            sequence += [
                nn.Conv2d(nc * nfMultPrev, nc * nfMult, kernel_size=kw, stride=2, padding=padw),
                norm(nc * nfMult),
                nn.LeakyReLU(0.2)
            ]

        nfMultPrev = nfMult
        nfMult = min(2**layers, 8)
        sequence += [
            nn.Conv2d(nc * nfMultPrev, nc * nfMult, kernel_size=kw, stride=1, padding=padw),
            norm(nc * nfMult),
            nn.LeakyReLU(0.2)
        ]

        sequence += [nn.Conv2d(nc * nfMult, 1, kernel_size=kw, stride=1, padding=padw)]

        sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class ABDataset(data.Dataset):

    def __init__(self, args):
        super(ABDataset, self).__init__()
        self.directory = args.data
        self.base_dir = os.path.join(args.data, args.phase)
        self.dir_A = self.base_dir + 'A'
        self.dir_B = self.base_dir + 'B'
        self.dir_C = self.base_dir + 'C'

        self.paths_A = sorted(folder.make_dataset(self.dir_A))
        self.paths_B = sorted(folder.make_dataset(self.dir_B))
        self.paths_C = sorted(folder.make_dataset(self.dir_C))
        self.size_A = len(self.paths_A)
        self.size_B = len(self.paths_B)

        transformList = []
        transformList.append(transforms.ToTensor())
        transformList.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transformList)
        self.transformMask = transforms.Compose([transforms.ToTensor()])

        self.rescalesize = args.rescalesize
        self.cropsize = args.cropsize


    def __getitem__(self, index):
        # load A
        path_A = self.paths_A[index]
        bname = os.path.basename(path_A).split('.')[0]
        A = Image.open(path_A).convert('RGB')
        A = A.resize((self.rescalesize, self.rescalesize), Image.BICUBIC)
        A = self.transform(A)

        # load B
        index_B = random.randint(0, self.size_B - 1)
        path_B = self.paths_B[index_B]
        B = Image.open(path_B).convert('RGB')
        B = B.resize((self.rescalesize, self.rescalesize), Image.BICUBIC)
        B = self.transform(B)

        # load C
        path_C = self.paths_C[index_B]
        C = Image.open(path_C)
        C = C.resize((self.rescalesize, self.rescalesize), Image.BICUBIC)
        C = self.transformMask(C) * 255

        # random crop
        w = A.size(2)
        h = A.size(1)
        wOffset = random.randint(0, max(0, w - self.cropsize - 1))
        hOffset = random.randint(0, max(0, h - self.cropsize - 1))
        wOffset2 = random.randint(0, max(0, w - self.cropsize - 1))
        hOffset2 = random.randint(0, max(0, h - self.cropsize - 1))

        A = A[:, hOffset:(hOffset+self.cropsize), wOffset:(wOffset+self.cropsize)]
        B = B[:, hOffset2:(hOffset2+self.cropsize), wOffset2:(wOffset2+self.cropsize)]
        C = C[:, hOffset2:(hOffset2+self.cropsize), wOffset2:(wOffset2+self.cropsize)]


        return {'A': B, 'B': A, 'C': C, 'Name': bname}


    def __len__(self):
        return self.size_A


class Preprocesser():

    def __init__(self):
        parser = ArgumentParser()

        # for general parameters
        parser.add_argument("--data", type=str, default="./datasets/facades")
        parser.add_argument("--taskname", type=str, default="facades_unet")
        parser.add_argument("--phase", type=str, default="test")

        # for network parameters
        parser.add_argument("--generator", type=str, default="unet256")
        parser.add_argument("--channel", type=int, default=64)
        parser.add_argument("--norm", type=str, default="batch")
        parser.add_argument("--layerD", type=int, default=3)
        parser.add_argument("--numberclasses", type=int, default=13)
        parser.add_argument("--alpha", type=float, default=100.0)
        parser.add_argument("--gamma", type=float, default=10.0)

        # for optimizer parameters
        parser.add_argument("--lr", type=float, default=0.0002)
        parser.add_argument("--epoch", type=int, default=200)
        parser.add_argument("--decayepoch", type=int, default=100)
        parser.add_argument("--batchsize", type=int, default=1)

        # for training time data argumentation
        parser.add_argument("--rescalesize", type=int, default=256)
        parser.add_argument("--cropsize", type=int, default=256)

        # for display
        parser.add_argument("--displayfrequency", type=int, default=97)
        parser.add_argument("--printfrequency", type=int, default=97)
        parser.add_argument("--savefrequency", type=int, default=10000)

        # for resume
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--currentepoch", type=int, default=1)

        # for GPU selections, only one
        parser.add_argument("--gpu", type=str, default="0")

        self.args = parser.parse_args()

        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu

        self.args.savepath = os.path.join("./result", self.args.taskname)
        self.args.loadpath = os.path.join("./result", self.args.taskname)
        if not os.path.exists(self.args.savepath):
            os.mkdir(self.args.savepath)

        if self.args.norm == 'batch':
            self.args.normlayer = functools.partial(nn.BatchNorm2d, affine=True)
        elif self.args.norm == 'instance':
            self.args.normlayer = functools.partial(nn.InstanceNorm2d, affine=False)
        else:
            raise Exception("[Error]: No this type of normalization.")

    def get_arguments(self):
        return self.args


def random_generator(numberclasses):
    randomVector = np.random.uniform(-1.0, 1.0, numberclasses)

    # make sure it is soft one-hot vector
    idx = np.argmax(randomVector)
    randomVector[idx] *= 2
    randomVector /= np.sum(np.abs(randomVector))

    return randomVector


if __name__ == '__main__':

    preprocesser = Preprocesser()
    args = preprocesser.get_arguments()

    dataset = ABDataset(args)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    model = DivCycleGAN(args)
    model.load()

    visualizer = Visualizer(args)

    webDirectory = os.path.join("./result", args.taskname, "sample")
    webpage = html.HTML(webDirectory, "Experiment = %s, Phase = test" % args.taskname)

    for i, data in enumerate(loader):
        multiDict = OrderedDict()
        for time in range(30):
            randomVector = random_generator(args.numberclasses)
            model.forward(data["A"], data["B"], data["C"], randomVector)
            visuals = model.get_current_visuals()
            multiDict["real_A"] = visuals["real_A"]
            multiDict["fake_B_%d" % time] = visuals["fake_B2"]
        
        visualizer.save_images(webpage, multiDict, data["Name"][0])

    webpage.save()

