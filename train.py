import argparse
from common import *
from GAN import Generator, Discriminator
from images_to_array import image_to_array

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

from torchsummary import summary

import torch
from torch import nn, optim
import torchvision.utils as vutils

import random


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train(epochs=5, batch_size=50, gpu=0, verbose = 1):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    dataloader = image_to_array(DATA_PATH, batch_size, N_WORKERS)

    generator = Generator(gpu).to(device)

    if verbose == 1:
        print(summary(generator, (N_NOISE, 1, 1)))



    discriminator = Discriminator(gpu).to(device)
    discriminator.apply(init_weights)
    if verbose == 1:
            print(summary(discriminator, (3, 64,64)))

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(25, N_NOISE, 1, 1, device=device)

    real_label = 1
    fake_label = 0

    optimizerDis = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimizerGen = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("TRAINING")
    im_number = 0
    for epoch in range(epochs):

        if verbose == 1:
            print('Epoch:', epoch)

        for i, data in enumerate(dataloader, 0):    

            discriminator.zero_grad()
           
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            
            output = discriminator(real_cpu).view(-1)
            
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()


            noise = torch.randn(b_size, N_NOISE, 1, 1, device=device)
            fake = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerDis.step()


            generator.zero_grad()
            label.fill_(real_label)
            output = discriminator(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerGen.step()

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            G_losses.append(errG.item())
            D_losses.append(errD.item())


            iters += 1


        if errD.item() < 0.001:
                break

        with torch.no_grad():
            fake = generator(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, nrow=5, padding=2, normalize=True))

            plt.imsave('output/{}.jpg'.format(im_number), np.transpose(np.asarray(img_list[-1]),(1,2,0)))
            im_number += 1

        torch.save(generator.state_dict(), 'generator.m')
        torch.save(discriminator.state_dict(), 'discriminator.m')
    
    plt.figure()
    plt.plot(G_losses, label='Generator')
    plt.plot(D_losses, label='Discriminator')
    plt.title('Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return img_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Insert training parameteres')
    parser.add_argument('epochs', metavar='epochs', type=int,
                        help='number of training epochs;')
    parser.add_argument('batch_size', metavar='batch_size', type=int,
                        help='number of samples in a single training batch;')
    parser.add_argument('gpu_number', metavar='gpu_number', type=int,
                        help='number of availible GPUs to perform training;')
    parser.add_argument('verbose', metavar='verbose', type=int,
                    help='verbose [0 / 1];')

    args = parser.parse_args()
    
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    
    img_list = train(args.epochs, args.batch_size, args.gpu_number, args.verbose)
    data = image_to_array(DATA_PATH)
