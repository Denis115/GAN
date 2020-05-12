import argparse
from common import *
from GAN import Generator, Discriminator
from images_to_array import get_dataloader

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

    # ЗЧИТАТИ ДАНІ
    dataloader = get_dataloader(DATA_PATH, batch_size, N_WORKERS)

    # ІНІЦІАЛІЗУВАТИ ГЕНЕРАТОР
    generator = Generator(gpu).to(device)
    if verbose == 1:
        print(summary(generator, (N_NOISE, 1, 1)))

    # ІНІЦІАЛІЗУВАТИ ДИСКРИМІНАТОР
    discriminator = Discriminator(gpu).to(device)
    discriminator.apply(init_weights)
    if verbose == 1:
            print(summary(discriminator, (3, 64,64)))

    # ІНІЦІАЛІЗУВАТИ ФУНКЦІЮ ВТРАТ
    criterion = nn.BCELoss()

    # СТАТИЧНИЙ ШУМ ДЛЯ КОНТРОЛЬНИХ ЗОБРАЖЕНЬ
    fixed_noise = torch.randn(25, N_NOISE, 1, 1, device=device)

    real_label = 1
    fake_label = 0

    # ІНІЦІАЛІЗУВАТИ ОПТИМІЗАТОРИ
    optimizerDis = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimizerGen = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

    # СЛУЖБОВІ СПИСКИ З ДОДАТКОВОЮ ІНФОРМАЦІЄЮ ЩОДО НАЧАННЯ
    images = []
    lossGenerator = []
    lossDiscriminator = []

    # ТРЕНУВАННЯ
    print("TRAINING")
    im_number = 0
    for epoch in range(epochs):

        if verbose == 1:
            print('Epoch:', epoch)

        for i, data in enumerate(dataloader, 0):    
            
            # ОНОВЛЕННЯ ДИСКРИМІНАТОРА

            # СПРАВЖНІ ЗОБРАЖЕННЯ
            discriminator.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            output = discriminator(real_cpu).view(-1)      
            errorDisc_real = criterion(output, label)
            errorDisc_real.backward()
            Accuracy = output.mean().item()

            # ЗГЕНЕРОВАНІ ЗОБРАЖЕННЯ
            noise = torch.randn(b_size, N_NOISE, 1, 1, device=device)
            fake = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake.detach()).view(-1)
            errorDisc_fake = criterion(output, label)
            errorDisc_fake.backward()
            AccuracyBefore = output.mean().item()
            errorDisc = errorDisc_real + errorDisc_fake

            optimizerDis.step()

            # ОНОВЛЕННЯ ГЕНЕРАТОРА
            generator.zero_grad()
            label.fill_(real_label)
            output = discriminator(fake).view(-1)
            errorGen = criterion(output, label)
            errorGen.backward()
            AccuracyAfter = output.mean().item()
            optimizerGen.step()

            # ВИВЕСТИ РЕЗУЛЬТАТ НА КОЖНІЙ 50-Й ІТЕРАЦІЇ
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, epochs, i, len(dataloader),
                        errorDisc.item(), errorGen.item(), Accuracy, AccuracyBefore, AccuracyAfter))

            lossGenerator.append(errorGen.item())
            lossDiscriminator.append(errorDisc.item())

        # ВИЙТИ У ВИПАДКУ ПЕРЕНАВЧАННЯ
        if errorDisc.item() < 0.001:
                break
        
        # ЗБЕРЕГТИ ПОТОЧНІ ЗГЕНЕРОВАНІ ЗОБРАЖЕННЯ
        with torch.no_grad():
            fake = generator(fixed_noise).detach().cpu()
            images.append(vutils.make_grid(fake, nrow=5, padding=2, normalize=True))

            plt.imsave('output/{}.jpg'.format(im_number), np.transpose(np.asarray(images[-1]),(1,2,0)))
            im_number += 1

        # ЗБЕРЕГТИ МОДЕЛІ
        torch.save(generator.state_dict(), 'generator.m')
        torch.save(discriminator.state_dict(), 'discriminator.m')
    
    plt.figure()
    plt.plot(lossGenerator, label='Generator')
    plt.plot(lossDiscriminator, label='Discriminator')
    plt.title('Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return images


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
    
    manualSeed = 42
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    images = train(args.epochs, args.batch_size, args.gpu_number, args.verbose)
