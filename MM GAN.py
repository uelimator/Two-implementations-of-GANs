from __future__ import absolute_import, division, print_function

import torch.nn.parallel
import torch.utils.data
# Python File der originalen Arbeit, Quelle: https://github.com/bioinf-jku/TTUR
import fid
import os
import numpy as np
import time
import tensorflow as tf
from scipy.stats import entropy
from torch.autograd import Variable
import torch.utils.data
from torchvision.models.inception import inception_v3
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.datasets as dset
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

#originaler Code des WGAN-GP, Quelle: https://github.com/igul222/improved_wgan_training

# Inspiration und Teile aus: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# Inspiration und Teile aus: https://github.com/igul222/improved_wgan_training
# Inspiration und teile aus: https://github.com/caogang/wgan-gp


# Die Implementierung ist mit Tensorflow 1.x geschrieben und ich habe tf 2.20 installiert
# Hier wird es kompatibel gemacht
#tf.compat.v1.disable_eager_execution()

# Speicherort des Datensatzes
dataroot = '~/Robin/dataset/celeba/29561_37705_bundle_archive/img_align_celeba'

############################

#      Hyperparameter      #

############################

# Anzahl an Arbeiter für das laden des Datensatzes
workers = 4

# Grösse eines Batches
batch_size = 128

#Bildgrösse
image_size = 64

# Anzahl an Kanälen (3 für Farbbilder)
channels = 3

# Grösse des latenten Vektors z
latent_dim = 128

# Grösse / Anzahl an Feature maps im Generator
ngf = 64

# Grösse / Anzahl an Feature maps im Diskriminator
ndf = 64

# Anzhal der Trainingsepochen
num_epochs = 150

# Lernrate für den Adam optimizer
lr = 0.0002

# Beta1 für den Adam optimizers
beta1 = 0.5

# Beta2 für den Adam optimizers
beta2 = 0.9

# Anzahl an Grafikkarten. 0 für CPU
ngpu = 1

# lambda isz ein Hyperparameter für die Gradientenstrafe
lambda_gp = 10

# Wie viel öfter der der Diskriminator (Kritiker) trainiert wird
n_critic = 5

# Intervall für das Speichern des Modelles, IS und FID
sample_interval = 1000

# variable für das zählen von Iterationen
iterator = torch.Tensor([0])

# Gesamtanzahl der Iterationen mit Batch_size = 64
iterations = 1583 * num_epochs

# initialisierung für den Gebrauch von Tensorboard
writer = SummaryWriter()
# Auswahl zwischen Cpu und GPU
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


os.makedirs("images4", exist_ok=True)


# initialisiert Gewichte, Quelle: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.preprocess = nn.Sequential(
            # Preprocessing
            nn.Linear(128, 8192),
            nn.BatchNorm1d(8192),
            # nn.ReLU(True),
        )
        self.main = nn.Sequential(
            # block1
            nn.ConvTranspose2d(8 * ngf, 4 * ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4 * ngf),
            nn.ReLU(True),

            # block1
            nn.ConvTranspose2d(4 * ngf, 2 * ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * ngf),
            nn.ReLU(True),

            # block1
            nn.ConvTranspose2d(2 * ngf, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # block1

            nn.ConvTranspose2d(ngf, channels, kernel_size=4, stride=2, padding=1),
            # activation
            nn.Tanh(),
        )

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 8 * ngf, 4, 4)
        output = self.main(output)
        output = output.view(-1, 3, 64, 64)

        return output


netG = Generator().to(device)
netG.apply(weights_init)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(channels, ndf, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(),

            nn.Conv2d(2 * ndf, 4 * ndf, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(),

            nn.Conv2d(4 * ndf, 8 * ndf, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(),
        )
        self.linear = nn.Linear(4 * 4 * 8 * ndf, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4 * 4 * 8 * ndf)
        output = self.linear(output)
        output = torch.sigmoid(output)
        return output



netD = Discriminator().to(device)
netD.apply(weights_init)

# gleiche Werte, wie originale Arbeit

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))


def generate_images(netG, iterator):
    with torch.no_grad():
        noise_s = torch.randn(batch_size, latent_dim)
        noise_s = noise_s.to(device)

        samples = netG(noise_s)

    save_image(samples.data[:25], "images4/%d.png" % iterator, nrow=5, normalize=True)



# Formatierung für den Datensatz
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Erstellen des dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# berechnet den IS, Quelle: https://github.com/sbarratt/inception-score-pytorch
def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True).type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        # print(x.size())
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)


criterion = nn.BCELoss()

# Labels
one = torch.tensor(1, dtype=torch.float)
mone = one * -1


one = one.to(device)
mone = mone.to(device)

iterator = iterator.to(device)

real_label = 1
fake_label = 0



##################################
#            Training            #
##################################

#Der Trainingsalgorithmus ist eine leicht angepasste Variante des Tutorials von
#Quelle: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        iterator += 1
        start_time=time.time()
        # Diskriminatortraining
        netD.zero_grad()
        #Training mit echten Bildern

        # Formatierung es Batches
        real_imgs = data[0].to(device)
        b_size = real_imgs.size(0)
        label = torch.full((b_size,), real_label, device=device)
        #Einstufung der Bilder
        output = netD(real_imgs).view(-1)
        # Berechnung des Verlusts
        errD_real = criterion(output, label)
        # Berechnung des Gradienten
        errD_real.backward()
        D_x = output.mean().item()

        #Training mit generierten Bildern

        # Generierung von Bilder

        noise = torch.randn(b_size, latent_dim)
        noise =noise.to(device)

        fake = netG(noise)
        label.fill_(fake_label)
        #Einstufung der Bilder
        output = netD(fake.detach()).view(-1)
        # Berechnung des Verlusts
        errD_fake = criterion(output, label)
        # Berechnung des Gradienten
        errD_fake.backward()

        # Zusammenfügen der Daten für die echten und generierten Bilder
        errD = errD_real + errD_fake
        # Update des Diskriminators
        optimizerD.step()

        # Generatortraining
        netG.zero_grad()
        label.fill_(real_label)
        # Einstufung von gefälschten Bilder
        output = netD(fake).view(-1)
        # Berechnung des Verlusts
        errG = criterion(output, label)
        # Berechnung des Gradienten
        errG.backward()

        # Update des Generators
        optimizerG.step()

        # Output von Statistiken
        if i % 100 == 0:
            print(
                 "[Iteration %d/%d] [Batch %d/%d] [D cost: %f] [G cost: %f] "
                 % ((iterator[0] - 1), iterations, i, len(dataloader), errD.item(), errG.item())
            )
            writer.add_scalar('cost/discriminator', errD.item(), iterator[0])
            writer.add_scalar('cost/generator', errG.item(), iterator[0])



            writer.flush()

        if iterator % sample_interval == 0:
            # Gibt Bilder aus
            generate_images(netG, iterator)
            # Speichert den Generator auf zwei Arten
            torch.save(netG, 'images4/Generator%d.pth.tar' % iterator)
            torch.save(netG.state_dict(), 'images4/Generator_param%d.pt' % iterator)
            """
            # Fréchet Inception Distance
            with torch.no_grad():
                noisef = torch.randn(200, latent_dim)
                noisef = noisef.to(device)
                fakef = netG(noisef)
                fakef = ((fakef + 1.0) * 127.5)
                fakef = fakef.type(torch.uint8)

                fakef = fakef.permute(0, 2, 3, 1)
                fakef = fakef.cpu()
                # Statistikdatei Quelle: http://bioinf.jku.at/research/ttur/
                stats_path = 'fid_stats_celeba.npz'  # Statistiken für den CelebA Datensatz
                inception_path = fid.check_or_download_inception(None)  # Download inception network

                # Laden der Statistiken für CelebA
                f = np.load(stats_path)
                mu_real, sigma_real = f['mu'][:], f['sigma'][:]
                f.close()

                fid.create_inception_graph(inception_path)
                with tf.compat.v1.Session() as sess:
                    sess.run(tf.compat.v1.global_variables_initializer())
                    mu_gen, sigma_gen = fid.calculate_activation_statistics(fakef, sess, batch_size=200)
                fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
                print("FID: %s" % fid_value)

            writer.add_scalar('distance/FID', fid_value, iterator[0])
            writer.flush()
            """
            # Inception Score
            ins = inception_score(fake, True, 64, True)
            print(ins)
            writer.add_scalar('score/IS', ins[0], iterator[0])
            writer.flush()


end_time = time.time()
duration = (end_time - start_time)
print("duration: %d" %duration)




