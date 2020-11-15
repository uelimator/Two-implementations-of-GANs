from __future__ import absolute_import, division, print_function

import torch.nn.parallel
import torch.utils.data
# Python File der originalen Arbeit zur FID, Quelle: https://github.com/bioinf-jku/TTUR
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
#from torch.utils.tensorboard import SummaryWriter
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
tf.compat.v1.disable_eager_execution()

# Speicherort des Datensatzes
#dataroot = '~/Robin/dataset/celeba/29561_37705_bundle_archive/img_align_celeba'

############################

#      Hyperparameter      #

############################

# Anzahl an Arbeiter für das laden des Datensatzes
workers = 2

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
num_epochs = 127*5

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
#writer = SummaryWriter()
# Auswahl zwischen Cpu und GPU
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


os.makedirs("images6", exist_ok=True)

# Laden der Statistiken für CelebA
stats_path = 'fid_stats_celeba.npz'  # Statistiken für den CelebA Datensatz
f = np.load(stats_path)
mu_real, sigma_real = f['mu'][:], f['sigma'][:]
f.close()


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

netG=Generator().to(device)
netG.load_state_dict(torch.load('images4/Generator_param26000.pt'))
netG.eval()


netG2=Generator().to(device)
netG2.load_state_dict(torch.load('images5/Generator_param979000.pt'))
netG2.eval()
def generate_images(netG,name):
    with torch.no_grad():
        noise_s = torch.randn(batch_size, latent_dim)
        noise_s = noise_s.to(device)

        samples = netG(noise_s)

    save_image(samples.data[:49], "samples"+name+".png"  , nrow=7, normalize=True)

name_mmgan='mmgan'
name_wgan_gp='wgangp'

generate_images(netG,name_mmgan)
generate_images(netG2,name_wgan_gp)
"""
with torch.no_grad():
    noisef = torch.randn(200, latent_dim)
    noisef = noisef.to(device)
    fakef = netG(noisef)
    fakef = ((fakef + 1.0) * 127.5)
    fakef = fakef.type(torch.uint8)

    fakef = fakef.permute(0, 2, 3, 1)
    fakef = fakef.cpu()
    # Statistikdatei Quelle: http://bioinf.jku.at/research/ttur/

    inception_path = fid.check_or_download_inception(None)  # Download inception network

    fid.create_inception_graph(inception_path)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        mu_gen, sigma_gen = fid.calculate_activation_statistics(fakef, sess, batch_size=200)
    fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    print("FID: %s" % fid_value)
"""