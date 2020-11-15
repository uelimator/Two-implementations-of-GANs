# Two-implementations-of-GANs
I implemented two versions of GANs for a my Maturitäts paper


The two implementations are written in Python with the framework Pytorch. 
I used the ressources marked in the Ressources tab for inspiration and marked snippets in the code that I reused. Note that the models are not very optimized for the best performance and were produced withe focus on a fair comparison.

# Ressources
Pytorch tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

The original WGAN-GP implementation: https://github.com/igul222/improved_wgan_training

An Pytorch Implementation: https://github.com/caogang/wgan-gp


FID: https://github.com/bioinf-jku/TTUR

# Code
I provided the two training algorithms. One for the WGAN-GP and one for the MMGAN. Additionally i provided two trained generators and a code to generate samples with these trained networks.
