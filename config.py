# This file is adapted from https://github.com/zalandoresearch/famos (MIT License)
# File: config.py
import argparse
import torch.nn as nn
import datetime
import os
parser = argparse.ArgumentParser()

##data path and loading parameters
parser.add_argument('--device', default=0, type=int, help='device ids assignment (e.g 1)')

parser.add_argument('--hp_od', default=20, type=float, nargs='+', help='OD threat model hyperparameter')
parser.add_argument('--od_threshold', default=0.5, type=float, nargs='+', help='OD threat model hyperparameter')
parser.add_argument('--psgan', type=int, default=1, nargs='+', help='use PSGAN dataloader')

parser.add_argument('--contentPath', default='./content/bikers.jpg', help='path to content image folder')

## coordinates to place patch [x1, y1, x2, y2], or [left, top, right, bottom], patch MUST be square
parser.add_argument('--coords', default=[45, 70, 141, 166], type=int, nargs='+', help='bikers, 96x96')
#parser.add_argument('--coords', default=[85, 66, 213, 194], type=int, nargs='+', help='overhead cars, 128x128')


parser.add_argument('--cropContentDisciminator', type=bool, default=False,help='reduce data being fed to whole image discriminator for faster processing')

parser.add_argument('--printability', type=bool, default=False, help='make patch printable')
parser.add_argument('--printabilityFile', default='color_palette.txt', help='non printability scoring')

parser.add_argument('--tensorboard', default=1, type=int, nargs='+', help='record tensorboard logs')
parser.add_argument('--mirror', type=bool, default=False,help='augment style image distribution for mirroring')
parser.add_argument('--contentScale', type=float, default=1.0,help='scale content images')
parser.add_argument('--textureScale', type=float, default=1.0,help='scale texture images')
parser.add_argument('--testImage',default='None', help='path to test image file')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)#0 means a single main process
parser.add_argument('--outputFolder', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--trainOverfit', type=bool, default=False,help='always use same image and same templates -- better in sample, worse out of sample')
##neural network parameters
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--ngf', type=int, default=120,help='number of channels of generator (at largest spatial resolution)')
parser.add_argument('--ndf', type=int, default=120,help='number of channels of discriminator (at largest spatial resolution)')
parser.add_argument('--nDep', type=int, default=5,help='depth of Unet Generator')
parser.add_argument('--nDepD', type=int, default=5,help='depth of Discriminator')
parser.add_argument('--N', type=int, default=30,help='count of memory templates')
parser.add_argument('--coordCopy', type=bool, default=True,help='copy  x,y coordinates of cropped memory template')
parser.add_argument('--multiScale', type=bool, default=False,help='multi-scales of mixing features; if False only full resolution; if True all levels')
parser.add_argument('--nBlocks', type=int, default=0,help='additional res blocks for complexity in the unet')
parser.add_argument('--blendMode', type=int, default=0,help='type of blending for parametric/nonparametric output')
parser.add_argument('--refine', type=bool, default=False,help='second unet after initial templates')
parser.add_argument('--skipConnections', type=bool, default=True,help='skip connections in  Unet -- allows better content reconstruct')
parser.add_argument('--Ubottleneck', type=int, default=-1,help='Unet bottleneck, leave negative for default wide bottleneck')
##regularization and loss criteria weighting parameters
parser.add_argument('--fContent', type=float, default=1.0,help='weight of content reconstruction loss')
parser.add_argument('--fAdvM', type=float, default=.0,help='weight of I_M adversarial loss')
parser.add_argument('--fContentM', type=float, default=1.0,help='weight of I_M content reconstruction loss')
parser.add_argument('--cLoss', type=int, default=0,help='type of perceptual distance metric for reconstruction loss')
parser.add_argument('--fAlpha', type=float, default=.1,help='regularization weight of norm of blending mask')
parser.add_argument('--fTV', type=float, default=.1,help='regularization weight of total variation of blending mask')
parser.add_argument('--fEntropy', type=float, default=.5,help='regularization weight of entropy -- forcing low entropy results in 0/1 values in mix tensor A')
parser.add_argument('--fDiversity', type=float, default=1,help='regularization weight of diversity of used templates')
parser.add_argument('--WGAN', type=bool, default=False,help='use WGAN-GP adversarial loss')
parser.add_argument('--LS', type=bool, default=False,help='use least squares GAN adversarial loss')
##Optimisation parameters
parser.add_argument('--niter', type=int, default=199, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--dIter', type=int, default=1, help='number of Discriminator steps -- for 1 Generator step')
##noise parameters
parser.add_argument('--zGL', type=int, default=20,help='noise channels, identical on every spatial position')
parser.add_argument('--zLoc', type=int, default=50,help='noise channels, sampled on each spatial position')
parser.add_argument('--zPeriodic', type=int, default=0,help='periodic spatial waves')
parser.add_argument('--firstNoise', type=bool, default=False,help='stochastic noise at bottleneck or input of Unet')
opt = parser.parse_args()

nDep = opt.nDep
##noise added to the deterministic content mosaic modules -- in some cases it makes a difference, other times can be ignored
bfirstNoise=opt.firstNoise
nz=opt.zGL+opt.zLoc+opt.zPeriodic
bMirror=opt.mirror##make for a richer distribution, 4x times more data
opt.fContentM *= opt.fContent

##GAN criteria changes given loss options LS or WGAN
if not opt.WGAN and not opt.LS:
    criterion = nn.BCELoss()
elif opt.LS:
    def crit(x,l):
        return ((x-l)**2).mean()
    criterion=crit
else:
    def dummy(val,label):
        return (val*(1-2*label)).mean()#so -1 fpr real. 1 fpr fake
    criterion=dummy

if opt.outputFolder=='.':
    split = os.path.splitext(opt.contentPath)[0]
    length = split[:-1].rfind('/')
    opt.outputFolder = "results/" +split[length + 1:] + '/'
    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    opt.outputFolder += stamp + "/"
try:
    os.makedirs(opt.outputFolder)
except OSError:
    pass
print ("outputFolder "+opt.outputFolder)

text_file = open(opt.outputFolder+"options.txt", "w")
text_file.write(str(opt))
text_file.close()
print (opt)