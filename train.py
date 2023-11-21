# This code is adapted from https://github.com/zalandoresearch/famos (MIT License)
# File: PSGAN.py
from __future__ import print_function
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from utils import TextureDataset, setNoise, learnedWN, build_dataloader, build_cropped_whole_disc_batch, crop_patch_from_whole_image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import sys
from network import weights_init,Discriminator, Vanilla, calc_gradient_penalty,NetG
from config import opt,nz,nDep,criterion
import time
from PIL import Image
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import object_detector
import os
from printability_score import NPSCalculator

# select a seed for training
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True

N=0
ngf = int(opt.ngf)
ndf = int(opt.ndf)
desc="fc"+str(opt.fContent)+"_ngf"+str(ngf)+"_ndf"+str(ndf)+"_dep"+str(nDep)+"-"+str(opt.nDepD)

if opt.WGAN:
    desc +='_WGAN'
if opt.LS:
        desc += '_LS'
if opt.mirror:
    desc += '_mirror'
if opt.textureScale !=1:
    desc +="_scale"+str(opt.textureScale)

device = "cuda:" + str(opt.device)
print ("device",device)

# instantiate patch discriminator
netD_patch = Discriminator(ndf, opt.nDepD, bSigm=not opt.LS and not opt.WGAN)

# instantiate whole image discriminator
model_config = {'max_features': 32, 'min_features': 32, 'num_blocks': 5, 'kernel_size': 3, 'padding': 0, 'in_channels': 3, 'normalization': True}
netD_whole = Vanilla(**model_config)

# instantiate patch generator
netG =NetG(ngf, nDep, nz)
Gnets=[netG]
if opt.zPeriodic:
    Gnets += [learnedWN]

# # load OD model
#model_kwargs = {
#    "max_size": 1280,
#    "min_size": 960,
#    "num_classes": 3 # can define relevant classes to target 
#}
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model = model.to(device)

# initialize custom weights of GAN models, defined in network.py        
for net in [netD_patch] + Gnets + [netD_whole]:
    try:
        net.apply(weights_init)
    except Exception as e:
        print (e,"weightinit")
    pass
    net=net.to(device)
    print(net)

patch_size = opt.coords[2] - opt.coords[0]
NZ = patch_size//2**nDep
noise = torch.FloatTensor(opt.batchSize, nz, NZ,NZ)
fixnoise = torch.FloatTensor(opt.batchSize, nz, NZ*4,NZ*4)
noise=noise.to(device)
fixnoise=fixnoise.to(device)

real_label = 1
fake_label = 0

# setup optimizers
optimizerD_patch = optim.Adam(netD_patch.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD_whole = optim.Adam(netD_whole.parameters(), lr=5e-4, betas=[0.5, 0.9])
optimizerU = optim.Adam([param for net in Gnets for param in list(net.parameters())], lr=opt.lr, betas=(opt.beta1, 0.999))


print("Building patch from coordinates: ", opt.coords)

whole_image = Image.open(opt.contentPath).convert('RGB').copy()

#recieve patch as tensor; only used with PSGAN
dataloader_patch = build_dataloader(whole_image)

# build real whole image batch for whole discriminator, range [-1,1]; only used without PSGAN
patch_disc_batch_train_real = torch.zeros([opt.batchSize, 3, patch_size, patch_size])

patch = crop_patch_from_whole_image(whole_image)
patch = patch.to(device)

# build real whole image batch for whole discriminator, range [-1,1]
whole_disc_batch_train_real = torch.zeros([opt.batchSize, 3, whole_image.size[1], whole_image.size[0]])

# build real whole image batch for OD, normalized range [0,1]
od_model_input_batch_real = torch.zeros([opt.batchSize, 3, whole_image.size[1], whole_image.size[0]])

# prepare whole image
transTensor = transforms.ToTensor()
whole_pic_tensor = transTensor(whole_image.copy()).to(device)

mean = torch.tensor([0.5,0.5,0.5]).to(device)
std = torch.tensor([0.5,0.5,0.5]).to(device)
# convert whole image to be [-1,1] 
if not opt.psgan:
    normalized_patch_tensor = (patch - mean[:,None,None]) / std[:, None, None] 
normalized_whole_pic_tensor = (whole_pic_tensor - mean[:,None,None]) / std[:, None, None] 
# build overlay batch
for sample in range(opt.batchSize):
    # add to OD model input; should range between 0 and 1
    od_model_input_batch_real[sample:,:,:] = whole_pic_tensor 
    if not opt.psgan:
        patch_disc_batch_train_real[sample:,:,:] = normalized_patch_tensor 
    # add to whole disc input; should range between -1 and 1
    whole_disc_batch_train_real[sample:,:,:] = normalized_whole_pic_tensor 

# if cropping is selected, crop whole image batch for whole disc
if opt.cropContentDisciminator:
    # takes in full image batch and returns it cropped 
     whole_disc_batch_train_real, patchExtCoords = build_cropped_whole_disc_batch(whole_disc_batch_train_real)
     vutils.save_image(whole_disc_batch_train_real,'%s/cropped_image_real.png' % (opt.outputFolder),normalize=True) 

whole_disc_batch_train_real = whole_disc_batch_train_real.to(device)
od_model_input_batch_real = od_model_input_batch_real.to(device)

# init tensorboard for visualizing training 
tensorboard_path = str(opt.outputFolder + "runs/")
writer = SummaryWriter(log_dir=tensorboard_path)

# instantiate printability if desired
if opt.printability:
    nps_calculator_patch= NPSCalculator(patch_side=patch_size, 
                                                   printability_file_1=opt.printabilityFile
                                                   ).to(device)

#begin training loop
for epoch in range(opt.niter):
    for i, data_patch in enumerate(dataloader_patch, 0):
        t0 = time.time()
        sys.stdout.flush()
        if opt.psgan:
            patch_disc_batch_train_real, _ = data_patch # return data and its index
        patch_disc_batch_train_real=patch_disc_batch_train_real.to(device)

        ########## Patch Disc #######
        # train patch discriminator with real data
        netD_patch.zero_grad()
        output = netD_patch(patch_disc_batch_train_real)

        errD_real_patch = criterion(output, output.detach()*0+real_label)
        errD_real_patch.backward()
        D_x_patch = output.mean()

        # generator produces output from input noise
        noise=setNoise(noise)
        patch_disc_batch_train_fake = netG(noise) #  this is used later too

        # train patch discriminator with fake
        output = netD_patch(patch_disc_batch_train_fake.detach())
        errD_fake_patch = criterion(output, output.detach()*0+fake_label)
        errD_fake_patch.backward()

        D_G_z1_patch = output.mean() # patch discriminator probability that fake input is real
        #errD_patch = errD_real_patch + errD_fake_patch
        if opt.WGAN:
            gradient_penalty_patch = calc_gradient_penalty(netD_patch, patch_disc_batch_train_real, patch_disc_batch_train_fake)
            gradient_penalty_patch.backward(retain_graph=True) 
        optimizerD_patch.step()

        ########## Whole Disc (VANILLA) #######
        # train whole discriminator with real 
        netD_whole.zero_grad()
        output = netD_whole(whole_disc_batch_train_real) # requires 4dim input

        errD_real_whole = criterion(output, output.detach()*0+real_label)
        errD_real_whole.backward()
        D_x_whole = output.mean()

        # build fake overlay batch
        whole_disc_batch_train_fake = whole_disc_batch_train_real.detach().clone() 
        whole_disc_batch_train_fake = whole_disc_batch_train_fake.to(device)
        for a, patch_image in enumerate(patch_disc_batch_train_fake):
            if opt.cropContentDisciminator:
                whole_disc_batch_train_fake[a:, :, patchExtCoords[1]:patchExtCoords[3], patchExtCoords[0]:patchExtCoords[2]]=patch_image
            else:    
                whole_disc_batch_train_fake[a:, :, opt.coords[1]:opt.coords[3], opt.coords[0]:opt.coords[2]]=patch_image

        # train whole discriminator with fake overlay image 
        output = netD_whole(whole_disc_batch_train_fake.detach().to(device))
        D_G_z1_whole = output.mean()

        errD_fake_whole = criterion(output, output.detach()*0+fake_label)
        errD_fake_whole.backward()
        #errD_whole = errD_real_whole + errD_fake_whole
        if opt.WGAN:
            # discriminator outputs large positive and negative values
            gradient_penalty_whole = calc_gradient_penalty(netD_whole, whole_disc_batch_train_real, whole_disc_batch_train_fake)
            gradient_penalty_whole.backward()
        optimizerD_whole.step()

        for net in Gnets:
            net.zero_grad()

        ########## Evaluation Patch Disc #######
        noise=setNoise(noise)

        patch_disc_batch_eval_fake = netG(noise) #  this is used later too
        output_patch_val = netD_patch(patch_disc_batch_eval_fake)

        loss_adv_patch = criterion(output_patch_val, output_patch_val.detach()*0+real_label)
        D_G_z2_patch = output_patch_val.mean()

        print(' PATCH [%d/%d][%d/%d] D(x): %.4f D(G(z)): %.4f / %.4f time %.4f'
              % (epoch, opt.niter, i, len(dataloader_patch), D_x_patch, D_G_z1_patch, D_G_z2_patch,time.time()-t0))
        writer.add_scalars('Patch Discriminator BCELoss', {"Patch_disc_fake/train":D_G_z1_patch,
                                'Patch_disc_real/train': D_x_patch,
                                'Patch_disc_fake/eval': D_G_z2_patch}, epoch)

        whole_disc_batch_eval_fake = whole_disc_batch_train_real.detach().clone() 
        whole_disc_batch_eval_fake = whole_disc_batch_eval_fake.to(device)
        od_batch_eval_fake = od_model_input_batch_real.detach().clone()
        od_batch_eval_fake = od_batch_eval_fake.to(device)

        # for non printability score as well as saving a visual when patches are adversarial
        patch_batch = torch.zeros([opt.batchSize, 3, patch_disc_batch_eval_fake.shape[2], patch_disc_batch_eval_fake.shape[3]])
        patch_batch = patch_batch.to(device)
        # build overlay batch
        for a, patch_image in enumerate(patch_disc_batch_eval_fake):
            normalized_patch = (patch_image - torch.min(patch_image)) / (torch.max(patch_image) - torch.min(patch_image))
            patch_batch[a] = normalized_patch
            #patch_batch[a] = normalized_patch
            od_batch_eval_fake[a:, :, opt.coords[1]:opt.coords[3], opt.coords[0]:opt.coords[2]] = normalized_patch
            if opt.cropContentDisciminator:
                whole_disc_batch_train_fake[a:, :, patchExtCoords[1]:patchExtCoords[3], patchExtCoords[0]:patchExtCoords[2]]=patch_image
            else:    
                whole_disc_batch_train_fake[a:, :, opt.coords[1]:opt.coords[3], opt.coords[0]:opt.coords[2]]=patch_image
        
        ########## Evaluation Whole Image Discriminator #######
        output_whole_val = netD_whole(whole_disc_batch_eval_fake.to(device))
        loss_adv_whole = criterion(output_whole_val, output_whole_val.detach()*0+real_label)
        D_G_z2_whole = output_whole_val.mean()

        print(' WHOLE [%d/%d][%d/%d] D(x): %.4f D(G(z)): %.4f / %.4f time %.4f'
            % (epoch, opt.niter,i, len(dataloader_patch),D_x_whole, D_G_z1_whole, D_G_z2_whole,time.time()-t0))
        writer.add_scalars('Whole Discriminator BCELoss', {"Whole_disc_fake/train":D_G_z1_whole,
                        'Whole_disc_real/train': D_x_whole,
                        'Whole_disc_fake/eval': D_G_z2_whole}, epoch)

        ###### Adversarial Object Detection #######
        model_losses, pred_visual, pred_classes, pred_scores = object_detector.calc_od_penalty(model, od_model_input_batch_real, od_batch_eval_fake, patch_batch, epoch,visualize=(i % 100 == 0))

        ########## Update generator #######
        # loss = patch_inconspicousness + scene_inconspicuousness
        errG = loss_adv_patch + loss_adv_whole 
        # subtract loss bc higher object detection loss means less error in adversarial generated data
        errG = errG - opt.hp_od*model_losses['loss_classifier'] # option to include model_losses['loss_box_reg']
        if i % 100 == 0:
                writer.add_scalar("Loss classifier", model_losses['loss_classifier'], epoch)
                #writer.add_scalar("Loss Box reg", model_losses['loss_box_reg'], epoch)
                writer.add_scalar("Generator Error with OD loss", errG, epoch)

        ###### Non Printability Score ####### 
        if opt.printability:
            nps_list = torch.zeros([opt.batchSize, 1])
            for val, patch in enumerate(patch_batch):
                nps = nps_calculator_patch(patch) # NPS color palette is [0,1]
                nps_list[val] = nps
            if i % 100 == 0:
                writer.add_scalar("Printability score", torch.mean(nps_list), epoch)

            #print("nps list", nps_list)
            errG = errG + torch.mean(nps_list)

        errG.backward()
        optimizerU.step()

        # save images for inference
        if i % 100 == 0:
            #vutils.save_image(patch_disc_batch_eval_fake,'%s/generated_textures_%03d_%s.png' % (opt.outputFolder, epoch,desc),normalize=True)  
            #vutils.save_image(cropped_disc_batch_eval,'%s/overlayed_patch_normalized_cropped_%03d_%s.png' % (opt.outputFolder,epoch,desc),  normalize=True)
            vutils.save_image(od_batch_eval_fake,'%s/overlayed_patch_normalized_whole_%03d_%s.png' % (opt.outputFolder,epoch,desc),  normalize=False)
            if isinstance(pred_visual, Image.Image):
                pred_visual.save('%s/object_detection_%03d_%s.png' % (opt.outputFolder,epoch,desc))
                # write static_y_list labels and scores to disk 
                txtname = str('object_detection_%03d_%s_labels_and_scores.txt' % (epoch,desc))
                filepath = os.path.join(opt.outputFolder, txtname)
                with open(filepath, 'w') as file:
                    for a, class_list in enumerate(pred_classes):
                        file.write("Sample " + str(a) + '\n')
                        file.write('labels: ')
                        file.write(str(class_list) + '\n')
                        for b, scores_list in enumerate(pred_scores):
                            if a == b:
                                file.write('scores: ')
                                file.write(str(scores_list) + '\n')
                                file.write('\n')
writer.flush()
writer.close()
            ##OPTIONAL
            ##save/load model for later use if desired
            #outModelName = '%s/netG_epoch_%d_%s.pth' % (opt.outputFolder, epoch*0,desc)
            #torch.save(netU.state_dict(),outModelName )
            #netU.load_state_dict(torch.load(outModelName))