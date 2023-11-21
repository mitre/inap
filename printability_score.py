# This code is adapted from https://github.com/VITA-Group/3D_Adversarial_Logo/tree/master (MIT License)
# File: load_data.py
import numpy as np
import torch.nn as nn
import torch

class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self,  patch_side, printability_file_1, printability_file_2=None):
        super(NPSCalculator, self).__init__()
        self.printability_array_1 = nn.Parameter(self.get_printability_array(printability_file_1, patch_side),requires_grad=False)
        if not(printability_file_2 == None):
            self.printability_array_2 = nn.Parameter(self.get_printability_array(printability_file_2, patch_side),requires_grad=False)

    def forward(self, adv_patch, key=1):
        # calculate euclidian distance between colors in patch and colors in printability_array 
        # square root of sum of squared difference

        if  (key == 1):
            color_dist = (adv_patch - self.printability_array_1+0.000001)  ##  torch.Size([30, 3, 300, 300])
        elif(key == 2):
            color_dist = (adv_patch - self.printability_array_2+0.000001)  ##  torch.Size([30, 3, 300, 300])
        color_dist = color_dist ** 2  ##                                 torch.Size([30, 3, 300, 300])
        color_dist = torch.sum(color_dist, 1)+0.000001  ##               torch.Size([30, 300, 300])
        color_dist = torch.sqrt(color_dist)  ##                          torch.Size([30, 300, 300])  
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0] #test: change prod for min (find distance to closest color)  ##  torch.Size([300, 300])
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod,0)  ##                                                                   torch.Size([300])
        nps_score = torch.sum(nps_score,0)  ##                                                                         torch.Size([])
        return nps_score/torch.numel(adv_patch)

    def get_printability_array(self, printability_file, side):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa

# def get_printable_vals():
#     printed_im = imread('/inap/color_palette.txt')

#     num_colors =30
#     indexed_image, color_map = rgb2ind(printed_im, num_colors)

#     printable_vals = round(color_map)
#     printable_vals = sortrows(printable_vals)
#     return printable_vals

# def norm(x,y):
#     return sum((x-y)*(x-y))

