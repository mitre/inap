import torch
from collections import Counter
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import PIL
import torch.nn as nn
from config import opt
import math
import torchvision
from torchvision.models import detection
import pickle
import cv2
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as transforms

# define global variables
first_time_through_calc_od_flag = True
COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
COCO_INSTANCE_CATEGORY_INTS = {'__background__': 0.0, 'person': 1.0, 'bicycle': 2.0, 'car': 3.0, 'motorcycle': 4.0, 'airplane': 5.0, 'bus': 6.0, 'train': 7.0, 'truck': 8.0, 'boat': 9.0, 'traffic light': 10.0, 'fire hydrant': 11.0, 'N/A': 83.0, 'stop sign': 13.0, 'parking meter': 14.0, 'bench': 15.0, 'bird': 16.0, 'cat': 17.0, 'dog': 18.0, 'horse': 19.0, 'sheep': 20.0, 'cow': 21.0, 'elephant': 22.0, 'bear': 23.0, 'zebra': 24.0, 'giraffe': 25.0, 'backpack': 27.0, 'umbrella': 28.0, 'handbag': 31.0, 'tie': 32.0, 'suitcase': 33.0, 'frisbee': 34.0, 'skis': 35.0, 'snowboard': 36.0, 'sports ball': 37.0, 'kite': 38.0, 'baseball bat': 39.0, 'baseball glove': 40.0, 'skateboard': 41.0, 'surfboard': 42.0, 'tennis racket': 43.0, 'bottle': 44.0, 'wine glass': 46.0, 'cup': 47.0, 'fork': 48.0, 'knife': 49.0, 'spoon': 50.0, 'bowl': 51.0, 'banana': 52.0, 'apple': 53.0, 'sandwich': 54.0, 'orange': 55.0, 'broccoli': 56.0, 'carrot': 57.0, 'hot dog': 58.0, 'pizza': 59.0, 'donut': 60.0, 'cake': 61.0, 'chair': 62.0, 'couch': 63.0, 'potted plant': 64.0, 'bed': 65.0, 'dining table': 67.0, 'toilet': 70.0, 'tv': 72.0, 'laptop': 73.0, 'mouse': 74.0, 'remote': 75.0, 'keyboard': 76.0, 'cell phone': 77.0, 'microwave': 78.0, 'oven': 79.0, 'toaster': 80.0, 'sink': 81.0, 'refrigerator': 82.0, 'book': 84.0, 'clock': 85.0, 'vase': 86.0, 'scissors': 87.0, 'teddy bear': 88.0, 'hair drier': 89.0, 'toothbrush': 90.0}
device = "cuda:"+str(opt.device)

def get_class_int(class_names_list):
    class_int_list = []
    for class_name in class_names_list:
        class_int_list.append( int(COCO_INSTANCE_CATEGORY_INTS[class_name]))
    return class_int_list

# expects boxes input [[x1,y1],[x2,y2]]
def prepare_boxes(box_list):
    boxes = []
    for pair in box_list:
        print(pair)
        new_list = [pair[0][0], pair[0][1], pair[1][0], pair[1][1]]
        boxes.append(new_list)
    # return format [x1,y1,x2,y2]
    return boxes

def calc_od_penalty(model, real_batch, synthetic_batch, patch_batch, epoch, visualize = False, threshold=opt.od_threshold, rect_th=2, text_size=0.45, text_th=2): 
    
    global first_time_through_calc_od_flag
    #this happens once
    if first_time_through_calc_od_flag:
        #global coco_names_as_ints
        first_time_through_calc_od_flag = False
        global boxes_static, pred_cls_static, scores_static
        boxes_static, pred_cls_static, scores_static = get_prediction(model, real_batch, threshold=threshold)
        # returns as uint8
        static_od_image_prediction = visualize_preds(real_batch, boxes_static, pred_cls_static, threshold=threshold, rect_th=rect_th, text_size=text_size, text_th=text_th)
 
        static_od_image_prediction.save('%s/static_object_detection.png' % (opt.outputFolder))

        global static_y_dict
        static_y_dict = {}
        static_y_dict['boxes'] = torch.tensor(boxes_static[0]).to(device)
        static_y_dict['labels'] = torch.tensor(get_class_int(pred_cls_static[0]), dtype=torch.long).to(device) # only need to worry about one since they're all the same picture
        static_y_dict['scores'] = torch.tensor(scores_static[0]).to(device)
    
        global static_y_list
        static_y_list = []

        for batch in range(opt.batchSize):
            static_y_list.append(static_y_dict)
        # write static_y_list labels and scores to disk 
        filepath = os.path.join(opt.outputFolder, 'static_labels_and_scores.txt')
        with open(filepath, 'w') as file:
            for key, value in static_y_dict.items():
                    file.write(f'{key}: {value}\n')

    # Calc FRCNN losses 
    with torch.no_grad():
        model.train()
        output = model(synthetic_batch.to(device), static_y_list) #'loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'

    model.eval()
    object_det_output_batch = 0
    pred_cls_attacked = 0
    scores_attacked = 0

    if visualize:          
        boxes_attacked, pred_cls_attacked, scores_attacked = get_prediction(model, synthetic_batch, threshold=threshold) 
        for num, classes in enumerate(pred_cls_attacked):
            # only visualize if something new was detected
            if len(classes) != len(static_y_list[num]['labels']):
                vutils.save_image(patch_batch,'%s/generated_textures_%03d_adversarial.png' % (opt.outputFolder, epoch),normalize=True)  
                object_det_output_batch = visualize_preds(synthetic_batch, boxes_attacked, pred_cls_attacked, threshold=threshold, rect_th=rect_th, text_size=text_size, text_th=text_th)
                break

    # return loss vals, batch OD visualization, attacked labels, attacked scores
    return output, object_det_output_batch, pred_cls_attacked, scores_attacked  


# recieves batch of images to make prediction
def visualize_preds(img_input, boxes, pred_cls, threshold=opt.od_threshold, rect_th=3, text_size=0.45, text_th=2) -> Image: 

    detection_batch = img_input.detach().clone()
    detection_batch = detection_batch.to(device)
    pil = transforms.ToPILImage()
    # each image in batch
    for a, image in enumerate(img_input): 

        uint8 = (image*255).to(torch.uint8)

        annotated_image_tensor = draw_bounding_boxes(uint8, boxes=torch.tensor(boxes[a]),
                    labels=pred_cls[a],
                    colors="yellow",
                    width=4, font_size=30)

        detection_batch[a] = annotated_image_tensor.to(torch.float32)

    detection = batch_to_PIL(detection_batch.to(torch.uint8))
    #detection.save('%s/static_object_detection.png' % (opt.outputFolder))
    return detection

def batch_to_PIL(input_tensor_batch):
    height, width = input_tensor_batch.shape[2], input_tensor_batch.shape[3]

    final_image = Image.new("RGB", (width * opt.batchSize, height), "white")
    pil = transforms.ToPILImage()
    for i, image in enumerate(input_tensor_batch):

        final_image.paste(pil(image), (i*width, 0))

    return final_image

# recieves batch
def get_prediction(model, img, threshold):

  pred_boxes_batch = []
  pred_class_batch = []
  scores_batch = []
  with torch.no_grad():
    model.eval()
 
    pred = model(img.to(device))
    # grab each image predictions
    for image_pred in pred: 

        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(image_pred['labels'].detach().cpu().numpy())]
        pred_boxes = [[i[0], i[1], i[2], i[3]] for i in list(image_pred['boxes'].detach().cpu().numpy())]
        pred_score = list(image_pred['scores'].detach().cpu().numpy())
        if pred_score and max(pred_score) >= threshold:
            # Only consider the model's confident predictions
            pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
            scores = [x for x in pred_score if x>threshold]
            pred_boxes = pred_boxes[:pred_t+1]
            pred_class = pred_class[:pred_t+1]
            pred_boxes_batch.append(pred_boxes)
            pred_class_batch.append(pred_class)
            scores_batch.append(scores)
        else:
            print("Object Detector has no predictions")
            pred_boxes_batch.append([])
            pred_class_batch.append([])
            scores_batch.append([])

   # returns three lists
  return pred_boxes_batch, pred_class_batch, scores_batch




