import cv2
import pdb
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import numpy as np 
from collections import OrderedDict
from .ptsemseg.pspnet import pspnet
import argparse
import warnings
warnings.filterwarnings("ignore")

Transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_coord(input_paths, ana_part_ids=[2,3,4,5,8]):
    checkpoint="/root/workspace/datasets/chestxdet/pspnet_chestxray_best_model_4.pkl" # default
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = ['Left Clavicle', 'Right Clavicle', 'Left Scapula', 'Right Scapula',
        'Left Lung', 'Right Lung', 'Left Hilus Pulmonis', 'Right Hilus Pulmonis',
        'Heart', 'Aorta', 'Facies Diaphragmatica', 'Mediastinum',  'Weasand', 'Spine']
    
    # load model
    n_classes = len(classes)
    model = pspnet(n_classes)
    model_path = checkpoint
    state = convert_state_dict(torch.load(model_path)["model_state"])
    model.load_state_dict(state)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.eval()
    model.to(device)
    
    anatomical_dict = dict()
    for input_path in tqdm(input_paths):
        # preprocess image
        img = cv2.imread(input_path, 1)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
        img = Transform(img)
        img = img.unsqueeze(0)
        img = img.to(device)
        
        # prediction
        outputs = model(img)
        pred = outputs.data.cpu().numpy()
        pred = 1 / (1 + np.exp(-pred))  # sigmoid
        pred[pred < 0.5] = 0
        pred[pred > 0.5] = 1
        
        categ = { # switch left to right right to left 
            "2": "right_scapula",
            "3": "left_scapula",
            "4": "right_lung",
            "5": "left_lung",
            "8": "heart"
        }
        anatomical_coord = dict()
        for ana_part_id in ana_part_ids:
            im_array = (pred[0 , ana_part_id] * 255).astype('uint8') # also threshold array
            
            contours, hierarchy = cv2.findContours(im_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            x_min, y_min, w, h = cv2.boundingRect(contours[0])
            x_max, y_max = x_min + w, y_min + h

            x_min, y_min = x_min/512, y_min/512
            x_max, y_max = x_max/512, y_max/512
            
            anatomical_coord[categ[str(ana_part_id)]] = [x_min, y_min, x_max, y_max]
        anatomical_dict[input_path] = anatomical_coord
    
    return anatomical_dict

if __name__ == "__main__":

    '''
    Anatomical part id:
        Left Scapula: 2 
        Right Scapula: 3
        Left Lung: 4
        Right Lung: 5
        Heart: 8
    '''
    # ana_part_id = [2,3]
    # 69559.png 55931.png 46395.png
    # input_path = "/root/workspace/repo/mmdetection/mmdet/models/roi_heads/sar_modules/spatial_relation_module/pretrainedPSPNet/demo.png"
    input_path = "/root/workspace/datasets/chestxdet/train/36201.png"
    image_size = (1024, 1024)
    checkpoint_path = "/root/workspace/pspnet_chestxray_best_model_4.pkl"
    print(get_coord(input_path, image_size))
