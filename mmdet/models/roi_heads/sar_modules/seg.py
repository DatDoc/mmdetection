import cv2
import pdb
import os
import glob
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import numpy as np 
from collections import OrderedDict
from .ptsemseg.pspnet import pspnet
from .seg_dataset import PSPNetDataset
import argparse
import warnings
warnings.filterwarnings("ignore")
from albumentations import (
    Resize, Rotate, HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2


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


def get_ana_bboxes(im_array, ana_part_ids):
    categ = { # switch left to right right to left 
        "2": "right_scapula",
        "3": "left_scapula",
        "4": "right_lung",
        "5": "left_lung",
        "8": "heart"
    }
    anatomical_coord = dict()
    for ana_image, ana_part_id in zip(im_array,ana_part_ids):
        contours, hierarchy = cv2.findContours(ana_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x_min, y_min, w, h = cv2.boundingRect(contours[0])
        x_max, y_max = x_min + w, y_min + h
        x_min, y_min = x_min/512, y_min/512
        x_max, y_max = x_max/512, y_max/512
        anatomical_coord[categ[str(ana_part_id)]] = [x_min, y_min, x_max, y_max]
    return anatomical_coord


def get_coord(img_prefix, pretrained_path, batch_size, ana_part_ids=[2,3,4,5,8]):
    checkpoint = pretrained_path # default
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
    model.to(device)
    model.eval()

    # dataloader
    Transform = Compose([
            Resize(512, 512, always_apply=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
            ToTensorV2(p=1.0),
            ], p=1.)
    dataset = PSPNetDataset(img_prefix, Transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        drop_last=False,
        shuffle=False)

    # Start predicting
    anatomical_dict = dict()
    for image_ids, images in tqdm(dataloader):
        images = images.to(device)
        # prediction
        outputs = model(images)
        preds = outputs.data.cpu().numpy()
        preds = 1 / (1 + np.exp(-preds))  # sigmoid
        preds[preds < 0.5] = 0
        preds[preds > 0.5] = 1

        im_arrays = (preds[:, ana_part_ids] * 255).astype('uint8') # (4, 5, 512, 512)
        for image_id, im_array in zip(image_ids, im_arrays):
            anatomical_dict[image_id] = get_ana_bboxes(im_array, ana_part_ids)
    # anatomical_dict = dict()
    # for image_id in tqdm(os.listdir(img_prefix)):
    #     image_path = os.path.join(img_prefix, image_id)
    #     anatomical_dict[image_path] = {
    #         'right_scapula': [0.0, 0.0, 1.0, 1.0], 
    #         'left_scapula': [0.0, 0.0, 1.0, 1.0], 
    #         'right_lung': [0.0, 0.0, 1.0, 1.0], 
    #         'left_lung': [0.0, 0.0, 1.0, 1.0], 
    #         'heart': [0.0, 0.0, 1.0, 1.0]
    #         }
    return anatomical_dict
