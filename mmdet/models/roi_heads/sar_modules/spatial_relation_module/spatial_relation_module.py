import torch
import torch.nn as nn
from tqdm import tqdm
import math
import copy
import numpy as np
from .positional_encoding import PositionalEncoding
from .pretrainedPSPNet.seg import get_coord

class SpatialRelationModule(nn.Module):
    def __init__(self, input_paths, d_model):
        super(SpatialRelationModule, self).__init__()
        self.d_model = d_model
        self.anatomical_dict = get_coord(input_paths)

    def get_spatial_vector(self, rois, anatomical_parts, image_size):
        M = np.empty((0,40), float)
        image_width, image_height = image_size
        
        for roi in rois:
            Mp = np.empty(40)
            # import pdb; pdb.set_trace()
            image_id = int(roi[0])
            _anatomical_parts = copy.deepcopy(anatomical_parts[image_id])
            
            for part in _anatomical_parts:
                _anatomical_parts[part][0] *= image_width
                _anatomical_parts[part][1] *= image_height
                _anatomical_parts[part][2] *= image_width
                _anatomical_parts[part][3] *= image_height
            
            roi = roi[1:] # tensor([  0.0000,  55.4688, 300.0000, 300.0000, 710.9375], device='cuda:0')
            wl = _anatomical_parts["right_lung"][2] - _anatomical_parts["left_lung"][0]
            hl = _anatomical_parts["right_lung"][3] - _anatomical_parts["left_lung"][1]
            for idx, part in enumerate(_anatomical_parts):
                ana_part_coord = _anatomical_parts[part]
                Mp[idx*8+0] = (roi[0] - ana_part_coord[0]) / wl
                Mp[idx*8+1] = (roi[1] - ana_part_coord[1]) / hl
                Mp[idx*8+2] = (roi[0] - ana_part_coord[2]) / wl
                Mp[idx*8+3] = (roi[1] - ana_part_coord[3]) / hl
                Mp[idx*8+4] = (roi[2] - ana_part_coord[0]) / wl
                Mp[idx*8+5] = (roi[3] - ana_part_coord[1]) / hl
                Mp[idx*8+6] = (roi[2] - ana_part_coord[2]) / wl
                Mp[idx*8+7] = (roi[3] - ana_part_coord[3]) / hl
            M = np.append(M, np.array([Mp]), axis=0)
        return M

    def positional_encoding(self, M):
        fspa = np.empty((M.shape[0],0), float)
        for j in range(self.d_model):
            _M = M * 1/(math.pow(1000, j/self.d_model))
            fspa = np.concatenate((fspa, np.concatenate((np.sin(_M), np.cos(_M)), axis=1)), axis=1)
        return fspa

    def forward(self, rois, img_metas):
        image_paths = [item["filename"] for item in img_metas]
        image_size = img_metas[0]["img_shape"][:2]
        res = [self.anatomical_dict[image_path] for image_path in image_paths]
        M = self.get_spatial_vector(rois, res, image_size)
        # M.shape = (4096, 40)
        fspa = self.positional_encoding(M)
        return fspa