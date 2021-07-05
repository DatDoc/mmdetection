import torch
import torch.nn as nn
from tqdm import tqdm
import math
import os
import copy
import numpy as np
from .seg import get_coord
from mmdet.models.builder import SAR_MODULES

@SAR_MODULES.register_module()
class SpatialRelationModule(nn.Module):
    def __init__(self, img_prefix, d_model, pretrainedPSPNet, batch_size):
        super(SpatialRelationModule, self).__init__()
        self.d_model = d_model
        self.anatomical_dict = get_coord(img_prefix, pretrainedPSPNet, batch_size)

    def compute_Mp_vector(self, roi, _anatomical_part):
        Mp = np.empty(40, float)
        for idx, part in enumerate(_anatomical_part):
            ana_part_coord = _anatomical_part[part]
            ana_width = ana_part_coord[2] - ana_part_coord[0]
            ana_height = ana_part_coord[3] - ana_part_coord[1]
            Mp[idx*self.d_model+0] = (roi[0] - ana_part_coord[0]) / ana_width
            Mp[idx*self.d_model+1] = (roi[1] - ana_part_coord[1]) / ana_height
            Mp[idx*self.d_model+2] = (roi[0] - ana_part_coord[2]) / ana_width
            Mp[idx*self.d_model+3] = (roi[1] - ana_part_coord[3]) / ana_height
            Mp[idx*self.d_model+4] = (roi[2] - ana_part_coord[0]) / ana_width
            Mp[idx*self.d_model+5] = (roi[3] - ana_part_coord[1]) / ana_height
            Mp[idx*self.d_model+6] = (roi[2] - ana_part_coord[2]) / ana_width
            Mp[idx*self.d_model+7] = (roi[3] - ana_part_coord[3]) / ana_height
        return Mp

    def get_spatial_vector(self, rois, anatomical_parts, image_sizes):
        M = np.empty((0,40), float) # size = [n, 40]
        for idx, anatomical_part in enumerate(anatomical_parts):
            image_rois = rois[rois[:,0]==float(idx)]
            image_width, image_height = image_sizes[idx][:2]
            _anatomical_part = copy.deepcopy(anatomical_part)
            for part in _anatomical_part:
                _anatomical_part[part][0] *= image_width
                _anatomical_part[part][1] *= image_height
                _anatomical_part[part][2] *= image_width
                _anatomical_part[part][3] *= image_height
            # wl = _anatomical_part["right_lung"][2] - _anatomical_part["left_lung"][0]
            # hl = _anatomical_part["right_lung"][3] - _anatomical_part["left_lung"][1]
            
            image_rois = image_rois.detach().cpu().numpy()

            Mp = np.apply_along_axis(
                self.compute_Mp_vector, 1, image_rois[:,1:], _anatomical_part)
            M = np.append(M, Mp, axis=0)
        return M

    def positional_encoding(self, M):
        fspa = np.empty((M.shape[0],0), float)
        for j in range(self.d_model):
            _M = M * 1/(np.power(1000, j/self.d_model))
            fspa = np.concatenate((fspa, np.concatenate((np.sin(_M), np.cos(_M)), axis=1)), axis=1)
        return fspa

    def forward(self, rois, img_metas):
        image_paths = [item["filename"] for item in img_metas]
        image_sizes = [item["img_shape"] for item in img_metas]
        res = [self.anatomical_dict[image_path] for image_path in image_paths]
        M = self.get_spatial_vector(rois, res, image_sizes)
        # M.shape = (512, 40) if batch_sise = 1
        fspa = self.positional_encoding(M)
        return fspa