import torch
import torch.nn as nn
from tqdm import tqdm
import math
import numpy as np
from .positional_encoding import PositionalEncoding
from .pretrainedPSPNet.seg import get_coord

class SpatialRelationModule(nn.Module):
    def __init__(self, d_model):
        super(SpatialRelationModule, self).__init__()
        self.d_model = d_model

    def get_spatial_vector(self, rois, anatomical_parts):
        M = np.empty((0,40), float)
        for roi in rois:
            Mp = np.empty(40)
            # import pdb; pdb.set_trace()
            image_id = int(roi[0])
            roi = roi[1:] # tensor([  0.0000,  55.4688, 300.0000, 300.0000, 710.9375], device='cuda:0')
            wl = anatomical_parts[image_id]["right_lung"][2] - anatomical_parts[image_id]["left_lung"][0]
            hl = anatomical_parts[image_id]["right_lung"][3] - anatomical_parts[image_id]["left_lung"][1]
            for idx, part in enumerate(anatomical_parts[image_id]):
                ana_part_coord = anatomical_parts[image_id][part]
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

    def positional_encoding(self, M): # NOT YET DEBUGGED
        fspa = np.empty((M.shape[0],0), float)
        for j in range(self.d_model):
            _M = M * 1/(math.pow(1000, j/self.d_model))
            fspa = np.concatenate((fspa, np.concatenate((np.sin(_M), np.cos(_M)), axis=1)), axis=1)
        return fspa

    def forward(self, rois, img_metas):
        image_paths = [item["filename"] for item in img_metas]
        image_size = img_metas[0]["img_shape"][:2]
        res = []
        for image_path in image_paths:
            res.append(get_coord(image_path, image_size))
        M = self.get_spatial_vector(rois, res)
        # M.shape = (4096, 40)
        fspa = self.positional_encoding(M)
        return fspa