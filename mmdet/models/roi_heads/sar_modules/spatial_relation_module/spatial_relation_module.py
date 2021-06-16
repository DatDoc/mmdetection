import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding

class SpatialRelationModule(nn.Module):
    def __init__(self, d_model, max_len):
        super(SpatialRelationModule, self).__init__()

        self.pos_encoder = PositionalEncoding(d_model, drop_out, max_len)
        print('checkpoint 1: show positional encoding')
        print(self.pos_encoder.pe)
        

    def forward_train(self, rois):

        return 