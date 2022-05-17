import torch
import torch.nn as nn



class SIMCLR(nn.Module):
    def __init__(self,encoder,projection_dim,feature_dim,device):
        super(SIMCLR,self).__init__()

        self.encoder = encoder(num_classes=feature_dim)
        self.feature_dim = feature_dim
        self.encoder_fc_in_feature_dim= self.encoder.fc.in_features

        self.encoder.fc= nn.Identity()
        self.projection_head = nn.Sequential(nn.Linear(self.encoder_fc_in_feature_dim,feature_dim),nn.ReLU(),nn.Linear(feature_dim,projection_dim))

        self.device = device

    def forward(self,img_1,img_2):

        h_1 = self.encoder(img_1)
        z_1 = self.projection_head(h_1)

        h_2 = self.encoder(img_2)
        z_2 = self.projection_head(h_2)

        z_1 = nn.functional.normalize(z_1, dim=1)
        z_2 = nn.functional.normalize(z_2, dim=1)

        return h_1,z_1,h_2,z_2

