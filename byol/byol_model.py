import torch
import torch.nn as nn


class BYOL(nn.Module):
    def __init__(self, encoder, feature_dim, projection_dim, tau, device):
        super(BYOL, self).__init__()
        self.online_network = encoder(num_classes=feature_dim).to(device)
        self.target_network = encoder(num_classes=feature_dim).to(device)

        in_features = self.online_network.fc.in_features
        self.online_network.fc = nn.Identity()
        self.target_network.fc = nn.Identity()
        self.projection_dim = projection_dim
        self.projection_online = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )
        self.projection_target = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )

        self.tau = tau

        self.device = device

    def normalized_loss(self, q, z):

        q = nn.functional.normalize(q, dim=1)
        z = nn.functional.normalize(z, dim=1)

        loss = 2 - 2 * (q * z).sum(dim=1)

        return loss

    def update(self):
        self.momentum_update(self.online_network, self.target_network)
        self.momentum_update(self.projection_online, self.projection_target)

    def momentum_update(self, online, target):
        for param_o, param_t in zip(online.parameters(),target.parameters()):
            param_t.data = param_t.data * self.tau + param_o * (1.0 - self.tau)

    def forward(self, img_1, img_2):

        z_1 = self.online_network(img_1)
        z_2 = self.online_network(img_2)

        with torch.no_grad():
            target_z_1 = self.target_network(img_1).detach()
            target_z_2 = self.target_network(img_2).detach()
            target_z_1 = self.projection_target(target_z_1).detach()
            target_z_2 = self.projection_target(target_z_2).detach()

        h_1 = self.projection_online(z_1)
        h_2 = self.projection_online(z_2)

        return h_1, h_2, target_z_1, target_z_2
