'''
motivated from https://github.com/sthalles/SimCLR/blob/master/simclr.py
'''
import torch
import torch.nn as nn
class INFO_NCE_LOSS(nn.Module):
    def __init__(self,device,temperature):
        super(INFO_NCE_LOSS,self).__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.temperature = temperature

    def forward(self,z_1,z_2):

        batch_size = z_1.size()[0]

        positive_indices_second = torch.arange(batch_size)
        positive_labels = torch.zeros((2*batch_size,2*batch_size))
        positive_labels[batch_size+positive_indices_second,positive_indices_second] =1 # (2*batch_size,2*batch_size)
        positive_labels = positive_labels + positive_labels.T

        feature = torch.cat((z_1,z_2),dim=0) # (2*batch_size, feature_dim)
        similarity_matrix = torch.matmul(feature,feature.T) #(2*batch_size,2*batch_size)
        positive_samples = similarity_matrix[positive_labels.bool()].view(2*batch_size,-1) #(2*batch_size,1)
        negative_samples = similarity_matrix[~positive_labels.bool()&~(torch.eye(2*batch_size)).bool()].view(2*batch_size,-1) #(2*batch_size,2*batch-2)

        outputs = torch.cat((positive_samples,negative_samples),dim=1)   / self.temperature
        labels = torch.zeros(2*batch_size,dtype=torch.long).to(self.device) # first index is the true label
        loss = self.criterion(outputs,labels)
        return loss, outputs, labels







