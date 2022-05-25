import torch
import torch.nn as nn

class MOCO(nn.Module):

    def __init__(self,encoder,feature_dim,momentum_coef,softmax_temp,device,queue_size = 2000):
        '''
        encoder : backbone e.g.,Resnet50
        feature_dim
        '''
        super(MOCO,self).__init__()

        self.encoder_q = encoder(num_classes=feature_dim)
        self.encoder_k = encoder(num_classes=feature_dim)

        self.m = momentum_coef
        self.T = softmax_temp

        # queue
        self.queue_size = queue_size
        self.register_buffer('queue',torch.randn(feature_dim,queue_size))
        self.queue = nn.functional.normalize(self.queue,dim=0)
        self.register_buffer('queue_ptr',torch.zeros(1,dtype=torch.long))
        self.device =  device

    def momentum_update(self):

        for param_q,param_k in zip(self.encoder_q.parameters(),self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q *(1. - self.m)

    def deq_enq(self,keys):

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        left = min(self.queue_size,ptr+batch_size)
        self.queue[:,ptr:left] = keys.T[:,:left-ptr]  # transpose feature_dim x bathc_size
        self.queue_ptr[0] = (ptr+batch_size)%self.queue_size

    def forward(self,img_q,img_k):

        q = self.encoder_q(img_q)
        q = nn.functional.normalize(q, dim=1)
        k = self.encoder_k(img_k)
        k = nn.functional.normalize(k, dim=1)

        # k is positive sample, queue is negative sample
        l_pos = torch.einsum('nc,nc->n',[q, k]).unsqueeze(-1)  # n x 1
        l_neg = torch.einsum('nc,ck->nk',[q, self.queue.detach().clone()]) #n x k
        logits = torch.cat([l_pos,l_neg],dim=1).to(self.device) /self.T  # n x (k+1)
        labels = torch.zeros(logits.shape[0],dtype=torch.long).to(self.device) # n : the fisrt element (l_pos) is true label among multi class
        self.deq_enq(k)

        return logits,labels


