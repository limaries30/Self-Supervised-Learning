from dataloader import SampleTwoImg,create_dataloader,train_transform
from .simclr_model import SIMCLR
from .simclr_loss import INFO_NCE_LOSS
from sacred import Experiment
from tqdm import tqdm
import torchvision.models as models
import torch
import torch.nn as nn
from utils import accuracy,save_checkpoint



'''sacred setup'''
ex = Experiment('SIMCLR TEST')

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

@ex.config
def config():
    config = {
        'arch':'resnet50',
        'dataset':'CIFAR10',
        'num_epochs': 1000,
        'feature_dim': 128,
        'projection_dim':128,
        'nce_temp':0.07,
        'batch_size':512,
        'lr':0.03,
        'lr-momentum':0.9,
        'lr-wd':1e-4
    }

@ex.automain
def main(config):

    num_epochs = config['num_epochs']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    simclr = SIMCLR(models.__dict__[config['arch']],config['projection_dim'],config['feature_dim'] ,device).to(device)
    info_nce_loss = INFO_NCE_LOSS(device,config['nce_temp'])
    train_set = create_dataloader(config['dataset'],SampleTwoImg(train_transform))
    train_loader = torch.utils.data.DataLoader(train_set,batch_size = config['batch_size'],drop_last=True)
    optimizer = torch.optim.SGD(simclr.parameters(), config['lr'],
                                momentum=config['lr-momentum'],
                                weight_decay=config['lr-wd'])
    episode_length = len(train_loader)
    epoch_acc1_best = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_avg_acc1 = 0
        with tqdm(train_loader,unit="batch") as tepoch:
            for img,_ in tepoch: # don't need to use label

                 img_1,img_2 = img
                 img_1,img_2 = img_1.to(device), img_2.to(device)
                 h_1,z_1,h_2,z_2= simclr(img_1,img_2)

                 loss,output,target = info_nce_loss(z_1,z_2)

                 optimizer.zero_grad()
                 loss.backward()
                 optimizer.step()
                 epoch_loss += loss.item()
                 tepoch.set_postfix(epoch = epoch,loss=loss.item())

                 acc1,acc5 = accuracy(output,target,topk=(1,5))
                 epoch_avg_acc1 += acc1
                 tepoch.set_postfix(epoch = epoch,loss=loss.item(),acc1=acc1.item(),acc5=acc5.item())

            epoch_avg_acc1 = epoch_avg_acc1/episode_length
            if epoch_acc1_best  < epoch_avg_acc1  :
                epoch_acc1_best = epoch_avg_acc1
                save_checkpoint(
                    {
                        'epoch':epoch,
                        'arch':config['arch'],
                        'state_dict':simclr.state_dict(),
                        'optimizer':optimizer.state_dict()
                    },
                    filename=f'model/simclr_{epoch}_{int(epoch_acc1_best)}.pth.tar'
                )