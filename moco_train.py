import argparse
import torch
import torchvision.models as models
from model import MOCO
from sacred import Experiment
from sacred.observers import MongoObserver
from dataloader import create_dataloader,SampleTwoImg,train_transform
import torch.nn as nn
from tqdm import tqdm
from utils import accuracy, save_checkpoint

'''sacred setup'''
ex = Experiment('MOCO TEST')

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
        'moco-momentum':0.5,
        'moco-temp':0.07,
        'batch_size':512,
        'lr':0.03,
        'lr-momentum':0.9,
        'lr-wd':1e-4
    }

@ex.automain
def main(config):

    num_epochs = config['num_epochs']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    moco = MOCO(models.__dict__[config['arch']],config['feature_dim'],config['moco-momentum'] ,config['moco-temp'],device).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    train_set = create_dataloader(config['dataset'],SampleTwoImg(train_transform))
    train_loader = torch.utils.data.DataLoader(train_set,batch_size = config['batch_size'],drop_last=True)
    optimizer = torch.optim.SGD(moco.parameters(), config['lr'],
                                momentum=config['lr-momentum'],
                                weight_decay=config['lr-wd'])
    episode_length = len(train_loader)
    epoch_acc1_best = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_avg_acc1 = 0
        with tqdm(train_loader,unit="batch") as tepoch:
            for img,_ in tepoch: # don't need to use label

                q,k = img
                output , target = moco(q.to(device),k.to(device))
                loss = criterion(output,target)

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
                        'state_dict':moco.state_dict(),
                        'optimizer':optimizer.state_dict()
                    },
                    filename=f'model/{epoch}_{int(epoch_acc1_best)}.pth.tar'
                )








