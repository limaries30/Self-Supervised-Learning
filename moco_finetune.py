import argparse
import torch
import torchvision.models as models
from model import MOCO
from sacred import Experiment
from sacred.observers import MongoObserver
from dataloader import create_dataloader,train_transform
import torch.nn as nn
from tqdm import tqdm
from utils import accuracy, save_checkpoint

'''sacred setup'''
ex = Experiment('MOCO Finetune')

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
        'moco-path':'model/337_99.pth.tar',
        'batch_size':512,
        'lr':0.03,
        'lr-momentum':0.9,
        'lr-wd':1e-4
    }



@ex.automain
def main(config):

    num_epochs = config['num_epochs']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(config['moco-path'],map_location='cpu')


    model = models.__dict__[config['arch']]().to(device)
    for name,param in model.named_parameters():
        if name not in ['fc.weight','fc.bias']:
            param.required_grad = False

    model.fc.weight.data.normal_(mean=0.0,std=0.01)
    model.fc.bias.data.zero_()

    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
            state_dict[k[len('encoder_q.'):]] = state_dict[k]
        del state_dict[k]
    model.load_state_dict(state_dict,strict=False)

    criterion = nn.CrossEntropyLoss().to(device)
    #
    train_set = create_dataloader(config['dataset'],train_transform)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size = config['batch_size'],drop_last=True)

    val_set = create_dataloader(config['dataset'],train_transform,isTrain=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config['batch_size'], drop_last=True)

    fc_params = list(filter(lambda p:p.requires_grad,model.parameters()))
    optimizer = torch.optim.SGD(fc_params, config['lr'],
                                 momentum=config['lr-momentum'],
                                 weight_decay=config['lr-wd'])
    episode_length = len(train_loader)
    epoch_acc1_best = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_avg_acc1 = 0
        with tqdm(train_loader,unit="batch") as tepoch:
            for img,label in tepoch: # don't need to use label
                label = label.to(device)
                output = model(img.to(device))
                loss = criterion(output,label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                tepoch.set_postfix(epoch = epoch,loss=loss.item())

                acc1,acc5 = accuracy(output,label,topk=(1,5))
                epoch_avg_acc1 += acc1
                tepoch.set_postfix(epoch = epoch,loss=loss.item(),acc1=acc1.item(),acc5=acc5.item())

        if epoch%10 == 0:
            with torch.no_grad():
                val_loss = 0
                val_acc_1 = 0
                val_acc_5 = 0
                val_length = len(val_loader)
                for i, (img, label) in enumerate(val_loader):
                    img = img.to(device)
                    label = label.to(device)
                    output = model(img)
                    loss = criterion(output, label)

                    val_loss += loss.item()
                    acc1, acc5 = accuracy(output, label, topk=(1, 5))
                    val_acc_1 += acc1
                    val_acc_5 += acc5
                val_loss = val_loss / val_length
                val_acc_1 = val_acc_1 / val_length
                val_acc_5 = val_acc_5 / val_length
                print(f'val_loss:{val_loss},val_acc_1:{val_acc_1},val_acc_5:{val_acc_5}')

            #epoch_avg_acc1 = epoch_avg_acc1/episode_length
            # if epoch_acc1_best  < epoch_avg_acc1  :
            #     epoch_acc1_best = epoch_avg_acc1
            #     save_checkpoint(
            #         {
            #             'epoch':epoch,
            #             'arch':config['arch'],
            #             'state_dict':moco.state_dict(),
            #             'optimizer':optimizer.state_dict()
            #         },
            #         filename=f'model/{epoch}_{int(epoch_acc1_best)}.pth.tar'
            #     )








