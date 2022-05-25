from dataloader import SampleTwoImg,create_dataloader,train_transform
from .byol_model import BYOL
from sacred import Experiment
from tqdm import tqdm
import torchvision.models as models
import torch
import torch.nn as nn
from utils import accuracy,save_checkpoint


'''sacred setup'''
ex = Experiment('BYOL TEST')

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
        'tau':0.9,
        'lr':0.03,
        'lr-momentum':0.9,
        'lr-wd':1e-4
    }


@ex.automain
def main(config):

    num_epochs = config['num_epochs']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    byol = BYOL(models.__dict__[config['arch']],config['feature_dim'],config['projection_dim'],config['tau'],device).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    train_set = create_dataloader(config['dataset'],SampleTwoImg(train_transform))
    train_loader = torch.utils.data.DataLoader(train_set,batch_size = config['batch_size'],drop_last=True)
    optimizer = torch.optim.SGD(byol.parameters(), config['lr'],
                                momentum=config['lr-momentum'],
                                weight_decay=config['lr-wd'])
    episode_length = len(train_loader)
    epoch_loss_best = 1000

    for epoch in range(num_epochs):
        epoch_loss = 0
        with tqdm(train_loader,unit="batch") as tepoch:
            for img,_ in tepoch: # don't need to use label

                img_1,img_2 = img
                z_1,z_2,h_1,h_2 = byol(img_1.to(device),img_2.to(device))
                loss = (byol.normalized_loss(z_1, h_2) + byol.normalized_loss(z_2, h_1)).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                byol.update()
                tepoch.set_postfix(epoch = epoch,loss=loss.item())

                #acc1,acc5 = accuracy(output,target,topk=(1,5))
                #epoch_avg_acc1 += acc1
                #tepoch.set_postfix(epoch = epoch,loss=loss.item(),acc1=acc1.item(),acc5=acc5.item())

            #epoch_avg_acc1 = epoch_avg_acc1/episode_length
            if epoch_loss  < epoch_loss_best  :
                epoch_loss_best = epoch_loss
                save_checkpoint(
                    {
                        'epoch':epoch,
                        'arch':config['arch'],
                        'state_dict':byol.state_dict(),
                        'optimizer':optimizer.state_dict()
                    },
                    filename=f'model/byol_{epoch}_{int(epoch_loss_best)}.pth.tar'
                )
