import torchvision.datasets as datasets
from torchvision import datasets, transforms

normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ]
)

class SampleTwoImg:
    def __init__(self,transform):
        self.transform = transform

    def __call__(self, x):
        img_1 = self.transform(x)
        img_2 = self.transform(x)
        return img_1,img_2

def create_dataloader(dataset_name,transformer,isTrain=True):
    dataset = datasets.__dict__[dataset_name]("dataset",train=isTrain,download=True,transform = transformer)
    return dataset


