import torch

def accuracy(output,target,topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        values, indices  = output.topk(maxk,dim=1,largest=True,sorted=True)

        correct = indices.eq(target.view(-1,1))

        res = []
        for k in topk:
            correct_k = correct[:,:k].squeeze().sum()
            res.append(correct_k.mul(100.0/batch_size))
        return res


def save_checkpoint(state,filename='checkpoint.pth.tar'):
    torch.save(state, filename)



