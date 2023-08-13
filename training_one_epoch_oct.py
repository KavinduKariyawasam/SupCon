from utils import AverageMeter,warmup_learning_rate
import time
import torch
import sys
from sklearn.metrics import f1_score

def train_OCT(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    device = opt.device
    end = time.time()

    label_list = []
    output_list = []
    
    for idx, (images, labels,patient) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.to(device)

        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        if opt.method1 == 'SupCon':
            labels=labels.cuda()
            loss = criterion(features, labels)
        elif opt.method1 == 'Patient':
            labels = patient.cuda()
            loss = criterion(features,labels)
        elif opt.method1 == 'Patient_SupCon':
            labels1 = patient.cuda()
            labels2 = labels.cuda()
            loss = criterion(features,labels1) + criterion(features,labels2)
        elif opt.method1 == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method1))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #NEW
        label_list.append(labels.squeeze().detach().cpu().numpy())
        output_list.append(((torch.sigmoid(output)>=0.5)*1).squeeze().detach().cpu().numpy())

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    label_array = np.concatenate(label_list,axis = 0)
    output_array = np.concatenate(output_list,axis = 0)
    f = f1_score(label_array.astype(int),output_array.astype(int),average='macro')
    print(f"Epoch: {epoch}, Loss: {losses.avg:.4f}, F1 Score: {f:.4f}")
    
    return losses.avg
