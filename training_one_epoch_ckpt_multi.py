import torch
from utils import AverageMeter,warmup_learning_rate
import sys
import time
import numpy as np
from utils_supcon import set_loader
from config_linear import parse_option
from utils import set_loader_new, set_model, set_optimizer, adjust_learning_rate, accuracy_multilabel
from sklearn.metrics import average_precision_score,roc_auc_score,f1_score

def train_OCT_multilabel(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #top1 = AverageMeter()
    correct = AverageMeter()
    total = AverageMeter()
    
    device = opt.device
    end = time.time()
    
    #for idx, (image, bio_tensor,eye_id,bcva,cst,patient) in enumerate(train_loader):
    for idx, (image, bio_tensor) in enumerate(train_loader):        #edited
        data_time.update(time.time() - end)

        images = image.to(device)

        labels = bio_tensor
        labels = labels.float()
        bsz = labels.shape[0]
        labels=labels.to(device)
        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)

        output = classifier(features.detach())
        #print(output.shape,labels.shape)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Accuracy         
        pred_labels = ((output)>=0.5)*1
        correct_count = torch.sum((labels == pred_labels)*1).detach().cpu().item()
        total_count =  torch.numel(labels)
        correct.update(1,correct_count)
        total.update(1,total_count)
        #print(pred_labels)
                
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #print(pred_labels)
        #break
        #print(correct.count,total.count)
        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'.format(
                epoch, idx + 1, len(train_loader)))
            sys.stdout.flush()

    return losses.avg, correct.count/total.count

def submission_generate(val_loader, model, classifier, opt):
    """validation"""
    model.eval()
    classifier.eval()

    device = opt.device
    out_list = []
    with torch.no_grad():
        for idx, image in (enumerate(val_loader)):

            images = image.float().to(device)

            # forward
            output = classifier(model.encoder(images))
            output = ((output)>=0.5)*1        #threshold = 0.5
            #output = torch.round(torch.sigmoid(output))
            out_list.append(output.squeeze().detach().cpu().numpy())


    out_submisison = np.array(out_list)
    np.save('output',out_submisison)


def validate_multilabel(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()
    device = opt.device
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    label_list = []
    out_list = []
    with torch.no_grad():
        end = time.time()
        #for idx, (image, bio_tensor,eye_id,bcva,cst,patient) in enumerate(val_loader):
        for idx, (image, bio_tensor) in enumerate(val_loader):
            images = image.float().to(device)

            labels = bio_tensor
            labels = labels.float()
            print(idx)
            label_list.append(labels.detach().cpu().numpy())
            labels = labels.to(device)
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))

            loss = criterion(output, labels)

            out_list.append(output.detach().cpu().numpy())
            # update metric
            losses.update(loss.item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


    label_array = np.concatenate(label_list, axis=0)

    out_array = np.concatenate(out_list, axis=0)
    r = roc_auc_score(label_array, out_array, multi_class='ovr',average='weighted')
    f1 = f1_score(label_array, out_array, average='micro')


    b6 = f1_score(label_array[:, 5], out_array[:, 5], average='micro')
    b5 = f1_score(label_array[:, 4], out_array[:, 4], average='micro')
    b4 = f1_score(label_array[:, 3], out_array[:, 3], average='micro')
    b3 = f1_score(label_array[:, 2], out_array[:, 2], average='micro')
    b2 = f1_score(label_array[:, 1], out_array[:, 1], average='micro')
    b1 = f1_score(label_array[:, 0], out_array[:, 0], average='micro')

    overall = (b1+b2+b3+b4+b5+b6) / 6
    #print('Partial_Vit ' + str(average_precision_score(label_array[:, 4], out_array[:, 4], average='micro')))
    #print('Full_Vit ' + str(average_precision_score(label_array[:, 3], out_array[:, 3], average='micro')))
    #print('IR HRF ' + str(average_precision_score(label_array[:, 2], out_array[:, 2], average='micro')))
    #print('DME ' + str(average_precision_score(label_array[:, 1], out_array[:, 1], average='micro')))
    #print('Fluid IRF ' + str(average_precision_score(label_array[:, 0], out_array[:, 0], average='micro')))
    print(overall, f1)
    return losses.avg, r, overall
def main_multilabel():
    best_acc = 0
    opt = parse_option()

    # build data loader
    device = opt.device
    
    train_loader,  test_loader = set_loader_new(opt)
    #train_loader = set_loader(opt)

    acc_list = []
    prec_list = []
    rec_list = []
    # training routine
    for i in range(0,1):
        model, classifier, criterion = set_model(opt)

        optimizer = set_optimizer(opt, classifier)
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss, acc = train_OCT_multilabel(train_loader, model, classifier, criterion,
                              optimizer, epoch, opt)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
                epoch, time2 - time1, acc))

    # eval for one epoch
        #loss, r,overall = validate_multilabel(test_loader, model, classifier, criterion, opt)

        #acc_list.append(test_acc)
        submission_generate(test_loader, model, classifier, opt)

    '''
    with open(opt.results_dir, "a") as file:
        # Writing data to a file
        file.write(opt.ckpt + '\n')
        file.write(opt.biomarker + '\n')
        file.write(opt.train_csv_path + '\n')
        file.write('AUROC: ' + str(sum(acc_list)) + '\n')
        file.write('Par_vit: ' + str(par_vit) + '\n')
        file.write('Full_vit: ' + str(full_vit) + '\n')
        file.write('IR_HRF: ' + str(ir_hrf) + '\n')
        file.write('DME: ' + str(dme) + '\n')
        file.write('Fluid_irf: ' + str(fluid_irf) + '\n')
        file.write('Overall: ' + str(overall) + '\n')
    '''
