from torchvision import transforms, datasets
from oct_dataset import OCTDataset, TREX_NEW
from biomarker import BiomarkerDatasetAttributes
from utils import TwoCropTransform
from prime import PrimeDatasetAttributes
from prime_trex_combined import CombinedDataset
#from recovery import recovery
from trex import TREX

#from oct_cluster import OCTDatasetCluster
import torch
from resnet import SupConResNet
from loss import SupConLoss
import torch.backends.cudnn as cudnn
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass
import torch.nn as nn
def set_model_contrast(opt):


    model = SupConResNet(name=opt.model)

    criterion = SupConLoss(temperature=opt.temp,device=opt.device)
    device = opt.device
    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if opt.parallel == 1:
            model = torch.nn.DataParallel(model)
            model = model.cuda()
            criterion = criterion.cuda()
        else:
            model = model.to(device)
            criterion = criterion.to(device)
        cudnn.benchmark = True

    return model, criterion


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'OCT':

        mean = (.1904)
        std = (.2088)
    elif opt.dataset == 'Recovery' or opt.dataset == 'Recovery_Compressed':

        mean = (.1706)
        std = (.2112)
    elif opt.dataset == 'TREX_DME' or opt.dataset == 'Prime_TREX_DME_Fixed' \
            or opt.dataset == 'Prime_TREX_Alpha' or opt.dataset == 'Prime_TREX_DME_Discrete' \
            or opt.dataset == 'Patient_Split_2_Prime_TREX' or opt.dataset == 'Patient_Split_3_Prime_TREX':
        mean = (.1706)
        std = (.2112)
    elif opt.dataset == 'path':

        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)


    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])



    if opt.dataset =='OCT' :        
        csv_path_train = opt.train_csv_path
        data_path_train = opt.train_image_path
        train_dataset = OCTDataset(csv_path_train,data_path_train,transforms = TwoCropTransform(train_transform))
    elif opt.dataset == 'Prime_TREX_DME_Fixed':      #edited prime_trex part
        csv_path_train = opt.train_csv_path
        data_path_train = opt.train_image_path
        train_dataset = TREX_NEW(csv_path_train,data_path_train,transforms = TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler,drop_last=True)

    return train_loader


def set_model(opt):

    model = SupConResNet(name=opt.model)

    criterion = SupConLoss(temperature=opt.temp,device=opt.device)
    device = opt.device
    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if opt.parallel == 1:
            model = torch.nn.DataParallel(model)
            model = model.cuda()
            criterion = criterion.cuda()
        else:
            model = model.to(device)
            criterion = criterion.to(device)
        cudnn.benchmark = True

    return model, criterion
