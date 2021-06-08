import torch
import torch.nn as nn
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms

import sys
sys.path.append('/home/pang/Desktop/Proj_1PCoder/my_scripts')
from scripts.mydata import MyDataset1
from scripts.architecture import PredictiveCoder, PC_Params
from scripts.train_process import pretrain_process, finetune_process


def pretraining(device,dataroot,trainbatch,valbatch,timesteps=10, netnumber = 1, max_epoch = 500, CheckpointFlag1 = False, checkpoint = None):

    # Create model instance
    pc_params = PC_Params(in_channels=[3,128,128],out_features=[128,128,128],num_layers=3,timesteps=timesteps,alpha=0.1,beta=0.2,lambdax=0.1)
    net       = PredictiveCoder(pc_params)
    if CheckpointFlag1:
        net.load_state_dict(checkpoint['model_state_dict'])   
    net.to(device)

    # load data
    transform_train = transforms.Compose([
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomCrop(32, padding=4), 
                      transforms.ToTensor()])                                     
    transform_val   = transforms.Compose([transforms.ToTensor()])             
    train_sets      = CIFAR100(dataroot,train=True,download=True,transform=transform_train)
    val_sets        = CIFAR100(dataroot,train=False,download=True,transform=transform_val)
    train_loader    = torch.utils.data.DataLoader(train_sets, batch_size=trainbatch, shuffle=True, num_workers=4, drop_last=False)
    val_loader      = torch.utils.data.DataLoader(val_sets, batch_size=valbatch, shuffle=False, num_workers=4, drop_last=False)
    class_name      = train_sets.classes
    
    # pretrain the network with cifar100 dataset
    if CheckpointFlag1:
        pretrain_process(pc_params, net,train_loader,val_loader,device, netnumber, max_epoch=max_epoch, CheckpointFlag1=CheckpointFlag1,checkpoint=checkpoint)
    else:
        pretrain_process(pc_params, net,train_loader,val_loader,device, netnumber, max_epoch=max_epoch, CheckpointFlag1=CheckpointFlag1)

        
        
def finetuning(device,alpha,train_root,valid_root,root_path,trainbatch,valbatch,timesteps=10,net1number = 1,net2number=11,max_epoch = 25,CheckpointFlag1 = False,checkpoint1 = None,checkpoint2=None):
   
    # Create a net instance
    pc_params = PC_Params(in_channels=[3,128,128],out_features=[128,128,128],num_layers=3,timesteps=timesteps,alpha=alpha,beta=0.2,lambdax=0.1)
    net       = PredictiveCoder(pc_params)
    if CheckpointFlag1:        
        net.load_state_dict(checkpoint2['model_state_dict'])   
    else:        
        net.load_state_dict(checkpoint1['model_state_dict'])
        net.fc3 = nn.Linear(128,2)
    net.to(device)
    
    # Load data
    transform_train = transforms.Compose([transforms.ToTensor(),])
    transform_val   = transforms.Compose([transforms.ToTensor(),])    
    train_sets      = MyDataset1(train_root,transform_train)
    val_sets        = MyDataset1(valid_root,transform_val)
    trainset_size   = len(train_sets)
    valset_size     = len(val_sets)
    train_loader    = torch.utils.data.DataLoader(train_sets, batch_size=trainbatch, shuffle=True,  num_workers=4, drop_last=False)
    val_loader      = torch.utils.data.DataLoader(val_sets,   batch_size=valbatch,   shuffle=False, num_workers=4, drop_last=False)
    
    # finetune the network with shapes dataset
    if CheckpointFlag1:
        finetune_process(pc_params, net,train_loader,trainset_size,val_loader,valset_size,device, net2number,root_path, max_epoch=max_epoch, CheckpointFlag1=CheckpointFlag1,checkpoint=checkpoint)
    else:
        finetune_process(pc_params, net,train_loader,trainset_size,val_loader,valset_size,device, net2number,root_path,max_epoch=max_epoch, CheckpointFlag1=CheckpointFlag1)

