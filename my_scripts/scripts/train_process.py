import os
import time
import torch
import datetime
import numpy as np
import torch.optim as optim
from scripts.loss import ae_loss,cc_loss

def pretrain_process(pc_params, net,train_loader,val_loader,device, netnumber, max_epoch=500, CheckpointFlag1=False,checkpoint=None ):
    num_layers = pc_params.num_layers
    timesteps = pc_params.timesteps
    max_epoch = max_epoch

    ### Checkpoint 2/2
    if CheckpointFlag1:
        optimizer = optim.Adam(net.parameters(), lr=0.00005)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #scheduler = checkpoint['scheduler']
        TV_loss = checkpoint['TV_loss']
        Train_layerloss = checkpoint['Train_layerloss']
        Train_detailloss = checkpoint['Train_detailloss']
        Val_detailloss = checkpoint['Val_detailloss']
        start_epoch = checkpoint['start_epoch']
    else:
        optimizer = optim.Adam(net.parameters(), lr=0.00005)
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[5,10],gamma=0.1)
        TV_loss = np.zeros((2,max_epoch)) # for train and val parts along epochs
        Train_detailloss = np.zeros((max_epoch, num_layers, timesteps))
        Train_layerloss = np.zeros((max_epoch, num_layers))
        Val_detailloss = np.zeros((max_epoch, num_layers, timesteps))
        start_epoch = 0

    since = time.time()
    for epoch in torch.arange(start_epoch,max_epoch):   # loop over the dataset multiple times
        print('Epoch {}/{}'.format(epoch+1, max_epoch))
        print('_' * 10)
        
        # Training phase
        net.train()
        microt =datetime.datetime.now()   

        running_T_loss = 0.0 # TV_loss in each epoch 
        running_layerloss = [0.0 for c in range(num_layers)]
        running_T_detailloss = np.zeros((num_layers, timesteps))
        batchs_totalloss = 0.0 # running loss for 5~10 batches in each epoch
        batchs_layerloss = [0.0 for c in range(num_layers)]

        for i, (images,labels) in enumerate(train_loader):
            # forward and backward
            images, labels = images.to(device), labels.to(device) # out_palce method
            optimizer.zero_grad() 
            predictions = net(images)
            loss_total, loss_layer, loss_detail = ae_loss(net.autoen_loss,timesteps=timesteps)   
            loss_total.backward()
            optimizer.step()
            
            # statistics for showing   
            batchs_totalloss += loss_total.item() # item() used on a number  
            batchs_layerloss = [batchs_layerloss[i]+loss_layer[i].item() for i in range(len(loss_layer))]
            if i % 10 == 9:  
                batchs_totalloss /=10
                batchs_layerloss =[c/10 for c in batchs_layerloss]
                print('[%2d %7d %10.7f %10.7f %10.7f %10.7f %30s]' % (epoch +1, i, batchs_totalloss, batchs_layerloss[0], batchs_layerloss[1], batchs_layerloss[2], datetime.datetime.now()-microt))
                batchs_totalloss = 0.0
                batchs_layerloss =  [0.0 for c in batchs_layerloss]
                microt = datetime.datetime.now()
                
                
            # Sum data over batches
            running_T_loss += loss_total.item()
            running_layerloss = [running_layerloss[i]+loss_layer[i].item() for i in range(len(loss_layer))] 
            batch_detailloss = np.zeros((num_layers,timesteps))
            for v in range(num_layers):
                for t in range(timesteps):
                    batch_detailloss[v,t] = loss_detail[v,t].item() 
            running_T_detailloss =  np.sum([running_T_detailloss,batch_detailloss],axis=0) 
            
        # Average over batches in one epoch
        running_layerloss = [running_layerloss[c]/len(train_loader) for c in range(len(loss_layer))]
        running_T_loss /= len(train_loader)
        running_T_detailloss /= len(train_loader)
            
        # save data into one epoch    
        TV_loss[0,epoch] = running_T_loss
        Train_layerloss[epoch,:] = running_layerloss
        Train_detailloss[epoch,:,:] = running_T_detailloss
        print ('LossTotal: {:.7f}   LossLayer: [1]{:.7f} [2]{:.7f}{:.7f} [3]{:.7f}'.format(running_T_loss, running_layerloss[0],running_layerloss[1],running_T_detailloss[2,-1], running_layerloss[2] ))
    
        # Validation phase
        net.eval()
        running_V_loss = 0.0
        running_V_detailloss = np.zeros((num_layers, timesteps))

        for i, (images,labels) in enumerate(val_loader):
            # only forward pass
            images, labels = images.to(device), labels.to(device) # out_palce method
            predictions = net(images)
            loss_total, loss_layer, loss_detail = ae_loss(net.autoen_loss,timesteps=timesteps)
            
            # Sum data over batches
            running_V_loss += loss_total.item()
            batch_detailloss = np.zeros((num_layers,timesteps))
            for v in range(num_layers):
                for t in range(timesteps):
                    batch_detailloss[v,t] = loss_detail[v,t].item() 
            running_V_detailloss =  np.sum([running_V_detailloss,batch_detailloss],axis=0) 
            
        # Average over batches in one epoch   
        running_V_loss /=len(val_loader) 
        running_V_detailloss /= len(val_loader)
        
        # save data into one epoch   
        TV_loss[1,epoch] = running_V_loss
        Val_detailloss[epoch,:,:] = running_V_detailloss
        
        print ('Val LossTotal: {:.7f}  Layer1{:.7f} Layer2{:.7f}'.format(running_V_loss,running_V_detailloss[0,-1],running_V_detailloss[1,-1]))
        #scheduler.step()   
        
        time_eplapsed = time.time() - since
        print('Training complete in {:.0f}m  {:.0f}s'.format(time_eplapsed // 60, time_eplapsed % 60))

  
        if epoch % 10 == 9:
            # save model and variables
            PATH = f'train/train1_t{timesteps}_net{netnumber}/train1_t{timesteps}_net{netnumber}_ep{epoch}.pth'     
            torch.save({'model_state_dict':net.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'start_epoch':epoch+1,
                        'TV_loss': TV_loss,
                        'Train_layerloss':Train_layerloss,
                        'Train_detailloss': Train_detailloss,
                        'Val_detailloss':Val_detailloss
                    }, PATH)        

    # Finish training
    
def finetune_process(pc_params, net,train_loader,trainset_size,val_loader,valset_size,device, netnumber, root_path,max_epoch=30, CheckpointFlag1=False,checkpoint=None ):
    num_layers = pc_params.num_layers
    timesteps = pc_params.timesteps
    max_epoch = max_epoch


    if CheckpointFlag1:
        optimizer = optim.Adam(net.parameters(), lr=0.00005)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        TV_loss = checkpoint['TV_loss']
        TV_acc = checkpoint['TV_acc']
        Train_detail_losscc = checkpoint['Train_detail_losscc']
        Train_detail_acc = checkpoint['Train_detail_acc']
        Val_detail_losscc = checkpoint['Val_detail_losscc']
        Val_detail_acc = checkpoint['Val_detail_acc']        
        start_epoch = checkpoint['start_epoch']
    else:
        optimizer = optim.Adam(net.parameters(),lr=0.00005)
        TV_loss = np.zeros((2,max_epoch))
        TV_acc = np.zeros((2,max_epoch))
        Train_detail_losscc =np.zeros((max_epoch,timesteps))
        Train_detail_acc = np.zeros((max_epoch,timesteps))
        Val_detail_losscc =np.zeros((max_epoch,timesteps))
        Val_detail_acc = np.zeros((max_epoch,timesteps))
        start_epoch = 0

    since = time.time()
    for epoch in torch.arange(start_epoch,max_epoch):
        microt = datetime.datetime.now()

        # Train phase
        net.train()  
        print('Epoch {}/{}'.format(epoch+1, max_epoch))
        print('_' * 10)
        cnt = 0.
        
        running_loss = 0.0
        running_acc = 0.0
        running_detail_losscc = [0.0 for c in range(timesteps)]
        running_detail_acc =  [0.0 for c in range(timesteps)]
        batches_loss = 0.0

        for i, (images,labels,noises) in enumerate(train_loader):
            cnt +=1
            # forward and backward
            images, labels = images.to(device), labels.to(device) #out_palce method
            optimizer.zero_grad()
            outputs = net(images)
            ccloss,ccloss_detail = cc_loss(outputs, labels,time=timesteps)
            ccloss.backward()
            optimizer.step()
            
            # statistics for showing 
            batches_loss += ccloss.item()
            if i % 10 == 9:
                batches_loss = batches_loss/10
                print('[%2d %6d %10.6f %30s]' % (epoch +1, i, batches_loss, datetime.datetime.now()-microt))
                batches_loss =0.0
                microt = datetime.datetime.now()
                #print('deconvs2',list(net.deconvs[2].bias)[0]) # frozen parameters
                #print('fc3',list(net.fc3.bias)[0])   # trainable parameters       
                
            # Sum data over batches (loss and acc)    
            running_loss += ccloss.item()    
            running_detail_losscc = [running_detail_losscc[i]+ccloss_detail[i].item() for i in range(timesteps)]
            # class_acc_detail 
            batch_corr = 0.0
            for t in range(timesteps):
                cname = 'Classification_at_time{}'.format(t)
                _,preds = torch.max(outputs[cname],1)
                running_detail_acc[t] += torch.sum(preds == labels.data)
                batch_corr  +=torch.sum(preds == labels.data)
            batch_corr  /= timesteps
            running_acc += batch_corr 
        
        # Average over batches in one epoch
        running_loss /=cnt
        running_detail_losscc = [running_detail_losscc[c]/cnt for c in range(timesteps)]
        running_acc /= trainset_size
        running_detail_acc =  [running_detail_acc[c]/trainset_size for c in range(timesteps)]
        
        # save data into one epoch  
        TV_loss[0,epoch] = running_loss
        TV_acc[0,epoch] = running_acc
        Train_detail_losscc[epoch,:] = running_detail_losscc
        Train_detail_acc[epoch,:] = running_detail_acc
        print ('[%d] train loss: %.6f  train acc: %.6f' % (epoch+1, running_loss, running_acc))
        
        
        # Validation
        net.eval()  
        cnt =0.
        
        running_loss = 0.0
        running_acc = 0.0
        running_detail_losscc = [0.0 for c in range(timesteps)]
        running_detail_acc =  [0.0 for c in range(timesteps)]
        
        for i, (images,labels,noises) in enumerate(val_loader):
            cnt +=1
            # forward and backward
            images, labels = images.to(device), labels.to(device) #out_palce method
            outputs = net(images)
            ccloss,ccloss_detail = cc_loss(outputs, labels,time=timesteps)
            # Sum data over batches (loss and acc)   
            running_loss += ccloss.item()
            running_detail_losscc = [running_detail_losscc[i]+ccloss_detail[i].item() for i in range(timesteps)]
            # class_acc_detail 
            batch_corr = 0.0
            for t in range(timesteps):
                cname = 'Classification_at_time{}'.format(t)
                _,preds = torch.max(outputs[cname],1)
                running_detail_acc[t] += torch.sum(preds == labels.data)
                batch_corr  +=torch.sum(preds == labels.data)
            batch_corr  /= timesteps
            running_acc += batch_corr 

        # Average over batches in one epoch
        running_loss /=cnt
        running_detail_losscc = [running_detail_losscc[c]/cnt for c in range(timesteps)]
        running_acc /= valset_size
        running_detail_acc =  [running_detail_acc[c]/valset_size for c in range(timesteps)]        

        # save data into one epoch  
        TV_loss[1,epoch] = running_loss
        TV_acc[1,epoch] = running_acc
        Val_detail_losscc[epoch,:] = running_detail_losscc
        Val_detail_acc[epoch,:] = running_detail_acc  
        print('[%d] test loss: %.6f  test acc: %.6f' %(epoch+1, running_loss,running_acc))
        
        # save model and variables
        if epoch % 5 == 4:
            PATH =  os.path.join(root_path,f'train2_t{timesteps}_net{netnumber}_ep{epoch}.pth')
            torch.save({'model_state_dict':net.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'start_epoch':epoch+1,
                        'TV_loss':TV_loss,
                        'TV_acc':TV_acc,
                        'Train_detail_losscc':Train_detail_losscc,
                        'Train_detail_acc':Train_detail_acc,
                        'Val_detail_losscc':Val_detail_losscc,
                        'Val_detail_acc':Val_detail_acc,
                    }, PATH)

            # Training time
            time_eplapsed = time.time() - since
            print('Training in {:.0f}m  {:.0f}s'.format(time_eplapsed // 60, time_eplapsed % 60))