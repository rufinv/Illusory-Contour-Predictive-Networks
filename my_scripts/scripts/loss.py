import numpy as np
import torch.nn.functional as F

def ae_loss(aeloss, num_layers=3, timesteps=10): 
    loss_total = 0.0
    loss_detail = np.zeros((num_layers,timesteps))
    loss_layer = []    
    for i in range (num_layers):
        loss = 0.0
        for t in range(timesteps): # 0-9
            lname = lname = f'loss_layer{i}_time{t}' 
            loss_detail[i,t] = aeloss[lname] 
            loss += aeloss[lname]
        loss_layer.append(loss/timesteps)
    loss_total = sum(loss_layer)/num_layers  
    return loss_total, loss_layer, loss_detail  

def cc_loss(predictions, label,time=10):
    ccloss=0.
    ccloss_detail = []
    
    loss_running=0.
    for t in range(time):  
        cname = 'Classification_at_time{}'.format(t)
        loss_running = F.cross_entropy(predictions[cname],label)
        ccloss_detail.append(loss_running)
        ccloss +=  loss_running
    ccloss = ccloss/time   
    
    return ccloss,ccloss_detail