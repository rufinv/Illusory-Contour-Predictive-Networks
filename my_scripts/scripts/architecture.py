import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d, ConvTranspose2d, Sequential, BatchNorm2d


class PC_Params():
    def __init__(self, in_channels,out_features,num_layers,timesteps,alpha,beta,lambdax,dropout=0.):
        self.in_channels  = in_channels
        self.out_features = out_features
        self.num_layers   = num_layers
        self.timesteps    = timesteps
        self.alpha        = alpha
        self.beta         = beta
        self.lambdax      = lambdax
        self.dropout      = dropout

class PredictiveCoder(nn.Module):
    def __init__(self, pc_params):
        super(PredictiveCoder,self).__init__()

        '''
        #Arguments
            in_channels: The input channels for each layer. The first in_channel is 3 for RGB.
            out_features: number of features/kernals at each layer.
            alpha: gradient correction step.
            beta:  feedforward weight.
            lambdax: feedback weight.
            timesteps: number of pc_loops one intend to do.
            
        #Returns
            self.all_layers: A dictionary of all layers, including encoding layers, decoding layers in all timesteps.
            self.autoen_loss: A dictionary of auto_encoding losses to compile losses
            self.grad: A dictionary of gradients for encoding layers
            self.class_time: A dictionary of classification for each timestep. e.g. 10
        '''     

        # Parameters
        self.num_layers = pc_params.num_layers
        self.timesteps  = pc_params.timesteps
        self.alpha      = pc_params.alpha
        self.beta       = pc_params.beta
        self.lambdax    = pc_params.lambdax
        self.dropout    = pc_params.dropout
        
        # Returned data
        self.all_layers  = {}  # enco_layer{0123}_time{0-9} // deco_layer{210}_time{0-9}
        self.autoen_loss = {}  # layer0, 1, 2
        self.grad        = {}
        self.class_time  = {}  # From 0 to 9     
        

        # Layers
        self.convs   = nn.ModuleList([Conv2d(pc_params.in_channels[x], pc_params.out_features[x], kernel_size=5, stride=(2,2),padding=2)
                                            for x in range(self.num_layers)])
        self.deconvs = nn.ModuleList([ConvTranspose2d(pc_params.out_features[y], pc_params.in_channels[y], kernel_size=5, stride=(2,2),padding=2,output_padding=1)
                                            for y in range(self.num_layers)])
        self.dropout_layer = nn.Dropout2d(self.dropout)          
        self.bach_normal   = nn.BatchNorm1d(num_features=2048)
        self.fc1 = nn.Linear(2048,256)  
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,100)
     
    def forward(self, input_image):
        
        # FIRST: Make the Encodings for t = 0 // Initialize All Layers// all input picture for all timesteps
        prev_layer = input_image
        for i in range (self.num_layers+1): # layer 0, 1, 2, 3            
            if i == 0:
                for x in range (self.timesteps+1): # Note: time0, 1, .., 9 10
                    ename = 'enco_layer{}_time{}'.format(i,x)
                    self.all_layers[ename] = input_image
            else:
                ename = 'enco_layer{}_time{}'.format(i,0)
                self.all_layers[ename] = F.relu(self.convs[i-1](prev_layer))
                prev_layer = self.all_layers[ename] if self.dropout ==0. else self.dropout_layer(self.all_layers[ename])
                  
        # SECOND: Iterations Over Timesteps.
        for time in range(self.timesteps): # time0, 1, .., 9
            
            # Step 1/5: decoding layers.
            for i in range(self.num_layers-1,-1,-1): # only decode to lower layer 2, 1, 0 (no layer 3)
                ename = 'enco_layer{}_time{}'.format(i+1,time) # 3 2 1
                dname = 'deco_layer{}_time{}'.format(i,time) # 2 1 0
                self.all_layers[dname] = torch.sigmoid(self.deconvs[i](self.all_layers[ename])) if i ==0 else F.relu(self.deconvs[i](self.all_layers[ename]))
                
            # Step 2/5: calculate the loss for each layer.
            for i in range(self.num_layers): # 0 1 2
                ename    = 'enco_layer{}_time{}'.format(i,time)
                dname    = 'deco_layer{}_time{}'.format(i,time)                
                loss_ind = 'loss_layer{}_time{}'.format(i,time)  # 0 1 2
                #assert self.all_layers[ename].shape == self.all_layers[dname].shape;"The shapes are different!!"
                self.autoen_loss[loss_ind] = F.mse_loss(self.all_layers[dname],self.all_layers[ename])
            
            # Step 3/5: calculate the gradients using the loss calculated.       
            for i in range(self.num_layers):
                ename    = 'enco_layer{}_time{}'.format(i+1,time) # 1 2 3
                gname    = 'grad_layer{}_time{}'.format(i+1,time) # 1 2 3
                loss_ind = 'loss_layer{}_time{}'.format(i,time) # 0 1 2      
                layer_below_shape = self.all_layers[f'enco_layer{i}_time{time}'].shape
                K = layer_below_shape[2] * layer_below_shape[3] * layer_below_shape[1]
                C = 25 * layer_below_shape[1]
                #print('scaling factor is', K/math.sqrt(C))
                self.grad[gname] = (self.all_layers[ename].shape[0])*(K/math.sqrt(C)) * self.alpha * torch.autograd.grad(self.autoen_loss[loss_ind],self.all_layers[ename],create_graph=True)[0]
                #self.grad[gname] = self.alpha * torch.autograd.grad(self.autoen_loss[loss_ind],self.all_layers[ename],create_graph=True)[0] 
            
            # Step 4/5: updating the activations using the given equation.
            for i in range(1, self.num_layers+1): # 1 2 3 (without layer0 which is the input image)
                if i<self.num_layers:
                    ename_nt = 'enco_layer{}_time{}'.format(i,time+1) # Activation: current layer at "next time"
                    iname_nt = 'enco_layer{}_time{}'.format(i-1,time+1) # Input: lower layre at "next time"
                    ename_ct = 'enco_layer{}_time{}'.format(i,time) # # Memory: current layer at "current time"
                    dname    = 'deco_layer{}_time{}'.format(i,time) # Note:layer2 1 0 Feedback: "higher" layer at "current time"
                    gname    = 'grad_layer{}_time{}'.format(i,time) # Correction: gradient"current" layer at "current time"                    
                    
                    # Input: from "lower" layer at "next time" 
                    self.all_layers[ename_nt] = F.relu(self.convs[i-1](self.all_layers[iname_nt])) if self.dropout == 0. else F.relu(self.dropout_layer(self.convs[i-1](self.all_layers[iname_nt])))
                                       
                    self.all_layers[ename_nt] = self.beta * self.all_layers[ename_nt]-self.grad[gname]+self.lambdax*self.all_layers[dname]+(1-self.lambdax-self.beta)*self.all_layers[ename_ct]
                   
                elif i == self.num_layers: # without feedback
                    ename_nt = 'enco_layer{}_time{}'.format(i,time+1) # Activation: current layer at "next time"
                    iname_nt = 'enco_layer{}_time{}'.format(i-1,time+1) # Input: from "lower" layer at "next time"
                    ename_ct = 'enco_layer{}_time{}'.format(i,time) # Memory: current layer at "current time"
                    gname    = 'grad_layer{}_time{}'.format(i,time) # Correction: gradient at "current time"

                    self.all_layers[ename_nt] = F.relu(self.convs[i-1](self.all_layers[iname_nt])) if self.dropout == 0. else F.relu(self.dropout_layer(self.convs[i-1](self.all_layers[iname_nt])))
                    
                    self.all_layers[ename_nt] = self.beta*self.all_layers[ename_nt]-self.grad[gname]+(1-self.beta)*self.all_layers[ename_ct]


            # Step 5/5: build the head
            x = self.all_layers['enco_layer{}_time{}'.format(self.num_layers,time)] #Think, output the current time(0-9) or next time (1-10)
            x = x.reshape(x.size(0),-1)  # nn.Flatten(x)| x.view((x.size(0),-1)) | self.bach_normal(x) 
            x = self.bach_normal(x)  
            x = self.fc1(x) 
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)    
            x = self.fc3(x)
            x = F.softmax(x,dim=-1) # TODO 

            out_at_t = 'Classification_at_time{}'.format(time) # time0, 1, 2,...,9
            self.class_time[out_at_t] = x
        
        return self.class_time   # include 10 classification results