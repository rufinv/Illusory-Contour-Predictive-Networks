import torch
import time
import torch.nn as nn
from scripts.mydata import MyDataset3_fg
import torchvision.transforms as transforms
from scripts.architecture import PredictiveCoder, PC_Params
import numpy as np




def fg_compute(device,checkpoint,dataroot,batch_size,img_num,savepath):
    
    # Create model instance
    timesteps = 100
    pc_params = PC_Params(in_channels=[3,128,128],out_features=[128,128,128],num_layers=3,timesteps=timesteps,alpha=0.1,beta=0.2,lambdax=0.1)
    net       = PredictiveCoder(pc_params)
    net.fc3   = nn.Linear(128,2)
    net.load_state_dict(checkpoint['model_state_dict']) 
    net.to(device)
    
    # Load data
    transform_test = transforms.Compose([transforms.ToTensor(),])
    test_sets      = MyDataset3_fg(dataroot,transform_test)
    testset_size   = len(test_sets)
    test_loader    = torch.utils.data.DataLoader(test_sets, batch_size=batch_size,shuffle=False, num_workers=4, drop_last=False)

    
    # test dataset
    net.eval()  
    since = time.time()

    edge_info_square   = np.zeros((img_num,timesteps+1)) # plus original one
    edge_info_random   = np.zeros((img_num,timesteps+1))
    edge_info_Ics      = np.zeros((img_num,timesteps+1))
    edge_info_Controls = np.zeros((img_num,timesteps+1))
    cnt_Square   = 0
    cnt_Random   = 0
    cnt_Controls = 0
    cnt_Ics      = 0
    for j, (images,labels,imginfo0,imginfo1,imginfo2,imginfo3,imginfo4,imginfo5,imginfo6) in enumerate(test_loader):
        print('batch {}/{}'.format(j+1, len(test_loader)))
        images, labels,imginfo0 = images.to(device), labels.to(device),imginfo0.to(device) #out_palce method
        imginfo1, imginfo2,imginfo3= imginfo1.to(device), imginfo2.to(device),imginfo3.to(device)
        imginfo4, imginfo5,imginfo6= imginfo4.to(device), imginfo5.to(device),imginfo6.to(device)
        outputs = net(images)
        layers = net.all_layers   
        for i in range(len(labels)):
            if labels[i]==0:  # square 
                ename='enco_layer0_time0'
                vv = layers[ename]
                imgori = vv[i,:,:,:]
                edge_info_square[cnt_Square,0] = get_edge_square(imgori,imginfo0[i],imginfo1[i],imginfo2[i],imginfo3[i],imginfo4[i],imginfo5[i],imginfo6[i])
                for t in range(timesteps):
                    dname = f'deco_layer0_time{t}'
                    xx = layers[dname] # batch * channel * weight * hight
                    img = xx[i,:,:,:]
                    edge_info_square[cnt_Square,t+1] = get_edge_square(img,imginfo0[i],imginfo1[i],imginfo2[i],imginfo3[i],imginfo4[i],imginfo5[i],imginfo6[i])
                cnt_Square +=1
            elif labels[i]==1:  # random 
                ename='enco_layer0_time0'
                vv = layers[ename]
                imgori = vv[i,:,:,:]
                edge_info_random[cnt_Random,0] = get_edge_random(imgori,imginfo0[i],imginfo1[i],imginfo2[i],imginfo3[i],imginfo4[i],imginfo5[i],imginfo6[i])
                for t in range(timesteps):
                    dname = f'deco_layer0_time{t}'
                    xx = layers[dname] # batch * channel * weight * hight
                    img = xx[i,:,:,:]
                    edge_info_random[cnt_Random,t+1] = get_edge_random(img,imginfo0[i],imginfo1[i],imginfo2[i],imginfo3[i],imginfo4[i],imginfo5[i],imginfo6[i])
                cnt_Random +=1
            elif labels[i]==2:  # control
                ename='enco_layer0_time0'
                vv = layers[ename]
                imgori = vv[i,:,:,:]
                edge_info_Controls[cnt_Controls,0] = get_edge_allout(imgori,imginfo0[i],imginfo1[i],imginfo2[i],imginfo3[i],imginfo4[i],imginfo5[i],imginfo6[i])
                for t in range(timesteps):
                    dname = f'deco_layer0_time{t}'
                    xx = layers[dname] # batch * channel * weight * hight
                    img = xx[i,:,:,:]
                    edge_info_Controls[cnt_Controls,t+1] = get_edge_allout(img,imginfo0[i],imginfo1[i],imginfo2[i],imginfo3[i],imginfo4[i],imginfo5[i],imginfo6[i])
                cnt_Controls +=1
            elif labels[i]==3:  # ICs
                ename='enco_layer0_time0'
                vv = layers[ename]
                imgori = vv[i,:,:,:]
                edge_info_Ics[cnt_Ics,0] = get_edge_ICs(imgori,imginfo0[i],imginfo1[i],imginfo2[i],imginfo3[i],imginfo4[i],imginfo5[i],imginfo6[i])
                for t in range(timesteps):
                    dname = f'deco_layer0_time{t}'
                    xx = layers[dname] # batch * channel * weight * hight
                    img = xx[i,:,:,:]
                    edge_info_Ics[cnt_Ics,t+1] = get_edge_ICs(img,imginfo0[i],imginfo1[i],imginfo2[i],imginfo3[i],imginfo4[i],imginfo5[i],imginfo6[i])
                cnt_Ics +=1
    #check 4/4
    torch.save({'edge_info_square':edge_info_square,
                'edge_info_random':edge_info_random,
                'edge_info_Ics':edge_info_Ics,
                'edge_info_Controls':edge_info_Controls}, savepath)
    time_eplapsed = time.time() - since
    print('Training complete in {:.0f}m  {:.0f}s'.format(time_eplapsed // 60, time_eplapsed % 60))
    
    
    
    
    


#npixel=1 #0 (one pixel); #1 (total two pixels); #2 (total three pixels)

def get_edge_square(image_in,imginfo0,imginfo1,imginfo2,imginfo3,imginfo4,imginfo5,imginfo6): 
    npixel=1
    #temp_image = np.float64(np.copy(image_in.cpu().detach()))    
    #temp_image = np.float64(np.copy(image_in))  
    imginfo0 = np.float64(imginfo0.cpu().detach())  
    imginfo1 = np.float64(imginfo1.cpu().detach())  
    imginfo2 = np.float64(imginfo2.cpu().detach())  
    imginfo3 = np.float64(imginfo3.cpu().detach())  
    imginfo4 = np.float64(imginfo4.cpu().detach())  
    imginfo5 = np.float64(imginfo5.cpu().detach())  
    imginfo6 = np.float64(imginfo6.cpu().detach())  
    temp_image = np.float64(image_in.cpu().detach())  
    temp_image = temp_image.transpose(1, 2, 0)
    ssize = imginfo1
    row = imginfo3
    col = imginfo4
    color0 = imginfo5
    color1 =imginfo6
    
    
    chan = 3
    edges = 0.0
    for cc in range(chan):
        image = temp_image[:,:,cc]
        edge1 = 0.
        edge2 = 0.
        edge3 = 0.
        edge4 = 0.

        # left edge
        if ssize%2 == 0:
            edge1_in = image[int(row+0.5*ssize-1):int(row+0.5*ssize+1),int(col):int(col+npixel+1)]
            edge1_out = image[int(row+0.5*ssize-1):int(row+0.5*ssize+1),int(col-1-npixel):int(col)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge1 = np.sum(edge1_out-edge1_in)/(2*(npixel+1))
            else: 
                edge1 = np.sum(edge1_in-edge1_out)/(2*(npixel+1))
        else:
            edge1_in = image[int(row+(ssize-1)*0.5),int(col):int(col+npixel+1)]
            edge1_out = image[int(row+(ssize-1)*0.5),int(col-1-npixel):int(col)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge1 = np.sum(edge1_out-edge1_in)/(npixel+1)
            else: 
                edge1 = np.sum(edge1_in-edge1_out)/(npixel+1)

        # top edge
        if ssize%2 == 0:
            edge2_in = image[int(row):int(row+npixel+1),int(col+0.5*ssize-1):int(col+0.5*ssize+1)]
            edge2_out = image[int(row-1-npixel):int(row),int(col+0.5*ssize-1):int(col+0.5*ssize+1)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge2 = np.sum(edge2_out-edge2_in)/(2*(npixel+1))
            else: 
                edge2 = np.sum(edge2_in-edge2_out)/(2*(npixel+1))
        else:
            edge2_in = image[int(row):int(row+npixel+1),int(col+(ssize-1)*0.5)]
            edge2_out = image[int(row-1-npixel):int(row),int(col+(ssize-1)*0.5)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge2 = np.sum(edge2_out-edge2_in)/(npixel+1)
            else: 
                edge2 = np.sum(edge2_in-edge2_out)/(npixel+1)           

        # right edge  
        if ssize%2 == 0:
            edge3_in = image[int(row+0.5*ssize-1):int(row+0.5*ssize+1),int(col+ssize-1-npixel):int(col+ssize)]
            edge3_out = image[int(row+0.5*ssize-1):int(row+0.5*ssize+1),int(col+ssize):int(col+ssize+npixel+1)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge3 = np.sum(edge3_out-edge3_in)/(2*(npixel+1))
            else: 
                edge3 = np.sum(edge3_in-edge3_out)/(2*(npixel+1))
        else:
            edge3_in = image[int(row+(ssize-1)*0.5),int(col+ssize-1-npixel):int(col+ssize)]
            edge3_out = image[int(row+(ssize-1)*0.5),int(col+ssize):int(col+ssize+npixel+1)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge3 = np.sum(edge3_out-edge3_in)/(npixel+1)  
            else: 
                edge3 = np.sum(edge3_in-edge3_out)/(npixel+1)  

        # bottom edge
        if ssize%2 == 0:
            edge4_in = image[int(row+ssize-1-npixel):int(row+ssize),int(col+0.5*ssize-1):int(col+0.5*ssize+1)]
            edge4_out = image[int(row+ssize):int(row+ssize+npixel+1),int(col+0.5*ssize-1):int(col+0.5*ssize+1)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge4 = np.sum(edge4_out-edge4_in)/(2*(npixel+1))
            else: 

                edge4 = np.sum(edge4_in-edge4_out)/(2*(npixel+1))
        else:
            edge4_in = image[int(row+ssize-1-npixel):int(row+ssize),int(col+(ssize-1)*0.5)]
            edge4_out = image[int(row+ssize):int(row+ssize+npixel+1),int(col+(ssize-1)*0.5)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge4 = np.sum(edge4_out-edge4_in)/(npixel+1)  
            else: 
                edge4 = np.sum(edge4_in-edge4_out)/(npixel+1)  

        edges +=(edge1+edge2+edge3+edge4)*0.25
    edge = edges/chan
    return edge


def get_edge_ICs(image_in,imginfo0,imginfo1,imginfo2,imginfo3,imginfo4,imginfo5,imginfo6):   
    npixel=1
    #temp_image = np.float64(np.copy(image_in.cpu().detach()))    
    #temp_image = np.float64(np.copy(image_in))  
    imginfo0 = np.float64(imginfo0.cpu().detach())  
    imginfo1 = np.float64(imginfo1.cpu().detach())  
    imginfo2 = np.float64(imginfo2.cpu().detach())  
    imginfo3 = np.float64(imginfo3.cpu().detach())  
    imginfo4 = np.float64(imginfo4.cpu().detach())  
    imginfo5 = np.float64(imginfo5.cpu().detach())  
    imginfo6 = np.float64(imginfo6.cpu().detach())  
    temp_image = np.float64(image_in.cpu().detach())  
    temp_image = temp_image.transpose(1, 2, 0)
    csize = imginfo0
    ssize = imginfo1
    row = imginfo3
    col = imginfo4
    color0 = imginfo5
    color1 =imginfo6
    
    chan = 3
    edges = 0.0
    for cc in range(chan):
        image = temp_image[:,:,cc]
        edge1 = 0.
        edge2 = 0.
        edge3 = 0.
        edge4 = 0.

        # left edge
        if ssize%2 == 0:
            edge1_in = image[int(row+0.5*ssize+(csize-1)-1):int(row+0.5*ssize+(csize-1)+1),int(col+csize-1):int(col+csize+npixel)]
            edge1_out = image[int(row+0.5*ssize+(csize-1)-1):int(row+0.5*ssize+(csize-1)+1),int(col+csize-2-npixel):int(col+csize-1)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge1 = np.sum(edge1_in-edge1_out)/(2*(npixel+1))
            else: 
                edge1 = np.sum(edge1_out-edge1_in)/(2*(npixel+1))
        else:
            edge1_in = image[int(row+(csize-1)+(ssize-1)*0.5),int(col+csize-1):int(col+csize+npixel)]
            edge1_out = image[int(row+(csize-1)+(ssize-1)*0.5),int(col+csize-2-npixel):int(col+csize-1)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge1 = np.sum(edge1_in-edge1_out)/(npixel+1)
            else: 
                edge1 = np.sum(edge1_out-edge1_in)/(npixel+1)

        # top edge
        if ssize%2 == 0:
            edge2_in = image[int(row+csize-1):int(row+csize+npixel),int(col+0.5*ssize+(csize-1)-1):int(col+0.5*ssize+(csize-1)+1)]
            edge2_out = image[int(row+csize-2-npixel):int(row+csize-1),int(col+0.5*ssize+(csize-1)-1):int(col+0.5*ssize+(csize-1)+1)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge2 = np.sum(edge2_in-edge2_out)/(2*(npixel+1))
            else: 
                edge2 = np.sum(edge2_out-edge2_in)/(2*(npixel+1))
        else:
            edge2_in = image[int(row+csize-1):int(row+csize+npixel),int(col+(csize-1)+(ssize-1)*0.5)]
            edge2_out = image[int(row+csize-2-npixel):int(row+csize-1),int(col+(csize-1)+(ssize-1)*0.5)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge2 = np.sum(edge2_in-edge2_out)/(npixel+1)
            else: 
                edge2 = np.sum(edge2_out-edge2_in)/(npixel+1)           

        # right edge  
        if ssize%2 == 0:
            edge3_in = image[int(row+0.5*ssize+(csize-1)-1):int(row+0.5*ssize+(csize-1)+1),int(col+csize-1+ssize-1-npixel):int(col+csize-1+ssize)]
            edge3_out = image[int(row+0.5*ssize+(csize-1)-1):int(row+0.5*ssize+(csize-1)+1),int(col+csize-1+ssize):int(col+csize+ssize+npixel)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge3 = np.sum(edge3_in-edge3_out)/(2*(npixel+1))
            else: 
                edge3 = np.sum(edge3_out-edge3_in)/(2*(npixel+1))
        else:
            edge3_in = image[int(row+(csize-1)+(ssize-1)*0.5),int(col+csize-1+ssize-1-npixel):int(col+csize-1+ssize)]
            edge3_out = image[int(row+(csize-1)+(ssize-1)*0.5),int(col+csize-1+ssize):int(col+csize+ssize+npixel)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge3 = np.sum(edge3_in-edge3_out)/(npixel+1)  
            else: 
                edge3 = np.sum(edge3_out-edge3_in)/(npixel+1)  

        # bottom edge
        if ssize%2 == 0:
            edge4_in = image[int(row+csize-1+ssize-1-npixel):int(row+csize-1+ssize),int(col+0.5*ssize+(csize-1)-1):int(col+0.5*ssize+(csize-1)+1)]
            edge4_out = image[int(row+csize-1+ssize):int(row+csize+ssize+npixel),int(col+0.5*ssize+(csize-1)-1):int(col+0.5*ssize+(csize-1)+1)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge4 = np.sum(edge4_in-edge4_out)/(2*(npixel+1))
            else: 
                edge4 = np.sum(edge4_out-edge4_in)/(2*(npixel+1))
        else:
            edge4_in = image[int(row+csize-1+ssize-1-npixel):int(row+csize-1+ssize),int(col+(csize-1)+(ssize-1)*0.5)]
            edge4_out = image[int(row+csize-1+ssize):int(row+csize+ssize+npixel),int(col+(csize-1)+(ssize-1)*0.5)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge4 = np.sum(edge4_in-edge4_out)/(npixel+1)  
            else: 
                edge4 = np.sum(edge4_out-edge4_in)/(npixel+1)  

        edges += (edge1+edge2+edge3+edge4)*0.25
    edge = edges/chan
    return edge

def get_edge_allout(image_in,imginfo0,imginfo1,imginfo2,imginfo3,imginfo4,imginfo5,imginfo6): 
    npixel=1
    #temp_image = np.float64(np.copy(image_in.cpu().detach()))    
    #temp_image = np.float64(np.copy(image_in))    
    imginfo0 = np.float64(imginfo0.cpu().detach())  
    imginfo1 = np.float64(imginfo1.cpu().detach())  
    imginfo2 = np.float64(imginfo2.cpu().detach())  
    imginfo3 = np.float64(imginfo3.cpu().detach())  
    imginfo4 = np.float64(imginfo4.cpu().detach())  
    imginfo5 = np.float64(imginfo5.cpu().detach())  
    imginfo6 = np.float64(imginfo6.cpu().detach())  
    temp_image = np.float64(image_in.cpu().detach())  
    temp_image = temp_image.transpose(1, 2, 0)
    csize = imginfo0
    ssize = imginfo1
    row = imginfo3
    col = imginfo4
    color0 = imginfo5
    color1 =imginfo6
    
    chan = 3
    edges = 0.0
    for cc in range(chan):
        image = temp_image[:,:,cc]   
        edge1 = 0.
        edge2 = 0.
        edge3 = 0.
        edge4 = 0.

        # left edge
        if ssize%2 == 0:
            edge1_in = image[int(row+0.5*ssize+(csize-1)-1):int(row+0.5*ssize+(csize-1)+1),int(col+csize):int(col+csize+1+npixel)]
            edge1_out = image[int(row+0.5*ssize+(csize-1)-1):int(row+0.5*ssize+(csize-1)+1),int(col+csize-1-npixel):int(col+csize)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge1 = np.sum(edge1_in-edge1_out)/(2*(npixel+1))
            else: 
                edge1 = np.sum(edge1_out-edge1_in)/(2*(npixel+1))
        else:
            edge1_in = image[int(row+(csize-1)+(ssize-1)*0.5),int(col+csize):int(col+csize+1+npixel)]
            edge1_out = image[int(row+(csize-1)+(ssize-1)*0.5),int(col+csize-1-npixel):int(col+csize)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge1 = np.sum(edge1_in-edge1_out)/(npixel+1) 
            else: 
                edge1 = np.sum(edge1_out-edge1_in)/(npixel+1) 

        # top edge
        if ssize%2 == 0:
            edge2_in = image[int(row+csize):int(row+csize+1+npixel),int(col+0.5*ssize+(csize-1)-1):int(col+0.5*ssize+(csize-1)+1)]
            edge2_out = image[int(row+csize-1-npixel):int(row+csize),int(col+0.5*ssize+(csize-1)-1):int(col+0.5*ssize+(csize-1)+1)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge2 = np.sum(edge2_in-edge2_out)/(2*(npixel+1))
            else: 
                edge2 = np.sum(edge2_out-edge2_in)/(2*(npixel+1))
        else:
            edge2_in = image[int(row+csize):int(row+csize+1+npixel),int(col+(csize-1)+(ssize-1)*0.5)]
            edge2_out = image[int(row+csize-1-npixel):int(row+csize),int(col+(csize-1)+(ssize-1)*0.5)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge2 = np.sum(edge2_in-edge2_out)/(npixel+1) 
            else: 
                edge2 = np.sum(edge2_out-edge2_in)/(npixel+1)           

        # right edge  
        if ssize%2 == 0:
            edge3_in = image[int(row+0.5*ssize+(csize-1)-1):int(row+0.5*ssize+(csize-1)+1),int(col+csize-1+ssize-2-npixel):int(col+csize-1+ssize-1)]
            edge3_out = image[int(row+0.5*ssize+(csize-1)-1):int(row+0.5*ssize+(csize-1)+1),int(col+csize-1+ssize-1):int(col+csize-1+ssize+npixel)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge3 = np.sum(edge3_in-edge3_out)/(2*(npixel+1))
            else: 
                edge3 = np.sum(edge3_out-edge3_in)/(2*(npixel+1))
        else:
            edge3_in = image[int(row+(csize-1)+(ssize-1)*0.5),int(col+csize-1+ssize-2-npixel):int(col+csize-1+ssize-1)]
            edge3_out = image[int(row+(csize-1)+(ssize-1)*0.5),int(col+csize-1+ssize-1):int(col+csize-1+ssize+npixel)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge3 = np.sum(edge3_in-edge3_out)/(npixel+1) 
            else: 
                edge3 = np.sum(edge3_out-edge3_in)/(npixel+1) 

        # bottom edge
        if ssize%2 == 0:
            edge4_in = image[int(row+csize-1+ssize-2-npixel):int(row+csize-1+ssize-1),int(col+0.5*ssize+(csize-1)-1):int(col+0.5*ssize+(csize-1)+1)]
            edge4_out = image[int(row+csize-1+ssize-1):int(row+csize-1+ssize+npixel),int(col+0.5*ssize+(csize-1)-1):int(col+0.5*ssize+(csize-1)+1)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge4 = np.sum(edge4_in-edge4_out)/(2*(npixel+1))
            else: 
                edge4 = np.sum(edge4_out-edge4_in)/(2*(npixel+1))
        else:
            edge4_in = image[int(row+csize-1+ssize-2-npixel):int(row+csize-1+ssize-1),int(col+(csize-1)+(ssize-1)*0.5)]
            edge4_out = image[int(row+csize-1+ssize-1):int(row+csize-1+ssize+npixel),int(col+(csize-1)+(ssize-1)*0.5)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge4 = np.sum(edge4_in-edge4_out)/(npixel+1) 
            else: 
                edge4 = np.sum(edge4_out-edge4_in)/(npixel+1) 

        edges += (edge1+edge2+edge3+edge4)*0.25
    edge = edges/chan
    return edge

def get_edge_random(image_in,imginfo0,imginfo1,imginfo2,imginfo3,imginfo4,imginfo5,imginfo6):   
    npixel=1
    #temp_image = np.float64(np.copy(image_in.cpu().detach()))   
    imginfo0 = np.float64(imginfo0.cpu().detach())  
    imginfo1 = np.float64(imginfo1.cpu().detach())  
    imginfo2 = np.float64(imginfo2.cpu().detach())  
    imginfo3 = np.float64(imginfo3.cpu().detach())  
    imginfo4 = np.float64(imginfo4.cpu().detach())  
    imginfo5 = np.float64(imginfo5.cpu().detach())  
    imginfo6 = np.float64(imginfo6.cpu().detach())  
    temp_image = np.float64(image_in.cpu().detach())  
    temp_image = temp_image.transpose(1, 2, 0)
    csize = int(imginfo0)
    ssize = int(imginfo1)
    row = int(imginfo3)
    col = int(imginfo4)
    color0 = imginfo5
    color1 =imginfo6

    
    chan = 3
    edges = 0.0
    for cc in range(chan):
        image = temp_image[:,:,cc]
        edge1 = 0.
        edge2 = 0.
        edge3 = 0.
        edge4 = 0.

        # left edge
        if ssize%2 == 0:
            edge1_in = image[int(row+0.5*ssize+(csize-1)-1):int(row+0.5*ssize+(csize-1)+1),int(col+csize):int(col+csize+1+npixel)]
            edge1_out = image[int(row+0.5*ssize+(csize-1)-1):int(row+0.5*ssize+(csize-1)+1),int(col+csize-2-npixel):int(col+csize-1)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge1 = np.sum(edge1_in-edge1_out)/(2*(npixel+1))
            else: 
                edge1 = np.sum(edge1_out-edge1_in)/(2*(npixel+1))
        else:
            edge1_in = image[int(row+(csize-1)+(ssize-1)*0.5),int(col+csize):int(col+csize+1+npixel)]
            edge1_out = image[int(row+(csize-1)+(ssize-1)*0.5),int(col+csize-2-npixel):int(col+csize-1)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge1 = np.sum(edge1_in-edge1_out)/(npixel+1) 
            else: 
                edge1 = np.sum(edge1_out-edge1_in)/(npixel+1) 

        # top edge
        if ssize%2 == 0:
            edge2_in = image[int(row+csize):int(row+csize+npixel+1),int(col+0.5*ssize+(csize-1)-1):int(col+0.5*ssize+(csize-1)+1)]
            edge2_out = image[int(row+csize-2-npixel):int(row+csize-1),int(col+0.5*ssize+(csize-1)-1):int(col+0.5*ssize+(csize-1)+1)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge2 = np.sum(edge2_in-edge2_out)/(2*(npixel+1))
            else: 
                edge2 = np.sum(edge2_out-edge2_in)/(2*(npixel+1))
        else:
            edge2_in = image[int(row+csize):int(row+csize+npixel+1),int(col+(csize-1)+(ssize-1)*0.5)]
            edge2_out = image[int(row+csize-2-npixel):int(row+csize-1),int(col+(csize-1)+(ssize-1)*0.5)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge2 = np.sum(edge2_in-edge2_out)/(npixel+1)
            else: 
                edge2 = np.sum(edge2_out-edge2_in)/(npixel+1)          

        # right edge  
        if ssize%2 == 0:
            edge3_in = image[int(row+0.5*ssize+(csize-1)-1):int(row+0.5*ssize+(csize-1)+1),int(col+csize-1+ssize-2-npixel):int(col+csize-1+ssize-1)]
            edge3_out = image[int(row+0.5*ssize+(csize-1)-1):int(row+0.5*ssize+(csize-1)+1),int(col+csize-1+ssize):int(col+csize+ssize+npixel)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge3 = np.sum(edge3_in-edge3_out)/(2*(npixel+1))
            else: 
                edge3 = np.sum(edge3_out-edge3_in)/(2*(npixel+1))
        else:
            edge3_in = image[int(row+(csize-1)+(ssize-1)*0.5),int(col+csize-1+ssize-2-npixel):int(col+csize-1+ssize-1)]
            edge3_out = image[int(row+(csize-1)+(ssize-1)*0.5),int(col+csize-1+ssize):int(col+csize+ssize+npixel)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge3 = np.sum(edge3_in-edge3_out)/(npixel+1) 
            else: 
                edge3 = np.sum(edge3_out-edge3_in)/(npixel+1) 

        # bottom edge
        if ssize%2 == 0:
            edge4_in = image[int(row+csize-1+ssize-2-npixel):int(row+csize-1+ssize-1),int(col+0.5*ssize+(csize-1)-1):int(col+0.5*ssize+(csize-1)+1)]
            edge4_out = image[int(row+csize-1+ssize):int(row+csize-1+ssize+npixel+1),int(col+0.5*ssize+(csize-1)-1):int(col+0.5*ssize+(csize-1)+1)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge4 = np.sum(edge4_in-edge4_out)/(2*(npixel+1))
            else: 
                edge4 = np.sum(edge4_out-edge4_in)/(2*(npixel+1))
        else:
            edge4_in = image[int(row+csize-1+ssize-2-npixel):int(row+csize-1+ssize-1),int(col+(csize-1)+(ssize-1)*0.5)]
            edge4_out = image[int(row+csize-1+ssize):int(row+csize-1+ssize+npixel+1),int(col+(csize-1)+(ssize-1)*0.5)]
            if color0 > color1: # background is lighter, IC is ligher inside
                edge4 = np.sum(edge4_in-edge4_out)/(npixel+1) 
            else: 
                edge4 = np.sum(edge4_out-edge4_in)/(npixel+1) 

        edges += (edge1+edge2+edge3+edge4)*0.25
    edge = edges/chan
    return edge