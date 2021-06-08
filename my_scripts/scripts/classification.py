import torch
import time
import torch.nn as nn
from scripts.mydata import MyDataset2_class
import torchvision.transforms as transforms
from scripts.architecture import PredictiveCoder, PC_Params


def classification(device,checkpoint,dataroot,batch,saveroot):
    
    # Create model instance
    timesteps = 100
    pc_params = PC_Params(in_channels=[3,128,128],out_features=[128,128,128],num_layers=3,timesteps=timesteps,alpha=0.1,beta=0.2,lambdax=0.1)
    net = PredictiveCoder(pc_params)
    net.fc3 = nn.Linear(128,2)
    net.load_state_dict(checkpoint['model_state_dict']) 
    net.to(device)
    
    # Load data
    transform_test = transforms.Compose([transforms.ToTensor(),])
    test_sets   = MyDataset2_class(dataroot,transform_test)
    testset_size = len(test_sets)
    test_loader   = torch.utils.data.DataLoader(test_sets, batch_size=batch,shuffle=False, num_workers=8, drop_last=False)

    
    since = time.time()
    net.eval()
    prob_square = {i: None for i in range(4)}
    for i, (images,labels) in enumerate(test_loader):
        print('batch {}/{}'.format(i+1, len(test_loader)))
        images, labels = images.to(device), labels.to(device)
        out = net(images)
        probs = torch.cat([out['Classification_at_time' + str(t)][:, 0:1] for t in range(net.timesteps)], dim=1).detach().to('cpu')

        for cl in range(4):
            vals = probs[labels == cl].clone()

            if prob_square[cl] is None:
                prob_square[cl] = vals
            else:
                prob_square[cl] = torch.cat([prob_square[cl], vals], dim=0)            
    print(prob_square[0].shape)
    time_eplapsed = time.time() - since
    torch.save(prob_square,saveroot)
    print('Training complete in {:.0f}m  {:.0f}s'.format(time_eplapsed // 60, time_eplapsed % 60))