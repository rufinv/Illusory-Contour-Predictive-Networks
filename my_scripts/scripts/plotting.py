import torch
import numpy as np
import matplotlib.pyplot as plt 


def plotting_pretrain(checkpoint,epoch_total,epochshow1,epochshow2,timesteps):


    plt.subplots_adjust(left=0.4, right=2.2, wspace = 0.8, hspace = 1)
    plt.subplot(131)
    xdata1 = np.arange(epoch_total)
    xdata2 = np.arange(timesteps)
    TV_loss = checkpoint['TV_loss']
    #Val_detailloss = checkpoint['Val_detailloss']
    Val_detailloss = checkpoint['Train_detailloss']

    plt.plot(xdata1,TV_loss[0,:epoch_total],label='train',color='k')
    plt.plot(xdata1,TV_loss[1,:epoch_total],label='validate',color='#006699')
    plt.legend()
    plt.title('Loss (train vs. validate)')
    plt.xlabel('epoch')
    plt.ylabel('loss')


    plt.subplot(263)
    plt.plot(Val_detailloss[epochshow1,0,:],color='#cc3300')
    plt.plot(Val_detailloss[epochshow1,1,:],color='#ff9900')
    plt.plot(Val_detailloss[epochshow1,2,:],color='#009900')
    plt.title(f'Epoch {epochshow1}')
    plt.xlabel('timesteps')
    plt.ylabel('loss')
    plt.subplot(264)
    plt.plot(Val_detailloss[epochshow1,0,:],color='#cc3300')
    plt.title('layer 1')
    plt.xlabel('timesteps')
    plt.subplot(265)
    plt.plot(Val_detailloss[epochshow1,1,:],color='#ff9900')
    plt.title('layer 2')
    plt.xlabel('timesteps')
    plt.subplot(266)
    plt.plot(Val_detailloss[epochshow1,2,:],color='#009900')
    plt.title('layer 3')
    plt.xlabel('timesteps')


    plt.subplot(269)
    plt.plot(Val_detailloss[epochshow2,0,:],color='#cc3300')
    plt.plot(Val_detailloss[epochshow2,1,:],color='#ff9900')
    plt.plot(Val_detailloss[epochshow2,2,:],color='#009900')
    plt.title(f'Epoch {epochshow2}')
    plt.xlabel('timesteps')
    plt.ylabel('loss')
    plt.subplot(2,6,10)
    plt.plot(Val_detailloss[epochshow2,0,:],color='#cc3300')
    plt.xlabel('timesteps')
    plt.subplot(2,6,11)
    plt.plot(Val_detailloss[epochshow2,1,:],color='#ff9900')
    plt.xlabel('timesteps')
    plt.subplot(2,6,12)
    plt.plot(Val_detailloss[epochshow2,2,:],color='#009900')
    plt.xlabel('timesteps')
    #plt.savefig('pre-train_a1000_net1.pdf',bbox_inches='tight',pad_inches=0)
    
    
def plotting_finetuning(checkpoint,epochs):   
    
    TV_loss = checkpoint['TV_loss']
    TV_acc = checkpoint['TV_acc']
    Val_detail_losscc = checkpoint['Val_detail_losscc']
    Val_detail_losscc = Val_detail_losscc[:epochs,:]
    val_losscc = np.mean(Val_detail_losscc,1)
    Val_detail_acc = checkpoint['Val_detail_acc']
    Val_detail_acc = Val_detail_acc[:epochs,:]
    val_acc = np.mean(Val_detail_acc,1)


    plt.figure() 
    plt.subplots_adjust(left=0.4, top= 1.3, right = 1, bottom = 0.1, wspace = 0.8, hspace = 0.5)
    plt.subplot(221)
    xdata = np.arange(epochs)
    plt.plot(xdata, TV_loss[0,:epochs], 'k',label='train',alpha=0.5) 
    plt.plot(xdata, TV_loss[1,:epochs], 'g',label='validate') 
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss (train vs. valid.)')
    plt.legend()
    
    plt.subplot(222)
    xdata = np.arange(epochs)
    plt.plot(xdata, TV_acc[0,:epochs], 'k',label='train',alpha=0.5) 
    plt.plot(xdata, TV_acc[1,:epochs], 'g',label='valid.') 
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    #plt.ylim(0.91,1)
    plt.title('Accuracy (train vs. validate)')
    plt.legend()
    #plt.savefig('../pic/finetune_net11.pdf',bbox_inches='tight',pad_inches=0)

    
    
    

def plotting_class(prob_square,saveflag,saveroot,img_num):
    
    timesteps = 100
    
    class0_prob_mean = np.mean(prob_square[0].numpy(), axis=0)
    class1_prob_mean = np.mean(prob_square[1].numpy(), axis=0)
    class2_prob_mean = np.mean(prob_square[2].numpy(), axis=0)
    class3_prob_mean = np.mean(prob_square[3].numpy(), axis=0)

    class0_prob_std = np.std(prob_square[0].numpy(),axis=0)/np.sqrt(img_num)
    class1_prob_std = np.std(prob_square[1].numpy(),axis=0)/np.sqrt(img_num)
    class2_prob_std = np.std(prob_square[2].numpy(),axis=0)/np.sqrt(img_num)
    class3_prob_std = np.std(prob_square[3].numpy(),axis=0)/np.sqrt(img_num)

    plt.figure() 
    plt.subplots_adjust(left=0.4, top= 1.2, right = 1.7, bottom = 0.5, wspace = 0.4, hspace = 0.5)
    plt.subplot(1,4,1)
    xdata = np.arange(timesteps)
    plt.errorbar(xdata,class0_prob_mean,label='Square',yerr=class2_prob_std,c='g',ecolor='g',errorevery=10)
    plt.errorbar(xdata,class1_prob_mean,label='Random', yerr=class1_prob_std,c='k',ecolor='k',errorevery=10)
    plt.errorbar(xdata,class2_prob_mean,label='All-out',yerr=class2_prob_std,c='#0072bd',ecolor='#0072bd',errorevery=10)
    plt.errorbar(xdata,class3_prob_mean,label='All-in', yerr=class3_prob_std,c='#d95319',ecolor='#d95319',errorevery=10)
    plt.legend()

    plt.xlabel('timesteps')
    plt.ylabel('prob')
    print(class3_prob_mean[99])
        #plt.ylim(0,0.14)
    if saveflag:
        plt.savefig(saveroot,bbox_inches='tight',pad_inches=0)
       
        

def plotting_fg(checkpoint,saveflag,saveroot,img_num):
    
    timesteps = 100
    n1_edge_square = checkpoint['edge_info_square']
    n1_edge_random = checkpoint['edge_info_random']
    n1_edge_Controls = checkpoint['edge_info_Controls']
    n1_edge_Ics = checkpoint['edge_info_Ics']
    n1m_edge_square = np.mean(n1_edge_square,axis=0)
    n1m_edge_random = np.mean(n1_edge_random,axis=0)
    n1m_edge_Controls = np.mean(n1_edge_Controls,axis=0)
    n1m_edge_Ics = np.mean(n1_edge_Ics,axis=0)
    n1s_edge_square = np.std(n1_edge_square,axis=0)/np.sqrt(img_num)
    n1s_edge_random = np.std(n1_edge_random,axis=0)/np.sqrt(img_num)
    n1s_edge_Controls = np.std(n1_edge_Controls,axis=0)/np.sqrt(img_num)
    n1s_edge_Ics = np.std(n1_edge_Ics,axis=0)/np.sqrt(img_num)
    
    

    xdata = np.arange(timesteps)
    plt.figure() 
    plt.subplots_adjust(left=0.1, top= 2, right = 1.5, bottom = 0.5, wspace = 0.5, hspace = 0.5)


    plt.subplot(241)
    plt.plot(xdata,np.ones(100)*n1m_edge_random[0],'k--',alpha = 0.7)
    plt.plot(xdata,np.ones(100)*n1m_edge_Controls[0],linestyle = 'dashed', c= '#0072bd',alpha = 0.7)
    plt.plot(xdata,np.ones(100)*n1m_edge_Ics[0],linestyle = 'dashed',c= '#d95319',alpha = 0.7)
    #plt.errorbar(xdata,n1m_edge_square,label='Square', yerr=n1s_edge_square,c='g',ecolor='g',errorevery=10)
    plt.errorbar(xdata,n1m_edge_random[1:],label='Random', yerr=n1s_edge_random[1:],c='k',ecolor='k',errorevery=10)
    plt.errorbar(xdata,n1m_edge_Controls[1:],label='All-out', yerr=n1s_edge_Controls[1:],c='#0072bd',ecolor='#0072bd',errorevery=10)
    plt.errorbar(xdata,n1m_edge_Ics[1:],label='All-in', yerr=n1s_edge_Ics[1:],c='#d95319',ecolor='#d95319',errorevery=10)
    plt.legend()
    plt.xlabel('timesteps')
    plt.ylabel('FG')
    plt.ylim(-0.015,0.065)
    if saveflag:
        plt.savefig(saveroot,bbox_inches='tight',pad_inches=0)
