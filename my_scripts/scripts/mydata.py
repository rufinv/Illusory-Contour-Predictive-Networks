from PIL import Image
from torch.utils.data import Dataset

class MyDataset1(Dataset):
    def __init__(self, txt_path, transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1]), float(words[2])))
        self.imgs = imgs        
        self.transform = transform

    def __getitem__(self, index):
        fn, label,noise = self.imgs[index]
        img = Image.open(fn).convert('RGB')    
        if self.transform is not None:
            img = self.transform(img)  
        return img, label,noise

    def __len__(self):
        return len(self.imgs)
    

class MyDataset2_class(Dataset):
    def __init__(self, txt_path, transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs       
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')    
        if self.transform is not None:
            img = self.transform(img) 
        return img, label

    def __len__(self):
        return len(self.imgs)
    
class MyDataset3_fg(Dataset):
    def __init__(self, txt_path, transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1]), [float(words[2]), float(words[3]), float(words[4]), float(words[5]), float(words[6]), float(words[7]),float(words[8])]))
        self.imgs = imgs       
        self.transform = transform

    def __getitem__(self, index):
        fn, label, imginfo = self.imgs[index]
        img = Image.open(fn).convert('RGB')    
        if self.transform is not None:
            img = self.transform(img) 
        return img, label,imginfo[0],imginfo[1],imginfo[2],imginfo[3],imginfo[4],imginfo[5],imginfo[6] 

    def __len__(self):
        return len(self.imgs)