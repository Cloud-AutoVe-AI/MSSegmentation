import numpy as np
import os
import glob
from torch.utils.data import Dataset
from PIL import Image
from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationCameraToPose, loadCalibrationRigid
from kitti360scripts.helpers.project import CameraPerspective, CameraFisheye    
from torchvision import io

EXTENSIONS = ['.jpg', '.png']
subfix_name = '_gtFine_trainIds.png'

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label_ori(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith(subfix_name)

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class KITTI360_CAM_LiDAR_seperable(Dataset):
    def __init__(self, dataset_root, co_transform=None, subset='train', subsamplingRate=1 ):

        self.dataset_root= dataset_root
        self.subset = subset
        print(self.dataset_root)
        filenameGT_path = self.dataset_root + '/semantic_labels/' + subset + '/*/semantic_trainids/*.png'
        filenamesGt = glob.glob(filenameGT_path)
        filenamesGt.sort()
        
        ##  학습의 효율을 위해 1/subsamplingRate 만 사용 ###
        for i in range(len(filenamesGt)-1, 0, -1):
            if i % subsamplingRate != 0:         
                del filenamesGt[i]
                
        
    
        num_images = len(filenamesGt)
        print(f'number of used {subset} images: {num_images}')           
        
        self.filenamesGt = filenamesGt
        self.co_transform = co_transform
        
    def __getitem__(self, index):
        filenameGt = self.filenamesGt[index]
        
        filenameLD = filenameGt.replace('/semantic_labels/'+ self.subset + '/', '/data_3d_raw/')        
        filenameLD = filenameLD.replace('/semantic_trainids/', '/velodyne_points/data/')
        filenameLD = filenameLD.replace('.png', '.bin')
        
        with open(filenameLD, 'rb') as f:
            LiDAR_points = np.fromfile(f, dtype=np.float32)
            LiDAR_points = np.reshape(LiDAR_points,[-1,4])        

        filename = filenameGt.replace('/semantic_labels/', '/semantic_images/')        
        filename = filename.replace('/semantic_trainids/', '/data_rect/')
        
        with open(filename, 'rb') as f:
            image = load_image(f).convert('RGB')            
            
        with open(filenameGt, 'rb') as f:
            label = load_image(f).convert('P')            
            
        if self.co_transform is not None:
            inputs, inputs2, label = self.co_transform(LiDAR_points, image,  label)

        return inputs, inputs2, label            
            
    def __len__(self):
        return len(self.filenamesGt)

class KITTI360_2CAMs(Dataset):
    def __init__(self, dataset_root, co_transform=None, subset='train', subsamplingRate=1 ):

        self.dataset_root= dataset_root
        self.subset = subset
        print(self.dataset_root)
        filenameGT_path = self.dataset_root + '/semantic_labels/' + subset + '/*/semantic_trainids/*.png'
        filenamesGt = glob.glob(filenameGT_path)
        filenamesGt.sort()
        
        ##  학습의 효율을 위해 1/subsamplingRate 만 사용 ###
        for i in range(len(filenamesGt)-1, 0, -1):
            if i % subsamplingRate != 0:         
                del filenamesGt[i]
                
        
    
        num_images = len(filenamesGt)
        print(f'number of used {subset} images: {num_images}')           
        
        self.filenamesGt = filenamesGt
        self.co_transform = co_transform
        
    def __getitem__(self, index):
        filenameGt = self.filenamesGt[index]
        

        filename = filenameGt.replace('/semantic_labels/', '/semantic_images/')        
        filename2 = filename.replace('/semantic_trainids/', '/data_rect_r/')
        filename = filename.replace('/semantic_trainids/', '/data_rect/')
        
        with open(filename, 'rb') as f:
            image_l = load_image(f).convert('RGB')            
        
        with open(filename2, 'rb') as f:
            image_r = load_image(f).convert('RGB')            
        
        with open(filenameGt, 'rb') as f:
            label = load_image(f).convert('P')            
            
        if self.co_transform is not None:
            inputs, inputs2, label = self.co_transform(image_l, image_r,  label)

        return inputs, inputs2, label            
            
    def __len__(self):
        return len(self.filenamesGt)        

class KITTI360_LiDAR(Dataset):
    def __init__(self, dataset_root, co_transform=None, subset='train', subsamplingRate=1 ):

        self.dataset_root= dataset_root
        self.subset = subset
        print(self.dataset_root)
        filenameGT_path = self.dataset_root + '/semantic_labels/' + subset + '/*/semantic_trainids/*.png'
        filenamesGt = glob.glob(filenameGT_path)
        filenamesGt.sort()
        
        ##  학습의 효율을 위해 1/subsamplingRate 만 사용 ###
        for i in range(len(filenamesGt)-1, 0, -1):
            if i % subsamplingRate != 0:         
                del filenamesGt[i]
                
        
    
        num_images = len(filenamesGt)
        print(f'number of used {subset} images: {num_images}')           
        
        self.filenamesGt = filenamesGt
        self.co_transform = co_transform
        
    def __getitem__(self, index):
        filenameGt = self.filenamesGt[index]
        
        filename = filenameGt.replace('/semantic_labels/'+ self.subset + '/', '/data_3d_raw/')        
        filename = filename.replace('/semantic_trainids/', '/velodyne_points/data/')
        filename = filename.replace('.png', '.bin')
        
        with open(filename, 'rb') as f:
            LiDAR_points = np.fromfile(f, dtype=np.float32)
            LiDAR_points = np.reshape(LiDAR_points,[-1,4])        
            
            
        with open(filenameGt, 'rb') as f:
            label = load_image(f).convert('P')            
            
        if self.co_transform is not None:
            LiDAR_points, label = self.co_transform(LiDAR_points, label)

        return LiDAR_points, label            
            
    def __len__(self):
        return len(self.filenamesGt)


class KITTI360_2CAMs_LiDAR2(Dataset):
    def __init__(self, transform=None, subset='train', args=None):    
        if args == None:
            subsamplingRate = 3
            datadir1 = '/home/tekken/dataset/KITTI360/semantic_data'
        else:
            subsamplingRate = args.subsamplingRate
            datadir1 = args.datadir1
        
        
        self.dataset_root= datadir1
        self.subset = subset
  
        
        print(f"Dataset root for {subset} set: {self.dataset_root+'/semantic_labels/' + subset}")
        if subset == 'train':
            filenameGT_path = self.dataset_root + '/semantic_labels/*/*/semantic_trainids/*.png'
        else:
            filenameGT_path = self.dataset_root + '/semantic_labels/' + subset + '/*/semantic_trainids/*.png'
        filenamesGt = glob.glob(filenameGT_path)
        filenamesGt.sort()
        
        ##  학습의 효율을 위해 1/n 만 사용 ###
        for i in range(len(filenamesGt)-1, 0, -1):
            if i % subsamplingRate != 0:         
                del filenamesGt[i]
    
        num_images = len(filenamesGt)
        print(f'number of used {subset} images: {num_images}')           
        
        self.filenamesGt = filenamesGt
        self.co_transform = transform
        
    def __getitem__(self, index):
        filenameGt = self.filenamesGt[index]
        filenameLD = filenameGt.replace('/semantic_data/semantic_labels/train/', '/data_3d_raw/')        
        filenameLD = filenameLD.replace('/semantic_data/semantic_labels/val/', '/data_3d_raw/')        
#         filenameLD = filenameGt.replace('/semantic_data/semantic_labels/'+ self.subset + '/', '/data_3d_raw/')        
        filenameLD = filenameLD.replace('/semantic_trainids/', '/velodyne_points/data/')
        filenameLD = filenameLD.replace('.png', '.bin')
        
        with open(filenameLD, 'rb') as f:
            LiDAR_points = np.fromfile(f, dtype=np.float32)
            LiDAR_points = np.reshape(LiDAR_points,[-1,4])        

        filename = filenameGt.replace('/semantic_labels/', '/semantic_images/')        
        filename2 = filename.replace('/semantic_trainids/', '/data_rect_r/')
        filename = filename.replace('/semantic_trainids/', '/data_rect/')
        
        image = io.read_image(filename)
        if image.shape[0] == 4:
            image = image[:3]           
      
        image_r = io.read_image(filename2)
        if image_r.shape[0] == 4:
            image_r = image_r[:3]      

        label = io.read_image(filenameGt)          
            
        if self.co_transform is not None:
            PriorTransData = [LiDAR_points, image, image_r, label]
            PostTransData = self.co_transform(PriorTransData)            

        return PostTransData         
            

         
        
    def __len__(self):
        return len(self.filenamesGt)           
