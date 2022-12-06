import numpy as np
import os
import glob
from torch.utils.data import Dataset
from PIL import Image


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

class cityscapes(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')
        
        self.images_root += subset
        self.labels_root += subset

        print (self.images_root)
        #self.filenames = [image_basename(f) for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        #[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(".")) for f in fn]
        #self.filenamesGt = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

        self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)

class KITTI360(Dataset):
    def __init__(self, label_root, co_transform=None, subset='train', subsamplingRate=1 ):
        train_val = 0
        if subset == 'train':
            train_val = 1
        if subset == 'val':
            train_val = 2
        self.train_val = train_val
        self.label_root= label_root
        print(self.label_root)
#         filenameGT_path = self.label_root + '/data_2d_semantics/train/*/semantic/*.png'
        filenameGT_path = self.label_root + '/semantic_labels/' + subset + '/*/semantic_trainids/*.png'
#         print(filenameGT_path)
        filename_path = self.label_root + '/semantic_images/' + subset + '/*/data_rect/*.png'
        
        filenamesGt = glob.glob(filenameGT_path)
        filenamesGt.sort()
        
        ##  학습의 효율을 위해 1/subsamplingRate 만 사용 ###
        for i in range(len(filenamesGt)-1, 0, -1):
            if i % subsamplingRate != 0:         
                del filenamesGt[i]        
        
        self.filenamesGt = filenamesGt


        self.co_transform = co_transform
        
    def __getitem__(self, index):
        filenameGt = self.filenamesGt[index]

        filename = filenameGt.replace('/semantic_labels/', '/semantic_images/')        
        filename = filename.replace('/semantic_trainids/', '/data_rect/')
        
        with open(filename, 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(filenameGt, 'rb') as f:
            label = load_image(f).convert('P')            
            
        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, label            
            
    def __len__(self):
        return len(self.filenamesGt)


from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationCameraToPose, loadCalibrationRigid
from kitti360scripts.helpers.project import CameraPerspective, CameraFisheye    

class KITTI360_CAM_LiDAR(Dataset):
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
            inputs, label = self.co_transform(LiDAR_points, image,  label)

        return inputs, label            
            
    def __len__(self):
        return len(self.filenamesGt)

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