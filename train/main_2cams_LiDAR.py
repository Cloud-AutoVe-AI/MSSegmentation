####### import modules ######
import os
import random
import time
import numpy as np
import torch
import math
import losses as L
import importlib
from shutil import copyfile


from PIL import Image
from argparse import ArgumentParser

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ColorJitter, ToTensor, ToPILImage
import torchvision.transforms as T
from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationCameraToPose, loadCalibrationRigid
from kitti360scripts.helpers.project import CameraPerspective, CameraFisheye   

from dataset import KITTI360_2CAMs_LiDAR2
from transform import Relabel, ToLabel
from iouEval import iouEval


###### pre-processing images ###### 
class MyCoTransform(object):
    def __init__(self, augment=True, datadir='/home/tekken/dataset/KITTI360'):
        self.augment = augment
        self.datadir = datadir    

        self.label_map = np.arange(256)        
        self.label_map[0] = 0        
        self.label_map[1] = 0
        self.label_map[2] = 1        
        self.label_map[3] = 1 
        self.label_map[4] = 1
        self.label_map[5] = 2        
        self.label_map[6] = 2        
        self.label_map[7] = 2
        self.label_map[8] = 3        
        self.label_map[9] = 3 
        self.label_map[10] = 4
        self.label_map[11] = 5                
        self.label_map[12] = 5        
        self.label_map[13] = 6
        self.label_map[14] = 6        
        self.label_map[15] = 6 
        self.label_map[16] = 6
        self.label_map[17] = 6 
        self.label_map[18] = 6        
        self.label_map[255] = 7                   


    def encode(self, label):
        label = self.label_map[label]
        return torch.from_numpy(label)    
    
    def __call__(self, PriorTransData):
        points = PriorTransData[0]
        img_l = PriorTransData[1]
        img_r = PriorTransData[2]
        GTmask = PriorTransData[3]        
#         print(f"LiDAR: {points.shape}\nimg_r: {img_l.shape}\nimg_l: {img_r.shape}\nlabel: {GTmask.shape}\n")
        
        ### LiDAR point projection to camera coord. ###

        camera = CameraPerspective(root_dir= self.datadir)

        # cam_0 to velo
        fileCameraToVelo = os.path.join(self.datadir, 'calibration', 'calib_cam_to_velo.txt')
        TrCam0ToVelo = loadCalibrationRigid(fileCameraToVelo)        

        # all cameras to system center 
        fileCameraToPose = os.path.join(self.datadir, 'calibration', 'calib_cam_to_pose.txt')
        TrCamToPose = loadCalibrationCameraToPose(fileCameraToPose)      

        # velodyne to all cameras
        TrVeloToCam = {}
        for k, v in TrCamToPose.items():
            # Tr(cam_k -> velo) = Tr(cam_k -> cam_0) @ Tr(cam_0 -> velo)
            TrCamkToCam0 = np.linalg.inv(TrCamToPose['image_00']) @ TrCamToPose[k]
            TrCamToVelo = TrCam0ToVelo @ TrCamkToCam0
            # Tr(velo -> cam_k)
            TrVeloToCam[k] = np.linalg.inv(TrCamToVelo)    

        TrVeloToRect = np.matmul(camera.R_rect, TrVeloToCam['image_%02d' % 0])

        point_intensity = np.zeros(points[:,3].shape)
        point_intensity += points[:,3]
        points[:,3] = 1           

        # transfrom velodyne points to camera coordinate
        pointsCam = np.matmul(TrVeloToRect, points.T).T
        pointsCam = pointsCam[:,:3]
        # project to image space
        u,v, depth= camera.cam2image(pointsCam.T)
        u = u.astype(int)
        v = v.astype(int)

        # prepare depth map for visualization
        depthMap = np.zeros((camera.height, camera.width))
        intensityMap = np.zeros((camera.height, camera.width))  
        mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<camera.width), v>=0), v<camera.height)

        # visualize points within 30 meters
        mask = np.logical_and(np.logical_and(mask, depth>0), depth<50)
        depthMap[v[mask],u[mask]] = depth[mask]
        intensityMap[v[mask],u[mask]] = point_intensity[mask]
        
        depthMap = np.expand_dims(depthMap, axis=2)
        intensityMap = np.expand_dims(intensityMap, axis=2)
        
        LiDAR_points = np.concatenate( (depthMap, intensityMap), axis=2)
        LiDAR_points = ToTensor()(LiDAR_points)

        PostTransData = LiDAR_points, img_l, img_r, self.encode(GTmask.squeeze().numpy()).long()
        
        return PostTransData    
#     def __init__(self, augment=True, resize_width=1280, crop_height=512, crop_width=512):
#         self.augment = augment
#         self.resize_width = resize_width
#         self.crop_height = crop_height
#         self.crop_width = crop_width        
#         pass
#     def __call__(self, input, input_r, target):
#         re_width = int(self.resize_width)
#         re_height = int(input.size[1] * re_width/input.size[0] +0.5)        

#         if(self.augment):
#             # Random hflip
#             if (random.random() < 0.5):
#                 input = input.transpose(Image.FLIP_LEFT_RIGHT)
#                 input_r = input_r.transpose(Image.FLIP_LEFT_RIGHT)
#                 target = target.transpose(Image.FLIP_LEFT_RIGHT)
#             # color jitter
#             input = ColorJitter(hue=.2, saturation=.2)(input)
#             input_r = ColorJitter(hue=.2, saturation=.2)(input_r)
            
#             # resize image
#             resize_factor = (0.8 + 0.4 * random.random())
#             re_width = int(resize_factor * re_width + 0.5)
#             re_height = int(resize_factor * re_height + 0.5)                           

#         input =  Resize((int(re_height), int(re_width)), T.InterpolationMode.BILINEAR)(input)
#         input_r =  Resize((int(re_height), int(re_width)), T.InterpolationMode.BILINEAR)(input_r)
#         target = Resize((int(re_height), int(re_width)), T.InterpolationMode.NEAREST)(target)              
#         transX = random.randint(0, re_width-self.crop_width)            
#         transY = random.randint(0, re_height-self.crop_height)

#         if re_width < self.crop_width or re_height < self.crop_height:
#             print('crop size must be smaller than input size !')
#             return
        
#         input = input.crop((transX, transY, transX+self.crop_width, transY+self.crop_height))
#         input_r = input_r.crop((transX, transY, transX+self.crop_width, transY+self.crop_height))
#         target = target.crop((transX, transY, transX+self.crop_width, transY+self.crop_height))    
        
#         input_r =  Resize((int(self.crop_height/2), int(self.crop_width/2)), T.InterpolationMode.BILINEAR)(input_r)
        
#         input = ToTensor()(input)
#         input_r = ToTensor()(input_r)
#         target = ToLabel()(target)

#         target = Relabel(0, 1)(target)        
#         target = Relabel(1, 1)(target)
#         target = Relabel(2, 2)(target)
#         target = Relabel(3, 2)(target)
#         target = Relabel(4, 2)(target)
#         target = Relabel(5, 3)(target)
#         target = Relabel(6, 3)(target)
#         target = Relabel(7, 3)(target)
#         target = Relabel(8, 4)(target)
#         target = Relabel(9, 4)(target)
#         target = Relabel(10, 5)(target)
#         target = Relabel(11, 6)(target)
#         target = Relabel(12, 6)(target)
#         target = Relabel(13, 7)(target)
#         target = Relabel(14, 7)(target)
#         target = Relabel(15, 7)(target)
#         target = Relabel(16, 7)(target)
#         target = Relabel(17, 7)(target)
#         target = Relabel(18, 7)(target)
#         target = Relabel(255, 0)(target)
            
#         return input, input_r, target

def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
    own_state = model.state_dict()

    for name, param in own_state.items():
        if name not in state_dict:
            print('not loaded:', name)
            continue
        own_state[name].copy_(param)
    return model        

class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)

###### training procedure ######
def train(args, model):
    device = torch.device('cuda')
    weight = torch.ones(args.num_classes)
    weight[0] = 20.0
    weight[1] = 2.8149201869965	
    weight[2] = 7.8698215484619
    weight[3] = 9.5110931396484	
    weight[4] = 4.6323022842407	
    weight[5] = 7.8698215484619
    weight[6] = 9.5168733596802	
    weight[7] = 6.6616044044495	
   

    if args.cuda:
        weight = weight.cuda()
    criterion = CrossEntropyLoss2d(weight)
        
    assert os.path.exists(args.datadir1), "Error: datadir1 (dataset directory) could not be loaded"

    ### input data pre-processing ###
    co_transform = MyCoTransform(augment=True, datadir=args.datadir1)
    co_transform_val = MyCoTransform(augment=False, datadir=args.datadir1)
    dataset_train = KITTI360_2CAMs_LiDAR2( co_transform, 'train', args)
    dataset_val = KITTI360_2CAMs_LiDAR2(co_transform_val, 'val', args)
    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True,drop_last=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,drop_last=True)

    savedir = f'../save/{args.savedir}'

    automated_log_path = savedir + "/automated_log.txt"
    modeltxtpath = savedir + "/model.txt"
    
    if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))

    optimizer = Adam(model.parameters(), args.init_learningrate, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)
    start_epoch = 1
    best_acc = 0
    best_epoch = 1
    if args.resume:
        filenameCheckpoint = savedir + '/checkpoint.pth.tar'
        assert os.path.exists(filenameCheckpoint), "Error: resume option was used but checkpoint was not found in folder"
        checkpoint = torch.load(filenameCheckpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))
  
   
    lambda1 = lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9)  ## scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)                             ## scheduler 2

    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        epoch_loss = []
        time_train = []
     
        doIouVal =  args.iouVal

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        for step, PostTransData in enumerate(loader):

            start_time = time.time()

            LiDARpts = PostTransData[0]
            img = PostTransData[1]
            img2 = PostTransData[2]
            lbl = PostTransData[3]

            optimizer.zero_grad(set_to_none=True)

            LiDARpts = LiDARpts.to(device).float()
            img = img.to(device).float()  # .float()를 써서 Byte input <-> float weight 의 괴리 해결
            img2 = img2.to(device).float()  
            lbl = lbl.to(device)
            
            outputs = model(img, img2, LiDARpts)

            if args.loss_type=='CE':
                loss = criterion(outputs, lbl.long())
            if args.loss_type=='lovasz':
                outputs2 = torch.nn.functional.softmax(outputs, dim=1)     
                loss = L.lovasz_softmax(outputs2, lbl.long(), ignore=args.ignore_class, only_present=True) 
           
                
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        iouTrain = 0
        
        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        epoch_loss_val = []
        time_val = []

        
        iouEvalVal = iouEval(args.num_classes,ignoreIndex=args.ignore_class)

        for step, PostTransData in enumerate(loader_val):
            start_time = time.time()
            LiDARpts = PostTransData[0]
            img = PostTransData[1]
            img2 = PostTransData[2]
            lbl = PostTransData[3]
            
            LiDARpts = LiDARpts.to(device).float()
            img = img.to(device).float()  # .float()를 써서 Byte input <-> float weight 의 괴리 해결
            img2 = img2.to(device).float()  
            lbl = lbl.to(device)   
                
            outputs = model(img, img2, LiDARpts)
            
            if args.loss_type=='CE':
                loss = criterion(outputs, lbl.long())            
            if args.loss_type=='lovasz':
                outputs2 = torch.nn.functional.softmax(outputs, dim=1)            
                loss = L.lovasz_softmax(outputs2, lbl.long(), ignore=args.ignore_class, only_present=True) 
            
            
            epoch_loss_val.append(loss.item())
            time_val.append(time.time() - start_time)

            iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, lbl.data)

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                print(f'VAL loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))
                       

        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)

        iouVal = 0
        if (doIouVal):
            iouVal, iou_classes = iouEvalVal.getIoU()
            iouStr = '{:0.2f}'.format(iouVal*100) + '\033[0m'
            print ("EPOCH IoU on VAL set: ", iouStr, "%") 

        if iouVal == 0:
            current_acc = -average_epoch_loss_val
        else:
            current_acc = iouVal 
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)
        filenameCheckpoint = savedir + '/checkpoint.pth.tar'
        filenameBest = savedir + '/model_best.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        #SAVE MODEL AFTER EPOCH

        filename = f'{savedir}/model-{epoch:03}.pth'
        filenamebest = f'{savedir}/model_best.pth'
        
        if args.epochs_save > 0 and step > 0 and step % args.epochs_save == 0:
            torch.save(model.state_dict(), filename)
            print(f'save: {filename} (epoch: {epoch})')
        if (is_best):
            best_epoch = epoch
            torch.save(model.state_dict(), filenamebest)
            print(f'save: {filenamebest} (epoch: {epoch})')

            with open(savedir + "/best.txt", "w") as myfile:
                myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))   
         

        #SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        #Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr ))
        scheduler.step()    ## scheduler 2
        print ("Best mIoU is", best_acc.item(), "at", best_epoch , "-th epoch" ) 
    return(model)   #return model (convenience for encoder-decoder training)

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)


def main(args):
    savedir = f'../save/{args.savedir}'

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    #Load Model
    assert os.path.exists(args.model + ".py"), "Error: model definition not found"
    model_file = importlib.import_module(args.model)
    model = model_file.Net(args.num_classes)
    copyfile(args.model + ".py", savedir + '/' + "network.py")
    
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
    
    if args.load_init_model:                    
        filenameCheckpoint = args.loaddir + '/model_best.pth.tar'   
        weight_path = args.loaddir + '/model_best.pth'
        assert os.path.exists(filenameCheckpoint), "Error: load_init_model option was used but checkpoint was not found in folder"
        checkpoint = torch.load(filenameCheckpoint)
        load_my_state_dict(model, torch.load(weight_path))
#         model.load_state_dict(checkpoint['state_dict'], strict=False)             
        print("=> mIoU of loaded checkpoint : {})".format(checkpoint['best_acc']))           

    print("========== EnDecoder TRAINING ===========")    
    model = train(args, model)   #Train decoder
    print("========== TRAINING FINISHED ===========")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)  
    parser.add_argument('--model', default="network")

    parser.add_argument('--num-workers', type=int, default=12)
    parser.add_argument('--steps-loss', type=int, default=50)    
    parser.add_argument('--epochs-save', type=int, default=0)    #You can use this value to save model every X epochs
    parser.add_argument('--savedir', required=True)    
    parser.add_argument('--iouVal', action='store_true', default=True) # 
    parser.add_argument('--resume', action='store_true')    #Use this flag to load last checkpoint for training  
    parser.add_argument('--load_init_model', action='store_true')    #Use this flag to load certain checkpoint for training  
    parser.add_argument('--loaddir', default=None)
      
    parser.add_argument('--datadir1', default=None)
    parser.add_argument('--datadir2', default=None)
    parser.add_argument('--datadir3', default=None)
    parser.add_argument('--datadir4', default=None)    
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=24)

    parser.add_argument('--init_learningrate', type=float, default=5e-4)     
    parser.add_argument('--loss_type', default='lovasz')
    parser.add_argument('--crop_height', type=int, default=384)    
    parser.add_argument('--crop_width', type=int, default=384)        
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--ignore_class', type=int, default=-1)
    parser.add_argument('--resize_width', type=int, default=1280)    
    parser.add_argument('--subsamplingRate', type=int, default=10)  
    parser.add_argument('--dataset_name', default='KITTI360')
    parser.add_argument('--alpha', type=float, default=0.25)    
    parser.add_argument('--gamma', type=float, default=2.0)        
    
    main(parser.parse_args())
    
    
